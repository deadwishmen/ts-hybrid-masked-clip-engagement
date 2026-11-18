"""
T-S Hybrid Masked Transformer for Student Engagement Recognition
================================================================
Thiết kế dựa trên:
1. Nghiên cứu Falcon Perception (arXiv:2603.27365 - https://arxiv.org/abs/2603.27365):
   - Unified Dense Transformer với Early Fusion & Hybrid Attention Masking.
   - Coordinate MLP mã hóa quỹ đạo tọa độ liên tục [x_min, y_min, x_max, y_max] in [0, 1]^4
     lên không gian nhúng chiều cao D=768 (Chain-of-Perception localization).
   - Inter-query / Spatiotemporal Masking cô lập luồng thông tin tránh nhiễu chéo và OOM.
2. Cấu trúc dữ liệu thực tế ViMELD_Dataset trên Google Drive (Chế độ READ-ONLY tuyệt đối):
   - Hỗ trợ đọc trực tiếp từ thư mục gốc ViMELD_Dataset (sessions_summary_train_valid_test.json,
     <Session_ID>.mp4, <Session_ID>_list_objects.csv, skeleton_vitpose_plus_l_upper_body.json,
     behaviors.json) hoặc từ bảng tổng hợp manifest.parquet / manifest.csv.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import CLIPVisionConfig, CLIPVisionModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ==============================================================================
# 0. ÁNH XẠ NHÃN ENGAGEMENT & HẰNG SỐ DỮ LIỆU ViMELD_Dataset
# ==============================================================================

# Bộ dữ liệu ViMELD gốc có 4 nhãn: eng_disengaged, eng_normal, eng_engaged, eng_very_engaged.
# Quy về 3 lớp (Low, Medium, High Engagement) theo yêu cầu Module 4:
ENGAGEMENT_3CLASS_MAP: Dict[str, int] = {
    "eng_disengaged": 0,    # Low Engagement
    "Low": 0,
    "low": 0,
    "eng_normal": 1,        # Medium Engagement
    "Medium": 1,
    "medium": 1,
    "eng_engaged": 2,       # High Engagement
    "eng_very_engaged": 2,  # High Engagement (gộp chung với eng_engaged)
    "High": 2,
    "high": 2,
}

ENGAGEMENT_4CLASS_MAP: Dict[str, int] = {
    "eng_disengaged": 0,
    "eng_normal": 1,
    "eng_engaged": 2,
    "eng_very_engaged": 3,
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ==============================================================================
# 1. READ-ONLY DATASET & DATALOADER CHO ViMELD_Dataset
# ==============================================================================

class ViMELDEngagementDataset(Dataset):
    """
    Dataset chỉ đọc (Read-Only) tương thích với cấu trúc thư mục ViMELD_Dataset trên Google Drive.
    Tuyệt đối KHÔNG ghi hay sửa bất kỳ tệp nào trong thư mục dữ liệu của người dùng.

    Hỗ trợ 2 chế độ đọc dữ liệu (tự động nhận diện):
    1. Chế độ Manifest (`manifest_path` trỏ tới `manifest.parquet` hoặc `manifest.csv`):
       Đọc các cột `clip_path`, `scene_image_path`, `bbox_norm` (hoặc `bbox_pixel` chia cho
       `frame_width=3200`, `frame_height=1800`), và `engagement_label`.
    2. Chế độ Thư mục Gốc `ViMELD_Dataset` (`root_dir`):
       - Đọc `sessions_summary_train_valid_test.json` để lọc các session theo `split` ('train', 'valid', 'test').
       - Duyệt `<Date>/Period <P>/Angle <A>/<Session_ID>/<Session_ID>_rgb/ID_xxxxx/seg_<idx>_<start>_<end>/`.
       - Đọc `skeleton_vitpose_plus_l_upper_body.json` (hoặc `<Session_ID>_list_objects.csv`) để lấy
         tọa độ Bounding Box chuẩn hóa [x_min, y_min, x_max, y_max] in [0, 1] qua ByteTrack ID.
       - Đọc nhãn `eng_*` từ `behaviors.json` khớp theo `roi_id` (`ID_xxxxx`) và khoảng thời gian `[start_sec, end_sec]`.
       - Trích xuất T frames toàn cảnh lớp học từ video `<Session_ID>.mp4` tại đúng `source_video_frame`
         (hoặc đọc từ các frame PNG có sẵn nếu `use_full_scene_video=False`).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 224,
        num_classes: int = 3,
        manifest_path: Optional[Union[str, Path]] = None,
        default_video_width: int = 3200,
        default_video_height: int = 1800,
        use_full_scene_video: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames  # T frames (ví dụ: T=8 lấy mẫu đều từ 16 frames của segment)
        self.image_size = image_size
        self.num_classes = num_classes
        self.default_video_width = default_video_width
        self.default_video_height = default_video_height
        self.use_full_scene_video = use_full_scene_video
        self.label_map = ENGAGEMENT_3CLASS_MAP if num_classes == 3 else ENGAGEMENT_4CLASS_MAP

        self.samples: List[Dict[str, Any]] = []
        if manifest_path is not None and Path(manifest_path).exists():
            self._load_from_manifest(Path(manifest_path))
        elif self.root_dir.exists():
            self._index_from_vimeld_directory()

    @staticmethod
    def uniform_temporal_indices(total_frames: int, target_frames: int) -> np.ndarray:
        """Lấy mẫu đều T frames từ chuỗi gốc (ví dụ lấy 8 frames từ 16 frames của mỗi segment)."""
        if total_frames <= 0:
            return np.zeros(target_frames, dtype=np.int64)
        return np.linspace(0, total_frames - 1, target_frames, dtype=np.int64)

    def _load_from_manifest(self, manifest_path: Path) -> None:
        """Đọc danh sách sample ở chế độ chỉ đọc từ manifest.parquet hoặc manifest.csv."""
        if pd is None:
            raise ImportError("Cần cài đặt pandas để đọc file manifest.")
        if manifest_path.suffix == ".parquet":
            df = pd.read_parquet(manifest_path)
        else:
            df = pd.read_csv(manifest_path)

        if "split" in df.columns and self.split:
            df = df[df["split"] == self.split].reset_index(drop=True)

        for _, row in df.iterrows():
            raw_label = str(row.get("engagement_label", ""))
            if raw_label not in self.label_map:
                continue

            bbox_norm = row.get("bbox_norm", None)
            if isinstance(bbox_norm, str):
                bbox_norm = ast.literal_eval(bbox_norm)

            self.samples.append({
                "sample_id": row.get("sample_id", ""),
                "session": row.get("session", ""),
                "object_id": row.get("object_id", ""),
                "clip_path": str(row.get("clip_path", "")),
                "scene_image_path": str(row.get("scene_image_path", "")),
                "video_path": str(row.get("video_path", "")),
                "bbox_norm": bbox_norm,
                "label": self.label_map[raw_label],
                "raw_label": raw_label,
            })

    def _index_from_vimeld_directory(self) -> None:
        """
        Duyệt cây thư mục ViMELD_Dataset ở chế độ READ-ONLY để ghép nối:
        - sessions_summary_train_valid_test.json
        - behaviors.json (nhãn eng_*)
        - skeleton_vitpose_plus_l_upper_body.json (tọa độ bbox trên video gốc 3200x1800)
        - <Session_ID>.mp4 hoặc 16 frames trong thư mục seg_*
        """
        split_file = self.root_dir / "sessions_summary_train_valid_test.json"
        allowed_sessions: Optional[set] = None
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if self.split in splits_data:
                allowed_sessions = set(splits_data[self.split])

        # Tìm các file behaviors.json trong cây thư mục
        for behaviors_path in sorted(self.root_dir.rglob("behaviors.json")):
            session_dir = behaviors_path.parent
            session_id = session_dir.name
            if allowed_sessions is not None and session_id not in allowed_sessions:
                continue

            rgb_root = session_dir / f"{session_id}_rgb"
            if not rgb_root.exists():
                continue

            video_path = session_dir.parent / f"{session_id}.mp4"

            with open(behaviors_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)

            # Lập chỉ mục nhãn engagement theo (roi_id, start_time, end_time)
            eng_by_roi: Dict[str, List[Tuple[float, float, str]]] = {}
            for ann in annotations:
                roi_id = ann.get("roi_id", "")
                labels = ann.get("labels", [])
                eng_labels = [lb for lb in labels if lb.startswith("eng_")]
                if not eng_labels:
                    continue
                eng_label = eng_labels[0]
                if eng_label not in self.label_map:
                    continue
                st = float(ann.get("start_time", 0.0))
                et = float(ann.get("end_time", 0.0))
                eng_by_roi.setdefault(roi_id, []).append((st, et, eng_label))

            # Duyệt các học sinh ID_xxxxx trong <Session_ID>_rgb
            for student_dir in sorted(rgb_root.glob("ID_*")):
                roi_id = student_dir.name
                if roi_id not in eng_by_roi:
                    continue

                for seg_dir in sorted(student_dir.glob("seg_*")):
                    parts = seg_dir.name.split("_")
                    if len(parts) < 4:
                        continue
                    try:
                        seg_start = float(parts[-2])
                        seg_end = float(parts[-1])
                    except ValueError:
                        continue

                    # Tìm nhãn engagement giao với [seg_start, seg_end]
                    matched_label: Optional[str] = None
                    seg_mid = 0.5 * (seg_start + seg_end)
                    for st, et, eng_lb in eng_by_roi[roi_id]:
                        if st <= seg_mid <= et or (max(st, seg_start) < min(et, seg_end)):
                            matched_label = eng_lb
                            break

                    if matched_label is None:
                        continue

                    self.samples.append({
                        "sample_id": f"{session_id}_{roi_id}_{seg_dir.name}",
                        "session": session_id,
                        "object_id": roi_id,
                        "seg_dir": str(seg_dir),
                        "video_path": str(video_path) if video_path.exists() else "",
                        "skeleton_json": str(seg_dir / "skeleton_vitpose_plus_l_upper_body.json"),
                        "label": self.label_map[matched_label],
                        "raw_label": matched_label,
                    })

    def _load_boxes_and_frame_ids_from_segment(
        self, sample: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[int], List[Path]]:
        """
        Trích xuất T tọa độ Bounding Box đã chuẩn hóa [x_min, y_min, x_max, y_max] in [0, 1]
        và danh sách frame từ segment.
        """
        # Trường hợp 1: Đã có sẵn bbox_norm từ manifest
        if sample.get("bbox_norm") is not None:
            boxes_all = np.asarray(sample["bbox_norm"], dtype=np.float32)
            if boxes_all.ndim == 1 and boxes_all.size == 4:
                boxes_all = np.tile(boxes_all[None, :], (self.num_frames, 1))
            idx = self.uniform_temporal_indices(len(boxes_all), self.num_frames)
            boxes = np.clip(boxes_all[idx], 0.0, 1.0)
            return boxes, idx.tolist(), []

        # Trường hợp 2: Đọc từ skeleton_vitpose_plus_l_upper_body.json trong thư mục seg_*
        seg_dir = Path(sample["seg_dir"])
        frame_files = sorted(seg_dir.glob("frame_*.png"))
        skel_path = Path(sample.get("skeleton_json", ""))

        boxes_list: List[List[float]] = []
        video_frame_ids: List[int] = []

        if skel_path.exists():
            with open(skel_path, "r", encoding="utf-8") as f:
                skel_data = json.load(f)
            frames_meta = skel_data.get("frames", [])
            for fm in frames_meta:
                bbox_xyxy = fm.get("body_bbox_xyxy_video_from_meta", [0.0, 0.0, 0.0, 0.0])
                x1, y1, x2, y2 = bbox_xyxy
                norm_box = [
                    float(x1) / self.default_video_width,
                    float(y1) / self.default_video_height,
                    float(x2) / self.default_video_width,
                    float(y2) / self.default_video_height,
                ]
                boxes_list.append(norm_box)
                video_frame_ids.append(int(fm.get("source_video_frame", 0)))

        total_available = max(len(boxes_list), len(frame_files), 1)
        idx = self.uniform_temporal_indices(total_available, self.num_frames)

        if boxes_list:
            boxes_arr = np.asarray(boxes_list, dtype=np.float32)
            boxes = np.clip(boxes_arr[idx], 0.0, 1.0)
            sampled_video_frames = [video_frame_ids[i] for i in idx]
        else:
            boxes = np.zeros((self.num_frames, 4), dtype=np.float32)
            sampled_video_frames = idx.tolist()

        sampled_pngs = [frame_files[i] for i in idx] if frame_files else []
        return boxes, sampled_video_frames, sampled_pngs

    def _preprocess_frame(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Resize ảnh về (H=224, W=224) và chuẩn hóa theo chuẩn CLIP ViT."""
        if cv2 is not None:
            resized = cv2.resize(
                img_rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        elif Image is not None:
            pil_img = Image.fromarray(img_rgb).resize((self.image_size, self.image_size))
            resized = np.asarray(pil_img)
        else:
            raise RuntimeError("Cần cài đặt opencv-python hoặc Pillow để xử lý ảnh.")

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def _load_frames(
        self,
        sample: Dict[str, Any],
        video_frame_ids: List[int],
        sampled_pngs: List[Path],
    ) -> torch.Tensor:
        """
        Đọc T frames (T, 3, 224, 224).
        Ưu tiên trích xuất frame toàn cảnh lớp học từ <Session_ID>.mp4 nếu use_full_scene_video=True
        và có file video; ngược lại đọc từ file .npy hoặc các file frame_xxx.png trong segment.
        """
        frames_tensors: List[torch.Tensor] = []

        # 1. Đọc từ file .npy nếu sử dụng Restructured_Dataset_NPY_Engagement
        clip_path = sample.get("clip_path", "")
        if clip_path and Path(clip_path).exists() and clip_path.endswith(".npy"):
            arr = np.load(clip_path, mmap_mode="r")  # Chế độ đọc mmap tiết kiệm RAM
            idx = self.uniform_temporal_indices(len(arr), self.num_frames)
            for i in idx:
                frames_tensors.append(self._preprocess_frame(np.asarray(arr[i], dtype=np.uint8)))
            return torch.stack(frames_tensors, dim=0)

        # 2. Đọc frame toàn cảnh lớp học từ video <Session_ID>.mp4
        video_path = sample.get("video_path", "")
        if self.use_full_scene_video and video_path and Path(video_path).exists() and cv2 is not None:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                for fid in video_frame_ids:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(fid)))
                    ok, frame_bgr = cap.read()
                    if ok and frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        frames_tensors.append(self._preprocess_frame(frame_rgb))
                cap.release()
            if len(frames_tensors) == self.num_frames:
                return torch.stack(frames_tensors, dim=0)
            frames_tensors.clear()

        # 3. Fallback: đọc từ các file frame_xxx.png trong thư mục seg_*
        if sampled_pngs:
            for png_path in sampled_pngs:
                if cv2 is not None:
                    bgr = cv2.imread(str(png_path))
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif Image is not None:
                    rgb = np.asarray(Image.open(png_path).convert("RGB"))
                else:
                    rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                frames_tensors.append(self._preprocess_frame(rgb))
            return torch.stack(frames_tensors, dim=0)

        # Nếu chạy kiểm thử không có file ảnh vật lý
        return torch.zeros((self.num_frames, 3, self.image_size, self.image_size), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        boxes, video_frame_ids, sampled_pngs = self._load_boxes_and_frame_ids_from_segment(sample)
        frames = self._load_frames(sample, video_frame_ids, sampled_pngs)
        label = int(sample["label"])

        return {
            "frames": frames,                                      # Tensor: (T, 3, 224, 224)
            "boxes": torch.from_numpy(boxes).float(),              # Tensor: (T, 4) chuẩn hóa in [0, 1]
            "label": torch.tensor(label, dtype=torch.long),        # Tensor: ()
            "sample_id": sample.get("sample_id", f"sample_{index}"),
        }


