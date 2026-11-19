"""
Read-Only Dataset & High-Throughput DataLoader for ViMELD_Dataset on Google Drive.
Tuyệt đối KHÔNG chỉnh sửa dữ liệu gốc của người dùng.
Tối ưu hóa tốc độ đọc frame cho GPU A100 (sequential grab khi frame gần nhau + mmap cho .npy).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

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


class ViMELDEngagementDataset(Dataset):
    """
    Dataset chỉ đọc (Read-Only) tương thích với cấu trúc thư mục ViMELD_Dataset trên Google Drive.
    Tối ưu tốc độ I/O cho A100:
      - Đọc file .npy bằng `mmap_mode="r"` (zero-copy).
      - Khi trích xuất T frames từ video `<Session_ID>.mp4`, sử dụng `cap.grab()` tuần tự
        cho các frame kế tiếp nhau thay vì `cap.set(CAP_PROP_POS_FRAMES)` lặp đi lặp lại.
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
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_classes = num_classes
        self.default_video_width = default_video_width
        self.default_video_height = default_video_height
        self.use_full_scene_video = use_full_scene_video
        self.label_map = ENGAGEMENT_3CLASS_MAP if num_classes == 3 else ENGAGEMENT_4CLASS_MAP

        self._mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)

        self.samples: List[Dict[str, Any]] = []
        if manifest_path is not None and Path(manifest_path).exists():
            self._load_from_manifest(Path(manifest_path))
        elif self.root_dir.exists():
            self._index_from_vimeld_directory()

    @staticmethod
    def uniform_temporal_indices(total_frames: int, target_frames: int) -> np.ndarray:
        if total_frames <= 0:
            return np.zeros(target_frames, dtype=np.int64)
        return np.linspace(0, total_frames - 1, target_frames, dtype=np.int64)

    def _load_from_manifest(self, manifest_path: Path) -> None:
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
        split_file = self.root_dir / "sessions_summary_train_valid_test.json"
        allowed_sessions: Optional[set] = None
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if self.split in splits_data:
                allowed_sessions = set(splits_data[self.split])

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
        if sample.get("bbox_norm") is not None:
            boxes_all = np.asarray(sample["bbox_norm"], dtype=np.float32)
            if boxes_all.ndim == 1 and boxes_all.size == 4:
                boxes_all = np.tile(boxes_all[None, :], (self.num_frames, 1))
            idx = self.uniform_temporal_indices(len(boxes_all), self.num_frames)
            boxes = np.clip(boxes_all[idx], 0.0, 1.0)
            return boxes, idx.tolist(), []

        seg_dir = Path(sample["seg_dir"])
        frame_files = sorted(seg_dir.glob("frame_*.png"))
        skel_path = Path(sample.get("skeleton_json", ""))

        boxes_list: List[List[float]] = []
        video_frame_ids: List[int] = []

        if skel_path.exists():
            with open(skel_path, "r", encoding="utf-8") as f:
                skel_data = json.load(f)
            for fm in skel_data.get("frames", []):
                x1, y1, x2, y2 = fm.get("body_bbox_xyxy_video_from_meta", [0.0, 0.0, 0.0, 0.0])
                boxes_list.append([
                    float(x1) / self.default_video_width,
                    float(y1) / self.default_video_height,
                    float(x2) / self.default_video_width,
                    float(y2) / self.default_video_height,
                ])
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
        if cv2 is not None:
            resized = cv2.resize(
                img_rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        elif Image is not None:
            resized = np.asarray(
                Image.fromarray(img_rgb).resize((self.image_size, self.image_size))
            )
        else:
            raise RuntimeError("Cần cài đặt opencv-python hoặc Pillow để xử lý ảnh.")

        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self._mean) / self._std

    def _read_video_frames_fast(
        self, video_path: str, video_frame_ids: List[int]
    ) -> List[torch.Tensor]:
        """
        Tối ưu hóa đọc video H.264 cho A100:
        Chỉ gọi `cap.set()` một lần tại frame đầu tiên nếu các frame tăng dần và gần nhau,
        dùng `cap.grab()` + `cap.retrieve()` để tránh keyframe re-decoding overhead.
        """
        frames_tensors: List[torch.Tensor] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames_tensors

        target_ids = [max(0, int(fid)) for fid in video_frame_ids]
        if target_ids:
            first_fid = target_ids[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_fid)
            current_pos = first_fid

            for fid in target_ids:
                if fid < current_pos or (fid - current_pos) > 60:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                    current_pos = fid

                while current_pos < fid:
                    if not cap.grab():
                        break
                    current_pos += 1

                ok, frame_bgr = cap.read()
                current_pos += 1
                if ok and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_tensors.append(self._preprocess_frame(frame_rgb))

        cap.release()
        return frames_tensors

    def _load_frames(
        self,
        sample: Dict[str, Any],
        video_frame_ids: List[int],
        sampled_pngs: List[Path],
    ) -> torch.Tensor:
        clip_path = sample.get("clip_path", "")
        if clip_path and Path(clip_path).exists() and clip_path.endswith(".npy"):
            arr = np.load(clip_path, mmap_mode="r")
            idx = self.uniform_temporal_indices(len(arr), self.num_frames)
            return torch.stack(
                [self._preprocess_frame(np.asarray(arr[i], dtype=np.uint8)) for i in idx],
                dim=0,
            )

        video_path = sample.get("video_path", "")
        if self.use_full_scene_video and video_path and Path(video_path).exists() and cv2 is not None:
            frames_tensors = self._read_video_frames_fast(video_path, video_frame_ids)
            if len(frames_tensors) == self.num_frames:
                return torch.stack(frames_tensors, dim=0)

        if sampled_pngs:
            frames_tensors = []
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

        return torch.zeros((self.num_frames, 3, self.image_size, self.image_size), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        boxes, video_frame_ids, sampled_pngs = self._load_boxes_and_frame_ids_from_segment(sample)
        frames = self._load_frames(sample, video_frame_ids, sampled_pngs)
        return {
            "frames": frames,
            "boxes": torch.from_numpy(boxes).float(),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "sample_id": sample.get("sample_id", f"sample_{index}"),
        }
