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


# ==============================================================================
# 2. MODULE 1: MÃ HÓA ĐẦU VÀO (INPUT ENCODING MODULE)
# ==============================================================================

class CoordinateMLP(nn.Module):
    """
    Mạng Coordinate MLP: Linear(4 -> D) -> GELU -> Linear(D -> D)
    Nâng tọa độ Bounding Box [x_min, y_min, x_max, y_max] đã chuẩn hóa từ (T, 4) lên (T, D).
    Lấy cảm hứng từ Coordinate & Size Encoder trong Falcon Perception (arXiv:2603.27365, Section 2.2).
    """

    def __init__(self, in_features: int = 4, embed_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boxes: Tensor kích thước (B, T, 4) chứa [x_min, y_min, x_max, y_max] đã chuẩn hóa.
        Returns:
            E_box: Tensor kích thước (B, T, D).
        """
        return self.mlp(boxes)


class InputEncodingModule(nn.Module):
    """
    Module 1 — Mã hóa Đầu vào:
    Quy 3 thành phần đầu vào của 1 sample (1 học sinh trong 1 clip dài T frames)
    về cùng không gian nhúng có số chiều D = 768 (CLIP ViT-B/16):
      1. Task Token (E_task): Learnable Vector (1, D) -> mở rộng theo batch thành (B, 1, D).
      2. Box Trajectory (E_box): (B, T, 4) qua CoordinateMLP -> (B, T, D).
      3. Image Patches (E_patch): (B, T, C, H, W) qua Conv2D Patch Embedding của CLIP ViT
         (kernel=16x16, stride=16x16) -> (B, T, N, D) với N = (224/16)^2 = 196 patches/frame.
    """

    def __init__(
        self,
        patch_embedding_conv: nn.Conv2d,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Task Token (E_task): Learnable Vector khởi tạo ngẫu nhiên kích thước (1, D)
        self.task_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 2. Coordinate MLP cho Box Trajectory (E_box)
        self.coord_mlp = CoordinateMLP(in_features=4, embed_dim=embed_dim)

        # 3. Lớp Conv2D Patch Embedding mượn từ CLIP ViT-B/16 (đóng băng theo chiến lược PEFT)
        self.patch_embedding = patch_embedding_conv
        for param in self.patch_embedding.parameters():
            param.requires_grad = False

    def forward(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frames: (B, T, C, H, W) với H=W=224, C=3.
            boxes:  (B, T, 4) chứa [x_min, y_min, x_max, y_max] đã chuẩn hóa.
        Returns:
            e_task:  (B, 1, D)
            e_box:   (B, T, D)
            e_patch: (B, T, N, D) với N=196, D=768.
        """
        batch_size, num_frames, channels, height, width = frames.shape

        # 1. Task Token: (1, D) -> (B, 1, D)
        e_task = self.task_token.unsqueeze(0).expand(batch_size, -1, -1)

        # 2. Box Trajectory: (B, T, 4) -> (B, T, D)
        e_box = self.coord_mlp(boxes)

        # 3. Image Patches: gộp (B*T, C, H, W) chạy qua Conv2D của CLIP ViT
        frames_flat = frames.reshape(batch_size * num_frames, channels, height, width)
        # patch_embeds: (B*T, D, H/16, W/16) = (B*T, 768, 14, 14)
        patch_embeds = self.patch_embedding(frames_flat)
        # Trải phẳng không gian 14x14 -> N=196: (B*T, D, 196) -> (B*T, 196, D)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        num_patches = patch_embeds.shape[1]
        # Khôi phục chiều thời gian T: (B, T, N, D)
        e_patch = patch_embeds.reshape(batch_size, num_frames, num_patches, self.embed_dim)

        return e_task, e_box, e_patch


# ==============================================================================
# 3. MODULE 2: NHÚNG KHÔNG GIAN & THỜI GIAN (SPATIOTEMPORAL INJECTION)
# ==============================================================================

class SpatiotemporalInjectionModule(nn.Module):
    """
    Module 2 — Nhúng Không gian & Thời gian và Ghép chuỗi (Sequence Concatenation):
      - Spatial Positional Embedding (P_spatial): Mượn trực tiếp từ CLIP ViT, kích thước (N, D) = (196, 768).
        Cộng P_spatial vào các Image Patches ở từng frame.
      - Temporal Positional Embedding (P_temporal): Learnable Parameter kích thước (T, D).
        + Cộng P_temporal[t] vào Box Token thứ t.
        + Cộng P_temporal[t] vào tất cả N Image Patches thuộc frame thứ t.
      - Ghép chuỗi (Sequence Concatenation):
        Sequence = [E_task] (+) E_box (+) E_patch
        Chiều dài chuỗi: L = 1 + T + (T * N). Với T=8, N=196 -> L = 1 + 8 + 1568 = 1577 tokens.
    """

    def __init__(
        self,
        pretrained_spatial_pos_embed: torch.Tensor,
        num_frames: int = 8,
        num_patches: int = 196,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # P_spatial: (N, D) mượn trực tiếp từ CLIP ViT (bỏ vị trí 0 của class token gốc trong CLIP)
        if pretrained_spatial_pos_embed.shape[0] == num_patches + 1:
            spatial_weight = pretrained_spatial_pos_embed[1:, :].detach().clone()
        elif pretrained_spatial_pos_embed.shape[0] == num_patches:
            spatial_weight = pretrained_spatial_pos_embed.detach().clone()
        else:
            raise ValueError(
                f"Kích thước P_spatial không khớp: nhận {pretrained_spatial_pos_embed.shape}, "
                f"kỳ vọng ({num_patches}, {embed_dim}) hoặc ({num_patches + 1}, {embed_dim})."
            )
        # Đóng băng P_spatial của CLIP backbone theo chiến lược PEFT
        self.register_buffer("p_spatial", spatial_weight)

        # P_temporal: Learnable Parameter kích thước (T, D) được cập nhật qua Cross-Entropy Loss
        self.p_temporal = nn.Parameter(torch.randn(num_frames, embed_dim) * 0.02)

    def forward(
        self,
        e_task: torch.Tensor,
        e_box: torch.Tensor,
        e_patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            e_task:  (B, 1, D)
            e_box:   (B, T, D)
            e_patch: (B, T, N, D)
        Returns:
            sequence: (B, L, D) với L = 1 + T + T*N.
        """
        batch_size, t_frames, n_patches, d_dim = e_patch.shape
        if t_frames > self.num_frames:
            raise ValueError(
                f"Số frame đầu vào T={t_frames} vượt quá kích thước P_temporal={self.num_frames}."
            )

        p_temp = self.p_temporal[:t_frames, :]  # (T, D)

        # 1. Cộng P_temporal[t] vào Box Token thứ t: (B, T, D) + (1, T, D)
        e_box_injected = e_box + p_temp.unsqueeze(0)

        # 2. Cộng P_spatial (1, 1, N, D) và P_temporal[t] (1, T, 1, D) vào Image Patches (B, T, N, D)
        e_patch_injected = (
            e_patch
            + self.p_spatial.unsqueeze(0).unsqueeze(0)
            + p_temp.unsqueeze(0).unsqueeze(2)
        )

        # 3. Trải phẳng (flatten) các patch từ (B, T, N, D) -> (B, T*N, D)
        e_patch_flat = e_patch_injected.reshape(batch_size, t_frames * n_patches, d_dim)

        # 4. Nối 3 thành phần thành một chuỗi duy nhất: [E_task] (+) E_box (+) E_patch
        sequence = torch.cat([e_task, e_box_injected, e_patch_flat], dim=1)
        return sequence


# ==============================================================================
# 4. MODULE 3: T-S HYBRID MASKED TRANSFORMER & CHIẾN LƯỢC LoRA
# ==============================================================================

class LoRALinear(nn.Module):
    """
    Adapter LoRA (Low-Rank Adaptation) bơm vào ma trận Query (Q) và Value (V)
    của các lớp Transformer trong CLIP ViT-B/16.
    Trọng số gốc W_0 bị đóng băng hoàn toàn; chỉ huấn luyện hai ma trận hạng thấp A và B:
        h = W_0 x + (alpha / r) * B(A(x))
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Khởi tạo Kaiming cho A và 0 cho B để ΔW = 0 tại bước đầu tiên
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out


def build_ts_hybrid_attention_mask(
    num_frames: int = 8,
    num_patches: int = 196,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Xây dựng ma trận T-S Hybrid Attention Mask kích thước (L, L) với L = 1 + T + T*N.
    Quy ước cấu trúc chuỗi:
      - Index 0: Task Token
      - Index 1 .. T: Box Tokens của frame t in [0, T-1] (index = 1 + t)
      - Index 1 + T + t*N .. 1 + T + (t+1)*N - 1: N Image Patches thuộc frame t

    Quy tắc kiểm soát "ai được nhìn thấy ai" (XAI - Explainable AI):
      1. Task Token (idx 0):
         Được quyền nhìn chính nó (idx 0) & toàn bộ chuỗi Box Tokens (idx 1..T).
         Lý do XAI: Hút thông tin quỹ đạo hành động (ví dụ: gục đầu, giơ tay) để ra quyết định.
      2. Box Token thứ t (idx 1 + t):
         Được quyền nhìn tất cả Box Tokens xuyên thời gian T (idx 1..T, bao gồm chính nó)
         & tất cả các Image Patches cùng frame t.
         Lý do XAI: Hiểu bối cảnh xung quanh học sinh đó ở frame t, đồng thời liên kết
         sự thay đổi tư thế theo thời gian.
      3. Patch (t, i) (thuộc frame t):
         Chỉ được quyền nhìn Box Token cùng frame t (idx 1 + t) & các Patches khác cùng frame t.
         Lý do XAI: Tạo ngữ cảnh xã hội (social context) của lớp học tại một thời điểm,
         giới hạn không nhìn xuyên frame để tránh nhiễu và OOM.

    Returns:
        allowed_mask: Tensor kiểu bool kích thước (L, L), trong đó True = được phép nhìn (attend),
                      False = bị khóa (masked out).
    """
    seq_len = 1 + num_frames + (num_frames * num_patches)
    allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Task Token (index 0) -> nhìn chính nó (0) và toàn bộ Box Tokens (1 .. T)
    allowed[0, 0 : 1 + num_frames] = True

    # 2. Box Tokens (index 1 .. T) -> nhìn toàn bộ Box Tokens (1 .. T) xuyên thời gian
    allowed[1 : 1 + num_frames, 1 : 1 + num_frames] = True

    # 3. Liên kết nội bộ từng frame t (Box Token t <-> Patches cùng frame t, và Patches cùng frame t <-> nhau)
    for t in range(num_frames):
        box_idx = 1 + t
        patch_start = 1 + num_frames + t * num_patches
        patch_end = patch_start + num_patches

        # Box Token (t) nhìn các Patches cùng frame (t)
        allowed[box_idx, patch_start:patch_end] = True

        # Patch (t, i) nhìn Box Token cùng frame (t)
        allowed[patch_start:patch_end, box_idx] = True

        # Patch (t, i) nhìn các Patches khác cùng frame (t)
        allowed[patch_start:patch_end, patch_start:patch_end] = True

    return allowed


def boolean_to_additive_attention_mask(
    allowed_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Chuyển đổi boolean mask (L, L) sang additive attention mask (1, 1, L, L)
    để cộng trực tiếp vào ma trận attention scores (Q @ K^T / sqrt(d_k)):
      - Vị trí True  -> 0.0
      - Vị trí False -> -inf (torch.finfo(dtype).min)
    """
    additive = torch.zeros_like(allowed_mask, dtype=dtype)
    additive = additive.masked_fill(~allowed_mask, torch.finfo(dtype).min)
    return additive.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


class NativeCLIPViTBlock(nn.Module):
    """
    Block Transformer tương thích 1-1 với kiến trúc CLIP ViT-B/16 (D=768, 12 heads, QuickGELU/GELU).
    Dùng khi chạy offline hoặc khi khởi tạo từ cấu hình chuẩn của CLIP ViT-B/16.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    @staticmethod
    def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D)
            additive_attn_mask: (1, 1, L, L) với 0.0 cho vị trí hợp lệ và -inf cho vị trí bị khóa.
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        x = self.layer_norm1(hidden_states)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Sử dụng PyTorch Scaled Dot-Product Attention (FlashAttention / Memory-Efficient kernel trên T4)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=additive_attn_mask,
            dropout_p=0.0,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        hidden_states = residual + self.out_proj(attn_out)

        residual = hidden_states
        x = self.layer_norm2(hidden_states)
        x = self.fc2(self._quick_gelu(self.fc1(x)))
        hidden_states = residual + x
        return hidden_states


class TSHybridMaskedTransformer(nn.Module):
    """
    Module 3 — T-S Hybrid Masked Transformer với chiến lược LoRA (PEFT):
      - Đóng băng toàn bộ backbone CLIP ViT-B/16 (pre_layrnorm, 12 Transformer layers, post_layernorm).
      - Chỉ bơm adapter LoRA vào các ma trận Query (q_proj) và Value (v_proj) của tất cả K=12 lớp.
      - Áp dụng ma trận T-S Hybrid Attention Mask kích thước (L, L) trong suốt quá trình lan truyền.
    """

    def __init__(
        self,
        pre_layernorm: nn.LayerNorm,
        encoder_layers: nn.ModuleList,
        use_hf_layers: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.pre_layernorm = pre_layernorm
        self.layers = encoder_layers
        self.use_hf_layers = use_hf_layers

        # 1. Đóng băng toàn bộ tham số gốc của CLIP ViT backbone
        for param in self.pre_layernorm.parameters():
            param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False

        # 2. Bơm LoRA vào đúng các ma trận Query (Q) và Value (V) của K=12 lớp Transformer
        self._inject_lora_into_qv(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    def _inject_lora_into_qv(self, r: int, alpha: float, dropout: float) -> None:
        for layer in self.layers:
            if self.use_hf_layers and hasattr(layer, "self_attn"):
                layer.self_attn.q_proj = LoRALinear(
                    layer.self_attn.q_proj, r=r, alpha=alpha, dropout=dropout
                )
                layer.self_attn.v_proj = LoRALinear(
                    layer.self_attn.v_proj, r=r, alpha=alpha, dropout=dropout
                )
            else:
                layer.q_proj = LoRALinear(layer.q_proj, r=r, alpha=alpha, dropout=dropout)
                layer.v_proj = LoRALinear(layer.v_proj, r=r, alpha=alpha, dropout=dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sequence:     (B, L, D) chuỗi 1577 tokens từ Module 2.
            allowed_mask: (L, L) ma trận boolean T-S Hybrid Attention Mask.
        Returns:
            hidden_states: (B, L, D) đầu ra sau K=12 lớp Transformer.
        """
        hidden_states = self.pre_layernorm(sequence)
        additive_mask = boolean_to_additive_attention_mask(
            allowed_mask, dtype=hidden_states.dtype
        )

        if self.use_hf_layers:
            # HuggingFace CLIPEncoderLayer nhận causal_attention_mask: (B, 1, L, L)
            batch_size = hidden_states.shape[0]
            hf_mask = additive_mask.expand(batch_size, -1, -1, -1)
            for layer in self.layers:
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=hf_mask,
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, additive_attn_mask=additive_mask)

        return hidden_states


# ==============================================================================
# 5. MODULE 4: MODULE PHÂN LOẠI (ENGAGEMENT HEAD)
# ==============================================================================

class EngagementHead(nn.Module):
    """
    Module 4 — Module Phân loại (Engagement Head):
    Sau khi đi qua K=12 lớp Transformer Blocks, chỉ trích xuất vector đầu ra
    của Task Token (index 0) ở lớp cuối cùng và đưa qua mạng MLP nhỏ:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Logits / Softmax
    Đầu ra là logits và xác suất của 3 lớp (Low, Medium, High Engagement).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 384,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_output: (B, L, D) đầu ra từ lớp cuối cùng của T-S Hybrid Masked Transformer.
        Returns:
            logits: (B, num_classes) dùng trực tiếp cho nn.CrossEntropyLoss().
            probs:  (B, num_classes) xác suất Softmax của (Low, Medium, High Engagement).
        """
        # Chỉ lấy vector đầu ra của Task Token ở vị trí index 0: (B, D)
        task_token_final = sequence_output[:, 0, :]
        logits = self.mlp_head(task_token_final)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


# ==============================================================================
# 6. KIẾN TRÚC TỔNG HỢP: T-S HYBRID MASKED CLIP FOR ENGAGEMENT RECOGNITION
# ==============================================================================

class StudentEngagementTSHybridCLIP(nn.Module):
    """
    Kiến trúc tổng hợp kết nối 4 Module:
      1. InputEncodingModule        (E_task, E_box qua CoordinateMLP, E_patch qua CLIP Conv2D)
      2. SpatiotemporalInjection    (P_spatial từ CLIP + P_temporal học được -> Sequence L=1577)
      3. TSHybridMaskedTransformer  (12 lớp CLIP ViT-B/16 với Hybrid Mask (L, L) & LoRA trên Q, V)
      4. EngagementHead             (Task Token [:, 0, :] -> LayerNorm -> Linear -> GELU -> Dropout -> Linear)
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        num_frames: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_classes: int = 3,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        head_dropout: float = 0.2,
        load_pretrained_clip: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 196 patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Khởi tạo hoặc nạp trọng số CLIP ViT-B/16
        patch_conv, spatial_pos_embed, pre_ln, encoder_layers, use_hf = (
            self._build_or_load_clip_backbone(
                clip_model_name=clip_model_name,
                embed_dim=embed_dim,
                patch_size=patch_size,
                num_patches=self.num_patches,
                num_layers=num_layers,
                num_heads=num_heads,
                load_pretrained=load_pretrained_clip,
            )
        )

        # Module 1: Mã hóa Đầu vào
        self.input_encoder = InputEncodingModule(
            patch_embedding_conv=patch_conv,
            embed_dim=embed_dim,
        )

        # Module 2: Nhúng Không gian & Thời gian
        self.spatiotemporal_injector = SpatiotemporalInjectionModule(
            pretrained_spatial_pos_embed=spatial_pos_embed,
            num_frames=num_frames,
            num_patches=self.num_patches,
            embed_dim=embed_dim,
        )

        # Module 3: T-S Hybrid Masked Transformer với LoRA trên Q, V
        self.transformer = TSHybridMaskedTransformer(
            pre_layernorm=pre_ln,
            encoder_layers=encoder_layers,
            use_hf_layers=use_hf,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Lưu sẵn ma trận T-S Hybrid Attention Mask (L, L) dưới dạng buffer không cần tính đạo hàm
        default_mask = build_ts_hybrid_attention_mask(
            num_frames=num_frames,
            num_patches=self.num_patches,
        )
        self.register_buffer("hybrid_attention_mask", default_mask)

        # Module 4: Engagement Classification Head
        self.engagement_head = EngagementHead(
            embed_dim=embed_dim,
            hidden_dim=embed_dim // 2,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    @staticmethod
    def _build_or_load_clip_backbone(
        clip_model_name: str,
        embed_dim: int,
        patch_size: int,
        num_patches: int,
        num_layers: int,
        num_heads: int,
        load_pretrained: bool,
    ) -> Tuple[nn.Conv2d, torch.Tensor, nn.LayerNorm, nn.ModuleList, bool]:
        """
        Tải trọng số CLIP ViT-B/16 từ HuggingFace (`openai/clip-vit-base-patch16`).
        Nếu môi trường không có kết nối mạng hoặc chỉ chạy kiểm thử kiến trúc,
        tự động khởi tạo cấu hình CLIP ViT-B/16 chuẩn tương đương.
        """
        if HAS_TRANSFORMERS and load_pretrained:
            try:
                clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
                vm = clip_vision.vision_model
                patch_conv = vm.embeddings.patch_embedding
                spatial_pos = vm.embeddings.position_embedding.weight  # (197, 768)
                pre_ln = vm.pre_layrnorm
                encoder_layers = vm.encoder.layers
                return patch_conv, spatial_pos, pre_ln, encoder_layers, True
            except Exception as exc:
                print(
                    f"[Thông báo] Không thể tải trực tuyến '{clip_model_name}' ({exc}). "
                    f"Sử dụng backbone CLIP ViT-B/16 nội bộ có cùng kích thước (D={embed_dim}, K={num_layers})."
                )

        # Khởi tạo kiến trúc CLIP ViT-B/16 nội bộ chuẩn xác tới từng tensor
        patch_conv = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        spatial_pos = torch.randn(num_patches + 1, embed_dim) * 0.02
        pre_ln = nn.LayerNorm(embed_dim, eps=1e-5)
        encoder_layers = nn.ModuleList([
            NativeCLIPViTBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        return patch_conv, spatial_pos, pre_ln, encoder_layers, False

    def forward(
        self,
        frames: torch.Tensor,
        boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, 224, 224) — T frames ảnh toàn cảnh lớp học.
            boxes:  (B, T, 4) — Tọa độ Bounding Box chuẩn hóa [x_min, y_min, x_max, y_max] của học sinh.
        Returns:
            Dict chứa:
              - 'logits': (B, num_classes)
              - 'probs':  (B, num_classes)
              - 'task_embedding': (B, D) vector đại diện của Task Token ở lớp cuối cùng
              - 'sequence_length': chiều dài chuỗi L = 1 + T + T*N
        """
        t_frames = frames.shape[1]

        # Module 1: Mã hóa Đầu vào -> E_task (B, 1, D), E_box (B, T, D), E_patch (B, T, N, D)
        e_task, e_box, e_patch = self.input_encoder(frames, boxes)

        # Module 2: Nhúng Không gian & Thời gian + Ghép chuỗi -> (B, L, D)
        sequence = self.spatiotemporal_injector(e_task, e_box, e_patch)

        # Lấy hoặc tạo ma trận Hybrid Attention Mask (L, L) tương ứng với số frame T
        if t_frames == self.num_frames:
            mask = self.hybrid_attention_mask
        else:
            mask = build_ts_hybrid_attention_mask(
                num_frames=t_frames,
                num_patches=self.num_patches,
                device=sequence.device,
            )

        # Module 3: Đẩy chuỗi qua K=12 lớp Transformer với Hybrid Mask & LoRA
        encoded_sequence = self.transformer(sequence, mask)

        # Module 4: Phân loại từ Task Token (index 0) ở lớp cuối cùng
        logits, probs = self.engagement_head(encoded_sequence)

        return {
            "logits": logits,
            "probs": probs,
            "task_embedding": encoded_sequence[:, 0, :],
        }

    def get_trainable_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Phân nhóm các tham số được phép cập nhật qua hàm Loss Cross-Entropy:
          1. LoRA adapters (Q, V của 12 lớp Transformer)
          2. Task Token (E_task)
          3. Coordinate MLP (mã hóa Box Trajectory)
          4. Temporal Positional Embeddings (P_temporal)
          5. Engagement Head (MLP phân loại)
        """
        lora_params: List[nn.Parameter] = []
        for name, param in self.transformer.named_parameters():
            if param.requires_grad and ("lora_A" in name or "lora_B" in name):
                lora_params.append(param)

        return {
            "lora_qv": lora_params,
            "task_token": [self.input_encoder.task_token],
            "coord_mlp": list(self.input_encoder.coord_mlp.parameters()),
            "temporal_embed": [self.spatiotemporal_injector.p_temporal],
            "engagement_head": list(self.engagement_head.parameters()),
        }

    def print_parameter_summary(self) -> Tuple[int, int]:
        """Thống kê số lượng tham số đóng băng và tham số huấn luyện (PEFT)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ratio = 100.0 * trainable_params / max(total_params, 1)

        print("=" * 72)
        print("THỐNG KÊ THAM SỐ MÔ HÌNH (T-S HYBRID MASKED CLIP + LoRA)")
        print("=" * 72)
        print(f"Tổng số tham số (Total Parameters)     : {total_params:,}")
        print(f"Tham số huấn luyện (Trainable Params)  : {trainable_params:,} ({ratio:.2f}%)")
        print(f"Tham số đóng băng (Frozen CLIP Params) : {total_params - trainable_params:,}")
        print("-" * 72)
        for group_name, params in self.get_trainable_parameter_groups().items():
            count = sum(p.numel() for p in params)
            print(f"  + {group_name:<20}: {count:>10,} params")
        print("=" * 72)
        return total_params, trainable_params


# ==============================================================================
# 7. HÀM HUẤN LUYỆN & TỐI ƯU HÓA VỚI CROSS-ENTROPY LOSS
# ==============================================================================

def configure_optimizer(
    model: StudentEngagementTSHybridCLIP,
    lr_lora: float = 2e-4,
    lr_new_modules: float = 5e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """
    Khởi tạo AdamW Optimizer chỉ cập nhật trọng số cho:
    LoRA (Q, V), Task Token, Coordinate MLP, Temporal Embeddings và Engagement Head.
    """
    groups = model.get_trainable_parameter_groups()
    optimizer_grouped_parameters = [
        {"params": groups["lora_qv"], "lr": lr_lora, "weight_decay": weight_decay},
        {"params": groups["task_token"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["temporal_embed"], "lr": lr_new_modules, "weight_decay": 0.0},
        {"params": groups["coord_mlp"], "lr": lr_new_modules, "weight_decay": weight_decay},
        {"params": groups["engagement_head"], "lr": lr_new_modules, "weight_decay": weight_decay},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters)


def train_one_step(
    model: StudentEngagementTSHybridCLIP,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """
    Một bước huấn luyện (Forward -> Cross-Entropy Loss -> Backward -> Update):
    chỉ cập nhật LoRA, Task Token, Coordinate MLP, Temporal Embeddings và Engagement Head.
    """
    model.train()
    optimizer.zero_grad()

    frames = batch["frames"].to(device)  # (B, T, 3, 224, 224)
    boxes = batch["boxes"].to(device)    # (B, T, 4)
    labels = batch["label"].to(device)   # (B,)

    outputs = model(frames=frames, boxes=boxes)
    logits = outputs["logits"]
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    return float(loss.item()), outputs["probs"].detach()


# ==============================================================================
# 8. KIỂM THỬ TOÀN DIỆN KIẾN TRÚC (VERIFICATION SUITE)
# ==============================================================================

def verify_architecture() -> None:
    """
    Kiểm thử kích thước tensor qua từng Module, kiểm tra quy tắc XAI của ma trận
    Hybrid Attention Mask (1577 x 1577) và kiểm tra cập nhật gradient với Cross-Entropy Loss.
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 2          # Batch size
    T = 8          # Số frame mỗi clip
    C, H, W = 3, 224, 224
    N = 196        # Số patch mỗi frame (224/16)^2
    D = 768        # Số chiều không gian nhúng CLIP ViT-B/16
    L = 1 + T + T * N  # 1 + 8 + 1568 = 1577 tokens

    print(f"[Kiểm thử] Thiết bị chạy: {device} | Chiều dài chuỗi kỳ vọng L = {L} tokens")

    # 1. Khởi tạo mô hình
    model = StudentEngagementTSHybridCLIP(
        num_frames=T,
        image_size=H,
        patch_size=16,
        embed_dim=D,
        num_layers=12,
        num_heads=12,
        num_classes=3,  # Low, Medium, High Engagement
        lora_r=8,
        lora_alpha=16.0,
        load_pretrained_clip=False,  # Đặt True khi chạy trên máy/Colab có kết nối HuggingFace
    ).to(device)

    model.print_parameter_summary()

    # 2. Kiểm tra quy tắc XAI của ma trận Hybrid Attention Mask (1577, 1577)
    mask = model.hybrid_attention_mask
    assert mask.shape == (L, L), f"Sai kích thước mask: {mask.shape} != {(L, L)}"

    # - Task Token (0) chỉ nhìn thấy chính nó (0) và T Box Tokens (1..T) -> tổng cộng 1 + T = 9 tokens
    assert int(mask[0].sum().item()) == 1 + T, "Task Token chỉ được nhìn chính nó và T Box Tokens!"
    assert not mask[0, 1 + T :].any(), "Task Token KHÔNG được nhìn trực tiếp vào Image Patches!"

    # - Box Token ở frame t (ví dụ t=0 -> idx=1) nhìn T Box Tokens (8) + N Patches cùng frame (196) = 204 tokens
    for t in range(T):
        box_idx = 1 + t
        assert int(mask[box_idx].sum().item()) == T + N, (
            f"Box Token frame {t} phải nhìn đúng T={T} Box Tokens và N={N} Patches cùng frame!"
        )

    # - Patch (t, i) chỉ nhìn Box Token cùng frame t (1) + N Patches cùng frame t (196) = 197 tokens
    for t in range(T):
        first_patch_idx = 1 + T + t * N
        assert int(mask[first_patch_idx].sum().item()) == 1 + N, (
            f"Patch ở frame {t} chỉ được nhìn 1 Box Token cùng frame và {N} Patches cùng frame!"
        )
    print(f"[OK] Ma trận T-S Hybrid Attention Mask {tuple(mask.shape)} tuân thủ tuyệt đối quy tắc XAI!")

    # 3. Kiểm tra luồng tensor qua từng Module
    dummy_frames = torch.randn(B, T, C, H, W, device=device)
    dummy_boxes = torch.rand(B, T, 4, device=device)  # [x_min, y_min, x_max, y_max] chuẩn hóa in [0, 1]
    dummy_labels = torch.tensor([0, 2], dtype=torch.long, device=device)  # 0: Low, 2: High

    e_task, e_box, e_patch = model.input_encoder(dummy_frames, dummy_boxes)
    assert e_task.shape == (B, 1, D), f"Sai kích thước E_task: {e_task.shape}"
    assert e_box.shape == (B, T, D), f"Sai kích thước E_box: {e_box.shape}"
    assert e_patch.shape == (B, T, N, D), f"Sai kích thước E_patch: {e_patch.shape}"
    print(
        f"[OK] Module 1 (Input Encoding): E_task={tuple(e_task.shape)}, "
        f"E_box={tuple(e_box.shape)}, E_patch={tuple(e_patch.shape)}"
    )

    seq = model.spatiotemporal_injector(e_task, e_box, e_patch)
    assert seq.shape == (B, L, D), f"Sai kích thước Sequence: {seq.shape} != {(B, L, D)}"
    print(f"[OK] Module 2 (Spatiotemporal Injection & Concatenation): Sequence={tuple(seq.shape)}")

    # 4. Kiểm tra Forward + Backward + Optimizer Step với Cross-Entropy Loss
    optimizer = configure_optimizer(model)
    criterion = nn.CrossEntropyLoss()
    loss_val, probs = train_one_step(
        model=model,
        batch={"frames": dummy_frames, "boxes": dummy_boxes, "label": dummy_labels},
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )
    assert probs.shape == (B, 3), f"Sai kích thước xác suất đầu ra: {probs.shape}"
    assert model.input_encoder.task_token.grad is not None, "Task Token phải có gradient!"
    assert model.spatiotemporal_injector.p_temporal.grad is not None, "P_temporal phải có gradient!"
    print(
        f"[OK] Module 3 & 4 (Forward + Cross-Entropy Backward): Loss = {loss_val:.4f} | "
        f"Probs shape = {tuple(probs.shape)}"
    )


if __name__ == "__main__":
    verify_architecture()
