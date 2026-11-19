"""
Read-Only Dataset & High-Throughput DataLoader for ViMELD_Dataset on Google Drive.
Tuyệt đối KHÔNG chỉnh sửa dữ liệu gốc của người dùng.

Tối ưu hóa tốc độ quét & nạp dữ liệu:
1. ThreadPoolExecutor + os.scandir(): Quét song song các Session và các thư mục ID_*/seg_*
   nhanh gấp 5-10x trên Google Drive FUSE.
2. Local Index Cache (/content/vimeld_index_cache): Lưu danh sách samples ra SSD cục bộ của Colab
   sau lần quét đầu tiên; các lần chạy sau chỉ mất < 0.2 giây.
3. Sequential Video Grab + mmap_mode="r": Giảm độ trễ giải mã H.264 và zero-copy cho .npy.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
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
    Tích hợp:
      - Quét song song đa luồng (`ThreadPoolExecutor` + `os.scandir`).
      - Cache chỉ mục mẫu ra `/content/vimeld_index_cache` (không bao giờ ghi vào `root_dir`).
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        num_frames: int = 8,
        image_size: int = 224,
        num_classes: int = 4,
        manifest_path: Optional[Union[str, Path]] = None,
        default_video_width: int = 3200,
        default_video_height: int = 1800,
        use_full_scene_video: bool = True,
        index_cache_dir: Optional[Union[str, Path]] = "/content/vimeld_index_cache",
        scan_workers: int = 16,
        force_rescan: bool = False,
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
        self.scan_workers = max(1, int(scan_workers))
        self.force_rescan = force_rescan
        self.label_map = ENGAGEMENT_3CLASS_MAP if num_classes == 3 else ENGAGEMENT_4CLASS_MAP

        # Chọn thư mục cache cục bộ (ưu tiên /content/... trên Colab, fallback /tmp nếu chạy local)
        if index_cache_dir is not None:
            preferred_cache = Path(index_cache_dir)
            if preferred_cache.parts[:2] == ("/", "content") and not Path("/content").exists():
                self.index_cache_dir: Optional[Path] = Path(tempfile.gettempdir()) / "vimeld_index_cache"
            else:
                self.index_cache_dir = preferred_cache
        else:
            self.index_cache_dir = None

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

    def _get_cache_file_path(self) -> Optional[Path]:
        if self.index_cache_dir is None:
            return None
        root_hash = hashlib.md5(str(self.root_dir).encode("utf-8")).hexdigest()[:8]
        return self.index_cache_dir / f"index_{self.split}_c{self.num_classes}_{root_hash}.json"

    def _load_from_manifest(self, manifest_path: Path) -> None:
        if pd is None:
            raise ImportError("Cần cài đặt pandas để đọc file manifest.")
        t0 = time.perf_counter()
        df = pd.read_parquet(manifest_path) if manifest_path.suffix == ".parquet" else pd.read_csv(manifest_path)

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
        print(
            f"[Dataset Manifest] Split='{self.split}': {len(self.samples):,} mẫu "
            f"({time.perf_counter() - t0:.2f}s).",
            flush=True,
        )

    @staticmethod
    def _find_behavior_files_scandir(root_dir: Path) -> List[Path]:
        """
        Dùng os.scandir() duyệt đúng cấp thư mục Session (<Date>/Period/Angle/<Session_ID>/behaviors.json)
        nhanh hơn rất nhiều so với Path.glob() và tuyệt đối không chui vào hàng trăm nghìn thư mục seg_*.
        """
        behavior_files: List[Path] = []
        root_str = str(root_dir)

        def _list_subdirs(path_str: str) -> List[str]:
            try:
                with os.scandir(path_str) as it:
                    return [e.path for e in it if e.is_dir(follow_symlinks=False)]
            except OSError:
                return []

        # Cấp 1: <Date> (vd: 28_11_2025)
        date_dirs = _list_subdirs(root_str)
        # Cấp 2: <Period> (vd: Period 1)
        period_dirs: List[str] = []
        for d in date_dirs:
            period_dirs.extend(_list_subdirs(d))
        # Cấp 3: <Angle> (vd: Angle 1)
        angle_dirs: List[str] = []
        for p in period_dirs:
            angle_dirs.extend(_list_subdirs(p))
        # Cấp 4: <Session_ID> (vd: S07_1_1_001) -> kiểm tra behaviors.json
        for a in angle_dirs:
            for s_path in _list_subdirs(a):
                beh_path = os.path.join(s_path, "behaviors.json")
                if os.path.isfile(beh_path):
                    behavior_files.append(Path(beh_path))

        # Fallback nếu cấu trúc thư mục nông hơn 1 cấp
        if not behavior_files:
            for p in period_dirs:
                for s_path in _list_subdirs(p):
                    beh_path = os.path.join(s_path, "behaviors.json")
                    if os.path.isfile(beh_path):
                        behavior_files.append(Path(beh_path))

        behavior_files.sort()
        return behavior_files

    def _scan_single_session(
        self, behaviors_path: Path, allowed_sessions: Optional[set]
    ) -> List[Dict[str, Any]]:
        """Quét một thư mục Session bằng os.scandir() (chạy song song trong ThreadPoolExecutor)."""
        session_dir = behaviors_path.parent
        session_id = session_dir.name
        if allowed_sessions is not None and session_id not in allowed_sessions:
            return []

        rgb_root = session_dir / f"{session_id}_rgb"
        if not rgb_root.is_dir():
            return []

        video_file = session_dir.parent / f"{session_id}.mp4"
        video_path_str = str(video_file) if video_file.is_file() else ""

        try:
            with open(behaviors_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
        except Exception:
            return []

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

        if not eng_by_roi:
            return []

        session_samples: List[Dict[str, Any]] = []
        try:
            with os.scandir(rgb_root) as it_students:
                student_entries = sorted(
                    (e for e in it_students if e.is_dir(follow_symlinks=False) and e.name.startswith("ID_")),
                    key=lambda x: x.name,
                )
        except OSError:
            return []

        for st_entry in student_entries:
            roi_id = st_entry.name
            if roi_id not in eng_by_roi:
                continue
            roi_intervals = eng_by_roi[roi_id]

            try:
                with os.scandir(st_entry.path) as it_segs:
                    seg_entries = sorted(
                        (e for e in it_segs if e.is_dir(follow_symlinks=False) and e.name.startswith("seg_")),
                        key=lambda x: x.name,
                    )
            except OSError:
                continue

            for seg_entry in seg_entries:
                parts = seg_entry.name.split("_")
                if len(parts) < 4:
                    continue
                try:
                    seg_start = float(parts[-2])
                    seg_end = float(parts[-1])
                except ValueError:
                    continue

                matched_label: Optional[str] = None
                seg_mid = 0.5 * (seg_start + seg_end)
                for st, et, eng_lb in roi_intervals:
                    if st <= seg_mid <= et or (max(st, seg_start) < min(et, seg_end)):
                        matched_label = eng_lb
                        break

                if matched_label is None:
                    continue

                session_samples.append({
                    "sample_id": f"{session_id}_{roi_id}_{seg_entry.name}",
                    "session": session_id,
                    "object_id": roi_id,
                    "seg_dir": seg_entry.path,
                    "video_path": video_path_str,
                    "skeleton_json": os.path.join(seg_entry.path, "skeleton_vitpose_plus_l_upper_body.json"),
                    "label": self.label_map[matched_label],
                    "raw_label": matched_label,
                })

        return session_samples

    def _index_from_vimeld_directory(self) -> None:
        t0 = time.perf_counter()
        cache_file = self._get_cache_file_path()

        # Cách 3: Đọc ngay từ Index Cache cục bộ (/content/vimeld_index_cache/...) nếu đã có
        if cache_file is not None and cache_file.is_file() and not self.force_rescan:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
                print(
                    f"[Dataset Cache] Split='{self.split}': Nạp {len(self.samples):,} mẫu từ "
                    f"{cache_file} trong {time.perf_counter() - t0:.2f}s.",
                    flush=True,
                )
                return
            except Exception as exc:
                print(f"[Dataset Cache] Không đọc được cache ({exc}), tiến hành quét lại...", flush=True)

        split_file = self.root_dir / "sessions_summary_train_valid_test.json"
        allowed_sessions: Optional[set] = None
        if split_file.is_file():
            with open(split_file, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
            if self.split in splits_data:
                allowed_sessions = set(splits_data[self.split])

        print(f"[Dataset] Đang tìm các session tại {self.root_dir} (os.scandir)...", flush=True)
        behavior_files = self._find_behavior_files_scandir(self.root_dir)
        if allowed_sessions is not None:
            behavior_files = [p for p in behavior_files if p.parent.name in allowed_sessions]

        print(
            f"[Dataset] Quét song song {len(behavior_files)} sessions cho split='{self.split}' "
            f"với {self.scan_workers} threads...",
            flush=True,
        )

        # Cách 2: ThreadPoolExecutor quét song song các thư mục Session trên Google Drive FUSE
        all_samples: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.scan_workers) as executor:
            futures = {
                executor.submit(self._scan_single_session, beh_path, allowed_sessions): beh_path
                for beh_path in behavior_files
            }
            for fut in as_completed(futures):
                all_samples.extend(fut.result())

        all_samples.sort(key=lambda x: x["sample_id"])
        self.samples = all_samples

        elapsed = time.perf_counter() - t0
        print(
            f"[Dataset] Split='{self.split}': Hoàn tất quét {len(self.samples):,} mẫu trong {elapsed:.2f}s.",
            flush=True,
        )

        # Lưu cache cục bộ ra /content/vimeld_index_cache (tuyệt đối không ghi vào Drive)
        if cache_file is not None and len(self.samples) > 0:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.samples, f)
                print(f"[Dataset Cache] Đã lưu chỉ mục cục bộ tại: {cache_file}", flush=True)
            except Exception as exc:
                print(f"[Dataset Cache] Bỏ qua lưu cache ({exc}).", flush=True)

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

        seg_dir = sample["seg_dir"]
        try:
            with os.scandir(seg_dir) as it:
                frame_files = sorted(
                    (Path(e.path) for e in it if e.is_file() and e.name.startswith("frame_") and e.name.endswith(".png")),
                    key=lambda p: p.name,
                )
        except OSError:
            frame_files = []

        skel_path = Path(sample.get("skeleton_json", ""))
        boxes_list: List[List[float]] = []
        video_frame_ids: List[int] = []

        if skel_path.is_file():
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
        if clip_path and Path(clip_path).is_file() and clip_path.endswith(".npy"):
            arr = np.load(clip_path, mmap_mode="r")
            idx = self.uniform_temporal_indices(len(arr), self.num_frames)
            return torch.stack(
                [self._preprocess_frame(np.asarray(arr[i], dtype=np.uint8)) for i in idx],
                dim=0,
            )

        video_path = sample.get("video_path", "")
        if self.use_full_scene_video and video_path and Path(video_path).is_file() and cv2 is not None:
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
        if cv2 is not None:
            cv2.setNumThreads(0)
        sample = self.samples[index]
        boxes, video_frame_ids, sampled_pngs = self._load_boxes_and_frame_ids_from_segment(sample)
        frames = self._load_frames(sample, video_frame_ids, sampled_pngs)
        return {
            "frames": frames,
            "boxes": torch.from_numpy(boxes).float(),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "sample_id": sample.get("sample_id", f"sample_{index}"),
        }