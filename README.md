# T-S Hybrid Masked Transformer for Student Engagement Recognition

Kiến trúc nhận diện mức độ tham gia của học sinh (**Student Engagement Recognition**) dựa trên nguyên lý **Unified Dense Transformer với Early Fusion & Hybrid Attention Masking** từ nghiên cứu [Falcon Perception (arXiv:2603.27365)](https://arxiv.org/abs/2603.27365) ([HTML](https://arxiv.org/html/2603.27365v1), [GitHub](https://github.com/tiiuae/Falcon-Perception)), được tối ưu hóa với **LoRA (PEFT)** trên backbone **CLIP ViT-B/16** (`D = 768`) và tương thích trực tiếp ở chế độ **chỉ đọc (Read-Only)** với bộ dữ liệu [ViMELD_Dataset](https://drive.google.com/drive/folders/17s0ROqYsZwXH3IKdCTrI6KR8p1DwWEtD).

## Cấu trúc thư mục dự án (GitHub Repository)

```text
ts-hybrid-masked-clip-engagement/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                                  # Script CLI huấn luyện & kiểm thử kiến trúc
├── ts_hybrid_masked_clip_engagement.py       # File độc lập (All-in-One) chứa toàn bộ 4 Module + Dataset
├── src/
│   ├── __init__.py
│   ├── dataset.py                            # Read-Only DataLoader cho ViMELD_Dataset & Manifest
│   ├── model.py                              # 4 Module: Input Encoding, Spatiotemporal, Hybrid Mask + LoRA, Head
│   └── trainer.py                            # Vòng lặp huấn luyện FP16 Mixed Precision (T4 GPU) & lưu PEFT checkpoint
└── notebooks/
    └── colab_train_ts_hybrid_clip.ipynb      # Google Colab Notebook tải repo từ GitHub và chạy trực tiếp
```

## Kiến trúc 4 Module

1. **Module 1 — Mã hóa Đầu vào (`InputEncodingModule`)**:
   - **Task Token ($E_{task}$)**: Learnable Vector $(1, 768)$ đại diện cho tác vụ nhận diện Engagement.
   - **Box Trajectory ($E_{box}$)**: Tọa độ Bounding Box chuẩn hóa $(T, 4)$ chứa $[x_{min}, y_{min}, x_{max}, y_{max}] \in [0, 1]$ của học sinh qua ByteTrack đi qua `CoordinateMLP` (`Linear(4, 768) -> GELU -> Linear(768, 768)`) $\rightarrow (T, 768)$.
   - **Image Patches ($E_{patch}$)**: $T$ frames ảnh $(T, 3, 224, 224)$ đi qua lớp `Conv2D` Patch Embedding ($16 \times 16$) của CLIP ViT-B/16 $\rightarrow (T, 196, 768)$.
2. **Module 2 — Nhúng Không gian & Thời gian (`SpatiotemporalInjectionModule`)**:
   - Cộng $P_{spatial}$ $(196, 768)$ mượn từ CLIP ViT vào $E_{patch}$ ở mỗi frame.
   - Cộng $P_{temporal}[t]$ $(T, 768)$ học được vào cả Box Token thứ $t$ và $196$ Image Patches thuộc frame thứ $t$.
   - Ghép chuỗi: $\text{Sequence} = [E_{task}] \oplus E_{box} \oplus E_{patch}$ có chiều dài $L = 1 + T + T \times N$ (với $T=8, N=196 \rightarrow L = 1577$ tokens).
3. **Module 3 — T-S Hybrid Masked Transformer & Chiến lược LoRA (`TSHybridMaskedTransformer`)**:
   - Ma trận **Hybrid Attention Mask** kích thước $(L, L) = (1577, 1577)$:
     - **Task Token (index $0$)**: Chỉ nhìn chính nó và $T$ Box Tokens (index $1 \dots T$).
     - **Box Token ($t$)**: Nhìn toàn bộ Box Tokens xuyên thời gian $T$ và $N=196$ Patches cùng frame $t$.
     - **Patch ($t, i$)**: Chỉ nhìn Box Token cùng frame $t$ và các Patches khác cùng frame $t$ (khóa không nhìn xuyên frame để tránh nhiễu nền và tránh OOM trên GPU T4).
   - **Chiến lược LoRA**: Đóng băng toàn bộ backbone CLIP ViT-B/16, chỉ bơm adapter LoRA vào các ma trận Query ($Q$) và Value ($V$) của 12 lớp Transformer.
4. **Module 4 — Module Phân loại (`EngagementHead`)**:
   - Trích xuất vector đầu ra của Task Token ở lớp cuối cùng (`sequence_output[:, 0, :]`), đưa qua `LayerNorm -> Linear -> GELU -> Dropout -> Linear` và tối ưu hóa bằng hàm `CrossEntropyLoss` cho 3 lớp (`Low`, `Medium`, `High Engagement`).

## Hướng dẫn đưa lên GitHub

Sau khi tải và giải nén file `ts_hybrid_masked_clip_github_repo.zip`:

```bash
cd ts-hybrid-masked-clip-engagement
git init
git branch -M main
git add .
git commit -m "Initial commit: T-S Hybrid Masked CLIP for Student Engagement Recognition"
git remote add origin https://github.com/<USERNAME>/<REPO_NAME>.git
git push -u origin main
```
