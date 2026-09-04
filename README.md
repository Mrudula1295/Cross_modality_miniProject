# Cross-Modality Person Re-Identification (RGB–Infrared)

An end-to-end research codebase built in Python and PyTorch for **Cross-Modality Person Re-Identification (RGB to Infrared / Thermal)**. This repository is specifically tuned for hardware constraints such as an **NVIDIA RTX 4050 Laptop GPU (6GB VRAM)** on **Windows 11**.

---

## Key Features & Highlights

- **Zero Manual Dataset Setup**: Automatically fetches **Market-1501** (real RGB pedestrian dataset) via `torchreid` / maintained mirrors and generates a paired **Synthetic Infrared (synthetic-IR)** dataset automatically.
- **Vision Transformer Backbone**: Pretrained `vit_small_patch16_224` (timm) with Gradient Checkpointing and a shared projection head mapping RGB and IR features into a normalized embedding space.
- **Hybrid Cross-Modality Loss**: Label-Smoothed Cross-Entropy Loss + Online Batch-Hard Triplet Loss across RGB and IR modalities.
- **6GB VRAM Optimization**: Automatic Mixed Precision (AMP `torch.cuda.amp`), Gradient Accumulation (effective batch 32 with physical batch 16), and peak VRAM profiling (`--dry-run` probe).
- **YOLOv8 Detection Pipeline**: Modular YOLOv8 (`detect.py`) person detection and cropping utility for raw image/video stream testing.
- **Future-Proof Real Dataset Migration**: Includes `data/prepare_sysu_regdb.py` to ingest real SYSU-MM01 or RegDB archives with zero code modification.

---

## Hardware Specification & Recommended Tuning

- **Target GPU**: NVIDIA GeForce RTX 4050 Laptop (6GB VRAM)
- **Target CPU**: Intel Core i7-14650HX (16GB RAM)
- **Target OS**: Windows 11 (CUDA 12.1 / 12.4)

---

## 1. Setup & Installation (Windows 11)

### Step 1: Install PyTorch with CUDA Support
For NVIDIA RTX 4050 on Windows 11, install CUDA-enabled PyTorch (CUDA 12.1 build recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Project Dependencies
Install remaining requirements from `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 2. Dataset Strategy: Synthetic-IR Modality Rationale

### Why Synthetic Infrared?
Real cross-modality datasets such as **SYSU-MM01** and **RegDB** are gated by the original authors, requiring manual email agreements and individual account approvals. To satisfy the zero manual dataset work constraint, this repository automatically constructs a paired two-modality dataset:

1. **Market-1501 RGB Acquisition**: Downloaded automatically into `./data_store/market1501/Market-1501-v15.09.15/`.
2. **Synthetic-IR Generation**: Every RGB image undergoes:
   - **Grayscale Conversion**: Eliminates color spectrum cues.
   - **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0, tileGridSize=(8,8)`).
   - **Thermal Color Remapping**: OpenCV `COLORMAP_INFERNO` color mapping.
3. **Identical Metadata**: Saved under `./data_store/market1501/synthetic_ir/` preserving identical Person IDs (`PID`) and Camera IDs (`CamID`).

> **Warning Banner**: A warning banner is displayed at the start of training to explicitly state that synthetic IR data is in use as a stand-in for real thermal camera data.

---

## 3. Quick Start & Execution Commands

### A. VRAM Probe / Dry Run (Recommended First Step)
Probe model memory footprint on your RTX 4050 without starting a full training run:

```bash
python train.py --dry-run
```

*Outputs peak GPU memory allocation (e.g. `Peak GPU VRAM: 3.82 GB / 6.00 GB`) for 1 forward+backward pass.*

### B. Single-Command Training
Runs automated dataset download, synthetic IR generation, and starts training end-to-end:

```bash
python train.py
```

- TensorBoard logs and metric CSVs are written to `./runs/`.
- Model checkpoints (`checkpoint_epoch_X.pth` keeping last 3, and `best_model.pth`) are saved automatically.

### C. Cross-Modality Evaluation
Evaluate Rank-1, Rank-5, Rank-10, and mAP on cross-modality splits (`RGB -> Synthetic IR` and `Synthetic IR -> RGB`):

```bash
python eval.py --weights ./runs/best_model.pth
```

### D. YOLOv8 Person Detection & Cropping
Detect person bounding boxes in raw uncropped imagery and extract Re-ID embeddings:

```bash
python detect.py --image path/to/raw_photo.jpg --weights ./runs/best_model.pth
```

---

## 4. Migration to Real SYSU-MM01 / RegDB Datasets

If you obtain approval for real thermal datasets in the future:

1. Place extracted SYSU-MM01 or RegDB archives into `./data_store/sysu_mm01/` or `./data_store/regdb/`.
2. In `configs/default.yaml`, update `dataset.name` to `"sysu_mm01"` or `"regdb"`.
3. The unified `BaseCrossModalDataset` interface and `data/prepare_sysu_regdb.py` will automatically ingest the dataset splits without requiring code changes.

---

## 5. Out-Of-Memory (OOM) Troubleshooting for 6GB VRAM

If CUDA Out-Of-Memory errors occur on 6GB VRAM:

1. **Verify Gradient Checkpointing**: Ensure `grad_checkpointing: true` under `model` in `configs/default.yaml`.
2. **Reduce Physical Batch Size**: Change `batch_size: 16` to `batch_size: 8` and increase `grad_accum_steps: 4` to maintain an effective batch size of 32.
3. **Adjust Windows Workers**: On Windows 11, if multiprocessing issues arise, reduce `num_workers: 2` or `0`.
4. **Mixed Precision**: Ensure `use_amp: true` is enabled in `configs/default.yaml`.

---

## Project Structure

```
miniprojectfinal/
├── configs/
│   └── default.yaml             # RTX 4050 tuned hyperparameters
├── data/
│   ├── __init__.py
│   ├── synthesize_ir.py          # Grayscale + CLAHE + INFERNO synthetic IR converter
│   ├── dataset.py                # Market1501Manager & BaseCrossModalDataset PyTorch loader
│   └── prepare_sysu_regdb.py     # Parser stub for real SYSU-MM01 / RegDB datasets
├── models/
│   ├── __init__.py
│   ├── vit_reid.py              # ViT backbone with gradient checkpointing + projection head
│   └── losses.py                # Hybrid Label Smoothing CE + Batch-Hard Triplet Loss
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Rank-1, Rank-5, Rank-10 & mAP evaluation protocol
│   └── logger.py                # Peak VRAM profiling & TensorBoard / CSV logger
├── train.py                     # Main training entrypoint with --dry-run VRAM check
├── eval.py                      # Evaluation script reporting Rank-N and mAP
├── detect.py                    # YOLOv8 person detector & cropper for raw footage
├── requirements.txt             # PyTorch ecosystem dependencies
└── README.md                    # Project documentation
```
