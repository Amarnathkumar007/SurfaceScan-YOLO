# Pothole Detection with YOLOv8

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Colab-GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"/>
  <img src="https://img.shields.io/badge/Roboflow-Dataset-6706CE?style=for-the-badge&logo=roboflow&logoColor=white" alt="Roboflow"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

This project builds an end-to-end pipeline that detects potholes on road surfaces from both still images and video footage. It fine-tunes the **YOLOv8 medium** architecture on a custom annotated dataset and runs entirely inside Google Colab with a T4 GPU.

---

## Why This Matters

Damaged roads are a leading cause of vehicle wear, accidents, and injury — particularly in regions where maintenance funding is limited. An automated system that can scan dashcam or drone footage and flag pothole locations has direct applications in municipal road surveys, fleet management, and autonomous driving safety layers.

---

## How It Works

The workflow follows four stages:

1. **Data ingestion** — Annotated road images (bounding-box format) are pulled from Roboflow via its Python API and converted to the YOLO directory layout (`train/`, `valid/`, `test/` splits with corresponding label files).

2. **Transfer learning** — A pre-trained `yolov8m.pt` checkpoint (25.8M parameters, 79.1 GFLOPs) is loaded and its detection head is re-configured for a single class (`pothole`). The model is then fine-tuned for 70–100 epochs at 640×640 input resolution using AdamW with automatic learning-rate selection.

3. **Validation** — After each epoch the model is evaluated on the held-out validation split. Metrics tracked include box precision, recall, mAP@0.5, and mAP@0.5:0.95. A confusion matrix and per-epoch loss curves are saved automatically.

4. **Inference** — The best checkpoint (`best.pt`) is applied to unseen test images and full-length road videos. For video, the raw `.avi` output is re-encoded to H.264 MP4 via FFmpeg so it can be played back inline inside the notebook.

---

## Repository Contents

```
├── PotholeDetectionUsingYOLO.ipynb         # Primary notebook — 70-epoch training run
├── Potholes.ipynb                          # Extended experiment — 100-epoch training run
├── Copy_of_PotholeDetectionUsingYOLO.ipynb # Sandbox copy for ad-hoc experiments
├── LICENSE                                 # MIT
└── README.md
```

| Notebook | Epochs | Train images | Val images | Dataset workspace |
|---|---|---|---|---|
| `PotholeDetectionUsingYOLO.ipynb` | 70 | 77 | 11 | hiteshram / object-detection-bounding-box v1 |
| `Potholes.ipynb` | 100 | 174 | 19 | PotholeDetectionProjectyolov8 |

---

## Getting Started

### Run on Google Colab (quickest option)

1. Open one of the notebooks via the **"Open in Colab"** badge at the top of the `.ipynb` file.
2. Switch the runtime to **GPU**: `Runtime → Change runtime type → T4 GPU → Save`.
3. Execute all cells in order (`Runtime → Run all`).

The notebook handles package installation, dataset download, training, and inference in a single run.

### Run Locally

```bash
# 1. Clone
git clone <your-repo-url>
cd pothole_detection_yolov8

# 2. Install packages
pip install ultralytics roboflow

# 3. (Optional) For video compression and API serving
pip install fastapi kaleido python-multipart uvicorn
```

You will need to supply your own Roboflow API key inside the notebook cell that calls `Roboflow(api_key=...)`. If Roboflow access is unavailable, you can manually place the dataset on Google Drive and mount it in Colab.

---

## Training Details

| Setting | Value |
|---|---|
| Base checkpoint | `yolov8m.pt` (medium) |
| Parameters | ~25.8 M |
| GFLOPs | 79.1 |
| Input resolution | 640 × 640 |
| Optimizer | AdamW (auto-tuned LR ≈ 0.002, momentum 0.9) |
| Augmentations | Mosaic, flip-LR, HSV jitter, Blur, MedianBlur, CLAHE |
| Classes | 1 (`pothole`) |
| Hardware | NVIDIA T4 (15 GB VRAM) via Colab |

Training outputs are written to `runs/detect/train/`:

- `weights/best.pt` — checkpoint with highest mAP
- `weights/last.pt` — final-epoch checkpoint
- `confusion_matrix.png` — class-level prediction breakdown
- `results.png` — epoch-wise loss and mAP curves
- `val_batch*_pred.jpg` — sample predictions on validation images

---

## Running Inference

**On images:**

```bash
yolo detect predict model=runs/detect/train/weights/best.pt conf=0.25 source=path/to/images
```

**On video:**

```bash
yolo detect predict model=runs/detect/train/weights/best.pt conf=0.25 source=path/to/video.mp4
```

To compress the resulting `.avi` for notebook playback:

```bash
ffmpeg -i runs/detect/predict/video.avi -vcodec libx264 result_compressed.mp4
```

---

## Tools and Libraries

| Library | Role |
|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8 training, validation, and inference engine |
| [Roboflow](https://roboflow.com) | Dataset hosting, annotation export, and versioning |
| OpenCV | Image I/O and pre-processing |
| FFmpeg | Video re-encoding (AVI → MP4) |
| matplotlib / seaborn | Training curve and metric visualisation |
| IPython | Inline display of images and video inside notebooks |

---

## License

Released under the [MIT License](LICENSE).
