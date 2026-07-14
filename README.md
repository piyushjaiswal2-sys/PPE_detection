# PPE Detection with YOLOv8

Object detection model that identifies personal protective equipment (PPE) compliance in workplace images — hardhats, safety vests, gloves, goggles, masks — and flags their absence, plus fall-detection, ladders, and safety cones.

## Overview

This project fine-tunes a **YOLOv8n** model on a 14-class PPE dataset to detect both PPE items being worn correctly and PPE violations (e.g. `NO-Hardhat`, `NO-Safety Vest`).

| | |
|---|---|
| **Base model** | `yolov8n.pt` (Ultralytics) |
| **Framework** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| **Task** | Object Detection |
| **Classes** | 14 |
| **Training environment** | Kaggle (Tesla T4 GPU) |

## Classes

`Fall-Detected`, `Gloves`, `Goggles`, `Hardhat`, `Ladder`, `Mask`, `NO-Gloves`, `NO-Goggles`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`, `Safety Cone`, `Safety Vest`

## Dataset

- Source: [PPE Dataset YOLOv8](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8) (Kaggle, Roboflow-formatted)
- For this baseline run, a random subset was used (seed=42):
  - **Train:** 500 images
  - **Validation:** 300 images

## Training Configuration

```
Model:      yolov8n.pt
Epochs:     20
Image size: 512
Batch size: 16
Workers:    2
Patience:   5 (early stopping)
```

## Results (Baseline)

Overall validation performance:

| Metric | Value |
|---|---|
| Precision | 0.592 |
| Recall | 0.511 |
| mAP@50 | 0.503 |
| mAP@50-95 | 0.282 |

Per-class breakdown:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Fall-Detected | 33 | 33 | 0.763 | 0.489 | 0.581 | 0.245 |
| Gloves | 10 | 22 | 0.720 | 0.909 | 0.887 | 0.388 |
| Goggles | 21 | 24 | 0.467 | 0.542 | 0.577 | 0.307 |
| Hardhat | 110 | 304 | 0.616 | 0.622 | 0.606 | 0.293 |
| Ladder | 8 | 8 | 0.685 | 0.625 | 0.628 | 0.448 |
| Mask | 6 | 6 | 0.393 | 0.667 | 0.547 | 0.369 |
| NO-Gloves | 17 | 31 | 0.389 | 0.350 | 0.330 | 0.154 |
| NO-Goggles | 24 | 31 | 0.401 | 0.454 | 0.341 | 0.160 |
| NO-Hardhat | 28 | 56 | 0.479 | 0.623 | 0.598 | 0.359 |
| NO-Mask | 10 | 20 | 0.459 | 0.200 | 0.206 | 0.144 |
| NO-Safety Vest | 8 | 16 | 1.000 | 0.000 | 0.035 | 0.012 |
| Person | 8 | 11 | 0.853 | 0.909 | 0.912 | 0.700 |
| Safety Cone | 16 | 139 | 0.552 | 0.468 | 0.408 | 0.180 |
| Safety Vest | 20 | 54 | 0.516 | 0.297 | 0.392 | 0.195 |

**Notes on results:** This is a baseline run on a small subset (500 train / 300 val images) for only 20 epochs, so scores are modest. `NO-Safety Vest` in particular has very few instances (16) and collapsed recall — more data and longer training would likely help most classes, especially the underrepresented ones (`Mask`, `Ladder`, `Person`).

## Repository Structure

```
.
├── best.pt                # Trained model weights
├── Unloq_task.ipynb       # Training / evaluation notebook
├── README.md
└── requirements.txt
```

## Usage

### Install

```bash
pip install ultralytics
```

### Run inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="path/to/image_or_folder", conf=0.4, save=True)
```

### Validate on your own data

```python
from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="path/to/data.yaml")
print(metrics.results_dict)
```

