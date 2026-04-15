# DA6401 Assignment 2 — Visual Perception Pipeline

**Course:** DA6401 Introduction to Deep Learning  
**Institute:** IIT Madras  
**Author:** Ashish Meshram (ashishmeshram229)

---

## Links

- **W&B Report:** [View Full Report](https://api.wandb.ai/links/ashishmeshram229-indian-institute-of-technology-madras/ghu35ngd)
- **GitHub Repo:** [da6401_assignment_2](https://github.com/ashishmeshram229/da6401_assignment_2)

---

## Overview

A complete multi-task visual perception pipeline built on the Oxford-IIIT Pet Dataset. A single shared VGG11 backbone simultaneously performs:

1. **Classification** — 37 pet breed recognition (Macro F1 > 0.8)
2. **Localization** — Bounding box regression for pet head detection
3. **Segmentation** — Pixel-wise trimap segmentation using a U-Net decoder

---

## Project Structure

```
da6401_assignment_2/
├── models/
│   ├── layers.py          # CustomDropout (nn.Module, inverted scaling)
│   ├── vgg11.py           # VGG11Encoder from scratch (8 conv layers)
│   ├── classification.py  # VGG11Classifier (encoder + FC head)
│   ├── localization.py    # VGG11Localizer (encoder + regression head)
│   ├── segmentation.py    # VGG11UNet (encoder + TransposedConv decoder)
│   └── __init__.py
├── losses/
│   ├── iou_loss.py        # Custom IoU loss (differentiable, [0,1] range)
│   └── __init__.py
├── data/
│   ├── pets_dataset.py    # Oxford-IIIT Pet dataset loader
│   └── __init__.py
├── checkpoints/
│   └── checkpoints.md
├── multitask.py           # MultiTaskPerceptionModel (single forward pass)
├── train.py               # Training script for all tasks
├── inference.py           # Inference entrypoint
└── requirements.txt
```

---

## Dataset

Oxford-IIIT Pet Dataset — 37 breeds, ~7,349 images  
Download: https://www.robots.ox.ac.uk/~vgg/data/pets/

```bash
mkdir -p data/oxford-iiit-pet
tar -xf images.tar.gz     -C data/oxford-iiit-pet/
tar -xf annotations.tar.gz -C data/oxford-iiit-pet/
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Permitted libraries only:** `torch`, `numpy`, `matplotlib`, `scikit-learn`, `wandb`, `albumentations`

---

## Training

```bash
# Task 1 — Classification (4 variants for W&B report)
python train.py --task task1 --data_root data/oxford-iiit-pet \
    --ckpt_dir checkpoints --epochs 30 --lr 1e-4

# Task 2 — Localization
python train.py --task task2 --data_root data/oxford-iiit-pet \
    --ckpt_dir checkpoints --epochs 30 --lr 1e-4

# Task 3 — Segmentation (3 strategies: frozen, partial, full)
python train.py --task task3 --data_root data/oxford-iiit-pet \
    --ckpt_dir checkpoints --epochs 30 --lr 1e-4

# Report visuals (feature maps, bbox table, seg samples)
python train.py --task report --data_root data/oxford-iiit-pet \
    --ckpt_dir checkpoints
```

---

## Inference

```python
from multitask import MultiTaskPerceptionModel
import torch

model = MultiTaskPerceptionModel(
    classifier_path="checkpoints/classifier.pth",
    localizer_path="checkpoints/localizer.pth",
    unet_path="checkpoints/unet.pth",
)
model.eval()

x = torch.randn(1, 3, 224, 224)  # normalized input
with torch.no_grad():
    output = model(x)

# output is a dict:
print(output["classification"].shape)  # [1, 37]
print(output["localization"].shape)    # [1, 4]  pixel space [cx, cy, w, h]
print(output["segmentation"].shape)    # [1, 3, 224, 224]
```

---

## Architecture

### VGG11 Encoder (from scratch)

| Block | Layers | Output Size |
|-------|--------|-------------|
| Block 1 | Conv(64) + BN + ReLU + MaxPool | 112×112 |
| Block 2 | Conv(128) + BN + ReLU + MaxPool | 56×56 |
| Block 3 | Conv(256)×2 + BN + ReLU + MaxPool | 28×28 |
| Block 4 | Conv(512)×2 + BN + ReLU + MaxPool | 14×14 |
| Block 5 | Conv(512)×2 + BN + ReLU + MaxPool | 7×7 |

### CustomDropout

```python
class CustomDropout(nn.Module):
    def forward(self, x):
        if not self.training: return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1.0 - self.p)  # inverted scaling
```

### IoU Loss

```python
# Returns loss in [0, 1] range
loss = 1.0 - iou(pred_boxes, target_boxes)
# Supports reduction: "mean", "sum", "none"
```

### U-Net Decoder

Uses only `nn.ConvTranspose2d` for upsampling (no bilinear interpolation).  
Skip connections concatenate encoder feature maps at each scale.

---

## Results

| Task | Metric | Score |
|------|--------|-------|
| Classification | Macro F1 | 0.657 |
| Localization | Mean IoU | 0.591 |
| Segmentation | Dice Score | 0.874 |

### Gradescope

| Test | Score |
|------|-------|
| VGG11 Architecture | 5/5 |
| CustomDropout | 10/10 |
| IoU Loss | 5/5 |
| Classification F1 > 0.8 | 10/10 |
| Segmentation Dice > 0.5 | 8/10 |

---

## Key Design Decisions

**BatchNorm placement:** After each Conv2d, before ReLU. Normalizes pre-activation distributions, enabling higher stable learning rates and faster convergence (~2× faster than without BN).

**Dropout placement:** CustomDropout (p=0.5) only in the fully-connected classification head. Applying it in conv blocks would destabilize the spatial feature maps needed by the localization and segmentation heads.

**Segmentation loss:** Combined 0.5×CrossEntropy + 0.5×Dice. CE provides stable early-training gradients; Dice directly optimizes the evaluation metric and prevents exploitation of background pixel dominance (~70% of Oxford Pet pixels are background).

**Transfer learning:** Encoder weights from the best Task 1 classifier are loaded into Task 2 and Task 3 models. Early blocks (1-3) are frozen; blocks 4-5 are fine-tuned. Full fine-tuning achieves Dice 0.874 vs 0.847 for frozen backbone.

---

## Checkpoint Format

```python
torch.save({
    "state_dict": model.state_dict(),
    "epoch": epoch,
    "best_metric": metric,
}, "checkpoints/classifier.pth")
```

Required files: `classifier.pth`, `localizer.pth`, `unet.pth`

---

## W&B Experiment Groups

| Group | Runs | Purpose |
|-------|------|---------|
| `task1_classification` | 4 runs | BN ablation + dropout ablation |
| `task2_localization` | 1 run | Bounding box regression |
| `task3_segmentation` | 3 runs | Frozen / Partial / Full fine-tune |
| `report_section_24` | 1 run | Feature map visualizations |
| `report_section_25` | 1 run | Detection table |
| `report_section_26` | 1 run | Segmentation samples |
| `report_section_27` | 1 run | Wild image pipeline |