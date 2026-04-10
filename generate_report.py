"""
Run this in Colab after wandb.login()
Generates ALL report sections using your best runs by ID
"""

import wandb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

api     = wandb.Api()
ENTITY  = "ashishmeshram229-indian-institute-of-technology-madras"
PROJECT = "da6401-assignment2"

# ── Best run IDs from your scan ──────────────────────────────
BEST_RUNS = {
    "task1_bn_dp0.5":      "nzd3dn65",
    "task1_no_bn_dp0.5":   "87h1o0zo",
    "task1_bn_no_dropout":  "hzzx7wsq",
    "task1_bn_dp0.2":       "mm0hfc3i",
    "task2":               "x8jv5vg5",
    "task3_frozen":        "ylw0ff9g",
    "task3_partial":       "nof0qn8o",
    "task3_full":          "7djej2cl",
}

def get_history_by_id(run_id):
    run  = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(samples=500, pandas=True)
    return hist, run

def finish(msg):
    wandb.finish()
    print(f"Done: {msg}")

# ─────────────────────────────────────────────────────────────
# SECTION 2.1 — BatchNorm vs No-BatchNorm
# ─────────────────────────────────────────────────────────────
print("\n=== Section 2.1 ===")
wandb.init(project=PROJECT, group="report_section_21",
           name="bn_vs_no_bn", reinit="finish_previous")

hist_bn,    _ = get_history_by_id(BEST_RUNS["task1_bn_dp0.5"])
hist_no_bn, _ = get_history_by_id(BEST_RUNS["task1_no_bn_dp0.5"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, hist, label, col in [
    (axes[0], hist_bn,    "With BatchNorm",    "#1D9E75"),
    (axes[1], hist_no_bn, "Without BatchNorm", "#E24B4A"),
]:
    if "train/loss" in hist.columns:
        ax.plot(hist["train/loss"].dropna().values, label="Train loss", color=col, linestyle="--", alpha=0.7)
    if "val/loss" in hist.columns:
        ax.plot(hist["val/loss"].dropna().values, label="Val loss", color=col)
    if "val/f1" in hist.columns:
        ax2 = ax.twinx()
        ax2.plot(hist["val/f1"].dropna().values, label="Val F1", color="gray", linestyle=":", linewidth=1.5)
        ax2.set_ylabel("Val Macro F1", color="gray")
        ax2.tick_params(axis='y', labelcolor='gray')
    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=8)

plt.suptitle("Effect of BatchNorm on Training Convergence", fontsize=14)
plt.tight_layout()
wandb.log({"section2_1/bn_vs_no_bn_curves": wandb.Image(fig)})
plt.close()
finish("Section 2.1")

# ─────────────────────────────────────────────────────────────
# SECTION 2.2 — Dropout comparison (3 runs overlaid)
# ─────────────────────────────────────────────────────────────
print("\n=== Section 2.2 ===")
wandb.init(project=PROJECT, group="report_section_22",
           name="dropout_comparison", reinit="finish_previous")

hist_dp0,  _ = get_history_by_id(BEST_RUNS["task1_bn_no_dropout"])
hist_dp02, _ = get_history_by_id(BEST_RUNS["task1_bn_dp0.2"])
hist_dp05, _ = get_history_by_id(BEST_RUNS["task1_bn_dp0.5"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

runs_2_2 = [
    (hist_dp0,  "No Dropout",     "#E24B4A"),
    (hist_dp02, "Dropout p=0.2",  "#378ADD"),
    (hist_dp05, "Dropout p=0.5",  "#1D9E75"),
]

for hist, label, col in runs_2_2:
    if "train/loss" in hist.columns:
        axes[0].plot(hist["train/loss"].dropna().values, label=f"Train {label}", color=col, linestyle="--", alpha=0.5)
    if "val/loss" in hist.columns:
        axes[0].plot(hist["val/loss"].dropna().values, label=f"Val {label}", color=col)
    if "val/f1" in hist.columns:
        axes[1].plot(hist["val/f1"].dropna().values, label=label, color=col)

axes[0].set_title("Train vs Val Loss — Dropout Comparison")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=7)

axes[1].set_title("Validation Macro F1 — Dropout Comparison")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
axes[1].legend(fontsize=8)

plt.tight_layout()
wandb.log({"section2_2/dropout_overlay": wandb.Image(fig)})
plt.close()
finish("Section 2.2")

# ─────────────────────────────────────────────────────────────
# SECTION 2.3 — Transfer learning strategies
# ─────────────────────────────────────────────────────────────
print("\n=== Section 2.3 ===")
wandb.init(project=PROJECT, group="report_section_23",
           name="transfer_learning", reinit="finish_previous")

hist_frozen,  _ = get_history_by_id(BEST_RUNS["task3_frozen"])
hist_partial, _ = get_history_by_id(BEST_RUNS["task3_partial"])
hist_full,    _ = get_history_by_id(BEST_RUNS["task3_full"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

runs_2_3 = [
    (hist_frozen,  "Frozen backbone",   "#E24B4A"),
    (hist_partial, "Partial fine-tune", "#378ADD"),
    (hist_full,    "Full fine-tune",    "#1D9E75"),
]

# Plot 1: Val Dice overlay
for hist, label, col in runs_2_3:
    if "val/dice" in hist.columns:
        axes[0].plot(hist["val/dice"].dropna().values, label=label, color=col)
axes[0].set_title("Val Dice Score — All Strategies")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice Score")
axes[0].legend(); axes[0].set_ylim([0.7, 1.0])

# Plot 2: Val Pixel Accuracy overlay
for hist, label, col in runs_2_3:
    if "val/pixel_acc" in hist.columns:
        axes[1].plot(hist["val/pixel_acc"].dropna().values, label=label, color=col)
axes[1].set_title("Val Pixel Accuracy — All Strategies")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Pixel Accuracy")
axes[1].legend()

# Plot 3: Train vs Val Dice for best strategy (full)
if "train/dice" in hist_full.columns:
    axes[2].plot(hist_full["train/dice"].dropna().values, label="Train Dice", color="#1D9E75", linestyle="--")
if "val/dice" in hist_full.columns:
    axes[2].plot(hist_full["val/dice"].dropna().values, label="Val Dice", color="#1D9E75")
axes[2].set_title("Train vs Val Dice — Full Fine-tune")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Dice Score")
axes[2].legend()

plt.tight_layout()
wandb.log({"section2_3/transfer_learning_comparison": wandb.Image(fig)})
plt.close()
finish("Section 2.3")

# ─────────────────────────────────────────────────────────────
# SECTION 2.8 — Meta-analysis summary plots
# ─────────────────────────────────────────────────────────────
print("\n=== Section 2.8 ===")
wandb.init(project=PROJECT, group="report_section_28",
           name="meta_analysis", reinit="finish_previous")

hist_t1, _ = get_history_by_id(BEST_RUNS["task1_bn_dp0.5"])
hist_t2, _ = get_history_by_id(BEST_RUNS["task2"])
hist_t3, _ = get_history_by_id(BEST_RUNS["task3_full"])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: all three tasks val metrics
if "val/f1" in hist_t1.columns:
    axes[0,0].plot(hist_t1["val/f1"].dropna().values, color="#1D9E75")
    axes[0,0].set_title("Classification — Val Macro F1")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("F1")

if "val/iou" in hist_t2.columns:
    axes[0,1].plot(hist_t2["val/iou"].dropna().values, color="#378ADD")
    axes[0,1].set_title("Localization — Val IoU")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("IoU")

if "val/dice" in hist_t3.columns:
    axes[0,2].plot(hist_t3["val/dice"].dropna().values, color="#BA7517")
    axes[0,2].set_title("Segmentation — Val Dice (Full)")
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("Dice")

# Row 2: train vs val for each task
if "train/loss" in hist_t1.columns and "val/loss" in hist_t1.columns:
    axes[1,0].plot(hist_t1["train/loss"].dropna().values, label="Train", linestyle="--", color="#1D9E75", alpha=0.7)
    axes[1,0].plot(hist_t1["val/loss"].dropna().values,   label="Val",   color="#1D9E75")
    axes[1,0].set_title("Classification — Train vs Val Loss")
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("Loss"); axes[1,0].legend()

if "train/loss" in hist_t2.columns and "val/loss" in hist_t2.columns:
    axes[1,1].plot(hist_t2["train/loss"].dropna().values, label="Train", linestyle="--", color="#378ADD", alpha=0.7)
    axes[1,1].plot(hist_t2["val/loss"].dropna().values,   label="Val",   color="#378ADD")
    axes[1,1].set_title("Localization — Train vs Val Loss")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Loss"); axes[1,1].legend()

if "train/dice" in hist_t3.columns and "val/dice" in hist_t3.columns:
    axes[1,2].plot(hist_t3["train/dice"].dropna().values, label="Train", linestyle="--", color="#BA7517", alpha=0.7)
    axes[1,2].plot(hist_t3["val/dice"].dropna().values,   label="Val",   color="#BA7517")
    axes[1,2].set_title("Segmentation — Train vs Val Dice")
    axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Dice"); axes[1,2].legend()

plt.suptitle("Meta-Analysis: All Tasks Summary", fontsize=15)
plt.tight_layout()
wandb.log({"section2_8/meta_summary_plots": wandb.Image(fig)})
plt.close()
finish("Section 2.8")

print("\n===========================")
print("ALL AUTOMATED SECTIONS DONE")
print("Still needed manually:")
print("  - Section 2.1: Activation histogram (separate script below)")
print("  - Section 2.4: Feature maps (run train.py --task report)")
print("  - Section 2.5: BBox table   (run train.py --task report)")
print("  - Section 2.6: Seg samples  (run train.py --task report)")
print("  - Section 2.7: Wild images  (3 photos from internet)")
print("===========================")