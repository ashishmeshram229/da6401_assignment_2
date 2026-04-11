"""
Interactive activation histogram for Section 2.1.
Uses wandb.plot.bar — fully interactive in W&B, no static images.
Run: python activation_histogram.py
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.classification import VGG11Classifier
from data.pets_dataset import OxfordIIITPetDataset
from torch.utils.data import DataLoader

PROJECT = "da6401-assignment2"

# ── Config ────────────────────────────────────────────────────
DATA_ROOT = "/content/drive/MyDrive/dl_ass2/data/oxford-iiit-pet"
CKPT      = "/content/drive/MyDrive/dl_ass2/checkpoints/classifier.pth"
NUM_BINS  = 60
# ─────────────────────────────────────────────────────────────


def get_activations(model, imgs, layer):
    acts = []
    h = layer.register_forward_hook(lambda m, i, o: acts.append(o.detach().cpu()))
    with torch.no_grad():
        model(imgs)
    h.remove()
    return acts[0].numpy().flatten()


def make_histogram_table(values, num_bins, run_label):
    """
    Converts raw activation values into a wandb.Table suitable for bar chart.
    Returns table with columns ["bin_center", "count", "run"]
    """
    counts, edges = np.histogram(values, bins=num_bins)
    centers = [(edges[i] + edges[i+1]) / 2.0 for i in range(len(counts))]
    rows = [[round(float(c), 4), int(cnt), run_label]
            for c, cnt in zip(centers, counts)]
    return wandb.Table(data=rows, columns=["activation_value", "count", "run"])


def make_combined_histogram_table(values_dict, num_bins):
    """
    values_dict: {"label": np_array, ...}
    Returns single table with all runs — for overlaid bar chart
    """
    all_edges = np.histogram(
        np.concatenate(list(values_dict.values())), bins=num_bins)[1]

    rows = []
    for label, values in values_dict.items():
        counts, _ = np.histogram(values, bins=all_edges)
        centers = [(all_edges[i] + all_edges[i+1]) / 2.0 for i in range(len(counts))]
        for c, cnt in zip(centers, counts):
            rows.append([round(float(c), 4), int(cnt), label])

    return wandb.Table(data=rows, columns=["activation_value", "count", "run"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data
ds  = OxfordIIITPetDataset(DATA_ROOT, split="val", seed=42)
ldr = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
imgs = next(iter(ldr))["image"].to(device)
print(f"Loaded {imgs.shape[0]} images for activation extraction.")

# ── Load WITH BatchNorm ───────────────────────────────────────
print("Loading model WITH BatchNorm...")
model_bn = VGG11Classifier().to(device)
model_bn.load_state_dict(
    torch.load(CKPT, map_location=device)["state_dict"])
model_bn.eval()

# 3rd conv layer = block2's first Conv2d
acts_bn = get_activations(model_bn, imgs, model_bn.encoder.block2[0])
print(f"  With BN   — mean={acts_bn.mean():.4f}  std={acts_bn.std():.4f}")

# ── Load WITHOUT BatchNorm ────────────────────────────────────
print("Loading model WITHOUT BatchNorm...")
model_no_bn = VGG11Classifier().to(device)

def remove_bn(m):
    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            setattr(m, name, nn.Identity())
        else:
            remove_bn(child)

remove_bn(model_no_bn)
# Load with strict=False because BN params won't match Identity
model_no_bn.load_state_dict(
    torch.load(CKPT, map_location=device)["state_dict"], strict=False)
model_no_bn.eval()

acts_no_bn = get_activations(model_no_bn, imgs, model_no_bn.encoder.block2[0])
print(f"  Without BN — mean={acts_no_bn.mean():.4f}  std={acts_no_bn.std():.4f}")

# ── Log to W&B ───────────────────────────────────────────────
print("\nLogging to W&B...")
wandb.init(project=PROJECT, group="report_section_21",
           name="activation_histograms_interactive", reinit="finish_previous")

# Combined overlaid histogram table
combined_tbl = make_combined_histogram_table(
    {"With BatchNorm": acts_bn, "Without BatchNorm": acts_no_bn},
    num_bins=NUM_BINS
)

# Interactive bar chart — overlaid
wandb.log({"s2_1/activation_histogram_overlay": wandb.plot.bar(
    combined_tbl,
    label="activation_value",
    value="count",
    title="Section 2.1 — 3rd Conv Activation Distribution: BN vs No BN"
)})

# Separate table for each — individual bar charts
tbl_bn = make_histogram_table(acts_bn, NUM_BINS, "With BatchNorm")
wandb.log({"s2_1/activation_histogram_with_bn": wandb.plot.bar(
    tbl_bn,
    label="activation_value",
    value="count",
    title="Section 2.1 — Activations WITH BatchNorm (3rd Conv Layer)"
)})

tbl_no_bn = make_histogram_table(acts_no_bn, NUM_BINS, "Without BatchNorm")
wandb.log({"s2_1/activation_histogram_no_bn": wandb.plot.bar(
    tbl_no_bn,
    label="activation_value",
    value="count",
    title="Section 2.1 — Activations WITHOUT BatchNorm (3rd Conv Layer)"
)})

# Also log summary stats as a table
stats_table = wandb.Table(
    columns=["Model", "Mean", "Std", "Min", "Max"],
    data=[
        ["With BatchNorm",    round(float(acts_bn.mean()),4),
         round(float(acts_bn.std()),4),
         round(float(acts_bn.min()),4),
         round(float(acts_bn.max()),4)],
        ["Without BatchNorm", round(float(acts_no_bn.mean()),4),
         round(float(acts_no_bn.std()),4),
         round(float(acts_no_bn.min()),4),
         round(float(acts_no_bn.max()),4)],
    ]
)
wandb.log({"s2_1/activation_stats": stats_table})

wandb.finish()
print("\nDone. Logged:")
print("  s2_1/activation_histogram_overlay   <- overlaid bar chart")
print("  s2_1/activation_histogram_with_bn   <- BN only")
print("  s2_1/activation_histogram_no_bn     <- No BN only")
print("  s2_1/activation_stats               <- summary table")
print("\nEmbed these panels in your W&B report under Section 2.1.")