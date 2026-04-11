"""
Generates fully interactive W&B charts for all report sections.
Run in Colab: python generate_interactive_report.py
"""

import wandb
import pandas as pd
import numpy as np

ENTITY  = "ashishmeshram229-indian-institute-of-technology-madras"
PROJECT = "da6401-assignment2"

api = wandb.Api()

BEST_RUNS = {
    "task1_bn_dp0.5":     "nzd3dn65",
    "task1_no_bn_dp0.5":  "87h1o0zo",
    "task1_bn_no_dropout": "hzzx7wsq",
    "task1_bn_dp0.2":     "mm0hfc3i",
    "task2":              "x8jv5vg5",
    "task3_frozen":       "ylw0ff9g",
    "task3_partial":      "nof0qn8o",
    "task3_full":         "7djej2cl",
}


def fetch(run_id, keys):
    run  = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(samples=500, pandas=True)
    hist["epoch"] = range(1, len(hist) + 1)
    return hist


def to_table(histories_dict, x_key, y_key):
    """
    histories_dict: {"Run Name": dataframe, ...}
    Returns wandb.Table with columns [x_key, y_key, "run"]
    """
    rows = []
    for run_name, hist in histories_dict.items():
        if y_key not in hist.columns:
            continue
        for _, row in hist.iterrows():
            xval = row.get(x_key, row.get("epoch", None))
            yval = row.get(y_key, None)
            if xval is not None and yval is not None and not np.isnan(float(yval)):
                rows.append([int(xval), float(yval), run_name])
    table = wandb.Table(data=rows, columns=[x_key, y_key, "run"])
    return table


# ─────────────────────────────────────────────────────────────
# SECTION 2.1 — BN vs No BN
# ─────────────────────────────────────────────────────────────
print("Logging Section 2.1...")
wandb.init(project=PROJECT, group="report_section_21",
           name="bn_comparison_interactive", reinit="finish_previous")

hist_bn    = fetch(BEST_RUNS["task1_bn_dp0.5"],    ["epoch","train/loss","val/loss","val/f1"])
hist_no_bn = fetch(BEST_RUNS["task1_no_bn_dp0.5"], ["epoch","train/loss","val/loss","val/f1"])

histories_21 = {
    "With BatchNorm":    hist_bn,
    "Without BatchNorm": hist_no_bn,
}

# Interactive line chart: val/loss
table_21_loss = to_table(histories_21, "epoch", "val/loss")
wandb.log({"section2_1/val_loss_bn_comparison": wandb.plot.line(
    table_21_loss, x="epoch", y="val/loss",
    stroke="run",
    title="Section 2.1 — Val Loss: BatchNorm vs No BatchNorm"
)})

# Interactive line chart: val/f1
table_21_f1 = to_table(histories_21, "epoch", "val/f1")
wandb.log({"section2_1/val_f1_bn_comparison": wandb.plot.line(
    table_21_f1, x="epoch", y="val/f1",
    stroke="run",
    title="Section 2.1 — Val F1: BatchNorm vs No BatchNorm"
)})

# Interactive line chart: train/loss
table_21_tr = to_table(histories_21, "epoch", "train/loss")
wandb.log({"section2_1/train_loss_bn_comparison": wandb.plot.line(
    table_21_tr, x="epoch", y="train/loss",
    stroke="run",
    title="Section 2.1 — Train Loss: BatchNorm vs No BatchNorm"
)})

wandb.finish()
print("  Section 2.1 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.2 — Dropout comparison
# ─────────────────────────────────────────────────────────────
print("Logging Section 2.2...")
wandb.init(project=PROJECT, group="report_section_22",
           name="dropout_comparison_interactive", reinit="finish_previous")

hist_dp0  = fetch(BEST_RUNS["task1_bn_no_dropout"], ["epoch","train/loss","val/loss","val/f1"])
hist_dp02 = fetch(BEST_RUNS["task1_bn_dp0.2"],      ["epoch","train/loss","val/loss","val/f1"])
hist_dp05 = fetch(BEST_RUNS["task1_bn_dp0.5"],      ["epoch","train/loss","val/loss","val/f1"])

histories_22 = {
    "No Dropout":    hist_dp0,
    "Dropout p=0.2": hist_dp02,
    "Dropout p=0.5": hist_dp05,
}

table_22_val = to_table(histories_22, "epoch", "val/loss")
wandb.log({"section2_2/val_loss_dropout": wandb.plot.line(
    table_22_val, x="epoch", y="val/loss",
    stroke="run",
    title="Section 2.2 — Val Loss: Dropout Comparison"
)})

table_22_tr = to_table(histories_22, "epoch", "train/loss")
wandb.log({"section2_2/train_loss_dropout": wandb.plot.line(
    table_22_tr, x="epoch", y="train/loss",
    stroke="run",
    title="Section 2.2 — Train Loss: Dropout Comparison"
)})

table_22_f1 = to_table(histories_22, "epoch", "val/f1")
wandb.log({"section2_2/val_f1_dropout": wandb.plot.line(
    table_22_f1, x="epoch", y="val/f1",
    stroke="run",
    title="Section 2.2 — Val Macro F1: Dropout Comparison"
)})

wandb.finish()
print("  Section 2.2 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.3 — Transfer learning strategies
# ─────────────────────────────────────────────────────────────
print("Logging Section 2.3...")
wandb.init(project=PROJECT, group="report_section_23",
           name="transfer_learning_interactive", reinit="finish_previous")

hist_frozen  = fetch(BEST_RUNS["task3_frozen"],  ["epoch","val/dice","val/pixel_acc","val/loss","train/dice"])
hist_partial = fetch(BEST_RUNS["task3_partial"], ["epoch","val/dice","val/pixel_acc","val/loss","train/dice"])
hist_full    = fetch(BEST_RUNS["task3_full"],    ["epoch","val/dice","val/pixel_acc","val/loss","train/dice"])

histories_23 = {
    "Frozen backbone":   hist_frozen,
    "Partial fine-tune": hist_partial,
    "Full fine-tune":    hist_full,
}

table_23_dice = to_table(histories_23, "epoch", "val/dice")
wandb.log({"section2_3/val_dice_strategies": wandb.plot.line(
    table_23_dice, x="epoch", y="val/dice",
    stroke="run",
    title="Section 2.3 — Val Dice: Transfer Learning Strategies"
)})

table_23_acc = to_table(histories_23, "epoch", "val/pixel_acc")
wandb.log({"section2_3/val_pixel_acc_strategies": wandb.plot.line(
    table_23_acc, x="epoch", y="val/pixel_acc",
    stroke="run",
    title="Section 2.3 — Val Pixel Accuracy: Transfer Learning Strategies"
)})

table_23_loss = to_table(histories_23, "epoch", "val/loss")
wandb.log({"section2_3/val_loss_strategies": wandb.plot.line(
    table_23_loss, x="epoch", y="val/loss",
    stroke="run",
    title="Section 2.3 — Val Loss: Transfer Learning Strategies"
)})

# Train vs Val Dice for full fine-tune specifically
histories_23_tv = {
    "Train Dice (Full)": hist_full.rename(columns={"train/dice": "val/dice"}),
    "Val Dice (Full)":   hist_full,
}
table_23_tv = to_table(
    {"Train Dice": hist_full.rename(columns={"train/dice": "metric"}).assign(metric=hist_full.get("train/dice", hist_full.get("val/dice"))),
     "Val Dice":   hist_full},
    "epoch", "val/dice"
)
wandb.log({"section2_3/full_train_vs_val_dice": wandb.plot.line(
    table_23_tv, x="epoch", y="val/dice",
    stroke="run",
    title="Section 2.3 — Full Fine-tune: Train vs Val Dice"
)})

wandb.finish()
print("  Section 2.3 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.8 — Meta-analysis all tasks
# ─────────────────────────────────────────────────────────────
print("Logging Section 2.8...")
wandb.init(project=PROJECT, group="report_section_28",
           name="meta_analysis_interactive", reinit="finish_previous")

hist_t1 = fetch(BEST_RUNS["task1_bn_dp0.5"], ["epoch","train/loss","val/loss","val/f1"])
hist_t2 = fetch(BEST_RUNS["task2"],          ["epoch","train/loss","val/loss","val/iou","train/iou"])
hist_t3 = fetch(BEST_RUNS["task3_full"],     ["epoch","train/loss","val/loss","val/dice","train/dice"])

# Classification metrics
table_t1_f1 = to_table({"Classification": hist_t1}, "epoch", "val/f1")
wandb.log({"section2_8/task1_val_f1": wandb.plot.line(
    table_t1_f1, x="epoch", y="val/f1",
    stroke="run",
    title="Section 2.8 — Classification Val Macro F1"
)})

table_t1_loss = to_table(
    {"Train Loss": hist_t1.rename(columns={"train/loss":"val/loss"}),
     "Val Loss":   hist_t1},
    "epoch", "val/loss"
)
wandb.log({"section2_8/task1_train_val_loss": wandb.plot.line(
    table_t1_loss, x="epoch", y="val/loss",
    stroke="run",
    title="Section 2.8 — Classification Train vs Val Loss"
)})

# Localization metrics
table_t2_iou = to_table({"Localization": hist_t2}, "epoch", "val/iou")
wandb.log({"section2_8/task2_val_iou": wandb.plot.line(
    table_t2_iou, x="epoch", y="val/iou",
    stroke="run",
    title="Section 2.8 — Localization Val IoU"
)})

# Segmentation metrics
table_t3_dice = to_table({"Segmentation": hist_t3}, "epoch", "val/dice")
wandb.log({"section2_8/task3_val_dice": wandb.plot.line(
    table_t3_dice, x="epoch", y="val/dice",
    stroke="run",
    title="Section 2.8 — Segmentation Val Dice"
)})

# Dice vs Pixel Accuracy comparison
rows_dice_vs_acc = []
for _, row in hist_t3.iterrows():
    ep = int(row.get("epoch", 0))
    if "val/dice" in hist_t3.columns and not np.isnan(float(row.get("val/dice", float("nan")))):
        rows_dice_vs_acc.append([ep, float(row["val/dice"]),    "Val Dice"])
    if "val/pixel_acc" in hist_t3.columns and not np.isnan(float(row.get("val/pixel_acc", float("nan")))):
        rows_dice_vs_acc.append([ep, float(row["val/pixel_acc"]), "Pixel Accuracy"])

table_26 = wandb.Table(data=rows_dice_vs_acc, columns=["epoch", "score", "metric"])
wandb.log({"section2_6/dice_vs_pixel_acc": wandb.plot.line(
    table_26, x="epoch", y="score",
    stroke="metric",
    title="Section 2.6 — Dice Score vs Pixel Accuracy (why Dice is better)"
)})

wandb.finish()
print("  Section 2.8 done.")
print("  Section 2.6 Dice vs PixelAcc also logged.")


print("\n===== ALL INTERACTIVE CHARTS LOGGED =====")
print("Sections auto-logged: 2.1, 2.2, 2.3, 2.6 (Dice chart), 2.8")
print("Now go to W&B UI and embed these in your report.")
print("Still need:")
print("  Section 2.4: python train.py --task report  (feature maps)")
print("  Section 2.5: python train.py --task report  (bbox table)")
print("  Section 2.6: python train.py --task report  (seg sample images)")
print("  Section 2.7: run wild images script")