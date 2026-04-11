import wandb
import numpy as np
import pandas as pd

ENTITY  = "ashishmeshram229-indian-institute-of-technology-madras"
PROJECT = "da6401-assignment2"

api = wandb.Api()

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


def fetch(run_id):
    run  = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(samples=500, pandas=True)
    hist = hist.reset_index(drop=True)
    hist["epoch"] = range(1, len(hist) + 1)
    return hist


def make_table(run_histories, y_key):
    """
    run_histories: list of (label_str, dataframe)
    y_key: column name to plot on y axis
    Returns wandb.Table with columns ["epoch", y_key, "run"]
    """
    rows = []
    for label, hist in run_histories:
        if y_key not in hist.columns:
            continue
        for i, row in hist.iterrows():
            ep = int(row["epoch"])
            yv = row[y_key]
            if pd.isna(yv):
                continue
            rows.append([ep, float(yv), label])
    return wandb.Table(data=rows, columns=["epoch", y_key, "run"])


def log_line(key, run_histories, y_key, title):
    tbl = make_table(run_histories, y_key)
    wandb.log({key: wandb.plot.line(tbl, x="epoch", y=y_key, stroke="run", title=title)})


def make_table_two_metrics(hist, col_a, col_b, label_a, label_b):
    """For plotting two metrics from same run as separate lines"""
    rows = []
    for i, row in hist.iterrows():
        ep = int(row["epoch"])
        if col_a in hist.columns and not pd.isna(row.get(col_a)):
            rows.append([ep, float(row[col_a]), label_a])
        if col_b in hist.columns and not pd.isna(row.get(col_b)):
            rows.append([ep, float(row[col_b]), label_b])
    return wandb.Table(data=rows, columns=["epoch", "value", "metric"])


# ─────────────────────────────────────────────────────────────
print("Fetching run histories...")
h_bn     = fetch(BEST_RUNS["task1_bn_dp0.5"])
h_no_bn  = fetch(BEST_RUNS["task1_no_bn_dp0.5"])
h_dp0    = fetch(BEST_RUNS["task1_bn_no_dropout"])
h_dp02   = fetch(BEST_RUNS["task1_bn_dp0.2"])
h_dp05   = h_bn
h_t2     = fetch(BEST_RUNS["task2"])
h_frozen = fetch(BEST_RUNS["task3_frozen"])
h_partial= fetch(BEST_RUNS["task3_partial"])
h_full   = fetch(BEST_RUNS["task3_full"])
print("All histories fetched.\n")


# ─────────────────────────────────────────────────────────────
# SECTION 2.1 — BatchNorm vs No BatchNorm
# ─────────────────────────────────────────────────────────────
print("Section 2.1...")
wandb.init(project=PROJECT, group="report_section_21",
           name="bn_vs_no_bn", reinit="finish_previous")

bn_runs = [("With BatchNorm", h_bn), ("Without BatchNorm", h_no_bn)]

log_line("s2_1/val_loss",  bn_runs, "val/loss",  "2.1 — Val Loss: BN vs No BN")
log_line("s2_1/train_loss",bn_runs, "train/loss", "2.1 — Train Loss: BN vs No BN")
log_line("s2_1/val_f1",    bn_runs, "val/f1",     "2.1 — Val F1: BN vs No BN")

wandb.finish()
print("  Section 2.1 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.2 — Dropout comparison
# ─────────────────────────────────────────────────────────────
print("Section 2.2...")
wandb.init(project=PROJECT, group="report_section_22",
           name="dropout_comparison", reinit="finish_previous")

dp_runs = [
    ("No Dropout",    h_dp0),
    ("Dropout p=0.2", h_dp02),
    ("Dropout p=0.5", h_dp05),
]

log_line("s2_2/val_loss",   dp_runs, "val/loss",   "2.2 — Val Loss: Dropout Comparison")
log_line("s2_2/train_loss", dp_runs, "train/loss",  "2.2 — Train Loss: Dropout Comparison")
log_line("s2_2/val_f1",     dp_runs, "val/f1",      "2.2 — Val F1: Dropout Comparison")

wandb.finish()
print("  Section 2.2 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.3 — Transfer learning strategies
# ─────────────────────────────────────────────────────────────
print("Section 2.3...")
wandb.init(project=PROJECT, group="report_section_23",
           name="transfer_learning", reinit="finish_previous")

seg_runs = [
    ("Frozen backbone",   h_frozen),
    ("Partial fine-tune", h_partial),
    ("Full fine-tune",    h_full),
]

log_line("s2_3/val_dice",      seg_runs, "val/dice",      "2.3 — Val Dice: Transfer Strategies")
log_line("s2_3/val_pixel_acc", seg_runs, "val/pixel_acc", "2.3 — Val Pixel Acc: Transfer Strategies")
log_line("s2_3/val_loss",      seg_runs, "val/loss",      "2.3 — Val Loss: Transfer Strategies")
log_line("s2_3/train_dice",    seg_runs, "train/dice",    "2.3 — Train Dice: Transfer Strategies")

# Train vs Val Dice for full fine-tune
tbl_tv = make_table_two_metrics(h_full, "train/dice", "val/dice", "Train Dice", "Val Dice")
wandb.log({"s2_3/full_train_vs_val_dice": wandb.plot.line(
    tbl_tv, x="epoch", y="value", stroke="metric",
    title="2.3 — Full Fine-tune: Train vs Val Dice"
)})

wandb.finish()
print("  Section 2.3 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.6 — Dice vs Pixel Accuracy
# ─────────────────────────────────────────────────────────────
print("Section 2.6...")
wandb.init(project=PROJECT, group="report_section_26",
           name="dice_vs_pixel_acc", reinit="finish_previous")

tbl_26 = make_table_two_metrics(h_full, "val/dice", "val/pixel_acc", "Dice Score", "Pixel Accuracy")
wandb.log({"s2_6/dice_vs_pixel_acc": wandb.plot.line(
    tbl_26, x="epoch", y="value", stroke="metric",
    title="2.6 — Dice Score vs Pixel Accuracy (Segmentation)"
)})

wandb.finish()
print("  Section 2.6 done.")


# ─────────────────────────────────────────────────────────────
# SECTION 2.8 — Meta-analysis all tasks
# ─────────────────────────────────────────────────────────────
print("Section 2.8...")
wandb.init(project=PROJECT, group="report_section_28",
           name="meta_analysis", reinit="finish_previous")

# Task 1 — val F1
log_line("s2_8/task1_val_f1",   [("Classification", h_bn)],  "val/f1",  "2.8 — Task1 Val Macro F1")

# Task 1 — train vs val loss
tbl_t1 = make_table_two_metrics(h_bn, "train/loss", "val/loss", "Train Loss", "Val Loss")
wandb.log({"s2_8/task1_train_val_loss": wandb.plot.line(
    tbl_t1, x="epoch", y="value", stroke="metric",
    title="2.8 — Task1 Train vs Val Loss"
)})

# Task 2 — val IoU
log_line("s2_8/task2_val_iou",  [("Localization", h_t2)],   "val/iou", "2.8 — Task2 Val IoU")

# Task 2 — train vs val IoU
tbl_t2 = make_table_two_metrics(h_t2, "train/iou", "val/iou", "Train IoU", "Val IoU")
wandb.log({"s2_8/task2_train_val_iou": wandb.plot.line(
    tbl_t2, x="epoch", y="value", stroke="metric",
    title="2.8 — Task2 Train vs Val IoU"
)})

# Task 3 — val Dice
log_line("s2_8/task3_val_dice", [("Segmentation", h_full)], "val/dice", "2.8 — Task3 Val Dice")

# Task 3 — train vs val Dice
tbl_t3 = make_table_two_metrics(h_full, "train/dice", "val/dice", "Train Dice", "Val Dice")
wandb.log({"s2_8/task3_train_val_dice": wandb.plot.line(
    tbl_t3, x="epoch", y="value", stroke="metric",
    title="2.8 — Task3 Train vs Val Dice"
)})

# All 3 tasks val metrics on one chart (normalised view)
all_rows = []
for i, row in h_bn.iterrows():
    if "val/f1" in h_bn.columns and not pd.isna(row.get("val/f1")):
        all_rows.append([int(row["epoch"]), float(row["val/f1"]), "Classification F1"])
for i, row in h_t2.iterrows():
    if "val/iou" in h_t2.columns and not pd.isna(row.get("val/iou")):
        all_rows.append([int(row["epoch"]), float(row["val/iou"]), "Localization IoU"])
for i, row in h_full.iterrows():
    if "val/dice" in h_full.columns and not pd.isna(row.get("val/dice")):
        all_rows.append([int(row["epoch"]), float(row["val/dice"]), "Segmentation Dice"])

tbl_all = wandb.Table(data=all_rows, columns=["epoch", "metric_value", "task"])
wandb.log({"s2_8/all_tasks_summary": wandb.plot.line(
    tbl_all, x="epoch", y="metric_value", stroke="task",
    title="2.8 — All Tasks: F1 / IoU / Dice over Epochs"
)})

wandb.finish()
print("  Section 2.8 done.")


print("\n===== ALL INTERACTIVE CHARTS LOGGED =====")
print("Groups created:")
print("  report_section_21 -> s2_1/*")
print("  report_section_22 -> s2_2/*")
print("  report_section_23 -> s2_3/*")
print("  report_section_26 -> s2_6/dice_vs_pixel_acc")
print("  report_section_28 -> s2_8/*")
print("")
print("Still need (images/tables, not line charts):")
print("  2.1 activation histogram -> run activation_histogram.py")
print("  2.4 feature maps         -> python train.py --task report")
print("  2.5 bbox table           -> python train.py --task report")
print("  2.6 seg sample images    -> python train.py --task report")
print("  2.7 wild images          -> run wild_images.py")