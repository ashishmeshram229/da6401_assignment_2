import os
import sys
import argparse
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

SEED          = 42
IMAGE_SIZE    = 224
NUM_CLASSES   = 37
SEG_CLASSES   = 3
WANDB_PROJECT = "da6401-assignment2"


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_loaders(data_root, batch_size=32, num_workers=2, require_bbox=False):
    train_ds = OxfordIIITPetDataset(data_root, split="train", seed=SEED, require_bbox=require_bbox)
    val_ds   = OxfordIIITPetDataset(data_root, split="val",   seed=SEED, require_bbox=require_bbox)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def compute_iou_np(pred, gt, eps=1e-6):
    px1 = pred[:, 0] - pred[:, 2] / 2;  py1 = pred[:, 1] - pred[:, 3] / 2
    px2 = pred[:, 0] + pred[:, 2] / 2;  py2 = pred[:, 1] + pred[:, 3] / 2
    gx1 = gt[:, 0]   - gt[:, 2]   / 2;  gy1 = gt[:, 1]   - gt[:, 3]   / 2
    gx2 = gt[:, 0]   + gt[:, 2]   / 2;  gy2 = gt[:, 1]   + gt[:, 3]   / 2
    ix1 = np.maximum(px1, gx1);  iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2);  iy2 = np.minimum(py2, gy2)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    pa    = np.maximum(px2-px1, 0) * np.maximum(py2-py1, 0)
    ga    = np.maximum(gx2-gx1, 0) * np.maximum(gy2-gy1, 0)
    return inter / (pa + ga - inter + eps)


def dice_score(pred_mask, gt_mask, num_classes=3, eps=1e-6):
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float(); g = (gt_mask == c).float()
        inter = (p * g).sum(); denom = p.sum() + g.sum()
        scores.append(1.0 if denom < eps else ((2*inter+eps)/(denom+eps)).item())
    return float(np.mean(scores))

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C, H, W] (Raw, un-softmaxed outputs from the UNet)
        # targets: [B, H, W] (Integer class labels: 0, 1, 2)
        
        num_classes = logits.size(1)
        
        # 1. Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # 2. One-hot encode the targets so they match the shape of probs [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # 3. Calculate Intersection and Union (Cardinality) over the batch and spatial dims
        # dims=(0, 2, 3) means we average over the batch size, height, and width, leaving just the classes
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        # 4. Calculate Dice Score for each class, then take the mean
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return loss (1 - dice)
        return 1.0 - dice_score.mean()

def pixel_acc(pred, gt):
    return (pred == gt).float().mean().item()


def save_ckpt(model, epoch, metric, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "epoch": epoch,
                "best_metric": metric}, path)


def load_ckpt_if_exists(model, path, device):
    if os.path.exists(path):
        ckpt   = torch.load(path, map_location=device)
        sd     = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        epoch  = ckpt.get("epoch", 0)
        metric = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from {os.path.basename(path)} "
              f"(epoch {epoch}, metric {metric:.4f})")
        return epoch, metric
    return 0, 0.0


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss  = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]; g = (targets == c).float()
            inter = (p * g).sum(); denom = p.sum() + g.sum()
            loss += 1.0 - (2.0*inter+self.eps) / (denom+self.eps)
        return loss / self.num_classes


# ─────────────────────────────────────────────────────────────
# TASK 1 — Classification
# ─────────────────────────────────────────────────────────────

def train_task1(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    runs = [
        # Best config trains for full epochs — this becomes classifier.pth
        {"dropout_p": 0.5, "use_bn": True,  "run_name": "task1_bn_dp0.5",
         "epochs_override": args.epochs},
        # Ablation configs: 15 epochs each — enough for W&B report comparison
        {"dropout_p": 0.5, "use_bn": False, "run_name": "task1_no_bn_dp0.5",
         "epochs_override": 15},
        {"dropout_p": 0.0, "use_bn": True,  "run_name": "task1_bn_no_dropout",
         "epochs_override": 15},
        {"dropout_p": 0.2, "use_bn": True,  "run_name": "task1_bn_dp0.2",
         "epochs_override": 15},
    ]

    best_overall_f1, best_overall_ckpt = 0.0, None

    for cfg in runs:
        print(f"\n--- Task1: {cfg['run_name']} ---")
        ckpt_path     = os.path.join(args.ckpt_dir, f"{cfg['run_name']}.pth")
        periodic_path = os.path.join(args.ckpt_dir, f"{cfg['run_name']}_periodic.pth")

        run_epochs = cfg.get("epochs_override", args.epochs)
        wandb.init(project=WANDB_PROJECT, group="task1_classification",
                   name=cfg["run_name"], reinit="finish_previous",
                   config={**cfg, "lr": args.lr, "epochs": run_epochs})

        model = VGG11Classifier(num_classes=NUM_CLASSES,
                                dropout_p=cfg["dropout_p"]).to(device)

        if not cfg["use_bn"]:
            def rm_bn(m):
                for n, c in m.named_children():
                    if isinstance(c, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        setattr(m, n, nn.Identity())
                    else:
                        rm_bn(c)
            rm_bn(model)

        start_epoch, best_f1 = load_ckpt_if_exists(model, periodic_path, device)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=run_epochs,
            last_epoch=max(start_epoch - 1, -1))

        for epoch in range(start_epoch + 1, run_epochs + 1):
            model.train()
            tr_loss = 0.0
            tr_preds, tr_labels = [], []

            for batch in tqdm(train_loader,
                              desc=f"e{epoch}/{run_epochs}", leave=False):
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(imgs)              # single forward pass
                loss   = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
                tr_preds.extend(logits.detach().argmax(1).cpu().numpy())
                tr_labels.extend(lbls.cpu().numpy())

            scheduler.step()

            # ── Section 2.1: log block2 activation distribution once per run ──
            if epoch == 1:
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    sample_imgs  = sample_batch["image"].to(device)[:8]
                    _, feats = model.encoder(sample_imgs, return_features=True)
                    act_vals = feats["block2"].cpu().numpy().flatten()
                wandb.log({
                    "activations/block2_hist": wandb.Histogram(act_vals),
                    "activations/block2_mean": float(act_vals.mean()),
                    "activations/block2_std":  float(act_vals.std()),
                })
                model.train()
            # ─────────────────────────────────────────────────────────────────

            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    lbls = batch["label"].to(device)
                    out  = model(imgs)
                    val_loss += criterion(out, lbls).item() * imgs.size(0)
                    val_preds.extend(out.argmax(1).cpu().numpy())
                    val_labels.extend(lbls.cpu().numpy())

            tr_f1  = macro_f1(tr_preds,  tr_labels)
            val_f1 = macro_f1(val_preds, val_labels)
            n_tr   = len(train_loader.dataset)
            n_val  = len(val_loader.dataset)

            wandb.log({"epoch":      epoch,
                       "train/loss": tr_loss  / n_tr,
                       "train/f1":   tr_f1,
                       "val/loss":   val_loss / n_val,
                       "val/f1":     val_f1})
            print(f"  e{epoch:02d}/{run_epochs} | train_f1={tr_f1:.4f}  "
                  f"val_f1={val_f1:.4f}  val_loss={val_loss/n_val:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_ckpt(model, epoch, val_f1, ckpt_path)
                print(f"    -> Best saved ({val_f1:.4f})")

            if epoch % 5 == 0:
                save_ckpt(model, epoch, best_f1, periodic_path)
                print(f"    -> Periodic ckpt saved (epoch {epoch})")

            if val_f1 > best_overall_f1:
                best_overall_f1   = val_f1
                best_overall_ckpt = ckpt_path

        wandb.summary["best_val_f1"] = best_f1
        wandb.finish()

    if best_overall_ckpt:
        shutil.copy(best_overall_ckpt,
                    os.path.join(args.ckpt_dir, "classifier.pth"))
    print(f"\nTask1 done. Best F1={best_overall_f1:.4f} -> classifier.pth saved")


# ─────────────────────────────────────────────────────────────
# TASK 2 — Localization
# ─────────────────────────────────────────────────────────────

def train_task2(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers,require_bbox=True)

    ckpt_path     = os.path.join(args.ckpt_dir, "localizer.pth")
    periodic_path = os.path.join(args.ckpt_dir, "localizer_periodic.pth")

    wandb.init(project=WANDB_PROJECT, group="task2_localization",
               name="task2_localizer", reinit="finish_previous",
               config={"lr": args.lr, "epochs": args.epochs, "loss": "MSE+IoU"})

    model = VGG11Localizer().to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  Encoder loaded + ALL blocks frozen.")

    start_epoch, best_iou = load_ckpt_if_exists(model, periodic_path, device)

    iou_fn  = IoULoss(reduction="mean")
    huber_fn = nn.SmoothL1Loss()          # more robust than MSE for bbox regression
    params  = [p for p in model.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    # Set initial_lr on param groups so scheduler can resume correctly
    for pg in opt.param_groups:
        pg.setdefault("initial_lr", pg["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, last_epoch=max(start_epoch - 1, -1))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Gradually unfreeze encoder: after epoch 10 unfreeze block3,
        # after epoch 20 unfreeze block4+block5 for fine-tuning
        if epoch == 11:
            for p in model.encoder.block3.parameters():
                p.requires_grad = True
            opt.add_param_group({"params": model.encoder.block3.parameters(),
                                 "lr": args.lr * 0.1})
            print("  Unfroze encoder block3")
        if epoch == 21:
            # Collect only params not already in any optimizer group
            existing = {id(p) for group in opt.param_groups for p in group["params"]}
            new_params = [p for p in
                list(model.encoder.block4.parameters()) +
                list(model.encoder.block5.parameters())
                if id(p) not in existing]
            for p in new_params:
                p.requires_grad = True
            if new_params:
                opt.add_param_group({"params": new_params, "lr": args.lr * 0.01})
            print(f"  Unfroze encoder block4+5 ({len(new_params)} new params)")

        model.train()
        tr_loss, tr_ious = 0.0, []
        for batch in tqdm(train_loader,
                          desc=f"task2 e{epoch}/{args.epochs}", leave=False):
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            opt.zero_grad()
            pred = model(imgs)
            loss = huber_fn(pred, bboxes) + 2.0 * iou_fn(pred, bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            with torch.no_grad():
                tr_ious.extend(compute_iou_np(
                    pred.detach().cpu().numpy(),
                    bboxes.cpu().numpy()).tolist())
        sch.step()

        model.eval()
        val_loss, val_ious = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred   = model(imgs)
                val_loss += (huber_fn(pred, bboxes) +
                             2.0 * iou_fn(pred, bboxes)).item() * imgs.size(0)
                val_ious.extend(compute_iou_np(
                    pred.cpu().numpy(), bboxes.cpu().numpy()).tolist())

        miou  = float(np.mean(val_ious))
        n_tr  = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        wandb.log({"epoch":      epoch,
                   "train/loss": tr_loss / n_tr,
                   "train/iou":  float(np.mean(tr_ious)),
                   "val/loss":   val_loss / n_val,
                   "val/iou":    miou})
        print(f"  e{epoch:02d} | train_iou={float(np.mean(tr_ious)):.4f}  "
              f"val_iou={miou:.4f}")

        if miou > best_iou:
            best_iou = miou
            save_ckpt(model, epoch, miou, ckpt_path)
            print(f"    -> Best saved ({miou:.4f})")

        if epoch % 5 == 0:
            save_ckpt(model, epoch, best_iou, periodic_path)
            print(f"    -> Periodic ckpt saved (epoch {epoch})")

    wandb.summary["best_val_iou"] = best_iou
    wandb.finish()
    print(f"\nTask2 done. Best IoU={best_iou:.4f} -> localizer.pth saved")


# ─────────────────────────────────────────────────────────────
# TASK 3 — Segmentation
# ─────────────────────────────────────────────────────────────

def train_task3_strategy(args, strategy):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    ckpt_path     = os.path.join(args.ckpt_dir, f"unet_{strategy}.pth")
    periodic_path = os.path.join(args.ckpt_dir, f"unet_{strategy}_periodic.pth")

    wandb.init(project=WANDB_PROJECT, group="task3_segmentation",
               name=f"task3_{strategy}", reinit="finish_previous",
               config={"strategy": strategy, "lr": args.lr,
                       "epochs": args.epochs, "loss": "0.5*CE+0.5*Dice"})

    model = VGG11UNet(num_classes=SEG_CLASSES).to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print(f"  [{strategy}] Encoder loaded.")

    if strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        for p in (list(model.encoder.block1.parameters()) +
                  list(model.encoder.block2.parameters()) +
                  list(model.encoder.block3.parameters())):
            p.requires_grad = False

    start_epoch, best_dice = load_ckpt_if_exists(model, periodic_path, device)

    # Upweight foreground(1) and boundary(2) — background(0) is majority class
    seg_weights = torch.tensor([0.5, 1.5, 2.0]).to(device)
    ce_fn   = nn.CrossEntropyLoss(weight=seg_weights)
    dice_fn = DiceLoss()
    params  = [p for p in model.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    for pg in opt.param_groups:
        pg.setdefault("initial_lr", pg["lr"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, last_epoch=max(start_epoch - 1, -1))

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        tr_loss, tr_dice, tr_acc = 0.0, [], []
        for batch in tqdm(train_loader,
                          desc=f"[{strategy}] e{epoch}/{args.epochs}", leave=False):
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss   = 0.5 * ce_fn(logits, masks) + 0.5 * dice_fn(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            preds = logits.detach().argmax(1)
            tr_dice.append(dice_score(preds.cpu(), masks.cpu()))
            tr_acc.append(pixel_acc(preds.cpu(),  masks.cpu()))
        sch.step()

        model.eval()
        val_loss, val_dice, val_acc = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                val_loss += (0.5*ce_fn(logits, masks) +
                             0.5*dice_fn(logits, masks)).item() * imgs.size(0)
                preds = logits.argmax(1)
                val_dice.append(dice_score(preds.cpu(), masks.cpu()))
                val_acc.append(pixel_acc(preds.cpu(),  masks.cpu()))

        vdice = float(np.mean(val_dice))
        n_tr  = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        wandb.log({"epoch":           epoch,
                   "train/loss":      tr_loss  / n_tr,
                   "train/dice":      float(np.mean(tr_dice)),
                   "train/pixel_acc": float(np.mean(tr_acc)),
                   "val/loss":        val_loss / n_val,
                   "val/dice":        vdice,
                   "val/pixel_acc":   float(np.mean(val_acc))})
        print(f"  [{strategy}] e{epoch:02d} | "
              f"train_dice={float(np.mean(tr_dice)):.4f}  "
              f"val_dice={vdice:.4f}  val_acc={float(np.mean(val_acc)):.4f}")

        if vdice > best_dice:
            best_dice = vdice
            save_ckpt(model, epoch, vdice, ckpt_path)
            print(f"    -> Best saved ({vdice:.4f})")

        if epoch % 5 == 0:
            save_ckpt(model, epoch, best_dice, periodic_path)
            print(f"    -> Periodic ckpt saved (epoch {epoch})")

    wandb.summary["best_val_dice"] = best_dice
    wandb.finish()
    return ckpt_path, best_dice


def train_task3(args):
    best_ckpt, best_dice = None, 0.0
    for strategy in ["frozen", "partial", "full"]:
        print(f"\n--- Task3: {strategy} ---")
        ckpt, d = train_task3_strategy(args, strategy)
        if d > best_dice:
            best_dice, best_ckpt = d, ckpt
    shutil.copy(best_ckpt, os.path.join(args.ckpt_dir, "unet.pth"))
    print(f"\nTask3 done. Best Dice={best_dice:.4f} -> unet.pth saved")


# ─────────────────────────────────────────────────────────────
# W&B Report visuals
# ─────────────────────────────────────────────────────────────

def run_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_loaders(args.data_root, batch_size=1, num_workers=2)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    # ── Section 2.4: Feature Maps ────────────────────────────────────────────
    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        wandb.init(project=WANDB_PROJECT, group="report_section_24",
                   name="feature_maps", reinit="finish_previous")
        model = VGG11Classifier().to(device)
        model.load_state_dict(torch.load(clf_path, map_location=device)["state_dict"])
        model.eval()
        batch = next(iter(val_loader))
        img   = batch["image"].to(device)[:1]
        acts  = []
        h = model.encoder.block1[0].register_forward_hook(
            lambda m, i, o: acts.append(o.detach().cpu()))
        with torch.no_grad():
            _, feats = model.encoder(img, return_features=True)
        h.remove()

        def plot_fmaps(feat, title):
            f = feat[0].numpy(); n = min(16, f.shape[0])
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            axes = axes.flatten()
            for i in range(n):
                axes[i].imshow(f[i], cmap="viridis"); axes[i].axis("off")
            for i in range(n, 16):
                axes[i].axis("off")
            fig.suptitle(title); plt.tight_layout(); return fig

        # also show original image for context
        orig_np = (batch["image"][0].numpy().transpose(1,2,0) * std + mean).clip(0,1)
        fig_orig, ax_orig = plt.subplots(figsize=(3,3))
        ax_orig.imshow(orig_np); ax_orig.axis("off")
        ax_orig.set_title("Input image"); plt.tight_layout()

        wandb.log({
            "section2_4/input_image":  wandb.Image(fig_orig),
            "section2_4/first_conv":   wandb.Image(
                plot_fmaps(acts[0],          "Layer 1 – edge detectors (64 filters)")),
            "section2_4/last_conv":    wandb.Image(
                plot_fmaps(feats["block5"],  "Block 5 – high-level semantics (512 filters)")),
        })
        plt.close("all"); wandb.finish()
        print("Section 2.4 done.")

    # ── Section 2.5: Detection Table ─────────────────────────────────────────
    loc_path = os.path.join(args.ckpt_dir, "localizer.pth")
    if os.path.exists(loc_path):
        wandb.init(project=WANDB_PROJECT, group="report_section_25",
                   name="bbox_table", reinit="finish_previous")
        model = VGG11Localizer().to(device)
        model.load_state_dict(torch.load(loc_path, map_location=device)["state_dict"])
        model.eval()
        table = wandb.Table(columns=["image", "iou", "confidence_score", "verdict"])
        count = 0
        for batch in val_loader:
            if count >= 10: break
            imgs = batch["image"].to(device)
            gt   = batch["bbox"][0].numpy()
            with torch.no_grad():
                raw  = model.reg_head(model.reg_pool(model.encoder(imgs))) # [0,1] before *IMAGE_SIZE
                pred = (raw * IMAGE_SIZE)[0].cpu().numpy()
                # Confidence proxy: mean sigmoid output value (how "committed" the network is)
                conf = float(raw.mean().cpu())
            iou    = compute_iou_np(pred[None], gt[None])[0]
            verdict = "✅ Good" if iou >= 0.5 else ("⚠️ Miss" if iou >= 0.2 else "❌ Fail")
            img_np = (batch["image"][0].numpy().transpose(1,2,0)
                      * std + mean).clip(0, 1)
            fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(img_np)
            for box, color, lbl in [(gt, "green", "GT"),
                                     (pred, "red", f"Pred IoU={iou:.2f}")]:
                cx, cy, bw, bh = box
                ax.add_patch(patches.Rectangle(
                    (cx-bw/2, cy-bh/2), bw, bh, lw=2, ec=color, fc="none"))
                ax.text(cx-bw/2, cy-bh/2-5, lbl, color=color, fontsize=7,
                        bbox=dict(fc="white", alpha=0.6, pad=1))
            ax.axis("off"); ax.set_title(f"Conf={conf:.3f}  IoU={iou:.3f}", fontsize=8)
            plt.tight_layout()
            table.add_data(wandb.Image(fig), round(float(iou), 4),
                           round(conf, 4), verdict)
            plt.close(fig); count += 1
        wandb.log({"section2_5/detection_table": table})
        wandb.finish(); print("Section 2.5 done.")

    # ── Section 2.6: Segmentation Samples ────────────────────────────────────
    unet_path = os.path.join(args.ckpt_dir, "unet.pth")
    if os.path.exists(unet_path):
        wandb.init(project=WANDB_PROJECT, group="report_section_26",
                   name="seg_samples", reinit="finish_previous")
        model = VGG11UNet().to(device)
        model.load_state_dict(torch.load(unet_path, map_location=device)["state_dict"])
        model.eval()
        count = 0
        for batch in val_loader:
            if count >= 5: break
            imgs      = batch["image"].to(device)
            gt_mask   = batch["mask"][0].numpy()
            with torch.no_grad():
                logits    = model(imgs)
                pred_mask = logits.argmax(1)[0].cpu().numpy()
            img_np  = (batch["image"][0].numpy().transpose(1,2,0)
                       * std + mean).clip(0, 1)
            p_acc   = float((pred_mask == gt_mask).mean())
            d_score = dice_score(
                torch.from_numpy(pred_mask), torch.from_numpy(gt_mask))
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np);
            axes[0].set_title("Original Image"); axes[0].axis("off")
            axes[1].imshow(gt_mask,   cmap="tab10", vmin=0, vmax=2)
            axes[1].set_title("Ground Truth Trimap"); axes[1].axis("off")
            axes[2].imshow(pred_mask, cmap="tab10", vmin=0, vmax=2)
            axes[2].set_title(f"Prediction  Dice={d_score:.3f}  PixAcc={p_acc:.3f}")
            axes[2].axis("off")
            plt.tight_layout()
            wandb.log({f"section2_6/sample_{count}": wandb.Image(fig)})
            plt.close(fig); count += 1
        wandb.finish(); print("Section 2.6 done.")

    # ── Section 2.7: Wild Images ─────────────────────────────────────────────
    # Place any 3 pet .jpg images in  wild_images/  folder before running report
    wild_dir = os.path.join(os.path.dirname(args.data_root), "wild_images")
    if os.path.isdir(wild_dir):
        from models.multitask import MultiTaskPerceptionModel
        wandb.init(project=WANDB_PROJECT, group="report_section_27",
                   name="wild_images", reinit="finish_previous")
        mt = MultiTaskPerceptionModel(
            classifier_path=os.path.join(args.ckpt_dir, "classifier.pth"),
            localizer_path=os.path.join(args.ckpt_dir, "localizer.pth"),
            unet_path=os.path.join(args.ckpt_dir, "unet.pth"),
        ).to(device)
        mt.eval()

        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        infer_tf = A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

        CLASSES = [
            "Abyssinian","Bengal","Birman","Bombay","British Shorthair",
            "Egyptian Mau","Maine Coon","Persian","Ragdoll","Russian Blue",
            "Siamese","Sphynx","american bulldog","american pit bull terrier",
            "basset hound","beagle","boxer","chihuahua","english cocker spaniel",
            "english setter","german shorthaired","great pyrenees","havanese",
            "japanese chin","keeshond","leonberger","miniature pinscher",
            "newfoundland","pomeranian","pug","saint bernard","samoyed",
            "scottish terrier","shiba inu","staffordshire bull terrier",
            "wheaten terrier","yorkshire terrier",
        ]

        wild_imgs = sorted([
            os.path.join(wild_dir, f) for f in os.listdir(wild_dir)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])[:3]

        if not wild_imgs:
            print("  No wild images found in wild_images/ — skipping Section 2.7")
        else:
            for i, img_path in enumerate(wild_imgs):
                img_np_orig = np.array(Image.open(img_path).convert("RGB"))
                t = infer_tf(image=img_np_orig)["image"].float().unsqueeze(0).to(device)
                with torch.no_grad():
                    _out = mt(t)
                    cls_out, bbox_out, seg_out = _out["classification"], _out["localization"], _out["segmentation"]
                pred_cls  = int(cls_out.argmax(1).item())
                breed     = CLASSES[pred_cls] if pred_cls < len(CLASSES) else str(pred_cls)
                confidence= float(torch.softmax(cls_out, dim=1).max().item())
                box       = bbox_out[0].cpu().numpy()   # cx,cy,w,h in pixels
                seg_mask  = seg_out.argmax(1)[0].cpu().numpy()

                img_vis = (infer_tf(image=img_np_orig)["image"]
                           .numpy().transpose(1,2,0) * std + mean).clip(0,1)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # Panel 1: image + bbox
                axes[0].imshow(img_vis)
                cx,cy,bw,bh = box
                rect = patches.Rectangle((cx-bw/2, cy-bh/2), bw, bh,
                                         lw=2, ec="red", fc="none")
                axes[0].add_patch(rect)
                axes[0].set_title(f"Breed: {breed}\nConf: {confidence:.2%}", fontsize=9)
                axes[0].axis("off")
                # Panel 2: original
                axes[1].imshow(img_vis); axes[1].axis("off")
                axes[1].set_title("Original", fontsize=9)
                # Panel 3: segmentation
                axes[2].imshow(seg_mask, cmap="tab10", vmin=0, vmax=2)
                axes[2].axis("off")
                axes[2].set_title("Segmentation Mask", fontsize=9)
                plt.suptitle(f"Wild Image {i+1}: {os.path.basename(img_path)}")
                plt.tight_layout()
                wandb.log({f"section2_7/wild_{i+1}": wandb.Image(fig)})
                plt.close(fig)
                print(f"  Wild {i+1}: {breed}  conf={confidence:.2%}")
        wandb.finish(); print("Section 2.7 done.")
    else:
        print(f"Skipping Section 2.7 – create folder '{wild_dir}' with 3 pet images.")

    print("\nReport generation complete.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str,   default="./data/oxford-iiit-pet")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--task", type=str, default="all",
                   choices=["all", "task1", "task2", "task3", "report", "report27"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.task in ("all", "task1"):   train_task1(args)
    if args.task in ("all", "task2"):   train_task2(args)
    if args.task in ("all", "task3"):   train_task3(args)
    if args.task in ("all", "report"):  run_report(args)
    if args.task == "report27":
        # Re-run only Section 2.7 wild-image inference.
        # Usage:
        #   1. mkdir wild_images && cp your_pet1.jpg your_pet2.jpg your_pet3.jpg wild_images/
        #   2. python train.py --task report27
        run_report(args)   # run_report is idempotent – skips sections whose ckpts don't exist

    print("\nDone. Checkpoints: classifier.pth  localizer.pth  unet.pth")