import os
import sys
import argparse

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

SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 37
SEG_CLASSES = 3
WANDB_PROJECT = "da6401-assignment2"


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_loaders(data_root, batch_size=32, num_workers=4):
    train_ds = OxfordIIITPetDataset(data_root, split="train", seed=SEED)
    val_ds = OxfordIIITPetDataset(data_root, split="val", seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", zero_division=0)


def compute_iou_np(pred_boxes, gt_boxes, eps=1e-6):
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    gx1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    gy1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gx2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    gy2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)

    iw = np.maximum(ix2 - ix1, 0)
    ih = np.maximum(iy2 - iy1, 0)
    inter = iw * ih

    pa = (px2 - px1).clip(0) * (py2 - py1).clip(0)
    ga = (gx2 - gx1).clip(0) * (gy2 - gy1).clip(0)
    union = pa + ga - inter

    return inter / (union + eps)


def dice_score(pred_mask, gt_mask, num_classes=3, eps=1e-6):
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float()
        g = (gt_mask == c).float()
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        if denom < eps:
            scores.append(1.0)
        else:
            scores.append(((2 * inter + eps) / (denom + eps)).item())
    return float(np.mean(scores))


def pixel_acc(pred_mask, gt_mask):
    return (pred_mask == gt_mask).float().mean().item()


def save_checkpoint(model, epoch, metric, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_metric": metric}, path)


class DiceLoss(nn.Module):
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        loss = 0.0
        for c in range(self.num_classes):
            p = probs[:, c]
            g = (targets == c).float()
            inter = (p * g).sum()
            denom = p.sum() + g.sum()
            loss += 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        return loss / self.num_classes


# ─────────────────────────────────────────────────────────────
# TASK 1 — Classification
# ─────────────────────────────────────────────────────────────

def train_task1(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    configs_to_run = [
        {"dropout_p": 0.5, "use_bn": True,  "run_name": "task1_bn_dp0.5"},
        {"dropout_p": 0.5, "use_bn": False, "run_name": "task1_no_bn_dp0.5"},
        {"dropout_p": 0.0, "use_bn": True,  "run_name": "task1_bn_no_dropout"},
        {"dropout_p": 0.2, "use_bn": True,  "run_name": "task1_bn_dp0.2"},
    ]

    best_overall_ckpt = None
    best_overall_f1 = 0.0

    for cfg in configs_to_run:
        print(f"\n--- Task1: {cfg['run_name']} ---")
        run = wandb.init(
            project=WANDB_PROJECT, entity=args.wandb_entity,
            group="task1_classification", name=cfg["run_name"], reinit='finish_previous',
            config={**cfg, "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size},
        )

        model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=cfg["dropout_p"]).to(device)

        if not cfg["use_bn"]:
            def remove_bn(m):
                for name, child in m.named_children():
                    if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        setattr(m, name, nn.Identity())
                    else:
                        remove_bn(child)
            remove_bn(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_f1 = 0.0
        ckpt_path = os.path.join(args.ckpt_dir, f"{cfg['run_name']}.pth")

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss, tr_preds, tr_labels = 0.0, [], []
            for batch in tqdm(train_loader, desc=f"[{cfg['run_name']}] train e{epoch}", leave=False):
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
                tr_preds.extend(logits.argmax(1).cpu().numpy())
                tr_labels.extend(lbls.cpu().numpy())
            scheduler.step()

            model.eval()
            val_loss, val_preds, val_labels = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    lbls = batch["label"].to(device)
                    logits = model(imgs)
                    val_loss += criterion(logits, lbls).item() * imgs.size(0)
                    val_preds.extend(logits.argmax(1).cpu().numpy())
                    val_labels.extend(lbls.cpu().numpy())

            tr_f1 = macro_f1(tr_preds, tr_labels)
            vf1 = macro_f1(val_preds, val_labels)
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss / len(train_loader.dataset),
                "train/f1": tr_f1,
                "val/loss": val_loss / len(val_loader.dataset),
                "val/f1": vf1,
            })
            print(f"  e{epoch} | val_loss={val_loss/len(val_loader.dataset):.4f} val_f1={vf1:.4f}")

            if vf1 > best_f1:
                best_f1 = vf1
                save_checkpoint(model, epoch, vf1, ckpt_path)

            if vf1 > best_overall_f1:
                best_overall_f1 = vf1
                best_overall_ckpt = ckpt_path

        wandb.summary["best_val_f1"] = best_f1
        wandb.finish()

    # Save best as classifier.pth
    import shutil
    final_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if best_overall_ckpt:
        shutil.copy(best_overall_ckpt, final_path)
    print(f"\nTask1 done. classifier.pth saved. Best F1={best_overall_f1:.4f}")
    return final_path


# ─────────────────────────────────────────────────────────────
# TASK 2 — Localization
# ─────────────────────────────────────────────────────────────

def train_task2(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    run = wandb.init(
        project=WANDB_PROJECT, entity=args.wandb_entity,
        group="task2_localization", name="task2_localizer", reinit='finish_previous',
        config={"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                "loss": "MSE + IoU"},
    )

    model = VGG11Localizer().to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        from models.classification import VGG11Classifier
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print("  Loaded encoder weights from classifier.pth")
        for p in list(model.encoder.block1.parameters()) + \
                 list(model.encoder.block2.parameters()) + \
                 list(model.encoder.block3.parameters()):
            p.requires_grad = False

    iou_loss_fn = IoULoss(reduction="mean")
    mse_loss_fn = nn.MSELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_iou = 0.0
    ckpt_path = os.path.join(args.ckpt_dir, "localizer.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_ious = 0.0, []
        for batch in tqdm(train_loader, desc=f"task2 train e{epoch}", leave=False):
            imgs = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)  # pixel space
            optimizer.zero_grad()
            pred = model(imgs)
            loss = mse_loss_fn(pred, bboxes) + iou_loss_fn(pred, bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            with torch.no_grad():
                ious = compute_iou_np(pred.detach().cpu().numpy(), bboxes.cpu().numpy())
                tr_ious.extend(ious.tolist())
        scheduler.step()

        model.eval()
        val_loss, val_ious = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred = model(imgs)
                loss = mse_loss_fn(pred, bboxes) + iou_loss_fn(pred, bboxes)
                val_loss += loss.item() * imgs.size(0)
                ious = compute_iou_np(pred.cpu().numpy(), bboxes.cpu().numpy())
                val_ious.extend(ious.tolist())

        miou = float(np.mean(val_ious))
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss / len(train_loader.dataset),
            "train/iou": float(np.mean(tr_ious)),
            "val/loss": val_loss / len(val_loader.dataset),
            "val/iou": miou,
        })
        print(f"  e{epoch} | val_iou={miou:.4f}")

        if miou > best_iou:
            best_iou = miou
            save_checkpoint(model, epoch, miou, ckpt_path)

    wandb.summary["best_val_iou"] = best_iou
    wandb.finish()
    print(f"Task2 done. localizer.pth saved. Best IoU={best_iou:.4f}")
    return ckpt_path


# ─────────────────────────────────────────────────────────────
# TASK 3 — Segmentation (3 strategies)
# ─────────────────────────────────────────────────────────────

def train_task3_strategy(args, strategy):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    run = wandb.init(
        project=WANDB_PROJECT, entity=args.wandb_entity,
        group="task3_segmentation", name=f"task3_{strategy}", reinit='finish_previous',
        config={"strategy": strategy, "lr": args.lr, "epochs": args.epochs,
                "batch_size": args.batch_size, "loss": "0.5*CE + 0.5*Dice"},
    )

    model = VGG11UNet(num_classes=SEG_CLASSES).to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        from models.classification import VGG11Classifier
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())
        print(f"  [{strategy}] Loaded encoder weights from classifier.pth")

    if strategy == "frozen":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif strategy == "partial":
        for p in list(model.encoder.block1.parameters()) + \
                 list(model.encoder.block2.parameters()) + \
                 list(model.encoder.block3.parameters()):
            p.requires_grad = False
    else:  # full
        for p in model.parameters():
            p.requires_grad = True

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes=SEG_CLASSES)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0
    ckpt_path = os.path.join(args.ckpt_dir, f"unet_{strategy}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_dice, tr_acc = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"task3[{strategy}] e{epoch}", leave=False):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            tr_dice.append(dice_score(preds.cpu(), masks.cpu()))
            tr_acc.append(pixel_acc(preds.cpu(), masks.cpu()))
        scheduler.step()

        model.eval()
        val_loss, val_dice, val_acc = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                loss = 0.5 * ce_loss(logits, masks) + 0.5 * dice_loss(logits, masks)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                val_dice.append(dice_score(preds.cpu(), masks.cpu()))
                val_acc.append(pixel_acc(preds.cpu(), masks.cpu()))

        vd = float(np.mean(val_dice))
        va = float(np.mean(val_acc))
        wandb.log({
            "epoch": epoch,
            "train/loss": tr_loss / len(train_loader.dataset),
            "train/dice": float(np.mean(tr_dice)),
            "train/pixel_acc": float(np.mean(tr_acc)),
            "val/loss": val_loss / len(val_loader.dataset),
            "val/dice": vd,
            "val/pixel_acc": va,
        })
        print(f"  [{strategy}] e{epoch} | val_dice={vd:.4f} val_acc={va:.4f}")

        if vd > best_dice:
            best_dice = vd
            save_checkpoint(model, epoch, vd, ckpt_path)

    wandb.summary["best_val_dice"] = best_dice
    wandb.finish()
    return ckpt_path, best_dice


def train_task3(args):
    best_ckpt = None
    best_dice = 0.0
    for strategy in ["frozen", "partial", "full"]:
        print(f"\n--- Task3: strategy={strategy} ---")
        ckpt, d = train_task3_strategy(args, strategy)
        if d > best_dice:
            best_dice = d
            best_ckpt = ckpt

    import shutil
    final = os.path.join(args.ckpt_dir, "unet.pth")
    if best_ckpt:
        shutil.copy(best_ckpt, final)
    print(f"\nTask3 done. unet.pth saved. Best Dice={best_dice:.4f}")
    return final


# ─────────────────────────────────────────────────────────────
# W&B Report sections
# ─────────────────────────────────────────────────────────────

def run_report_sections(args, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    _, val_loader = get_loaders(args.data_root, batch_size=1, num_workers=2)

    # Section 2.4 — Feature maps
    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        from models.classification import VGG11Classifier
        run = wandb.init(project=WANDB_PROJECT, entity=args.wandb_entity,
                         group="report_section_24", name="feature_maps", reinit='finish_previous')
        model = VGG11Classifier().to(device)
        model.load_state_dict(torch.load(clf_path, map_location=device)["state_dict"])
        model.eval()
        batch = next(iter(val_loader))
        img = batch["image"].to(device)[:1]

        acts = []
        h = model.encoder.block1[0].register_forward_hook(lambda m, i, o: acts.append(o))
        with torch.no_grad():
            _, feats = model.encoder(img, return_features=True)
        h.remove()

        def plot_fmaps(feat, title, n=16):
            f = feat[0].cpu().numpy()
            n = min(n, f.shape[0])
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            axes = axes.flatten()
            for i in range(n):
                axes[i].imshow(f[i], cmap="viridis")
                axes[i].axis("off")
            for i in range(n, 16):
                axes[i].axis("off")
            fig.suptitle(title, fontsize=12)
            plt.tight_layout()
            return fig

        fig1 = plot_fmaps(acts[0], "First Conv Layer — low-level edge detectors")
        fig5 = plot_fmaps(feats["block5"], "Last Conv Block — high-level semantic features")
        wandb.log({"section2_4/first_conv": wandb.Image(fig1),
                   "section2_4/last_conv": wandb.Image(fig5)})
        plt.close("all")
        wandb.finish()
        print("Section 2.4 logged.")

    # Section 2.5 — BBox table
    loc_path = os.path.join(args.ckpt_dir, "localizer.pth")
    if os.path.exists(loc_path):
        from models.localization import VGG11Localizer
        run = wandb.init(project=WANDB_PROJECT, entity=args.wandb_entity,
                         group="report_section_25", name="bbox_table", reinit='finish_previous')
        model = VGG11Localizer().to(device)
        model.load_state_dict(torch.load(loc_path, map_location=device)["state_dict"])
        model.eval()

        table = wandb.Table(columns=["image", "iou", "confidence_proxy"])
        count = 0
        for batch in val_loader:
            if count >= 10:
                break
            imgs = batch["image"].to(device)
            gt_box = batch["bbox"][0].numpy()
            with torch.no_grad():
                pred_box = model(imgs)[0].cpu().numpy()
            iou = compute_iou_np(pred_box[None], gt_box[None])[0]

            img_np = batch["image"][0].numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean).clip(0, 1)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img_np)
            for box, color, label in [(gt_box, "green", "GT"), (pred_box, "red", f"Pred IoU={iou:.2f}")]:
                cx, cy, bw, bh = box
                x1, y1 = cx - bw / 2, cy - bh / 2
                ax.add_patch(patches.Rectangle((x1, y1), bw, bh, lw=2, ec=color, fc="none"))
                ax.text(x1, y1 - 3, label, color=color, fontsize=7,
                        bbox=dict(fc="white", alpha=0.5, pad=1))
            ax.axis("off")
            plt.tight_layout()
            table.add_data(wandb.Image(fig), round(float(iou), 4), round(float(iou), 4))
            plt.close(fig)
            count += 1

        wandb.log({"section2_5/detection_table": table})
        wandb.finish()
        print("Section 2.5 logged.")

    # Section 2.6 — Segmentation samples
    unet_path = os.path.join(args.ckpt_dir, "unet.pth")
    if os.path.exists(unet_path):
        from models.segmentation import VGG11UNet
        run = wandb.init(project=WANDB_PROJECT, entity=args.wandb_entity,
                         group="report_section_26", name="seg_samples", reinit='finish_previous')
        model = VGG11UNet().to(device)
        model.load_state_dict(torch.load(unet_path, map_location=device)["state_dict"])
        model.eval()

        count = 0
        for batch in val_loader:
            if count >= 5:
                break
            imgs = batch["image"].to(device)
            gt_mask = batch["mask"][0].numpy()
            with torch.no_grad():
                pred_mask = model(imgs).argmax(1)[0].cpu().numpy()

            img_np = batch["image"][0].numpy().transpose(1, 2, 0)
            img_np = (img_np * std + mean).clip(0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=2); axes[1].set_title("GT Mask"); axes[1].axis("off")
            axes[2].imshow(pred_mask, cmap="tab10", vmin=0, vmax=2); axes[2].set_title("Predicted"); axes[2].axis("off")
            plt.tight_layout()
            wandb.log({f"section2_6/sample_{count}": wandb.Image(fig)})
            plt.close(fig)
            count += 1

        wandb.finish()
        print("Section 2.6 logged.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--task", type=str, default="all",
                   choices=["all", "task1", "task2", "task3", "report"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.task in ("all", "task1"):
        train_task1(args)

    if args.task in ("all", "task2"):
        train_task2(args)

    if args.task in ("all", "task3"):
        train_task3(args)

    if args.task in ("all", "report"):
        run_report_sections(args, device)

    print("\nAll done. Checkpoints: classifier.pth, localizer.pth, unet.pth")
