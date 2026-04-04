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

SEED        = 42
IMAGE_SIZE  = 224
NUM_CLASSES = 37
SEG_CLASSES = 3
WANDB_PROJECT = "da6401-assignment2"


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_loaders(data_root, batch_size=32, num_workers=2):
    train_ds = OxfordIIITPetDataset(data_root, split="train", seed=SEED)
    val_ds   = OxfordIIITPetDataset(data_root, split="val",   seed=SEED)
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
    pa = np.maximum(px2-px1,0) * np.maximum(py2-py1,0)
    ga = np.maximum(gx2-gx1,0) * np.maximum(gy2-gy1,0)
    return inter / (pa + ga - inter + eps)


def dice_score(pred_mask, gt_mask, num_classes=3, eps=1e-6):
    scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float(); g = (gt_mask == c).float()
        inter = (p * g).sum(); denom = p.sum() + g.sum()
        scores.append(1.0 if denom < eps else ((2*inter+eps)/(denom+eps)).item())
    return float(np.mean(scores))


def pixel_acc(pred, gt):
    return (pred == gt).float().mean().item()


def save_ckpt(model, epoch, metric, path):
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
        {"dropout_p": 0.5,  "use_bn": True,  "run_name": "task1_bn_dp0.5"},
        {"dropout_p": 0.5,  "use_bn": False, "run_name": "task1_no_bn_dp0.5"},
        {"dropout_p": 0.0,  "use_bn": True,  "run_name": "task1_bn_no_dropout"},
        {"dropout_p": 0.2,  "use_bn": True,  "run_name": "task1_bn_dp0.2"},
    ]

    best_f1, best_ckpt = 0.0, None

    for cfg in runs:
        print(f"\n--- Task1: {cfg['run_name']} ---")
        wandb.init(project=WANDB_PROJECT, group="task1_classification",
                   name=cfg["run_name"], reinit="finish_previous",
                   config={**cfg, "lr": args.lr, "epochs": args.epochs})

        model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=cfg["dropout_p"]).to(device)

        if not cfg["use_bn"]:
            def rm_bn(m):
                for n, c in m.named_children():
                    if isinstance(c, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        setattr(m, n, nn.Identity())
                    else:
                        rm_bn(c)
            rm_bn(model)

        criterion  = nn.CrossEntropyLoss()
        optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        ckpt_path  = os.path.join(args.ckpt_dir, f"{cfg['run_name']}.pth")
        run_best   = 0.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss, tr_p, tr_l = 0.0, [], []
            for batch in tqdm(train_loader, desc=f"train e{epoch}", leave=False):
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * imgs.size(0)
                tr_p.extend(model(imgs).argmax(1).detach().cpu().numpy())
                tr_l.extend(lbls.cpu().numpy())
            scheduler.step()

            model.eval()
            vl, vp, vl2 = 0.0, [], []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    lbls = batch["label"].to(device)
                    out  = model(imgs)
                    vl  += criterion(out, lbls).item() * imgs.size(0)
                    vp.extend(out.argmax(1).cpu().numpy())
                    vl2.extend(lbls.cpu().numpy())

            vf1 = macro_f1(vp, vl2)
            wandb.log({"epoch": epoch,
                       "train/loss": tr_loss / len(train_loader.dataset),
                       "val/loss":   vl / len(val_loader.dataset),
                       "val/f1":     vf1})
            print(f"  e{epoch} | val_f1={vf1:.4f}")

            if vf1 > run_best:
                run_best = vf1
                save_ckpt(model, epoch, vf1, ckpt_path)

            if vf1 > best_f1:
                best_f1 = vf1
                best_ckpt = ckpt_path

        wandb.summary["best_val_f1"] = run_best
        wandb.finish()

    shutil.copy(best_ckpt, os.path.join(args.ckpt_dir, "classifier.pth"))
    print(f"Task1 done. Best F1={best_f1:.4f}. Saved classifier.pth")


# ─────────────────────────────────────────────────────────────
# TASK 2 — Localization
# ─────────────────────────────────────────────────────────────

def train_task2(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    wandb.init(project=WANDB_PROJECT, group="task2_localization",
               name="task2_localizer", reinit="finish_previous",
               config={"lr": args.lr, "epochs": args.epochs, "loss": "MSE+IoU"})

    model = VGG11Localizer().to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())
        for p in list(model.encoder.block1.parameters()) + \
                 list(model.encoder.block2.parameters()) + \
                 list(model.encoder.block3.parameters()):
            p.requires_grad = False
        print("Encoder loaded and partially frozen.")

    iou_fn = IoULoss(reduction="mean")
    mse_fn = nn.MSELoss()
    params  = [p for p in model.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sch     = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    best_iou, ckpt_path = 0.0, os.path.join(args.ckpt_dir, "localizer.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_ious = 0.0, []
        for batch in tqdm(train_loader, desc=f"task2 e{epoch}", leave=False):
            imgs   = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            opt.zero_grad()
            pred = model(imgs)
            loss = mse_fn(pred, bboxes) + iou_fn(pred, bboxes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            with torch.no_grad():
                tr_ious.extend(compute_iou_np(pred.detach().cpu().numpy(), bboxes.cpu().numpy()).tolist())
        sch.step()

        model.eval()
        vl, vious = 0.0, []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device); bboxes = batch["bbox"].to(device)
                pred = model(imgs)
                vl  += (mse_fn(pred, bboxes) + iou_fn(pred, bboxes)).item() * imgs.size(0)
                vious.extend(compute_iou_np(pred.cpu().numpy(), bboxes.cpu().numpy()).tolist())

        miou = float(np.mean(vious))
        wandb.log({"epoch": epoch, "train/loss": tr_loss/len(train_loader.dataset),
                   "train/iou": float(np.mean(tr_ious)), "val/loss": vl/len(val_loader.dataset),
                   "val/iou": miou})
        print(f"  e{epoch} | val_iou={miou:.4f}")

        if miou > best_iou:
            best_iou = miou
            save_ckpt(model, epoch, miou, ckpt_path)

    wandb.summary["best_val_iou"] = best_iou
    wandb.finish()
    print(f"Task2 done. Best IoU={best_iou:.4f}. Saved localizer.pth")


# ─────────────────────────────────────────────────────────────
# TASK 3 — Segmentation
# ─────────────────────────────────────────────────────────────

def train_task3_strategy(args, strategy):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    wandb.init(project=WANDB_PROJECT, group="task3_segmentation",
               name=f"task3_{strategy}", reinit="finish_previous",
               config={"strategy": strategy, "lr": args.lr, "epochs": args.epochs,
                       "loss": "0.5*CE + 0.5*Dice"})

    model = VGG11UNet(num_classes=SEG_CLASSES).to(device)

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    if os.path.exists(clf_path):
        clf = VGG11Classifier()
        clf.load_state_dict(torch.load(clf_path, map_location="cpu")["state_dict"])
        model.encoder.load_state_dict(clf.encoder.state_dict())

    if strategy == "frozen":
        for p in model.encoder.parameters(): p.requires_grad = False
    elif strategy == "partial":
        for p in list(model.encoder.block1.parameters()) + \
                 list(model.encoder.block2.parameters()) + \
                 list(model.encoder.block3.parameters()):
            p.requires_grad = False
    # full: all params trainable

    ce   = nn.CrossEntropyLoss()
    dice = DiceLoss(num_classes=SEG_CLASSES)
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_dice, ckpt_path = 0.0, os.path.join(args.ckpt_dir, f"unet_{strategy}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_d, tr_a = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"task3[{strategy}] e{epoch}", leave=False):
            imgs  = batch["image"].to(device); masks = batch["mask"].to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss   = 0.5 * ce(logits, masks) + 0.5 * dice(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(1)
            tr_d.append(dice_score(preds.cpu(), masks.cpu()))
            tr_a.append(pixel_acc(preds.cpu(), masks.cpu()))
        sch.step()

        model.eval()
        vl, vd, va = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device); masks = batch["mask"].to(device)
                logits = model(imgs)
                vl += (0.5*ce(logits,masks) + 0.5*dice(logits,masks)).item() * imgs.size(0)
                preds = logits.argmax(1)
                vd.append(dice_score(preds.cpu(), masks.cpu()))
                va.append(pixel_acc(preds.cpu(), masks.cpu()))

        vdice = float(np.mean(vd))
        wandb.log({"epoch": epoch,
                   "train/loss": tr_loss/len(train_loader.dataset),
                   "train/dice": float(np.mean(tr_d)), "train/pixel_acc": float(np.mean(tr_a)),
                   "val/loss": vl/len(val_loader.dataset),
                   "val/dice": vdice, "val/pixel_acc": float(np.mean(va))})
        print(f"  [{strategy}] e{epoch} | val_dice={vdice:.4f}")

        if vdice > best_dice:
            best_dice = vdice
            save_ckpt(model, epoch, vdice, ckpt_path)

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
    print(f"Task3 done. Best Dice={best_dice:.4f}. Saved unet.pth")


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
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])

    # Section 2.4 — feature maps
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
        h = model.encoder.block1[0].register_forward_hook(lambda m,i,o: acts.append(o.detach().cpu()))
        with torch.no_grad():
            _, feats = model.encoder(img, return_features=True)
        h.remove()

        def plot_fmaps(feat, title):
            f = feat[0].numpy(); n = min(16, f.shape[0])
            fig, axes = plt.subplots(2, 8, figsize=(16, 4))
            [axes.flatten()[i].imshow(f[i], cmap="viridis") or axes.flatten()[i].axis("off") for i in range(n)]
            [axes.flatten()[i].axis("off") for i in range(n, 16)]
            fig.suptitle(title); plt.tight_layout(); return fig

        wandb.log({"section2_4/first_conv": wandb.Image(plot_fmaps(acts[0], "First Conv — edges")),
                   "section2_4/last_conv":  wandb.Image(plot_fmaps(feats["block5"], "Last Conv — semantics"))})
        plt.close("all"); wandb.finish(); print("Section 2.4 done.")

    # Section 2.5 — bbox table
    loc_path = os.path.join(args.ckpt_dir, "localizer.pth")
    if os.path.exists(loc_path):
        wandb.init(project=WANDB_PROJECT, group="report_section_25",
                   name="bbox_table", reinit="finish_previous")
        model = VGG11Localizer().to(device)
        model.load_state_dict(torch.load(loc_path, map_location=device)["state_dict"])
        model.eval()
        table = wandb.Table(columns=["image", "iou", "confidence_proxy"])
        count = 0
        for batch in val_loader:
            if count >= 10: break
            imgs = batch["image"].to(device); gt = batch["bbox"][0].numpy()
            with torch.no_grad(): pred = model(imgs)[0].cpu().numpy()
            iou = compute_iou_np(pred[None], gt[None])[0]
            img_np = (batch["image"][0].numpy().transpose(1,2,0) * std + mean).clip(0,1)
            fig, ax = plt.subplots(figsize=(4,4)); ax.imshow(img_np)
            for box, color, lbl in [(gt,"green","GT"), (pred,"red",f"IoU={iou:.2f}")]:
                cx,cy,bw,bh = box
                ax.add_patch(patches.Rectangle((cx-bw/2,cy-bh/2),bw,bh,lw=2,ec=color,fc="none"))
                ax.text(cx-bw/2, cy-bh/2-3, lbl, color=color, fontsize=7,
                        bbox=dict(fc="white",alpha=0.5,pad=1))
            ax.axis("off"); plt.tight_layout()
            table.add_data(wandb.Image(fig), round(float(iou),4), round(float(iou),4))
            plt.close(fig); count += 1
        wandb.log({"section2_5/detection_table": table}); wandb.finish(); print("Section 2.5 done.")

    # Section 2.6 — segmentation samples
    unet_path = os.path.join(args.ckpt_dir, "unet.pth")
    if os.path.exists(unet_path):
        wandb.init(project=WANDB_PROJECT, group="report_section_26",
                   name="seg_samples", reinit="finish_previous")
        from models.segmentation import VGG11UNet
        model = VGG11UNet().to(device)
        model.load_state_dict(torch.load(unet_path, map_location=device)["state_dict"])
        model.eval()
        count = 0
        for batch in val_loader:
            if count >= 5: break
            imgs = batch["image"].to(device); gt_mask = batch["mask"][0].numpy()
            with torch.no_grad(): pred_mask = model(imgs).argmax(1)[0].cpu().numpy()
            img_np = (batch["image"][0].numpy().transpose(1,2,0) * std + mean).clip(0,1)
            fig, axes = plt.subplots(1,3,figsize=(12,4))
            axes[0].imshow(img_np);                              axes[0].set_title("Original");    axes[0].axis("off")
            axes[1].imshow(gt_mask,   cmap="tab10",vmin=0,vmax=2); axes[1].set_title("GT Mask");  axes[1].axis("off")
            axes[2].imshow(pred_mask, cmap="tab10",vmin=0,vmax=2); axes[2].set_title("Predicted");axes[2].axis("off")
            plt.tight_layout(); wandb.log({f"section2_6/sample_{count}": wandb.Image(fig)})
            plt.close(fig); count += 1
        wandb.finish(); print("Section 2.6 done.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   default="./data/oxford-iiit-pet")
    p.add_argument("--ckpt_dir",     type=str,   default="./checkpoints")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--task", type=str, default="all",
                   choices=["all", "task1", "task2", "task3", "report"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if args.task in ("all", "task1"):  train_task1(args)
    if args.task in ("all", "task2"):  train_task2(args)
    if args.task in ("all", "task3"):  train_task3(args)
    if args.task in ("all", "report"): run_report(args)

    print("\nDone. Checkpoints: classifier.pth  localizer.pth  unet.pth")
