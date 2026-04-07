import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super(IoULoss, self).__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Both tensors: [B, 4] in (cx, cy, w, h)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

        gt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
        gt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
        gt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
        gt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

        # 1. Standard IoU
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        gt_area   = (gt_x2 - gt_x1).clamp(min=0) * (gt_y2 - gt_y1).clamp(min=0)
        
        union_area = pred_area + gt_area - inter_area
        iou = inter_area / (union_area + self.eps)

        # 2. Generalized IoU (GIoU) Addition
        # Find the coordinates of the smallest enclosing box
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)

        enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)

        # GIoU formula pushes boxes together even when there is zero overlap
        giou = iou - (enclose_area - union_area) / (enclose_area + self.eps)
        loss = 1.0 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss