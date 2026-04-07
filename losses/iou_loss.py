import torch
import torch.nn as nn

class GIoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super(GIoULoss, self).__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Both tensors expected to be: [B, 4] in (cx, cy, w, h) format
        
        # 1. Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2.0
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2.0
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2.0
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2.0

        gt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2.0
        gt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2.0
        gt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2.0
        gt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2.0

        # 2. Calculate Standard IoU
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)

        # Clamp width and height to 0 to avoid false positive intersections
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        pred_w = torch.clamp(pred_x2 - pred_x1, min=0)
        pred_h = torch.clamp(pred_y2 - pred_y1, min=0)
        pred_area = pred_w * pred_h

        gt_w = torch.clamp(gt_x2 - gt_x1, min=0)
        gt_h = torch.clamp(gt_y2 - gt_y1, min=0)
        gt_area = gt_w * gt_h
        
        union_area = pred_area + gt_area - inter_area
        iou = inter_area / (union_area + self.eps)

        # 3. Calculate Generalized IoU (GIoU) Addition
        # Find the coordinates of the smallest enclosing box
        enclose_x1 = torch.min(pred_x1, gt_x1)
        enclose_y1 = torch.min(pred_y1, gt_y1)
        enclose_x2 = torch.max(pred_x2, gt_x2)
        enclose_y2 = torch.max(pred_y2, gt_y2)

        enclose_w = torch.clamp(enclose_x2 - enclose_x1, min=0)
        enclose_h = torch.clamp(enclose_y2 - enclose_y1, min=0)
        enclose_area = enclose_w * enclose_h

        # GIoU formula: IoU - (Area of empty space in enclosing box / Total enclosing area)
        giou = iou - (enclose_area - union_area) / (enclose_area + self.eps)
        
        # Loss ranges from 0 (perfect match) to 2.0 (infinitely far apart)
        loss = 1.0 - giou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss