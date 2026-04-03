import os
import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

IMAGE_SIZE = 224


def _load_state(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        super(MultiTaskPerceptionModel, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build individual models and load weights
        clf = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        loc = VGG11Localizer(in_channels=in_channels)
        seg = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        if os.path.exists(classifier_path):
            clf.load_state_dict(_load_state(classifier_path, device))
        if os.path.exists(localizer_path):
            loc.load_state_dict(_load_state(localizer_path, device))
        if os.path.exists(unet_path):
            seg.load_state_dict(_load_state(unet_path, device))

        # Shared backbone — take weights from classifier (best trained)
        self.encoder = clf.encoder

        # Classification head
        self.avgpool = clf.avgpool
        self.cls_head = clf.classifier

        # Localization head
        self.reg_head = loc.reg_head

        # Segmentation decoder
        self.up5 = seg.up5
        self.dec5 = seg.dec5
        self.up4 = seg.up4
        self.dec4 = seg.dec4
        self.up3 = seg.up3
        self.dec3 = seg.dec3
        self.up2 = seg.up2
        self.dec2 = seg.dec2
        self.up1 = seg.up1
        self.dec1 = seg.dec1
        self.seg_head = seg.seg_head
        self.seg_dropout = seg.dropout

    def _pad_cat(self, upsampled, skip):
        import torch.nn.functional as F
        dh = skip.size(2) - upsampled.size(2)
        dw = skip.size(3) - upsampled.size(3)
        if dh > 0 or dw > 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x: torch.Tensor):
        # Single forward pass through shared encoder
        e5, feats = self.encoder(x, return_features=True)
        e1 = feats["block1"]
        e2 = feats["block2"]
        e3 = feats["block3"]
        e4 = feats["block4"]

        # Classification
        pooled = self.avgpool(e5)
        cls_out = self.cls_head(torch.flatten(pooled, 1))

        # Localization
        bbox_out = self.reg_head(e5)

        # Segmentation
        d5 = self.up5(e5)
        d5 = self._pad_cat(d5, e4)
        d5 = self.dec5(d5)
        d5 = self.seg_dropout(d5)

        d4 = self.up4(d5)
        d4 = self._pad_cat(d4, e3)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_cat(d3, e2)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, e1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        seg_out = self.seg_head(d1)

        return {
            "classification": cls_out,
            "localization": bbox_out,
            "segmentation": seg_out,
        }
