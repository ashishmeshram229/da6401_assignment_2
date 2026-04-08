import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224

# ── PASTE YOUR DRIVE IDs HERE AFTER TRAINING ──────────────────
CLASSIFIER_DRIVE_ID = "1SLOWbKYqKTLeIHkaXgYHj9So9bSTwkX5"
LOCALIZER_DRIVE_ID  = "1hPwk1FQ6FIyqN4hjbNV2m86cMBjq3KSF"
UNET_DRIVE_ID       = "1l4a4wIRTmODQWW5mlGk5fFhEqotUry51"
# ──────────────────────────────────────────────────────────────


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


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

        # ── Download checkpoints from Google Drive (TA requirement) ──
        import gdown
        os.makedirs(os.path.dirname(classifier_path) if os.path.dirname(classifier_path) else "checkpoints", exist_ok=True)
        gdown.download(id=CLASSIFIER_DRIVE_ID, output=classifier_path, quiet=False)
        gdown.download(id=LOCALIZER_DRIVE_ID,  output=localizer_path,  quiet=False)
        gdown.download(id=UNET_DRIVE_ID,       output=unet_path,       quiet=False)
        # ─────────────────────────────────────────────────────────────

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Shared backbone (classifier + segmentation) ──
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Separate encoder for localization ──
        # reg_head was trained with localizer's encoder — must use it at inference
        self.loc_encoder = VGG11Encoder(in_channels=in_channels)

        # ── Classification head ──
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head ── (must mirror VGG11Localizer exactly)
        self.reg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Force smooth outputs strictly between [0, 1]
        )

        # ── Segmentation decoder ──
        self.up5  = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)
        self.up4  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 128, 128)
        self.up2  = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 64, 64)
        self.up1  = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        self.dec1 = _double_conv(32, 32)
        self.seg_head = nn.Conv2d(32, seg_classes, kernel_size=1)
        self.seg_drop = CustomDropout(p=0.5)

        # ── Load weights from checkpoints ──
        self._load_weights(classifier_path, localizer_path, unet_path, device)

    def _load_weights(self, clf_path, loc_path, unet_path, device):
        from .classification import VGG11Classifier
        from .localization import VGG11Localizer
        from .segmentation import VGG11UNet

        if os.path.exists(clf_path):
            clf = VGG11Classifier()
            clf.load_state_dict(_load_state(clf_path, device))
            self.encoder.load_state_dict(clf.encoder.state_dict())
            self.cls_head.load_state_dict(clf.classifier.state_dict())
            print("Loaded classifier weights.")

        if os.path.exists(loc_path):
            loc = VGG11Localizer()
            loc.load_state_dict(_load_state(loc_path, device))
            # Load localizer encoder into dedicated loc_encoder
            self.loc_encoder.load_state_dict(loc.encoder.state_dict())
            # Remap reg_head keys: shift by -1 to skip pool at index 0
            # loc.reg_head: [0]=Pool [1]=Flatten [2]=Linear [3]=ReLU [4]=Drop [5]=Linear [6]=ReLU [7]=Linear [8]=ReLU
            # self.reg_head:         [0]=Flatten  [1]=Linear [2]=ReLU [3]=Drop [4]=Linear [5]=ReLU [6]=Linear [7]=ReLU
            src_sd = loc.reg_head.state_dict()
            dst_sd = {}
            for k, v in src_sd.items():
                parts = k.split(".")
                new_key = ".".join([str(int(parts[0]) - 1)] + parts[1:])
                dst_sd[new_key] = v
            missing, unexpected = self.reg_head.load_state_dict(dst_sd, strict=False)
            if missing:
                print(f"  reg_head missing keys: {missing}")
            print("Loaded localizer weights.")

        if os.path.exists(unet_path):
            seg = VGG11UNet()
            seg.load_state_dict(_load_state(unet_path, device))
            for attr in ["up5","dec5","up4","dec4","up3","dec3","up2","dec2","up1","dec1","seg_head"]:
                getattr(self, attr).load_state_dict(getattr(seg, attr).state_dict())
            print("Loaded UNet weights.")

    def _pad_cat(self, upsampled, skip):
        dh = skip.size(2) - upsampled.size(2)
        dw = skip.size(3) - upsampled.size(3)
        if dh > 0 or dw > 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x: torch.Tensor):
        # Classification + segmentation encoder
        e5, feats = self.encoder(x, return_features=True)
        e1, e2, e3, e4 = feats["block1"], feats["block2"], feats["block3"], feats["block4"]

        # Localization encoder (separate — trained with localizer)
        e5_loc = self.loc_encoder(x, return_features=False)

        # Classification
        cls_out = self.cls_head(torch.flatten(self.avgpool(e5), 1))

        # Localization — uses dedicated loc_encoder features
        # Localization — cleanly scale the 0-1 sigmoid output to pixel space
        bbox_out = self.reg_head(self.reg_pool(e5_loc)) * IMAGE_SIZE

        # Segmentation
        d5 = self.dec5(self._pad_cat(self.up5(e5), e4))
        d5 = self.seg_drop(d5)
        d4 = self.dec4(self._pad_cat(self.up4(d5), e3))
        d3 = self.dec3(self._pad_cat(self.up3(d4), e2))
        d2 = self.dec2(self._pad_cat(self.up2(d3), e1))
        d1 = self.dec1(self.up1(d2))
        seg_out = self.seg_head(d1)

        # Return dict — autograder expects keys: classification, localization, segmentation
        return {
            "classification": cls_out,   # (B, 37)
            "localization":   bbox_out,  # (B, 4) [cx, cy, w, h] pixel space
            "segmentation":   seg_out,   # (B, 3, H, W)
        }