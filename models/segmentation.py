import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11UNet, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Decoder: TransposedConv upsampling + skip connection concat
        # block5 out: 512ch 7x7 -> up to 14x14, cat with block4 (512ch)
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        # 14x14 -> 28x28, cat with block3 (256ch)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 256, 256)

        # 28x28 -> 56x56, cat with block2 (128ch)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 128, 128)

        # 56x56 -> 112x112, cat with block1 (64ch)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 64, 64)

        # 112x112 -> 224x224, no skip (input is 3ch, not useful here)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32, 32)

        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self.dropout = CustomDropout(p=dropout_p)

        self._init_decoder()

    def _init_decoder(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pad_cat(self, upsampled, skip):
        dh = skip.size(2) - upsampled.size(2)
        dw = skip.size(3) - upsampled.size(3)
        if dh > 0 or dw > 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feats = self.encoder(x, return_features=True)

        e1 = feats["block1"]
        e2 = feats["block2"]
        e3 = feats["block3"]
        e4 = feats["block4"]
        e5 = feats["block5"]

        d5 = self.up5(e5)
        d5 = self._pad_cat(d5, e4)
        d5 = self.dec5(d5)
        d5 = self.dropout(d5)

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

        return self.seg_head(d1)
