import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # AdaptiveAvgPool to 7x7 (more spatial info than 4x4)
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),                        # (B, 512*7*7) = (B, 25088)
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),               # lighter dropout for regression
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.ReLU(),                          # keep outputs >= 0, no sigmoid saturation
        )

        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f5 = self.encoder(x, return_features=False)
        # ReLU keeps values >= 0; clamp to valid pixel range at output
        out = self.reg_head(f5)
        return out.clamp(0, IMAGE_SIZE)