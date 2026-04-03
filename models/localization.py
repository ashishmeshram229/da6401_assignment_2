import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x, return_features=False)
        # output is in [0,1] normalized, scale to pixel space
        out = self.reg_head(features)
        # scale [cx, cy, w, h] from [0,1] to pixel coords
        return out * IMAGE_SIZE
