import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

IMAGE_SIZE = 224


class VGG11Localizer(nn.Module):
    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Localizer, self).__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # AdaptiveAvgPool to fixed size, then flatten, then FC
        self.reg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                        # (B, 512*4*4) = (B, 8192)
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),                        # output in [0, 1]
        )

        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f5 = self.encoder(x, return_features=False)
        # sigmoid output [0,1] scaled to pixel space [0, IMAGE_SIZE]
        return self.reg_head(f5) * IMAGE_SIZE
