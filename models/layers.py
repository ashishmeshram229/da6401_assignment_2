import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1.0 - self.p)

    def extra_repr(self):
        return f"p={self.p}"
