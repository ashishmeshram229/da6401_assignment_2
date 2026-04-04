from .layers import CustomDropout
from .vgg11 import VGG11Encoder, VGG11
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

# MultiTaskPerceptionModel is intentionally NOT imported here at module level.
# It calls gdown inside __init__() to download checkpoints, which would trigger
# network calls the moment anyone does `from models import ...`.
# Import it explicitly where needed:  from models.multitask import MultiTaskPerceptionModel

__all__ = [
    "CustomDropout",
    "VGG11",
    "VGG11Encoder",
    "VGG11Classifier",
    "VGG11Localizer",
    "VGG11UNet",
    # "MultiTaskPerceptionModel" — import directly from models.multitask
]
