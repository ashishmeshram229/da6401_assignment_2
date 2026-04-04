import torch
from models.multitask import MultiTaskPerceptionModel


def run_inference(image_tensor,
                  classifier_path="checkpoints/classifier.pth",
                  localizer_path="checkpoints/localizer.pth",
                  unet_path="checkpoints/unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskPerceptionModel(
        classifier_path=classifier_path,
        localizer_path=localizer_path,
        unet_path=unet_path,
    ).to(device)
    model.eval()
    with torch.no_grad():
        cls_out, bbox_out, seg_out = model(image_tensor.to(device))
    return cls_out, bbox_out, seg_out


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    cls_out, bbox_out, seg_out = run_inference(x)
    print("classification:", cls_out.shape)   # [1, 37]
    print("localization:  ", bbox_out.shape)  # [1, 4]
    print("segmentation:  ", seg_out.shape)   # [1, 3, 224, 224]
