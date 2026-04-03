import torch
from multitask import MultiTaskPerceptionModel


def run_inference(image_tensor, classifier_path="checkpoints/classifier.pth",
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
        out = model(image_tensor.to(device))
    return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    out = run_inference(x)
    print("classification:", out["classification"].shape)
    print("localization:  ", out["localization"].shape)
    print("segmentation:  ", out["segmentation"].shape)
