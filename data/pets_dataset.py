import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_SIZE = 224


def get_transforms(split):
    if split == "train":
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="albumentations", label_fields=["bbox_labels"]),
        )
    else:
        return A.Compose(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="albumentations", label_fields=["bbox_labels"]),
        )


class OxfordIIITPetDataset(Dataset):
    def __init__(self, root: str, split: str = "train", seed: int = 42):
        self.root = root
        self.split = split
        self.images_dir = os.path.join(root, "images")
        self.masks_dir  = os.path.join(root, "annotations", "trimaps")
        self.xmls_dir   = os.path.join(root, "annotations", "xmls")
        self.list_file  = os.path.join(root, "annotations", "list.txt")

        self.samples = []
        self._build_samples(seed)
        self.transform = get_transforms(split)

    def _build_samples(self, seed):
        all_data = []
        with open(self.list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                class_id = int(parts[1]) - 1

                img_path  = os.path.join(self.images_dir, img_name + ".jpg")
                mask_path = os.path.join(self.masks_dir,  img_name + ".png")
                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    continue

                all_data.append({
                    "img_path":  img_path,
                    "mask_path": mask_path,
                    "xml_path":  os.path.join(self.xmls_dir, img_name + ".xml"),
                    "class_id":  class_id,
                })

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(all_data))
        split_at = int(0.8 * len(all_data))
        chosen = [all_data[i] for i in (idx[:split_at] if self.split == "train" else idx[split_at:])]
        self.samples = chosen

    def _parse_bbox_normalized(self, xml_path, img_w, img_h):
        if not os.path.exists(xml_path):
            return [0.0, 0.0, 1.0, 1.0]
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj  = root.find("object")
            if obj is None:
                return [0.0, 0.0, 1.0, 1.0]
            bb   = obj.find("bndbox")
            xmin = float(bb.find("xmin").text) / img_w
            ymin = float(bb.find("ymin").text) / img_h
            xmax = float(bb.find("xmax").text) / img_w
            ymax = float(bb.find("ymax").text) / img_h
            return [max(0., min(1., xmin)), max(0., min(1., ymin)),
                    max(0., min(1., xmax)), max(0., min(1., ymax))]
        except Exception:
            return [0.0, 0.0, 1.0, 1.0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        image = np.array(Image.open(s["img_path"]).convert("RGB"))
        mask  = np.array(Image.open(s["mask_path"]).convert("L"))
        h, w  = image.shape[:2]

        bbox_norm = self._parse_bbox_normalized(s["xml_path"], w, h)

        transformed = self.transform(
            image=image, mask=mask,
            bboxes=[bbox_norm], bbox_labels=[0],
        )

        image_t = transformed["image"].float()

        # Fix: ToTensorV2 may return mask as Tensor already
        mask_raw = transformed["mask"]
        if isinstance(mask_raw, torch.Tensor):
            mask_t = mask_raw.long()
        else:
            mask_t = torch.from_numpy(np.array(mask_raw)).long()
        mask_t = torch.clamp(mask_t - 1, 0, 2)

        if len(transformed["bboxes"]) > 0:
            x1, y1, x2, y2 = transformed["bboxes"][0]
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0

        # Pixel-space cx, cy, w, h
        cx = ((x1 + x2) / 2.0) * IMAGE_SIZE
        cy = ((y1 + y2) / 2.0) * IMAGE_SIZE
        bw = (x2 - x1) * IMAGE_SIZE
        bh = (y2 - y1) * IMAGE_SIZE
        bbox_t = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)

        return {
            "image": image_t,
            "label": torch.tensor(s["class_id"], dtype=torch.long),
            "bbox":  bbox_t,
            "mask":  mask_t,
        }
