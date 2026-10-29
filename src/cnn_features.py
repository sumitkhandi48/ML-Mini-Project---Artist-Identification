"""Extract ImageNet pooled features in the canonical dataset order."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class PaintingDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image)


def get_model(name):
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        return torch.nn.Sequential(*(list(model.children())[:-1])), 2048
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()), 768
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten()), 1280
    raise ValueError(f"Unsupported backbone: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet50", "convnext_tiny", "efficientnet_b0"],
                        default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    seed_everything()

    cache = ROOT / "data" / f"cnn_features_{args.backbone}.npz"
    if cache.exists():
        print(f"Using existing cache: {cache}")
        return

    data = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    artists = data["artists"]
    paths = [ROOT / "data" / "images" / str(artists[y]) / str(filename)
             for filename, y in zip(data["files"], data["y"])]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first missing path: {missing[0]}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = PaintingDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(4, os.cpu_count() or 1), pin_memory=torch.cuda.is_available())
    model, feature_dim = get_model(args.backbone)
    model.to(DEVICE).eval()
    features = np.empty((len(dataset), feature_dim), dtype=np.float32)
    with torch.no_grad():
        for start, batch in enumerate(loader):
            batch = batch.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                output = model(batch).flatten(1)
            offset = start * args.batch_size
            features[offset:offset + len(output)] = output.float().cpu().numpy()
            if (start + 1) % 20 == 0:
                print(f"{min(offset + len(output), len(dataset))}/{len(dataset)}")
    np.savez_compressed(cache, X=features, y=data["y"], files=data["files"], artists=artists)
    print(f"Saved {cache} with shape {features.shape}")


if __name__ == "__main__":
    main()
