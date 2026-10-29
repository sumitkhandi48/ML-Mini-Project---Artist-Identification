"""Fine-tune an ImageNet backbone with a validation-selected checkpoint."""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import split_data

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_CLASSES = 15


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class PaintingDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index])


def make_model(backbone):
    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, NUM_CLASSES))
        head = model.fc
        trainable = [model.layer3, model.layer4]
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, NUM_CLASSES))
        head = model.classifier
        trainable = [model.features[-2], model.features[-1]]
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Flatten(1), nn.LayerNorm(in_features),
                                         nn.Dropout(0.4), nn.Linear(in_features, NUM_CLASSES))
        head = model.classifier
        trainable = [model.features[-2], model.features[-1]]
    return model, head, trainable


def set_trainable(model, head, blocks, phase_b):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    if phase_b:
        for block in blocks:
            for parameter in block.parameters():
                parameter.requires_grad = True


def accuracy_and_loss(model, loader, criterion, scaler):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                output = model(images)
                loss = criterion(output, labels)
            total_loss += loss.item() * len(labels)
            correct += (output.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["resnet50", "efficientnet_b0", "convnext_tiny"],
                        default="resnet50")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    args = parser.parse_args()
    seed_everything()

    data = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    all_paths = [ROOT / "data" / "images" / str(data["artists"][y]) / str(filename)
                 for filename, y in zip(data["files"], data["y"])]
    indices = np.arange(len(all_paths))
    i_tr, i_va, _, _, _, _ = split_data(indices, data["y"])
    # split_data returns arrays of indices in the same fixed stratified split.
    train_idx, val_idx = i_tr.astype(int), i_va.astype(int)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(PaintingDataset([all_paths[i] for i in train_idx], data["y"][train_idx],
                                               train_transform), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(PaintingDataset([all_paths[i] for i in val_idx], data["y"][val_idx],
                                             eval_transform), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model, head, blocks = make_model(args.backbone)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    rows, best_val, best_state = [], -1.0, None

    total_epochs = args.head_epochs + args.finetune_epochs
    for epoch in range(total_epochs):
        phase_b = epoch >= args.head_epochs
        set_trainable(model, head, blocks, phase_b)
        if epoch == 0 or epoch == args.head_epochs:
            groups = [{"params": [p for p in head.parameters() if p.requires_grad], "lr": 1e-3}]
            if phase_b:
                groups.append({"params": [p for b in blocks for p in b.parameters() if p.requires_grad],
                               "lr": 1e-4})
            optimizer = AdamW(groups, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - epoch))
        model.train()
        start = time.time()
        total_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                output = model(images)
                loss = criterion(output, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            correct += (output.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()
        val_loss, val_acc = accuracy_and_loss(model, val_loader, criterion, scaler)
        train_loss, train_acc = total_loss / total, correct / total
        elapsed = time.time() - start
        row = {"backbone": args.backbone, "trainable_layers": "head" if not phase_b else "head+last_two_stages",
               "epoch": epoch + 1, "phase": "A_head" if not phase_b else "B_finetune",
               "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
               "val_acc": val_acc, "time_s": elapsed}
        rows.append(row)
        print(f"epoch {epoch + 1}/{total_epochs} {row['phase']} "
              f"train={train_acc:.4f} val={val_acc:.4f} time={elapsed:.1f}s", flush=True)
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    checkpoint = ROOT / "models" / f"cnn_best_{args.backbone}.pt"
    torch.save({"model_state": model.state_dict(), "backbone": args.backbone,
                "val_acc": best_val, "artists": data["artists"].tolist()}, checkpoint)
    log_path = ROOT / "results" / f"cnn_training_log_{args.backbone}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.figure(figsize=(10, 4))
    epochs = [row["epoch"] for row in rows]
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in rows], label="train")
    plt.plot(epochs, [row["val_loss"] for row in rows], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_acc"] for row in rows], label="train")
    plt.plot(epochs, [row["val_acc"] for row in rows], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "results" / f"cnn_training_curves_{args.backbone}.png", dpi=150)
    plt.close()
    params = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"checkpoint": str(checkpoint), "best_val_acc": best_val,
                      "params": params, "size_mb": checkpoint.stat().st_size / 1e6}))
    print("Test set was not evaluated.")


if __name__ == "__main__":
    main()
