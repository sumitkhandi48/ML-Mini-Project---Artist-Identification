"""Perform the approved, one-time held-out test evaluation."""

import csv
import os
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_auc_score,
                             roc_curve, top_k_accuracy_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from common import split_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARTIFACTS = ROOT / "results"
ARTISTS = None


class ImageDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        return self.transform(image), int(self.labels[index]), str(self.paths[index])


def save_test_outputs(name, y_true, probabilities, files):
    global ARTISTS
    predictions = probabilities.argmax(axis=1)
    top1 = accuracy_score(y_true, predictions)
    top3 = top_k_accuracy_score(y_true, probabilities, k=3,
                                labels=np.arange(len(ARTISTS)))
    macro_f1 = f1_score(y_true, predictions, average="macro")
    report = classification_report(y_true, predictions, target_names=ARTISTS,
                                   output_dict=True, zero_division=0)
    if name == "cnn_finetuned_resnet50":
        with open(ARTIFACTS / "cnn_per_artist.csv", "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["artist", "precision", "recall", "f1", "support"])
            for artist in ARTISTS:
                row = report[artist]
                writer.writerow([artist, row["precision"], row["recall"],
                                 row["f1-score"], row["support"]])
        cm = confusion_matrix(y_true, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=ARTISTS, yticklabels=ARTISTS)
        plt.xlabel("Predicted artist"); plt.ylabel("True artist")
        plt.xticks(rotation=90); plt.tight_layout()
        plt.savefig(ARTIFACTS / "cnn_confusion_matrix.png", dpi=160)
        plt.close()

        pairs = []
        for i in range(len(ARTISTS)):
            for j in range(i + 1, len(ARTISTS)):
                count = int(cm[i, j] + cm[j, i])
                pairs.append((count, ARTISTS[i], ARTISTS[j]))
        pairs.sort(reverse=True)
        with open(ARTIFACTS / "cnn_most_confused_pairs.csv", "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["artist_a", "artist_b", "symmetric_confusions"])
            writer.writerows((a, b, count) for count, a, b in pairs[:10])

        wrong = np.flatnonzero(predictions != y_true)[:8]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for axis, index in zip(axes.flat, wrong):
            image = Image.open(files[index]).convert("RGB")
            axis.imshow(image)
            axis.set_title(f"T: {ARTISTS[y_true[index]]}\nP: {ARTISTS[predictions[index]]}", fontsize=8)
            axis.axis("off")
        for axis in axes.flat[len(wrong):]:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "cnn_misclassified_grid.png", dpi=160)
        plt.close()

        bootstrap = np.random.default_rng(SEED)
        estimates = np.empty(10000)
        correct = (predictions == y_true).astype(np.float64)
        for index in range(len(estimates)):
            estimates[index] = bootstrap.choice(correct, len(correct), replace=True).mean()
        low, high = np.quantile(estimates, [0.025, 0.975])
        np.savez(ARTIFACTS / "cnn_test_predictions.npz", y_true=y_true,
                 probabilities=probabilities, predictions=predictions, files=files)
        with open(ARTIFACTS / "cnn_final_results.txt", "w", encoding="utf-8") as output:
            output.write(f"Model: fine-tuned ResNet-50, checkpoint epoch 13\n"
                         f"Test samples: {len(y_true)}\nTop-1 accuracy: {top1:.6f}\n"
                         f"Top-3 accuracy: {top3:.6f}\nMacro-F1: {macro_f1:.6f}\n"
                         f"Bootstrap 95% CI: [{low:.6f}, {high:.6f}]\n")
        with open(ARTIFACTS / "cnn_test_metrics.csv", "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["model", "top1_accuracy", "top3_accuracy", "macro_f1",
                             "bootstrap_low", "bootstrap_high"])
            writer.writerow([name, top1, top3, macro_f1, low, high])
    return top1, top3, macro_f1


def evaluate_cnn(y_test, test_files, artists):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.4),
                                   torch.nn.Linear(model.fc.in_features, len(artists)))
    checkpoint = torch.load(ROOT / "models" / "cnn_best_resnet50.pt",
                            map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    loader = DataLoader(ImageDataset(test_files, y_test), batch_size=16, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    probabilities = []
    with torch.no_grad():
        for images, _, _ in loader:
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                probabilities.append(torch.softmax(model(images.to(DEVICE)), dim=1).cpu().numpy())
    probabilities = np.concatenate(probabilities)
    return probabilities


def main():
    global ARTISTS
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    ARTISTS = data["artists"]
    _, _, test_indices, _, _, y_test = split_data(np.arange(len(data["y"])), data["y"])
    test_indices = test_indices.astype(int)
    test_paths = [ROOT / "data" / "images" / str(ARTISTS[data["y"][i]]) / str(data["files"][i])
                  for i in test_indices]

    cnn_probabilities = evaluate_cnn(y_test, test_paths, ARTISTS)
    cnn_metrics = save_test_outputs("cnn_finetuned_resnet50", y_test, cnn_probabilities, test_paths)

    frozen_metrics = []
    for backbone in ("resnet50", "convnext_tiny"):
        feature_data = np.load(ROOT / "data" / f"cnn_features_{backbone}.npz",
                               allow_pickle=True)
        _, _, feature_test, _, _, feature_y_test = split_data(feature_data["X"], feature_data["y"])
        bundle = joblib.load(ROOT / "models" / f"cnn_frozen_best_{backbone}.joblib")
        frozen_probabilities = bundle["model"].predict_proba(
            bundle["scaler"].transform(feature_test))
        metrics = (accuracy_score(feature_y_test, frozen_probabilities.argmax(1)),
                   top_k_accuracy_score(feature_y_test, frozen_probabilities, k=3,
                                        labels=np.arange(len(ARTISTS))),
                   f1_score(feature_y_test, frozen_probabilities.argmax(1), average="macro"))
        frozen_metrics.append((f"frozen_{backbone}_logreg_C0.1", *metrics))

    base = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    spatial = np.load(ROOT / "data" / "features_spatial.npz", allow_pickle=True)
    if not np.array_equal(base["files"], spatial["files"]):
        raise ValueError("Feature caches are not aligned")
    combined = np.hstack([base["X"], spatial["X"]])
    _, _, svm_test, _, _, svm_y_test = split_data(combined, base["y"])
    svm_bundle = joblib.load(ROOT / "models" / "svm.joblib")
    svm_probabilities = svm_bundle["model"].predict_proba(
        svm_bundle["pca"].transform(svm_bundle["scaler"].transform(svm_test)))
    svm_metrics = (accuracy_score(svm_y_test, svm_probabilities.argmax(1)),
                   top_k_accuracy_score(svm_y_test, svm_probabilities, k=3,
                                        labels=np.arange(len(ARTISTS))),
                   f1_score(svm_y_test, svm_probabilities.argmax(1), average="macro"))

    rows = [
        ("classical_svm", *svm_metrics, "current project result"),
        *frozen_metrics,
        ("fine_tuned_resnet50", *cnn_metrics, "approved headline model"),
        ("paper_svm", 0.681, "", "", "Chen & Deng paper"),
        ("paper_cnn", 0.747, "", "", "Chen & Deng paper"),
    ]
    with open(ARTIFACTS / "cnn_final_comparison.csv", "w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["model", "test_top1", "test_top3", "test_macro_f1", "note"])
        writer.writerows(rows)

    cnn_y_onehot = label_binarize(y_test, classes=np.arange(len(ARTISTS)))
    fpr, tpr, _ = roc_curve(cnn_y_onehot.ravel(), cnn_probabilities.ravel())
    macro_auc = roc_auc_score(cnn_y_onehot, cnn_probabilities, average="macro", multi_class="ovr")
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, label=f"fine-tuned ResNet-50 (macro AUC={macro_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("One-vs-rest micro ROC on held-out test set")
    plt.legend(); plt.tight_layout()
    plt.savefig(ARTIFACTS / "cnn_roc_auc.png", dpi=160)
    plt.close()
    with open(ARTIFACTS / "cnn_final_results.txt", "a", encoding="utf-8") as output:
        output.write(f"Macro one-vs-rest AUC: {macro_auc:.6f}\n")
    print(f"Fine-tuned ResNet-50 test metrics: top1={cnn_metrics[0]:.6f}, "
          f"top3={cnn_metrics[1]:.6f}, macro_f1={cnn_metrics[2]:.6f}, macro_auc={macro_auc:.6f}")
    for row in frozen_metrics:
        print(f"{row[0]} test metrics: top1={row[1]:.6f}, top3={row[2]:.6f}, macro_f1={row[3]:.6f}")
    print(f"classical_svm test metrics: top1={svm_metrics[0]:.6f}, "
          f"top3={svm_metrics[1]:.6f}, macro_f1={svm_metrics[2]:.6f}")
    print("Held-out test evaluation completed exactly once for the approved configurations.")


if __name__ == "__main__":
    main()
