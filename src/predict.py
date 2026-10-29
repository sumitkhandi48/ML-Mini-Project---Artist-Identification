import argparse
import os
import random
import numpy as np
import joblib
import torch
from PIL import Image
from torchvision import models, transforms
from common import split_data
from features import extract_features
from spatial_features import extract_spatial_features
from v3_features import extract_all_v3_features

current_version = "svm"
scaler = None
pca = None
model = None
artists = None
cnn_model = None
cnn_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model_bundle(version="svm"):
    global scaler, pca, model, artists, current_version, cnn_model
    current_version = version
    if version == "cnn":
        checkpoint = torch.load("models/cnn_best_resnet50.pt", map_location="cpu",
                                weights_only=True)
        cnn_model = models.resnet50(weights=None)
        cnn_model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4), torch.nn.Linear(cnn_model.fc.in_features, 15)
        )
        cnn_model.load_state_dict(checkpoint["model_state"])
        cnn_model.eval()
        artists = np.asarray(checkpoint["artists"])
        return checkpoint
    model_path = "models/v3_final.joblib" if version == "v3" else "models/svm.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    model = bundle["model"]
    artists = bundle["artists"]
    return bundle


# Default to existing SVM model for backward compatibility
load_model_bundle("svm")


def predict(path, version=None):
    if version is not None and version != current_version:
        load_model_bundle(version)

    if current_version == "cnn":
        with Image.open(path) as image:
            image = cnn_transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            scores = torch.softmax(cnn_model(image), dim=1)[0].numpy()
        top = np.argsort(scores)[::-1][:3]
        return [(artists[i], float(scores[i])) for i in top]
    if current_version == "v3":
        combined = extract_all_v3_features(path)
        if combined is None:
            print("Could not read image:", path)
            return None
    else:
        f = extract_features(path)
        if f is None:
            print("Could not read image:", path)
            return None
        spatial = extract_spatial_features(path)
        if spatial is None:
            print("Could not read image:", path)
            return None
        combined = np.concatenate([f, spatial])

    z = pca.transform(scaler.transform(combined.reshape(1, -1)))
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(z)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(z)[0]
    else:
        scores = model.predict(z)[0]

    top = np.argsort(scores)[::-1][:3]
    return [(artists[i], scores[i]) for i in top]


def evaluate_random_samples(count, version=None):
    """Predict a user-selected number of paintings from the fixed test split."""
    if version is not None and version != current_version:
        load_model_bundle(version)

    features_file = "data/features_v3.npz" if current_version == "v3" else "data/features.npz"
    d = np.load(features_file, allow_pickle=True)
    y, files = d["y"], d["files"]
    idx = np.arange(len(y)).reshape(-1, 1)
    _, _, idx_te, _, _, y_te = split_data(idx, y)
    idx_te = idx_te.ravel()
    if count > len(idx_te):
        raise ValueError(f"sample count must be between 1 and {len(idx_te)}")

    correct = 0
    print(f"Evaluating {count} random test samples using model version: {current_version.upper()}")
    for i in random.sample(list(idx_te), count):
        path = os.path.join("data/images", str(artists[y[i]]), str(files[i]))
        res = predict(path)
        if not res:
            continue
        ok = res[0][0] == artists[y[i]]
        correct += ok
        print(f"{files[i]:14s} true: {artists[y[i]]:28s} predicted: {res[0][0]:28s} "
              f"{'OK' if ok else 'WRONG'}   top3: {[str(a) for a, _ in res]}")
    print(f"\n{correct}/{count} correct on random test paintings ({correct/count*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Predict an artist for one image or test samples.")
    parser.add_argument("image", nargs="?", help="path to one image to classify")
    parser.add_argument(
        "-n", "--samples", type=int,
        help="number of random paintings to check from the fixed test split (default: 5)",
    )
    parser.add_argument(
        "--model", choices=["svm", "v3", "cnn"], default="svm",
        help="model version to use: 'svm' (default), 'v3', or fine-tuned 'cnn'",
    )
    args = parser.parse_args()

    if args.image and args.samples is not None:
        parser.error("use either IMAGE or --samples, not both")

    if args.model != current_version:
        load_model_bundle(args.model)

    if args.image:
        res = predict(args.image)
        if res:
            print("Predicted artist:", res[0][0])
            print("Top 3:", ", ".join(a for a, _ in res))
    else:
        evaluate_random_samples(args.samples if args.samples is not None else 5)


if __name__ == "__main__":
    main()