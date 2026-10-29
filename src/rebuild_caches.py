"""Rebuild the base and spatial feature caches with one deterministic row order."""

import os
import sys
import time

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from features import extract_features

IMG_DIR = "data/images"
EDGES = [0, 85, 170, 256]


def spatial_features(path):
    img = cv2.resize(cv2.imread(path), (256, 256))
    if img is None:
        raise ValueError(f"Unreadable image: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    colour, texture = [], []
    for i in range(3):
        for j in range(3):
            ys, ye = EDGES[i], EDGES[i + 1]
            xs, xe = EDGES[j], EDGES[j + 1]
            cell = hsv[ys:ye, xs:xe]
            for channel, bins, value_range in (
                (0, 8, [0, 180]), (1, 4, [0, 256]), (2, 4, [0, 256])
            ):
                hist = cv2.calcHist([cell], [channel], None, [bins], value_range).flatten()
                colour.append(hist / hist.sum())
            counts = np.bincount(
                lbp[ys:ye, xs:xe].astype(int).ravel(), minlength=10
            ).astype(float)
            texture.append(counts / counts.sum())
    mean_std = np.concatenate([img.reshape(-1, 3).mean(0),
                               img.reshape(-1, 3).std(0)]) / 255.0
    return np.concatenate(colour + texture + [mean_std])


def main():
    artists = sorted(
        name for name in os.listdir(IMG_DIR)
        if os.path.isdir(os.path.join(IMG_DIR, name))
    )
    base, spatial, labels, files = [], [], [], []
    start = time.time()
    for label, artist in enumerate(artists):
        folder = os.path.join(IMG_DIR, artist)
        for filename in sorted(os.listdir(folder)):
            path = os.path.join(folder, filename)
            base_feature = extract_features(path)
            if base_feature is None:
                raise ValueError(f"Unreadable image: {path}")
            base.append(base_feature)
            spatial.append(spatial_features(path))
            labels.append(label)
            files.append(filename)
        print(f"Done {artist}; rows={len(files)}; elapsed={time.time() - start:.0f}s")

    os.makedirs("data", exist_ok=True)
    np.savez("data/features.npz", X=np.asarray(base), y=np.asarray(labels),
             files=np.asarray(files), artists=np.asarray(artists))
    np.savez("data/features_spatial.npz", X=np.asarray(spatial),
             files=np.asarray(files), artists=np.asarray(artists))
    print(f"Base shape: {np.asarray(base).shape}")
    print(f"Spatial shape: {np.asarray(spatial).shape}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
