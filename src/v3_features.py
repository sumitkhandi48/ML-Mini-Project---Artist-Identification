"""Feature extraction for v3 spatial pyramid features."""

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from joblib import Parallel, delayed

from features import extract_features

SIZE = (256, 256)


def extract_spatial_pyramid(path):
    """
    Extract spatial HSV/LBP/mean-std features across 1x1, 2x2, and 4x4 grids.
    1x1: 1 cell * (16 colour + 10 lbp) = 26
    2x2: 4 cells * (16 colour + 10 lbp) = 104
    4x4: 16 cells * (16 colour + 10 lbp) = 416
    mean-std: 6
    Total spatial features = 552.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    colour, texture = [], []
    for grid in (1, 2, 4):
        edges = np.linspace(0, 256, grid + 1, dtype=int)
        for i in range(grid):
            for j in range(grid):
                ys, ye = edges[i], edges[i + 1]
                xs, xe = edges[j], edges[j + 1]
                cell_hsv = hsv[ys:ye, xs:xe]
                for channel, bins, value_range in (
                    (0, 8, [0, 180]), (1, 4, [0, 256]), (2, 4, [0, 256])
                ):
                    hist = cv2.calcHist([cell_hsv], [channel], None, [bins], value_range).flatten()
                    total = hist.sum()
                    colour.append(hist / total if total > 0 else hist)
                cell_lbp = lbp[ys:ye, xs:xe]
                counts = np.bincount(cell_lbp.astype(int).ravel(), minlength=10).astype(float)
                total = counts.sum()
                texture.append(counts / total if total > 0 else counts)

    mean_std = np.concatenate([img.reshape(-1, 3).mean(0),
                               img.reshape(-1, 3).std(0)]) / 255.0
    return np.concatenate(colour + texture + [mean_std])


def extract_all_v3_features(path):
    """Combine base (1795) and spatial pyramid (552) for 2347 features."""
    base = extract_features(path)
    if base is None:
        return None
    spatial = extract_spatial_pyramid(path)
    if spatial is None:
        return None
    return np.concatenate([base, spatial])


def build_and_cache_features_v3(dest="data/features_v3.npz"):
    d = np.load("data/features.npz", allow_pickle=True)
    files = d["files"]
    artists = d["artists"]
    y = d["y"]
    base_X = d["X"]

    print(f"Building spatial pyramid features for {len(files)} images...")
    paths = [os.path.join("data", "images", str(artists[y[i]]), str(files[i]))
             for i in range(len(files))]

    # Parallel extraction
    spatial_rows = Parallel(n_jobs=-1, batch_size=64)(
        delayed(extract_spatial_pyramid)(p) for p in paths
    )
    spatial_X = np.asarray(spatial_rows, dtype=np.float32)
    print("Spatial feature shape:", spatial_X.shape)
    assert spatial_X.shape == (len(files), 552)

    X_v3 = np.hstack([base_X, spatial_X])
    print("Combined v3 feature shape:", X_v3.shape)
    assert X_v3.shape == (len(files), 2347)

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    np.savez(dest, X=X_v3, y=y, files=files, artists=artists)
    print(f"Saved {dest} successfully.")
    return X_v3, y, files, artists


if __name__ == "__main__":
    build_and_cache_features_v3()
