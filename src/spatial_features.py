import cv2
import numpy as np
from skimage.feature import local_binary_pattern

EDGES = [0, 85, 170, 256]


def extract_spatial_features(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
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
