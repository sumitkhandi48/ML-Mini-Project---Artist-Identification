import os, time
import cv2
import numpy as np
from skimage.feature import hog

IMG_DIR = "data/images"
SIZE = (256, 256)

def extract_features(path):
    img = cv2.imread(path)                    # BGR color image
    if img is None:
        return None
    img = cv2.resize(img, SIZE)               # preprocessing: 256x256
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) HOG: edge directions -> 1764 numbers
    hog_f = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                cells_per_block=(2, 2), feature_vector=True)

    # 2) Color histogram: 8 bins per channel -> 24 numbers
    hist = []
    for ch in range(3):
        h = cv2.calcHist([img], [ch], None, [8], [0, 256]).flatten()
        hist.append(h / h.sum())              # normalize so values are 0-1
    hist_f = np.concatenate(hist)

    # 3) Hu moments: 7 numbers, log-scaled because raw values are tiny
    hu = cv2.HuMoments(cv2.moments(gray)).flatten()
    hu_f = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return np.concatenate([hog_f, hist_f, hu_f])

if __name__ == "__main__":
    artists = sorted(os.listdir(IMG_DIR))
    X, y, files = [], [], []
    start = time.time()
    for label, artist in enumerate(artists):
        folder = os.path.join(IMG_DIR, artist)
        for f in sorted(os.listdir(folder)):
            feat = extract_features(os.path.join(folder, f))
            if feat is None:
                print("Skipped unreadable:", f)
                continue
            X.append(feat); y.append(label); files.append(f)
        print(f"Done {artist} ({len(X)} images so far, {time.time()-start:.0f}s)")

    X = np.array(X); y = np.array(y)
    os.makedirs("data", exist_ok=True)
    np.savez("data/features.npz", X=X, y=y, files=np.array(files),
             artists=np.array(artists))
    print("\nFeature matrix shape:", X.shape)
    print("Saved data/features.npz")