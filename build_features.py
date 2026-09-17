import csv
import numpy as np
from features import extract_features, FEATURE_NAMES

def process(split):
    with open(f"{split}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X, y, failed = [], [], 0
    for row in rows:
        feats = extract_features(row)
        if feats is None:
            failed += 1
            continue
        X.append(feats)
        y.append(row["type"])

    X = np.array(X)
    y = np.array(y)
    np.savez(f"features_{split}.npz", X=X, y=y, feature_names=FEATURE_NAMES)
    print(f"{split}: {len(y)} ok, {failed} descartadas (parseo fallido), "
          f"shape={X.shape}")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process(split)
