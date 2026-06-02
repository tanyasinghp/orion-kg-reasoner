#!/usr/bin/env python3
"""Download the Amazon Product Reviews dataset from Kaggle."""

import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_dataset() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "amazon.csv")

    if os.path.exists(csv_path):
        print(f"Dataset already exists at: {csv_path}")
        return csv_path

    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        import kagglehub

    print("Downloading Amazon Product Reviews dataset from Kaggle...")
    path = kagglehub.dataset_download("yasserh/amazon-product-reviews-dataset")
    print(f"Downloaded to: {path}")

    import shutil
    for f in os.listdir(path):
        if f.endswith(".csv"):
            shutil.copy(os.path.join(path, f), csv_path)
            print(f"Copied to: {csv_path}")
            return csv_path

    raise FileNotFoundError("No CSV found in downloaded dataset.")


if __name__ == "__main__":
    download_dataset()
