"""
Download, process, and save the California Housing dataset to CSV
without using sklearn's broken downloader.
"""

import urllib.request
import tarfile
import numpy as np
import pandas as pd
import os

URL = "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.tgz"
ARCHIVE_NAME = "cal_housing.tgz"
OUTPUT_CSV = "california_housing.csv"

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

def main():
    # Download archive if not present
    if not os.path.exists(ARCHIVE_NAME):
        print("Downloading dataset...")
        urllib.request.urlretrieve(URL, ARCHIVE_NAME)

    # Extract and load data
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        file = tar.extractfile("CaliforniaHousing/cal_housing.data")
        data = np.loadtxt(file, delimiter=",")

    # Match sklearn column order
    columns_index = [8, 7, 2, 3, 4, 5, 6, 1, 0]
    data = data[:, columns_index]

    # Split target and features
    target = data[:, 0] / 100000.0  # MedHouseVal
    X = data[:, 1:]

    # Apply sklearn preprocessing
    X[:, 2] /= X[:, 5]      # AveRooms
    X[:, 3] /= X[:, 5]      # AveBedrms
    X[:, 5] = X[:, 4] / X[:, 5]  # AveOccup

    # Build DataFrame
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["MedHouseVal"] = target

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved dataset to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
