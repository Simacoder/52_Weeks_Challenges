from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_DIR = Path("predictor") / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "iris_rf.joblib"
STATS_PATH = MODEL_DIR / "iris_stats.json"
PLOTS_DIR = Path("predictor") / "static" / "predictor" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    data = load_iris()
    X, y = data.data, data.target
    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = y

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train Model ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(
        {
            "estimator": clf,
            "target_names": data.target_names,
            "feature_names": data.feature_names,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH.resolve()}")

    # --- Compute Stats ---
    feature_stats = df.describe().to_dict()
    class_counts = df["target"].value_counts().to_dict()

    stats = {
        "feature_stats": feature_stats,
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
        "target_names": list(data.target_names),
    }

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {STATS_PATH.resolve()}")

    # --- Plots ---
    # 1. Pairplot (distribution by class)
    sns.pairplot(df, hue="target", diag_kind="kde", palette="Set2")
    plt.suptitle("Feature Distributions by Class", y=1.02)
    pairplot_path = PLOTS_DIR / "pairplot.png"
    plt.savefig(pairplot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved pairplot to {pairplot_path.resolve()}")

    # 2. Correlation Heatmap
    plt.figure(figsize=(6, 4))
    corr = df.drop(columns="target").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {heatmap_path.resolve()}")

if __name__ == "__main__":
    main()
