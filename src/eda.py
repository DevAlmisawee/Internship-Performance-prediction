"""
eda.py
------
Exploratory Data Analysis for the Internship Performance dataset.
Generates summary statistics and saves plots (class balance, feature
distributions, correlation heatmap) to the logs/ directory.

Run directly:
    python src/eda.py
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, clean_data, FEATURE_COLUMNS, TARGET_COLUMN, CLASS_ORDER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "internship_data.csv")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def run_eda(csv_path: str = DATA_PATH, output_dir: str = LOGS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    df_raw = load_data(csv_path)
    print("=" * 60)
    print("RAW DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df_raw.shape}")
    print(f"\nMissing values per column:\n{df_raw.isna().sum()}")
    print(f"\nDuplicate rows: {df_raw.duplicated().sum()}")
    print(f"\nData types:\n{df_raw.dtypes}")

    df = clean_data(df_raw)

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS (post-cleaning)")
    print("=" * 60)
    print(df[FEATURE_COLUMNS].describe().T)

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(df[TARGET_COLUMN].value_counts())

    # --- Plot 1: Class balance ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER)
    ax.bar(counts.index, counts.values, color=["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"])
    ax.set_title("Class Distribution: Internship Performance")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=120)
    plt.close(fig)

    # --- Plot 2: Feature distributions by class (boxplots) ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLUMNS):
        data_by_class = [df[df[TARGET_COLUMN] == c][col].values for c in CLASS_ORDER]
        axes[i].boxplot(data_by_class, tick_labels=CLASS_ORDER)
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(len(FEATURE_COLUMNS), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions_by_class.png"), dpi=120)
    plt.close(fig)

    # --- Plot 3: Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_COLUMNS].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLUMNS)
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)

    print(f"\nEDA plots saved to: {output_dir}")
    return df


if __name__ == "__main__":
    run_eda()
