"""Göğüs kanseri veri seti: yükleme, EDA çıktıları ve görselleştirme."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["target_name"] = df["target"].map(
        {0: "malignant", 1: "benign"}
    )
    return df


def print_eda(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in {"target", "target_name"}]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns
    categorical_cols = df[feature_cols].select_dtypes(exclude="number").columns

    print("=== Veri özeti ===")
    print(f"Örnek sayısı: {len(df)}")
    print(f"Etiket sayısı: {df['target'].nunique()}")
    print(f"Etiketler: {sorted(df['target_name'].unique())}")
    print(f"Sayısal feature sayısı: {len(numeric_cols)}")
    print(f"Kategorik feature sayısı: {len(categorical_cols)}")

    print("\n=== info() ===")
    df.info()

    print("\n=== Sınıf dağılımı ===")
    counts = df["target_name"].value_counts()
    percents = df["target_name"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"{label}: {counts[label]} ({percents[label]:.1f}%)")

    print("\n=== describe() ===")
    print(df[numeric_cols].describe().T.to_string())


def plot_eda(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in {"target", "target_name"}]
    mean_features = [c for c in feature_cols if c.startswith("mean ")]

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="target_name", hue="target_name", ax=ax, legend=False)
    ax.set_title("Sınıf dağılımı")
    ax.set_xlabel("Sınıf")
    ax.set_ylabel("Adet")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sinif_dagilimi.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.7})
    ax.set_title("Özellik korelasyon ısı haritası")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "korelasyon.png", dpi=150)
    plt.close(fig)

    scaled = df[mean_features].copy()
    scaled[mean_features] = MinMaxScaler().fit_transform(scaled)
    scaled["target_name"] = df["target_name"]
    melted = scaled.melt(
        id_vars="target_name",
        value_vars=mean_features,
        var_name="feature",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="value", hue="target_name", ax=ax)
    ax.set_title("Ortalama özellikler: sınıfa göre dağılım (Min-Max)")
    ax.set_xlabel("Özellik")
    ax.set_ylabel("Normalize değer (0-1)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "boxplot_mean_ozellikler.png", dpi=150)
    plt.close(fig)

    print(f"\nGörseller kaydedildi: {OUTPUT_DIR}")


def main() -> None:
    df = load_data()
    print_eda(df)
    plot_eda(df)


if __name__ == "__main__":
    main()
