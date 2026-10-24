"""Stratified train/test ayrımı."""

import pandas as pd
from sklearn.model_selection import train_test_split

from eda import OUTPUT_DIR, load_data
from temizlik import feature_columns

TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_NAMES = {0: "malignant", 1: "benign"}


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cols = feature_columns(df)
    X = df[cols]
    y = df["target"]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def print_split_summary(
    y: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    print("=== Stratified train/test split ===")
    print(f"test_size={TEST_SIZE}, stratify=y, random_state={RANDOM_STATE}")
    print(f"Toplam: {len(y)}  |  Train: {len(y_train)}  |  Test: {len(y_test)}")

    print("\nSınıf dağılımı:")
    for name, series in (("Tüm veri", y), ("Train", y_train), ("Test", y_test)):
        counts = series.value_counts().sort_index()
        percents = series.value_counts(normalize=True).sort_index() * 100
        parts = [
            f"{TARGET_NAMES[label]}={counts[label]} ({percents[label]:.1f}%)"
            for label in counts.index
        ]
        print(f"  {name}: {', '.join(parts)}")


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    train = X_train.copy()
    train["target"] = y_train
    test = X_test.copy()
    test["target"] = y_test

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"\nKayıt: {train_path}")
    print(f"Kayıt: {test_path}")


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print_split_summary(df["target"], y_train, y_test)
    save_split(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
