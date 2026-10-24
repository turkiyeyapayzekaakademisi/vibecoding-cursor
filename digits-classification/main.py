import os
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

ARTIFACT_DIR = Path(__file__).resolve().parent / "models"

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_ann() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(64,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(8, 8, 1)),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_datasets():
    set_seeds(SEED)
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_ann = scaler.fit_transform(X_train)
    X_test_ann = scaler.transform(X_test)
    X_train_cnn = X_train_ann.reshape(-1, 8, 8, 1)
    X_test_cnn = X_test_ann.reshape(-1, 8, 8, 1)
    return X_train_ann, X_test_ann, X_train_cnn, X_test_cnn, y_train, y_test, scaler


def load_or_train_models():
    ann_path = ARTIFACT_DIR / "ann.keras"
    cnn_path = ARTIFACT_DIR / "cnn.keras"
    meta_path = ARTIFACT_DIR / "meta.joblib"
    if ann_path.exists() and cnn_path.exists() and meta_path.exists():
        meta = joblib.load(meta_path)
        return (
            keras.models.load_model(ann_path),
            keras.models.load_model(cnn_path),
            meta["scaler"],
            float(meta["ann_acc"]),
            float(meta["cnn_acc"]),
        )

    X_train_ann, X_test_ann, X_train_cnn, X_test_cnn, y_train, y_test, scaler = (
        prepare_datasets()
    )
    ann = build_ann()
    cnn = build_cnn()
    ann.fit(X_train_ann, y_train, epochs=50, batch_size=32, verbose=0)
    cnn.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)
    ann_acc = float(accuracy_score(y_test, ann.predict(X_test_ann, verbose=0).argmax(axis=1)))
    cnn_acc = float(accuracy_score(y_test, cnn.predict(X_test_cnn, verbose=0).argmax(axis=1)))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    ann.save(ann_path)
    cnn.save(cnn_path)
    joblib.dump({"scaler": scaler, "ann_acc": ann_acc, "cnn_acc": cnn_acc}, meta_path)
    return ann, cnn, scaler, ann_acc, cnn_acc


def train_and_evaluate(
    name: str,
    model: keras.Model,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple[float, np.ndarray]:
    print(f"\n=== {name} ===")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{name} test accuracy: {accuracy:.4f}")
    print(f"\n{name} classification report:")
    print(classification_report(y_test, y_pred))
    print(f"{name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy, y_pred


def main() -> None:
    X_train_ann, X_test_ann, X_train_cnn, X_test_cnn, y_train, y_test, _scaler = (
        prepare_datasets()
    )

    ann_acc, _ = train_and_evaluate(
        "ANN", build_ann(), X_train_ann, y_train, X_test_ann, y_test
    )
    cnn_acc, _ = train_and_evaluate(
        "CNN", build_cnn(), X_train_cnn, y_train, X_test_cnn, y_test
    )

    print("\n=== Comparison ===")
    print(f"ANN accuracy: {ann_acc:.4f}")
    print(f"CNN accuracy: {cnn_acc:.4f}")
    winner = "CNN" if cnn_acc > ann_acc else "ANN" if ann_acc > cnn_acc else "Tie"
    print(f"Better model: {winner}")


if __name__ == "__main__":
    main()
