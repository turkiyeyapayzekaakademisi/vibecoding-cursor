"""Karar ağacı + baseline: eğitim, metrikler, feature importance ve ağaç görseli."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

from eda import OUTPUT_DIR, load_data
from split import RANDOM_STATE, TARGET_NAMES, split_data

MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 5
CLASS_LABELS = [TARGET_NAMES[0], TARGET_NAMES[1]]


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[DecisionTreeClassifier, DummyClassifier]:
    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
    )
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    return tree, baseline


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_LABELS,
            zero_division=0,
        )
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=CLASS_LABELS,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    slug = name.lower().replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"confusion_{slug}.png", dpi=150)
    plt.close(fig)


def plot_roc(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC eğrisi")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_auc.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    stem: str = "feature_importance",
) -> None:
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": tree.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    importance = importance[importance["importance"] > 0]

    print("\n=== Feature importance ===")
    print(importance.to_string(index=False))
    importance.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=importance, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title("Karar ağacı feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Özellik")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)


def plot_decision_tree(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    stem: str = "karar_agaci",
) -> None:
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=CLASS_LABELS,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        f"Karar ağacı (max_depth={tree.max_depth}, "
        f"min_samples_leaf={tree.min_samples_leaf}, criterion={tree.criterion})"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    tree, baseline = train_models(X_train, y_train)
    models = {"Karar agaci": tree, "Baseline": baseline}

    print(
        f"Eğitim: DecisionTreeClassifier(max_depth={MAX_DEPTH}, "
        f"min_samples_leaf={MIN_SAMPLES_LEAF}) + DummyClassifier(most_frequent)"
    )
    for name, model in models.items():
        evaluate_model(name, model, X_test, y_test)

    plot_roc(models, X_test, y_test)
    plot_feature_importance(tree, list(X_train.columns))
    plot_decision_tree(tree, list(X_train.columns))
    print(f"\nGörseller kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
