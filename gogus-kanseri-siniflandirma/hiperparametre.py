"""Karar ağacı hiperparametre araması ve görselleştirme."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from eda import OUTPUT_DIR, load_data
from model import evaluate_model, plot_decision_tree, plot_feature_importance
from split import RANDOM_STATE, split_data

PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, 6, 8, None],
    "min_samples_leaf": [1, 2, 5, 8, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}
CV_FOLDS = 5


def run_search(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def results_frame(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)
    results["max_depth"] = results["param_max_depth"].apply(
        lambda v: "None" if v is None else str(int(v))
    )
    keep = [
        "param_criterion",
        "max_depth",
        "param_min_samples_leaf",
        "param_min_samples_split",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    out = results[keep].sort_values("mean_test_score", ascending=False)
    out.to_csv(OUTPUT_DIR / "hiperparametre_sonuclar.csv", index=False)
    return out


def print_best(search: GridSearchCV, results: pd.DataFrame) -> None:
    print("=== GridSearchCV ===")
    print(f"scoring=roc_auc, cv={CV_FOLDS}, random_state={RANDOM_STATE}")
    print(f"Aday sayısı: {len(results)}")
    print(f"En iyi parametreler: {search.best_params_}")
    print(f"En iyi CV ROC-AUC: {search.best_score_:.3f}")
    print("\nİlk 8 kombinasyon:")
    print(results.head(8).to_string(index=False))


def plot_heatmap(results: pd.DataFrame) -> None:
    pivot = (
        results.groupby(["max_depth", "param_min_samples_leaf"], as_index=False)[
            "mean_test_score"
        ]
        .max()
        .pivot(index="param_min_samples_leaf", columns="max_depth", values="mean_test_score")
    )
    depth_order = [c for c in ["2", "3", "4", "5", "6", "8", "None"] if c in pivot.columns]
    pivot = pivot[depth_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "CV ROC-AUC"},
    )
    ax.set_title("Hiperparametre ısısı: max_depth × min_samples_leaf")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("min_samples_leaf")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hiperparametre_heatmap.png", dpi=150)
    plt.close(fig)


def plot_depth_vs_score(results: pd.DataFrame) -> None:
    best_per_depth = (
        results.groupby("max_depth", as_index=False)
        .agg(cv_auc=("mean_test_score", "max"), train_auc=("mean_train_score", "max"))
    )
    order = [d for d in ["2", "3", "4", "5", "6", "8", "None"] if d in set(best_per_depth["max_depth"])]
    best_per_depth["max_depth"] = pd.Categorical(best_per_depth["max_depth"], order, ordered=True)
    best_per_depth = best_per_depth.sort_values("max_depth")

    melted = best_per_depth.melt(
        id_vars="max_depth",
        value_vars=["train_auc", "cv_auc"],
        var_name="set",
        value_name="roc_auc",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=melted, x="max_depth", y="roc_auc", hue="set", marker="o", ax=ax)
    ax.set_title("Derinlik arttıkça train / CV ROC-AUC")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.85, 1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hiperparametre_derinlik.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    search = run_search(X_train, y_train)
    results = results_frame(search)
    print_best(search, results)
    plot_heatmap(results)
    plot_depth_vs_score(results)

    best_tree = search.best_estimator_
    evaluate_model("En iyi agac", best_tree, X_test, y_test)
    plot_feature_importance(best_tree, list(X_train.columns), stem="feature_importance_best")
    plot_decision_tree(best_tree, list(X_train.columns), stem="karar_agaci_best")
    print(f"\nGörseller kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
