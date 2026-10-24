"""Compare improvement methods from plan_iyilelestirme.md on one split."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.linear_model import (
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from src.diabetes_regression import (
    DiabetesDataset,
    TrainTestSplit,
    evaluate_holdout_metrics,
    load_diabetes_dataset,
    split_dataset,
)
from src.logging_config import setup_logging

logger = setup_logging()

RANDOM_STATE = 42
N_SPLITS = 5
BASELINE_TEST_R2 = 0.452603
RESULTS_PATH = Path(__file__).resolve().parent.parent / "test_results.txt"
BMI_INDEX = 2
BP_INDEX = 3
S5_INDEX = 8


@dataclass
class ModelScore:
    """Holdout and cross-validation metrics for one candidate."""

    model_name: str
    cv_r2_mean: float
    cv_r2_std: float
    train_r2: float
    test_r2: float
    test_rmse: float
    test_mae: float

    @property
    def overfit_gap(self) -> float:
        """Absolute gap between train and test R2."""
        return abs(self.train_r2 - self.test_r2)


def make_cv() -> KFold:
    """Return the shared 5-fold splitter."""
    return KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def add_manual_interactions(features: np.ndarray) -> np.ndarray:
    """Add bmi*bp, bmi*s5, and bp*s5 interaction columns."""
    bmi_bp = features[:, BMI_INDEX] * features[:, BP_INDEX]
    bmi_s5 = features[:, BMI_INDEX] * features[:, S5_INDEX]
    bp_s5 = features[:, BP_INDEX] * features[:, S5_INDEX]
    return np.column_stack([features, bmi_bp, bmi_s5, bp_s5])


def drop_iqr_outliers(data_split: TrainTestSplit) -> TrainTestSplit:
    """Drop training rows whose target is outside the IQR fence."""
    q1 = np.percentile(data_split.y_train, 25)
    q3 = np.percentile(data_split.y_train, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (data_split.y_train >= lower) & (data_split.y_train <= upper)
    removed = int(np.sum(~mask))
    logger.info("Removed %s IQR outliers from the training set.", removed)
    return TrainTestSplit(
        x_train=data_split.x_train[mask],
        x_test=data_split.x_test,
        y_train=data_split.y_train[mask],
        y_test=data_split.y_test,
    )


def select_lasso_features(data_split: TrainTestSplit) -> TrainTestSplit:
    """Keep features with non-zero LassoCV coefficients."""
    lasso = LassoCV(cv=make_cv(), random_state=RANDOM_STATE, max_iter=10000)
    lasso.fit(data_split.x_train, data_split.y_train)
    keep_mask = np.abs(lasso.coef_) > 1e-12
    if not np.any(keep_mask):
        logger.info("Lasso zeroed every feature; keeping the original set.")
        return data_split
    logger.info("Lasso kept %s features.", int(np.sum(keep_mask)))
    return TrainTestSplit(
        x_train=data_split.x_train[:, keep_mask],
        x_test=data_split.x_test[:, keep_mask],
        y_train=data_split.y_train,
        y_test=data_split.y_test,
    )


def score_estimator(
    model_name: str,
    estimator,
    data_split: TrainTestSplit,
) -> ModelScore:
    """Fit one estimator and score it with CV plus holdout metrics."""
    logger.info("Evaluating candidate: %s", model_name)
    cv_scores = cross_val_score(
        estimator,
        data_split.x_train,
        data_split.y_train,
        scoring="r2",
        cv=make_cv(),
    )
    estimator.fit(data_split.x_train, data_split.y_train)
    train_pred = estimator.predict(data_split.x_train)
    test_pred = estimator.predict(data_split.x_test)
    train_metrics = evaluate_holdout_metrics(data_split.y_train, train_pred)
    test_metrics = evaluate_holdout_metrics(data_split.y_test, test_pred)
    score = ModelScore(
        model_name=model_name,
        cv_r2_mean=float(np.mean(cv_scores)),
        cv_r2_std=float(np.std(cv_scores)),
        train_r2=train_metrics["r2"],
        test_r2=test_metrics["r2"],
        test_rmse=test_metrics["rmse"],
        test_mae=test_metrics["mae"],
    )
    logger.info(
        "%s | cv_r2=%.4f +/- %.4f | train_r2=%.4f | test_r2=%.4f | rmse=%.4f | mae=%.4f",
        score.model_name,
        score.cv_r2_mean,
        score.cv_r2_std,
        score.train_r2,
        score.test_r2,
        score.test_rmse,
        score.test_mae,
    )
    return score


def build_svr() -> GridSearchCV:
    """Small RBF-SVR grid as specified in the improvement plan."""
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]
    )
    param_grid = {
        "model__C": [1.0, 10.0, 50.0, 100.0],
        "model__gamma": ["scale", 0.1, 1.0],
        "model__epsilon": [5.0, 10.0, 20.0],
    }
    return GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=make_cv(),
        n_jobs=-1,
    )


def build_candidates() -> list[tuple[str, object]]:
    """Return named estimators for every planned method."""
    ridge_alphas = np.logspace(-3, 3, 20)
    return [
        ("linear_regression_baseline", LinearRegression()),
        ("ridge_cv", RidgeCV(alphas=ridge_alphas)),
        ("lasso_cv", LassoCV(cv=make_cv(), random_state=RANDOM_STATE, max_iter=10000)),
        (
            "elastic_net_cv",
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                cv=make_cv(),
                random_state=RANDOM_STATE,
                max_iter=10000,
            ),
        ),
        (
            "poly2_ridge",
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", RidgeCV(alphas=ridge_alphas)),
                ]
            ),
        ),
        (
            "poly2_lasso",
            Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LassoCV(
                            cv=make_cv(),
                            random_state=RANDOM_STATE,
                            max_iter=20000,
                        ),
                    ),
                ]
            ),
        ),
        (
            "select_kbest_6_ridge",
            Pipeline(
                [
                    ("select", SelectKBest(f_regression, k=6)),
                    ("model", RidgeCV(alphas=ridge_alphas)),
                ]
            ),
        ),
        (
            "rfecv_ridge",
            Pipeline(
                [
                    (
                        "select",
                        RFECV(
                            estimator=Ridge(alpha=1.0),
                            step=1,
                            cv=make_cv(),
                            scoring="r2",
                        ),
                    ),
                    ("model", RidgeCV(alphas=ridge_alphas)),
                ]
            ),
        ),
        ("svr_rbf_grid", build_svr()),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "random_forest_shallow",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "huber_regressor",
            HuberRegressor(max_iter=2000),
        ),
    ]


def run_special_cases(data_split: TrainTestSplit) -> list[ModelScore]:
    """Run interaction, Lasso-selection, and IQR variants."""
    scores: list[ModelScore] = []

    interaction_split = TrainTestSplit(
        x_train=add_manual_interactions(data_split.x_train),
        x_test=add_manual_interactions(data_split.x_test),
        y_train=data_split.y_train,
        y_test=data_split.y_test,
    )
    scores.append(
        score_estimator(
            "manual_interactions_ridge",
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            interaction_split,
        )
    )

    lasso_split = select_lasso_features(data_split)
    scores.append(
        score_estimator(
            "lasso_selected_ridge",
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            lasso_split,
        )
    )

    iqr_split = drop_iqr_outliers(data_split)
    scores.append(
        score_estimator(
            "iqr_filtered_ridge",
            RidgeCV(alphas=np.logspace(-3, 3, 20)),
            iqr_split,
        )
    )
    return scores


def compare_models(dataset: DiabetesDataset | None = None) -> list[ModelScore]:
    """Train every planned candidate on the same diabetes split."""
    dataset = dataset or load_diabetes_dataset()
    data_split = split_dataset(dataset)
    scores = [
        score_estimator(name, estimator, data_split)
        for name, estimator in build_candidates()
    ]
    scores.extend(run_special_cases(data_split))
    scores.sort(key=lambda item: item.cv_r2_mean, reverse=True)
    return scores


def select_best_score(scores: list[ModelScore]) -> ModelScore:
    """Pick the best model by cross-validation R2, not holdout R2."""
    return max(scores, key=lambda item: item.cv_r2_mean)


def format_comparison_report(scores: list[ModelScore]) -> str:
    """Build a text report with the winner and every other result."""
    best = select_best_score(scores)
    best_test = max(scores, key=lambda item: item.test_r2)
    lines = [
        "Diabetes Model Comparison Results",
        "=================================",
        f"baseline_test_r2: {BASELINE_TEST_R2:.6f}",
        f"best_model_by_cv: {best.model_name}",
        f"best_cv_r2: {best.cv_r2_mean:.6f} +/- {best.cv_r2_std:.6f}",
        f"best_train_r2: {best.train_r2:.6f}",
        f"best_test_r2: {best.test_r2:.6f}",
        f"best_test_rmse: {best.test_rmse:.6f}",
        f"best_test_mae: {best.test_mae:.6f}",
        f"best_overfit_gap: {best.overfit_gap:.6f}",
        (
            f"highest_test_r2_model: {best_test.model_name} "
            f"({best_test.test_r2:.6f}, cv_r2={best_test.cv_r2_mean:.6f})"
        ),
        "",
        "All results ranked by CV R2",
        "---------------------------",
        (
            f"{'model':<28} {'cv_r2':>8} {'cv_std':>8} {'train_r2':>9} "
            f"{'test_r2':>8} {'rmse':>8} {'mae':>8} {'gap':>8}"
        ),
    ]
    for score in scores:
        lines.append(
            f"{score.model_name:<28} {score.cv_r2_mean:8.4f} {score.cv_r2_std:8.4f} "
            f"{score.train_r2:9.4f} {score.test_r2:8.4f} {score.test_rmse:8.2f} "
            f"{score.test_mae:8.2f} {score.overfit_gap:8.4f}"
        )
    return "\n".join(lines) + "\n"


def write_comparison_results(
    scores: list[ModelScore],
    output_path: Path = RESULTS_PATH,
) -> Path:
    """Write the comparison report to a text file."""
    output_path.write_text(format_comparison_report(scores), encoding="utf-8")
    logger.info("Wrote comparison results to %s", output_path)
    return output_path


def run_model_comparison() -> list[ModelScore]:
    """Run the full comparison and persist the report."""
    scores = compare_models()
    write_comparison_results(scores)
    return scores


if __name__ == "__main__":
    comparison_scores = run_model_comparison()
    winner = select_best_score(comparison_scores)
    logger.info("Best model by CV R2: %s (%.6f)", winner.model_name, winner.cv_r2_mean)
