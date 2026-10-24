"""Unit tests for the diabetes linear regression pipeline."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from src.diabetes_regression import (
    evaluate_holdout_metrics,
    evaluate_r2_score,
    load_diabetes_dataset,
    run_diabetes_linear_regression,
    split_dataset,
    train_linear_regression,
)
from src.model_comparison import (
    add_manual_interactions,
    format_comparison_report,
    run_model_comparison,
    select_best_score,
)

TEST_RESULTS_PATH = Path(__file__).resolve().parent.parent / "test_results.txt"


def test_load_diabetes_dataset() -> None:
    """The diabetes dataset should contain samples and ten features."""
    dataset = load_diabetes_dataset()
    assert dataset.features.shape[0] > 0
    assert dataset.features.shape[1] == 10
    assert dataset.target.shape[0] == dataset.features.shape[0]


def test_split_dataset() -> None:
    """The train and test split should keep all rows."""
    dataset = load_diabetes_dataset()
    data_split = split_dataset(dataset)
    total_rows = data_split.x_train.shape[0] + data_split.x_test.shape[0]
    assert total_rows == dataset.features.shape[0]
    assert data_split.x_train.shape[0] > data_split.x_test.shape[0]


def test_train_linear_regression() -> None:
    """Training should return a fitted linear regression model."""
    dataset = load_diabetes_dataset()
    data_split = split_dataset(dataset)
    model = train_linear_regression(data_split)
    assert isinstance(model, LinearRegression)
    assert model.coef_.shape[0] == dataset.features.shape[1]


def test_evaluate_r2_score() -> None:
    """R2 score on the diabetes test split should be a finite number."""
    dataset = load_diabetes_dataset()
    data_split = split_dataset(dataset)
    model = train_linear_regression(data_split)
    score = evaluate_r2_score(model, data_split)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_evaluate_holdout_metrics() -> None:
    """Holdout metrics should include r2, rmse, and mae."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = evaluate_holdout_metrics(y_true, y_pred)
    assert metrics["r2"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_add_manual_interactions() -> None:
    """Manual interactions should add three extra columns."""
    features = np.ones((5, 10))
    expanded = add_manual_interactions(features)
    assert expanded.shape == (5, 13)


def test_run_diabetes_linear_regression_baseline() -> None:
    """The original baseline pipeline should stay above a valid R2 range."""
    score = run_diabetes_linear_regression()
    assert -1.0 <= score <= 1.0


def test_run_model_comparison_and_write_results() -> None:
    """Compare planned methods and persist every score as a text file."""
    scores = run_model_comparison()
    best = select_best_score(scores)
    report = TEST_RESULTS_PATH.read_text(encoding="utf-8")
    assert TEST_RESULTS_PATH.exists()
    assert "best_model_by_cv:" in report
    assert best.model_name in report
    assert "All results ranked by CV R2" in format_comparison_report(scores)
    assert len(scores) >= 10
    assert -1.0 <= best.cv_r2_mean <= 1.0
