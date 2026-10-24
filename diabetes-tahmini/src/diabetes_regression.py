"""Linear regression model trained on the sklearn diabetes dataset."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.logging_config import setup_logging

logger = setup_logging()

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass
class DiabetesDataset:
    """Container for diabetes features, target, and feature names."""

    features: np.ndarray
    target: np.ndarray
    feature_names: list[str]


@dataclass
class TrainTestSplit:
    """Train and test arrays produced from the diabetes dataset."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_diabetes_dataset() -> DiabetesDataset:
    """Load the sklearn diabetes dataset."""
    logger.info("Loading the sklearn diabetes dataset.")
    diabetes = load_diabetes()
    dataset = DiabetesDataset(
        features=diabetes.data,
        target=diabetes.target,
        feature_names=list(diabetes.feature_names),
    )
    logger.info(
        "Loaded %s samples with %s features.",
        dataset.features.shape[0],
        dataset.features.shape[1],
    )
    return dataset


def split_dataset(
    dataset: DiabetesDataset,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainTestSplit:
    """Split features and target into train and test sets."""
    logger.info(
        "Splitting data with test_size=%s and random_state=%s.",
        test_size,
        random_state,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
    )
    return TrainTestSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )


def train_linear_regression(data_split: TrainTestSplit) -> LinearRegression:
    """Fit a linear regression model on the training split."""
    logger.info("Training the linear regression model.")
    model = LinearRegression()
    model.fit(data_split.x_train, data_split.y_train)
    logger.info("Linear regression training completed.")
    return model


def evaluate_r2_score(
    model: LinearRegression,
    data_split: TrainTestSplit,
) -> float:
    """Evaluate model predictions with the R2 score."""
    logger.info("Evaluating predictions with the R2 score.")
    y_pred = model.predict(data_split.x_test)
    score = float(r2_score(data_split.y_test, y_pred))
    logger.info("R2 score: %s", score)
    return score


def evaluate_holdout_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R2, RMSE, and MAE for a prediction set."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}


def run_diabetes_linear_regression() -> float:
    """Train and evaluate the diabetes linear regression pipeline."""
    dataset = load_diabetes_dataset()
    data_split = split_dataset(dataset)
    model = train_linear_regression(data_split)
    return evaluate_r2_score(model, data_split)


if __name__ == "__main__":
    r2_result = run_diabetes_linear_regression()
    logger.info("Final R2 score: %s", r2_result)
