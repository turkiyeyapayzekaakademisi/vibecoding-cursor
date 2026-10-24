"""Runtime checks for the digits ANN/CNN pipeline."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from main import SEED, build_ann, build_cnn, set_seeds


def _prepare():
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
    return digits, X, y_train, y_test, X_train_ann, X_test_ann, X_train_cnn, X_test_cnn


def test_split_has_all_classes() -> None:
    _, _, y_train, y_test, *_ = _prepare()
    assert sorted(np.unique(y_train)) == list(range(10))
    assert sorted(np.unique(y_test)) == list(range(10))


def test_cnn_reshape_matches_ann_and_images() -> None:
    digits, X, _, _, X_train_ann, _, X_train_cnn, _ = _prepare()
    assert np.allclose(X_train_cnn.reshape(len(X_train_ann), 64), X_train_ann)
    assert np.allclose(X[0].reshape(8, 8), digits.images[0])
    assert list(X_train_cnn.shape[1:]) == [8, 8, 1]


def test_eval_sets_share_labels() -> None:
    _, _, _, y_test, _, X_test_ann, _, X_test_cnn = _prepare()
    assert len(X_test_ann) == len(X_test_cnn) == len(y_test)


def test_prediction_contract() -> None:
    _, _, _, y_test, _, X_test_ann, _, X_test_cnn = _prepare()
    ann_proba = build_ann().predict(X_test_ann, verbose=0)
    cnn_proba = build_cnn().predict(X_test_cnn, verbose=0)
    assert ann_proba.shape == (len(y_test), 10)
    assert cnn_proba.shape == (len(y_test), 10)


def test_seeded_ann_is_stable() -> None:
    _, _, y_train, y_test, X_train_ann, X_test_ann, _, _ = _prepare()
    accs = []
    for _ in range(2):
        set_seeds(SEED)
        model = build_ann()
        model.fit(X_train_ann, y_train, epochs=8, batch_size=32, verbose=0)
        pred = model.predict(X_test_ann, verbose=0).argmax(axis=1)
        accs.append(float((pred == y_test).mean()))
    assert accs[0] == accs[1]


if __name__ == "__main__":
    test_split_has_all_classes()
    test_cnn_reshape_matches_ann_and_images()
    test_eval_sets_share_labels()
    test_prediction_contract()
    test_seeded_ann_is_stable()
    print("pipeline checks finished")
