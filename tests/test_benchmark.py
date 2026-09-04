import numpy as np
import pytest

import src.benchmark as benchmark_module
from src.benchmark import build_benchmark


def test_decision_tree_benchmark_ignores_missing_labels():
    config = {
        "type": "decision_tree",
        "max_depth": 2,
        "min_samples_leaf": 1,
        "class_weight": None,
        "use_missingness_mask": True,
        "random_state": 42,
    }
    features = np.array([[0.0], [1.0], [0.0], [1.0]])
    feature_mask = np.zeros_like(features)
    availability = np.array([[1], [1], [1], [0]], dtype=bool)
    targets = np.array([[0.0], [1.0], [0.0], [0.0]])

    first = build_benchmark(config).fit(features, targets, feature_mask, availability)
    targets[-1, 0] = 99.0
    second = build_benchmark(config).fit(features, targets, feature_mask, availability)

    first_scores = first.predict_proba(features, feature_mask)
    second_scores = second.predict_proba(features, feature_mask)
    assert first_scores.shape == (4, 1)
    np.testing.assert_array_equal(first_scores, second_scores)


def test_xgboost_uses_decision_tree_fallback(monkeypatch):
    config = {
        "type": "xgboost",
        "fallback": "decision_tree",
        "use_missingness_mask": True,
        "random_state": 42,
        "xgboost": {"n_estimators": 2},
        "decision_tree": {"max_depth": 2},
    }
    features = np.array([[0.0], [1.0], [0.0], [1.0]])
    feature_mask = np.zeros_like(features)
    targets = np.array([[0.0], [1.0], [0.0], [1.0]])
    availability = np.ones_like(targets, dtype=bool)
    monkeypatch.setattr(benchmark_module, "XGBClassifier", None)

    with pytest.warns(UserWarning, match="using decision_tree"):
        benchmark = build_benchmark(config).fit(
            features, targets, feature_mask, availability
        )

    assert benchmark.backend_counts() == {"decision_tree": 1}
    assert benchmark.predict_proba(features, feature_mask).shape == (4, 1)


@pytest.mark.skipif(benchmark_module.XGBClassifier is None, reason="xgboost unavailable")
def test_xgboost_benchmark_predicts_one_probability_per_label():
    config = {
        "type": "xgboost",
        "fallback": "decision_tree",
        "use_missingness_mask": False,
        "random_state": 42,
        "xgboost": {"n_estimators": 2, "max_depth": 1, "n_jobs": 1},
    }
    features = np.array([[0.0], [1.0], [0.0], [1.0]])
    targets = np.array([[0.0], [1.0], [0.0], [1.0]])
    masks = np.zeros_like(features)
    availability = np.ones_like(targets, dtype=bool)

    benchmark = build_benchmark(config).fit(features, targets, masks, availability)
    probabilities = benchmark.predict_proba(features, masks)

    assert benchmark.backend_counts() == {"xgboost": 1}
    assert probabilities.shape == targets.shape
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
