from collections import Counter
from typing import Any
import warnings

import numpy as np
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class MaskedPerLabelBenchmark:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.use_missingness_mask = config.get("use_missingness_mask", True)
        self.models: list[float | Any] = []
        self.backends: list[str] = []

    def _inputs(
        self, features: np.ndarray, feature_missingness: np.ndarray
    ) -> np.ndarray:
        features = np.asarray(features)
        if self.use_missingness_mask:
            return np.concatenate([features, feature_missingness], axis=1)
        return features

    def _estimator(self, model_type: str, targets: np.ndarray) -> Any:
        random_state = self.config.get("random_state", 42)
        if model_type == "decision_tree":
            tree = self.config.get("decision_tree", self.config)
            return DecisionTreeClassifier(
                max_depth=tree.get("max_depth"),
                min_samples_leaf=tree.get("min_samples_leaf", 1),
                class_weight=tree.get("class_weight"),
                random_state=random_state,
            )
        if model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError("xgboost is not installed")
            xgb = self.config.get("xgboost", self.config)
            positives = targets.sum()
            scale_pos_weight = (
                (len(targets) - positives) / positives
                if xgb.get("balance_classes", True) and positives
                else 1.0
            )
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=xgb.get("n_estimators", 100),
                learning_rate=xgb.get("learning_rate", 0.05),
                max_depth=xgb.get("max_depth", 3),
                min_child_weight=xgb.get("min_child_weight", 1.0),
                subsample=xgb.get("subsample", 1.0),
                colsample_bytree=xgb.get("colsample_bytree", 1.0),
                reg_lambda=xgb.get("reg_lambda", 1.0),
                scale_pos_weight=scale_pos_weight,
                tree_method=xgb.get("tree_method", "hist"),
                device=xgb.get("device", "cpu"),
                n_jobs=xgb.get("n_jobs", -1),
                random_state=random_state,
            )
        raise ValueError(f"Unsupported benchmark type: {model_type}")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_missingness: np.ndarray,
        label_availability: np.ndarray,
    ) -> "MaskedPerLabelBenchmark":
        inputs = self._inputs(features, feature_missingness)
        targets = np.asarray(targets)
        availability = np.asarray(label_availability, dtype=bool)
        primary = self.config["type"].lower()
        fallback = self.config.get("fallback")
        self.models = []
        self.backends = []

        for index in range(targets.shape[1]):
            observed = availability[:, index]
            observed_targets = targets[observed, index]
            classes = np.unique(observed_targets)
            if len(classes) == 0:
                self.models.append(0.5)
                self.backends.append("constant")
                continue
            if len(classes) == 1:
                self.models.append(float(classes[0]))
                self.backends.append("constant")
                continue

            try:
                model = self._estimator(primary, observed_targets)
                model.fit(inputs[observed], observed_targets)
                backend = primary
            except Exception as error:
                if not fallback or fallback == primary:
                    raise
                warnings.warn(
                    f"{primary} failed for label {index}; using {fallback}: {error}",
                    stacklevel=2,
                )
                model = self._estimator(fallback, observed_targets)
                model.fit(inputs[observed], observed_targets)
                backend = fallback
            self.models.append(model)
            self.backends.append(backend)
        return self

    def backend_counts(self) -> dict[str, int]:
        return dict(Counter(self.backends))

    def predict_proba(
        self, features: np.ndarray, feature_missingness: np.ndarray
    ) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Benchmark must be fitted before prediction")
        inputs = self._inputs(features, feature_missingness)
        probabilities = np.empty((len(inputs), len(self.models)), dtype=np.float32)
        for index, model in enumerate(self.models):
            if isinstance(model, float):
                probabilities[:, index] = model
            else:
                positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
                probabilities[:, index] = model.predict_proba(inputs)[:, positive_index]
        return probabilities


def build_benchmark(config: dict[str, Any]) -> MaskedPerLabelBenchmark:
    return MaskedPerLabelBenchmark(config)
