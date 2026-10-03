"""Model definitions for the fruit classifier.

Every estimator is built from ``params.yaml`` so that a teammate can reproduce
or sweep a run with ``dvc exp run --set-param ...``.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

MODELS = (
    "sgd_svm",
    "logistic_regression",
    "linear_svc",
    "random_forest",
    "extra_trees",
    "mlp",
)


def build_model(cfg: dict[str, Any], seed: int) -> Any:
    """Instantiate the estimator described by the ``train`` section of params."""
    name = cfg.get("model", "sgd_svm")

    if name == "sgd_svm":
        return SGDClassifier(
            loss="hinge",
            alpha=cfg.get("alpha", 1e-5),
            max_iter=cfg.get("max_iter", 25),
            tol=cfg.get("tol", 1e-4),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", -1),
        )
    if name == "logistic_regression":
        return LogisticRegression(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 1000),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", -1),
        )
    if name == "linear_svc":
        return LinearSVC(
            C=cfg.get("C", 1.0),
            max_iter=cfg.get("max_iter", 3000),
            tol=cfg.get("tol", 1e-4),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", -1),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=seed,
            n_jobs=cfg.get("n_jobs", -1),
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", (256,))),
            alpha=cfg.get("alpha", 1e-4),
            max_iter=cfg.get("max_iter", 60),
            early_stopping=cfg.get("early_stopping", True),
            n_iter_no_change=cfg.get("n_iter_no_change", 5),
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")