"""Model definitions for the fruit classifier.

Every estimator is built from ``params.yaml`` so that a teammate can reproduce
or sweep a run with ``dvc exp run --set-param ...``.

Bad configuration raises :class:`ModelConfigError` with a message naming the
model and the offending key, instead of a bare scikit-learn traceback.
"""

from __future__ import annotations

from numbers import Integral, Real
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


class ModelConfigError(ValueError):
    """Raised when the ``train`` section of params.yaml is invalid.

    Subclasses ``ValueError`` so existing ``except ValueError`` code still works.
    """


def _number(
    cfg: dict[str, Any],
    key: str,
    default: Any,
    *,
    integer: bool = False,
    allow_none: bool = False,
) -> Any:
    """Read a strictly positive int/float from cfg, or raise ModelConfigError."""
    value = cfg.get(key, default)
    if value is None:
        if allow_none:
            return None
        raise ModelConfigError(f"'{key}' must not be null")

    expected = Integral if integer else Real
    # bool is a subclass of int, so reject it explicitly (``true`` in YAML).
    if isinstance(value, bool) or not isinstance(value, expected):
        kind = "an int" if integer else "a number"
        hint = ""
        if isinstance(value, str) and not integer:
            hint = " (PyYAML reads 1e-5 as text; write 1.0e-5 or 0.00001)"
        raise ModelConfigError(f"'{key}' must be {kind}, got {value!r}{hint}")
    if value <= 0:
        raise ModelConfigError(f"'{key}' must be > 0, got {value!r}")
    return value


def _n_jobs(cfg: dict[str, Any]) -> int:
    value = cfg.get("n_jobs", -1)
    if isinstance(value, bool) or not isinstance(value, Integral) or value == 0:
        raise ModelConfigError(f"'n_jobs' must be a non-zero int, got {value!r}")
    return value


def _flag(cfg: dict[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if not isinstance(value, bool):
        raise ModelConfigError(f"'{key}' must be true or false, got {value!r}")
    return value


def _hidden_layers(cfg: dict[str, Any]) -> tuple[int, ...]:
    value = cfg.get("hidden_layer_sizes", (256,))
    if isinstance(value, Integral) and not isinstance(value, bool):
        value = (value,)  # YAML ``hidden_layer_sizes: 256``
    valid = (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(
            isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in value
        )
    )
    if not valid:
        raise ModelConfigError(
            f"'hidden_layer_sizes' must be a positive int or a non-empty list "
            f"of positive ints, got {value!r}"
        )
    return tuple(value)


def _build(name: str, cfg: dict[str, Any], seed: int) -> Any:
    if name == "sgd_svm":
        return SGDClassifier(
            loss="hinge",
            alpha=_number(cfg, "alpha", 1e-5),
            max_iter=_number(cfg, "max_iter", 25, integer=True),
            tol=_number(cfg, "tol", 1e-4),
            random_state=seed,
            n_jobs=_n_jobs(cfg),
        )
    if name == "logistic_regression":
        return LogisticRegression(
            C=_number(cfg, "C", 1.0),
            max_iter=_number(cfg, "max_iter", 1000, integer=True),
            random_state=seed,
            n_jobs=_n_jobs(cfg),
        )
    if name == "linear_svc":
        return LinearSVC(
            C=_number(cfg, "C", 1.0),
            max_iter=_number(cfg, "max_iter", 3000, integer=True),
            tol=_number(cfg, "tol", 1e-4),
            random_state=seed,
        )
    if name in ("random_forest", "extra_trees"):
        cls = (
            RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        )
        return cls(
            n_estimators=_number(cfg, "n_estimators", 200, integer=True),
            max_depth=_number(cfg, "max_depth", None, integer=True, allow_none=True),
            min_samples_leaf=_number(cfg, "min_samples_leaf", 1, integer=True),
            random_state=seed,
            n_jobs=_n_jobs(cfg),
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=_hidden_layers(cfg),
            alpha=_number(cfg, "alpha", 1e-4),
            max_iter=_number(cfg, "max_iter", 60, integer=True),
            early_stopping=_flag(cfg, "early_stopping", True),
            n_iter_no_change=_number(cfg, "n_iter_no_change", 5, integer=True),
            random_state=seed,
        )
    raise AssertionError(f"unhandled model {name!r}")  # MODELS and _build out of sync


def build_model(cfg: dict[str, Any], seed: int) -> Any:
    """Instantiate the estimator described by the ``train`` section of params.

    Raises:
        ModelConfigError: ``cfg`` or ``seed`` is malformed, the model name is
            unknown, or a hyperparameter has the wrong type or range.
    """
    if not isinstance(cfg, dict):
        raise ModelConfigError(
            f"train config must be a mapping, got {type(cfg).__name__}"
        )
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ModelConfigError(f"seed must be an int, got {seed!r}")

    raw_name = cfg.get("model", "sgd_svm")
    if not isinstance(raw_name, str):
        raise ModelConfigError(f"'model' must be a string, got {raw_name!r}")
    name = raw_name.strip().lower()
    if name not in MODELS:
        raise ModelConfigError(f"unknown model {raw_name!r}; expected one of {MODELS}")

    try:
        return _build(name, cfg, int(seed))
    except ModelConfigError as exc:
        raise ModelConfigError(f"[train.model={name}] {exc}") from exc
    except (TypeError, ValueError) as exc:
        # scikit-learn rejected a value we could not catch ourselves.
        raise ModelConfigError(f"[train.model={name}] could not build: {exc}") from exc
