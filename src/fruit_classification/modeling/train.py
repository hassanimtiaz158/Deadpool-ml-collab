"""Stage 2: train the classifier and serialise it to ``models/``.

All randomness (split shuffling, estimator initialisation) is seeded from
``params.yaml`` so two runs of the same commit produce identical metrics.

Any failure that stops the stage is raised as :class:`TrainingError` with a
message that says what went wrong and what to try. ``main`` turns it into a
clean non-zero exit so ``dvc repro`` and CI fail without a long traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import warnings
from numbers import Integral
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from fruit_classification import config, prepare
from fruit_classification.modeling.models import build_model

logger = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when the train stage cannot complete."""


def current_commit_sha() -> str:
    """Best-effort Git SHA of the working tree (used for run provenance)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        # Not a git checkout, git not installed, or the directory is missing.
        return "unknown"
    return out.stdout.strip() or "unknown"


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/validation split used for model selection."""
    stratify_arg = y if stratify and len(np.unique(y)) > 1 else None
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_arg
    )


def _class_names(cache_tag: str) -> list[str]:
    """Class names for a cached feature set, read from its index.

    Returns ``[]`` when the index is unavailable or unreadable so a run never
    fails purely because the raw archive is missing.
    """
    index = prepare.INTERIM_SUBDIR / f"{cache_tag}_index.csv"
    if not index.is_file():
        return []
    try:
        import pandas as pd

        names = pd.read_csv(index)["label"].drop_duplicates().sort_values().tolist()
    except (OSError, ValueError, KeyError) as exc:
        # Empty file, bad CSV, or no 'label' column: labels are optional.
        logger.warning("could not read class names from %s: %s", index, exc)
        return []
    return [str(n) for n in names]


def _load_params(params: dict | None = None) -> dict:
    try:
        return config.params_with_defaults(params or config.load_params())
    except FileNotFoundError as exc:
        raise TrainingError(f"params.yaml not found: {exc}") from exc
    except Exception as exc:  # malformed YAML, wrong types, bad defaults, ...
        raise TrainingError(f"could not load params.yaml: {exc}") from exc


def _short(exc: Exception, limit: int = 300) -> str:
    """First ``limit`` characters of an error (sklearn messages can be huge)."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[:limit] + " ..."


def _section(params: dict, key: str) -> dict:
    value = params.get(key)
    if not isinstance(value, dict):
        raise TrainingError(f"params.yaml: '{key}' section is missing or not a mapping")
    return value


def _check_arrays(name: str, X: np.ndarray, y: np.ndarray) -> None:
    if X.shape[0] == 0:
        raise TrainingError(f"{name} features are empty; rebuild the cache")
    if X.shape[0] != len(y):
        raise TrainingError(
            f"{name} features and labels differ in length: {X.shape[0]} vs {len(y)}"
        )


def _save_bundle(bundle: dict, path: Path) -> None:
    """Write via a temp file so a failed dump never leaves a corrupt model."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        joblib.dump(bundle, tmp, compress=3)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def train(params: dict | None = None, progress: bool = True) -> dict:
    params = _load_params(params)
    seed = params.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TrainingError(f"params.yaml: 'seed' must be an int, got {seed!r}")
    split_cfg = _section(params, "split")
    runtime = _section(params, "runtime")
    train_cfg = dict(_section(params, "train"))

    cache_tag = params.get("cache_tag", "full")
    try:
        X, y, X_test, y_test = prepare.load(cache_tag)
    except FileNotFoundError as exc:
        raise TrainingError(
            f"features for cache_tag={cache_tag!r} not found ({exc}). "
            "Run `dvc pull`, or the prepare stage (`dvc repro`), first."
        ) from exc
    except (OSError, ValueError, KeyError) as exc:
        raise TrainingError(
            f"could not load features for cache_tag={cache_tag!r}: {exc}"
        ) from exc
    _check_arrays("train", X, y)
    _check_arrays("test", X_test, y_test)

    try:
        X_tr, X_val, y_tr, y_val = holdout_split(
            X,
            y,
            test_size=split_cfg.get("test_size", 0.2),
            seed=seed,
            stratify=split_cfg.get("stratify", True),
        )
    except ValueError as exc:
        raise TrainingError(
            f"train/validation split failed: {_short(exc)} "
            "(split.test_size must be in (0, 1); if a class has a single sample, "
            "set split.stratify to false)"
        ) from exc

    if train_cfg.get("scale_features", False):
        # Fitted on the training split only - never on validation or test data.
        from sklearn.preprocessing import StandardScaler

        try:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        except ValueError as exc:
            raise TrainingError(
                f"feature scaling failed (inf in features?): {_short(exc)}"
            ) from exc

    train_cfg.setdefault("n_jobs", runtime.get("n_jobs", -1))
    try:
        model = build_model(train_cfg, seed=seed)
    except ValueError as exc:  # includes ModelConfigError
        raise TrainingError(f"invalid train config: {exc}") from exc

    started = time.time()
    if progress:
        print(
            f"[train] fitting {train_cfg.get('model', 'sgd_svm')} on "
            f"{X_tr.shape[0]} rows x {X_tr.shape[1]} features ..."
        )
    try:
        with warnings.catch_warnings():
            # SGD-style estimators warn when max_iter is hit. That is a tuning
            # signal, not a pipeline failure, so record it instead of printing it.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_tr, y_tr)
    except MemoryError as exc:
        raise TrainingError(
            "ran out of memory while fitting; use a smaller feature cache "
            "or a lighter model (e.g. sgd_svm)"
        ) from exc
    except ValueError as exc:
        raise TrainingError(
            f"fit failed (NaN or inf in features, or bad labels?): {_short(exc)}"
        ) from exc
    fit_seconds = time.time() - started

    model_path = config.model_path(cache_tag)
    try:
        model_rel = str(model_path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        model_rel = str(model_path)  # path override outside the project root

    run_info = {
        "model": train_cfg.get("model", "sgd_svm"),
        "train_cfg": train_cfg,
        "commit_sha": current_commit_sha(),
        "seed": seed,
        "cache_tag": cache_tag,
        # Carried so `evaluate` can label a confusion matrix without the dataset.
        "classes": _class_names(cache_tag),
        "n_train_rows": int(X_tr.shape[0]),
        "n_val_rows": int(X_val.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "fit_seconds": round(fit_seconds, 2),
        "model_path": model_rel,
    }
    # The bundle keeps the run metadata next to the estimator so `evaluate` can
    # report provenance without re-running the fit.
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _save_bundle({"model": model, "run_info": run_info}, model_path)
    except OSError as exc:
        raise TrainingError(f"could not write model to {model_path}: {exc}") from exc
    return {
        "model": model,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "run_info": run_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fruit classifier")
    parser.add_argument("--model", help="override train.model from params.yaml")
    args = parser.parse_args()

    try:
        params = _load_params()
        if args.model:
            params.setdefault("train", {})["model"] = args.model
        result = train(params)
    except TrainingError as exc:
        sys.exit(f"[train] error: {exc}")
    except KeyboardInterrupt:
        sys.exit(130)
    print(json.dumps(result["run_info"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
