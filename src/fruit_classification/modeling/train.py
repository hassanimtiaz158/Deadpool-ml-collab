"""Stage 2: train the classifier and serialise it to ``models/``.

All randomness (split shuffling, estimator initialisation) is seeded from
``params.yaml`` so two runs of the same commit produce identical metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from fruit_classification import config, prepare
from fruit_classification.modeling.models import build_model


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
    except (subprocess.SubprocessError, FileNotFoundError):
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

    Returns ``[]`` when the index is unavailable so a run never fails purely
    because the raw archive is missing.
    """
    index = prepare.INTERIM_SUBDIR / f"{cache_tag}_index.csv"
    if not index.is_file():
        return []
    import pandas as pd

    names = pd.read_csv(index)["label"].drop_duplicates().sort_values().tolist()
    return [str(n) for n in names]


def train(params: dict | None = None, progress: bool = True) -> dict:
    params = config.params_with_defaults(params or config.load_params())
    seed = params["seed"]
    split_cfg = params["split"]
    runtime = params["runtime"]
    train_cfg = dict(params["train"])

    cache_tag = params.get("cache_tag", "full")
    X, y, X_test, y_test = prepare.load(cache_tag)

    X_tr, X_val, y_tr, y_val = holdout_split(
        X,
        y,
        test_size=split_cfg.get("test_size", 0.2),
        seed=seed,
        stratify=split_cfg.get("stratify", True),
    )
    if train_cfg.get("scale_features", False):
        # Fitted on the training split only - never on validation or test data.
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    train_cfg.setdefault("n_jobs", runtime.get("n_jobs", -1))
    model = build_model(train_cfg, seed=seed)

    started = time.time()
    if progress:
        print(
            f"[train] fitting {train_cfg.get('model', 'sgd_svm')} on "
            f"{X_tr.shape[0]} rows x {X_tr.shape[1]} features ..."
        )
    with warnings.catch_warnings():
        # SGD-style estimators warn when max_iter is hit. That is a tuning
        # signal, not a pipeline failure, so record it instead of printing it.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X_tr, y_tr)
    fit_seconds = time.time() - started

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.model_path(cache_tag)

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
        "model_path": str(model_path.relative_to(config.PROJECT_ROOT)),
    }
    # The bundle keeps the run metadata next to the estimator so `evaluate` can
    # report provenance without re-running the fit.
    joblib.dump({"model": model, "run_info": run_info}, model_path, compress=3)
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

    params = config.params_with_defaults(config.load_params())
    if args.model:
        params["train"]["model"] = args.model

    result = train(params)
    print(json.dumps(result["run_info"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()