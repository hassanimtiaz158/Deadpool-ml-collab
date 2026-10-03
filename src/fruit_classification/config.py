"""Project paths and configuration loading.

All paths are resolved relative to the repository root so that the pipeline runs
from any working directory and on any teammate's machine (no hardcoded paths).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# src/fruit_classification/config.py -> src/fruit_classification -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DATASET_ZIP = RAW_DIR / "fruits-360_100x100.zip"
SAMPLE_ZIP = RAW_DIR / "sample.zip"

PARAMS_FILES = (PROJECT_ROOT / "params.yaml", PROJECT_ROOT / "configs" / "params.yaml")

DEFAULT_SEED = 42

# Environment override, used by CI's smoke job to point at data/raw/sample.zip
# without editing params.yaml.
DATASET_ZIP_ENV = "FRUIT_DATASET_ZIP"


def dataset_zip(params: dict[str, Any] | None = None) -> Path:
    """Resolve the dataset archive from params, then env, then the default."""
    override = os.environ.get(DATASET_ZIP_ENV)
    if override:
        return Path(override).resolve()
    if params and params.get("data", {}).get("zip"):
        candidate = Path(params["data"]["zip"])
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return DATASET_ZIP


# The canonical tag owns the top-level artefact paths; every other tag (notably
# "smoke") is namespaced underneath. Without this a CI smoke run overwrites the
# real model and metrics with results from 112 images.
CANONICAL_TAG = "full"


def model_path(cache_tag: str = CANONICAL_TAG) -> Path:
    """Where the trained estimator for ``cache_tag`` is serialised."""
    name = "model" if cache_tag == CANONICAL_TAG else f"{cache_tag}_model"
    return MODELS_DIR / f"{name}.joblib"


def report_dir(cache_tag: str = CANONICAL_TAG) -> Path:
    """Directory holding metrics.json and the per-class table."""
    return REPORTS_DIR if cache_tag == CANONICAL_TAG else REPORTS_DIR / cache_tag


def figures_dir(cache_tag: str = CANONICAL_TAG) -> Path:
    """Directory holding the confusion matrix and other figures."""
    return FIGURES_DIR if cache_tag == CANONICAL_TAG else FIGURES_DIR / cache_tag


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline parameters from ``params.yaml``.

    Falls back to an empty dict when no params file exists so that library use
    (tests, notebooks) never depends on the repo layout.
    """
    candidates = (Path(path),) if path else PARAMS_FILES
    for candidate in candidates:
        if candidate and candidate.is_file():
            with candidate.open(encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def params_with_defaults(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge user params over the project defaults."""
    resolved: dict[str, Any] = {
        "seed": DEFAULT_SEED,
        "split": {"test_size": 0.2, "stratify": True},
        "features": {
            "image_size": 32,
            "hog_cells": 4,
            "hog_bins": 9,
            "color_hist": True,
        },
        "train": {
            "model": "sgd_svm",
            "alpha": 1.0e-5,
            "max_iter": 25,
            "tol": 1.0e-4,
            "C": 1.0,
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_leaf": 1,
            "hidden_layer_sizes": [256],
            "early_stopping": True,
            "n_iter_no_change": 5,
            "scale_features": False,
            "n_jobs": -1,
        },
        "runtime": {"n_jobs": -1, "max_images_per_class": None},
    }
    for key, value in (params or {}).items():
        if isinstance(value, dict) and isinstance(resolved.get(key), dict):
            resolved[key] = {**resolved[key], **value}
        else:
            resolved[key] = value
    return resolved