"""Stage 1: build the feature matrices used by every later stage.

Reads the DVC-tracked ZIP, indexes it, splits train/test and caches the
extracted features under ``data/interim/`` so that repeated experiments and
``dvc repro`` do not re-decode 189k JPEGs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fruit_classification import config, dataset

INTERIM_SUBDIR = config.INTERIM_DIR


def cache_paths(tag: str) -> tuple[Path, Path, Path]:
    """Return ``(features.npz, index.csv, manifest.json)`` for a cache tag."""
    return (
        INTERIM_SUBDIR / f"{tag}_features.npz",
        INTERIM_SUBDIR / f"{tag}_index.csv",
        INTERIM_SUBDIR / f"{tag}_manifest.json",
    )


def build(params: dict, force: bool = False, progress: bool = True, zip_path=None) -> dict:
    params = config.params_with_defaults(params)
    seed = params["seed"]
    features_cfg = params["features"]
    runtime = params["runtime"]
    tag = params.get("cache_tag", "full")
    zip_path = zip_path or config.dataset_zip(params)

    npz_path, index_path, manifest_path = cache_paths(tag)
    if npz_path.is_file() and index_path.is_file() and not force:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if progress:
            print(f"[prepare] cache hit: {npz_path.relative_to(config.PROJECT_ROOT)}")
        return manifest

    started = time.time()
    INTERIM_SUBDIR.mkdir(parents=True, exist_ok=True)

    if progress:
        print("[prepare] indexing dataset archive ...")
    index = dataset.build_index(zip_path)
    missing = index.attrs.get("classes_missing_from_test", [])

    train_requested = dataset.select_subset(
        index,
        dataset.SPLIT_TRAIN,
        max_per_class=runtime.get("max_images_per_class"),
        classes_limit=runtime.get("classes_limit"),
    )
    test_requested = dataset.select_subset(
        index,
        dataset.SPLIT_TEST,
        max_per_class=runtime.get("max_images_per_class"),
        classes_limit=runtime.get("classes_limit"),
    )

    shared = dict(
        image_size=features_cfg["image_size"],
        cells=features_cfg["hog_cells"],
        bins=features_cfg["hog_bins"],
        use_color=features_cfg.get("color_hist", True),
        n_jobs=runtime.get("n_jobs", -1),
        progress=progress,
    )

    if progress:
        print(f"[prepare] extracting train features for {len(train_requested)} images ...")
    X_train, y_train, kept_train = dataset.extract_features_for(
        train_requested, zip_path=zip_path, **shared
    )
    if progress:
        print(f"[prepare] extracting test features for {len(test_requested)} images ...")
    X_test, y_test, kept_test = dataset.extract_features_for(
        test_requested, zip_path=zip_path, **shared
    )

    undecodable = (len(train_requested) - len(kept_train)) + (len(test_requested) - len(kept_test))
    train_frame = train_requested.iloc[kept_train].reset_index(drop=True)
    test_frame = test_requested.iloc[kept_test].reset_index(drop=True)

    np.savez_compressed(
        npz_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    index.to_csv(index_path, index=False)

    manifest = {
        "cache_tag": tag,
        "seed": seed,
        "n_classes": len(dataset.class_labels(index)),
        "n_images_total": int(len(index)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_undecodable": int(undecodable),
        "feature_dim": int(X_train.shape[1]),
        "feature_config": features_cfg,
        "dataset_zip": str(zip_path),
        "classes_missing_from_test": missing,
        "elapsed_sec": round(time.time() - started, 2),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    if progress:
        print(f"[prepare] wrote {npz_path.name} ({X_train.shape} train / {X_test.shape} test)")
    return manifest


def load(tag: str = "full") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path, _, _ = cache_paths(tag)
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"{npz_path} not found. Run `dvc repro prepare` (or `python -m fruit_classification.dataset --build`)."
        )
    with np.load(npz_path) as data:
        return (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached image features")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    params = config.params_with_defaults(config.load_params())
    manifest = build(params, force=args.force, progress=not args.quiet)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
