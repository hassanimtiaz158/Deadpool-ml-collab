"""Dataset indexing and feature extraction from the DVC-tracked ZIP archive.

The raw Fruits-360 download is a single ``.zip`` tracked by DVC. Reading 189k
JPEG members directly from the archive avoids materialising ~189k files on disk
while still letting ``dvc pull`` be the only way data enters the repo.
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .features import extract_features

SPLIT_TRAIN = "Training"
SPLIT_TEST = "Test"
VALID_SPLITS = (SPLIT_TRAIN, SPLIT_TEST)

INDEX_COLUMNS = ("split", "label", "label_id", "member")


def parse_member(member: str) -> tuple[str, str, str] | None:
    """Return ``(split, label, member)`` for a ZIP member, or ``None``.

    Expected layout inside the archive::

        fruits-360/Training/<class>/<file>.jpg
        fruits-360/Test/<class>/<file>.jpg
    """
    if not member.lower().endswith(".jpg"):
        return None
    parts = member.strip("/").split("/")
    if len(parts) < 4:
        return None
    split, label = parts[-3], parts[-2]
    if split not in VALID_SPLITS:
        return None
    return split, label, member


def build_index(zip_path: str | Path | None = None, splits: Sequence[str] = VALID_SPLITS) -> pd.DataFrame:
    """Build the full image index DataFrame with stable integer label ids.

    Label ids are sorted by class name so they do not depend on ZIP ordering.
    """
    zip_path = Path(zip_path) if zip_path else config.dataset_zip()
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"raw dataset not found at {zip_path}. Run `dvc pull` to fetch it."
        )

    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            parsed = parse_member(member)
            if parsed is None:
                continue
            split, label, member = parsed
            if split in splits:
                records.append((split, label, member))

    index = pd.DataFrame(records, columns=["split", "label", "member"])
    if index.empty:
        raise ValueError(f"no images matched splits {tuple(splits)} inside {zip_path}")

    classes = sorted(index["label"].unique())
    label_ids = {label: i for i, label in enumerate(classes)}
    index["label_id"] = index["label"].map(label_ids).astype(np.int32)

    missing = set(classes) - set(index.loc[index["split"] == SPLIT_TEST, "label"].unique())
    index.attrs["classes"] = classes
    index.attrs["classes_missing_from_test"] = sorted(missing)
    return index


def class_labels(index: pd.DataFrame) -> list[str]:
    """Ordered class names matching the ``label_id`` column of ``index``."""
    return list(index.attrs.get("classes") or sorted(index["label"].unique()))


def select_subset(
    index: pd.DataFrame,
    split: str,
    max_per_class: int | None = None,
    classes_limit: int | None = None,
) -> pd.DataFrame:
    """Deterministically subsample one split for smoke runs and experiments."""
    frame = index[index["split"] == split].sort_values("member").reset_index(drop=True)
    if classes_limit:
        keep = frame["label_id"].sort_values().unique()[:classes_limit]
        frame = frame[frame["label_id"].isin(keep)]
    if max_per_class:
        frame = frame.groupby("label_id", group_keys=False).head(max_per_class)
    return frame.sort_values("member").reset_index(drop=True)


_WORKER_ZIP: Path | None = None
_WORKER_OPTS: dict | None = None
_WORKER_ARCHIVE: zipfile.ZipFile | None = None


def _init_worker(zip_path: str, opts: dict) -> None:  # pragma: no cover - process setup
    global _WORKER_ZIP, _WORKER_OPTS, _WORKER_ARCHIVE
    _WORKER_ZIP = Path(zip_path)
    _WORKER_OPTS = opts
    _WORKER_ARCHIVE = zipfile.ZipFile(_WORKER_ZIP)


def _close_worker() -> None:  # pragma: no cover - process teardown
    global _WORKER_ARCHIVE
    if _WORKER_ARCHIVE is not None:
        _WORKER_ARCHIVE.close()
        _WORKER_ARCHIVE = None


def _read_batch(members: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read a batch of members and return ``(features, member_ok_mask)``."""
    assert _WORKER_OPTS is not None
    rows: list[np.ndarray] = []
    ok: list[bool] = []
    for member in members:
        try:
            with _WORKER_ARCHIVE.open(member) as handle:
                img = Image.open(handle)
                img.load()
            rows.append(extract_features(img, **_WORKER_OPTS))
            ok.append(True)
        except Exception:  # corrupt member: report via mask, keep going
            ok.append(False)
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.zeros((0,), dtype=bool)
    return np.vstack(rows), np.asarray(ok, dtype=bool)


def _chunks(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def extract_features_for(
    frame: pd.DataFrame,
    zip_path: str | Path | None = None,
    image_size: int = 32,
    cells: int = 4,
    bins: int = 9,
    use_color: bool = True,
    n_jobs: int = -1,
    batch_size: int = 1000,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for every row of ``frame``.

    Returns ``(X, y, kept_index)`` where ``kept_index`` are positional indices
    into ``frame`` for the members that decoded successfully, so callers can
    drop corrupt images without losing label alignment.
    """
    zip_path = Path(zip_path) if zip_path else config.dataset_zip()
    members = frame["member"].tolist()
    opts = {
        "image_size": image_size,
        "cells": cells,
        "bins": bins,
        "use_color": use_color,
    }

    n_jobs = max(1, n_jobs if n_jobs > 0 else 1)
    if n_jobs == -1:
        import os

        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    batches = list(_chunks(members, batch_size))

    if n_jobs == 1:
        _init_worker(str(zip_path), opts)
        results = map(_read_batch, batches)
        pool = None
    else:
        pool = ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(str(zip_path), opts),
        )
        results = pool.map(_read_batch, batches)

    feature_blocks: list[np.ndarray] = []
    kept_index: list[np.ndarray] = []
    try:
        offset = 0
        for batch, (feats, ok) in zip(batches, results):
            if feats.size:
                feature_blocks.append(feats)
                kept_index.append(np.arange(offset, offset + len(batch))[ok])
            offset += len(batch)
            if progress:
                print(f"  extracted {offset}/{len(members)} images", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
        else:
            _close_worker()

    X = (
        np.vstack(feature_blocks)
        if feature_blocks
        else np.empty((0, 0), dtype=np.float32)
    )
    kept = np.concatenate(kept_index) if kept_index else np.zeros(0, dtype=int)
    y = frame["label_id"].to_numpy(dtype=np.int32)[kept]
    return X, y, kept


def iter_preview_images(
    index: pd.DataFrame,
    labels: Iterable[str],
    zip_path: str | Path | None = None,
    per_label: int = 1,
) -> dict[str, list[Image.Image]]:
    """Load a few decoded images per label, for notebooks and sanity checks."""
    zip_path = Path(zip_path) if zip_path else config.dataset_zip()
    wanted = set(labels)
    picked: dict[str, list[Image.Image]] = {label: [] for label in wanted}
    with zipfile.ZipFile(zip_path) as archive:
        for split in (SPLIT_TRAIN, SPLIT_TEST):
            frame = index[index["split"] == split]
            for label, group in frame.groupby("label", sort=False):
                if label not in wanted or len(picked[label]) >= per_label:
                    continue
                for member in group["member"].tolist()[:per_label]:
                    with archive.open(member) as handle:
                        img = Image.open(handle)
                        img.load()
                    picked[label].append(img.convert("RGB"))
    return picked