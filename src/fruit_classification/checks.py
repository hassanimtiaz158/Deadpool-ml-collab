"""Dataset data checks - schema, value ranges and null counts.

Run locally or in CI:

    python -m fruit_classification.checks

Checks the committed sample index by default so CI needs no full download, and
additionally validates the full index when it is already present locally.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fruit_classification import config, dataset

REQUIRED_COLUMNS = {
    "split": "category",
    "label": "string",
    "member": "string",
    "label_id": "Int64",
}

VALID_SPLITS = set(dataset.VALID_SPLITS)

SAMPLE_INDEX = config.RAW_DIR / "sample_index.csv"
FULL_INDEX = config.INTERIM_DIR / "full_index.csv"


class CheckFailure(Exception):
    """Raised when a dataset invariant is violated."""


def check_schema(frame: pd.DataFrame, name: str) -> None:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise CheckFailure(f"{name}: missing columns {sorted(missing)}")


def check_nulls(frame: pd.DataFrame, name: str) -> dict[str, int]:
    nulls = {col: int(frame[col].isna().sum()) for col in frame.columns}
    offenders = {col: n for col, n in nulls.items() if n}
    if offenders:
        raise CheckFailure(f"{name}: null values found {offenders}")
    return nulls


def check_ranges(frame: pd.DataFrame, name: str) -> None:
    bad_splits = set(frame["split"].unique()) - VALID_SPLITS
    if bad_splits:
        raise CheckFailure(f"{name}: unexpected split values {sorted(bad_splits)}")

    ids = frame["label_id"].to_numpy()
    if ids.min() < 0:
        raise CheckFailure(f"{name}: negative label_id (min {ids.min()})")
    if ids.max() != frame["label_id"].nunique() - 1:
        raise CheckFailure(
            f"{name}: label_id is not a dense 0..K-1 range "
            f"(max {ids.max()}, {frame['label_id'].nunique()} classes)"
        )

    if not frame["member"].str.endswith(".jpg").all():
        raise CheckFailure(f"{name}: some members are not .jpg files")
    if frame["member"].duplicated().any():
        dupes = frame.loc[frame["member"].duplicated(), "member"].head(5).tolist()
        raise CheckFailure(f"{name}: duplicate members, e.g. {dupes}")
    if (frame["label"].str.strip() == "").any():
        raise CheckFailure(f"{name}: blank class names present")


def check_splits(frame: pd.DataFrame, name: str) -> dict[str, int]:
    counts = frame["split"].value_counts().to_dict()
    if not counts:
        raise CheckFailure(f"{name}: index is empty")
    missing = VALID_SPLITS - set(counts)
    if missing:
        raise CheckFailure(f"{name}: no rows for split(s) {sorted(missing)}")
    train_classes = set(frame.loc[frame["split"] == dataset.SPLIT_TRAIN, "label"])
    test_classes = set(frame.loc[frame["split"] == dataset.SPLIT_TEST, "label"])
    only_train = sorted(train_classes - test_classes)
    if only_train:
        print(
            f"  [warn] {name}: {len(only_train)} class(es) present in Training but not "
            f"Test: {only_train[:5]}"
        )
    return {k: int(v) for k, v in counts.items()}


def check_file(path: Path) -> dict:
    if not path.is_file():
        raise CheckFailure(f"{path} is missing - run `dvc pull` or `make data-index`")
    frame = pd.read_csv(path, dtype={"label": str, "member": str})
    name = path.name
    check_schema(frame, name)
    nulls = check_nulls(frame, name)
    check_ranges(frame, name)
    counts = check_splits(frame, name)
    print(f"  [ok] {name}: {len(frame)} rows, {frame['label_id'].nunique()} classes, {counts}")
    return {"file": name, "rows": int(len(frame)), "nulls": nulls, "splits": counts}


def main() -> int:
    targets = [p for p in (SAMPLE_INDEX, FULL_INDEX) if p.is_file()]
    if not targets:
        raise CheckFailure(
            "no committed dataset index found; expected data/raw/sample_index.csv "
            "or data/interim/full_index.csv"
        )

    print("[checks] dataset invariants")
    results = [check_file(path) for path in targets]

    summary = {"status": "ok", "checked": results}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.REPORTS_DIR / "data_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"[checks] PASS ({len(results)} file(s))")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailure as exc:
        print(f"[checks] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)