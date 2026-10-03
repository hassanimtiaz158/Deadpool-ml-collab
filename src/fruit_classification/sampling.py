"""Build the small dataset sample that CI runs on.

The full archive is 847 MB, which is far too slow (and too large) to pull on
every pull request. This module carves out a stratified few-hundred-image sample
that is DVC-tracked separately, so the `data-checks` and `smoke-train` CI jobs
prove the pipeline runs end to end without the full download.

Usage
-----
    python -m fruit_classification.sampling --classes 10 --train 12 --test 4
    dvc add data/raw/sample.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from fruit_classification import config, dataset

DEFAULT_CLASSES = 10
DEFAULT_TRAIN = 12
DEFAULT_TEST = 4


def build_sample(
    out_path: Path | None = None,
    classes: int = DEFAULT_CLASSES,
    per_class_train: int = DEFAULT_TRAIN,
    per_class_test: int = DEFAULT_TEST,
    index_out: Path | None = None,
    zip_path: Path | None = None,
) -> tuple[Path, Path, int]:
    """Write a stratified sample archive plus its committed index CSV."""
    out_path = Path(out_path) if out_path else config.SAMPLE_ZIP
    index_out = Path(index_out) if index_out else config.RAW_DIR / "sample_index.csv"
    zip_path = Path(zip_path) if zip_path else config.dataset_zip()

    index = dataset.build_index(zip_path)
    labels = sorted(index["label"].unique())[:classes]

    picked: list[pd.DataFrame] = []
    for split, per_class in (
        (dataset.SPLIT_TRAIN, per_class_train),
        (dataset.SPLIT_TEST, per_class_test),
    ):
        frame = index[(index["split"] == split) & (index["label"].isin(labels))]
        picked.append(frame.groupby("label", group_keys=False).head(per_class))
    sample = pd.concat(picked, ignore_index=True).sort_values(["split", "label", "member"])
    sample = sample.reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as source, zipfile.ZipFile(
        out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as target:
        for member in sample["member"]:
            target.writestr(member, source.read(member))

    index_out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(index_out, index=False)

    size_kb = out_path.stat().st_size / 1024
    return out_path, index_out, int(size_kb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CI dataset sample")
    parser.add_argument("--classes", type=int, default=DEFAULT_CLASSES)
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST)
    args = parser.parse_args()

    archive, index_csv, size_kb = build_sample(
        classes=args.classes,
        per_class_train=args.train,
        per_class_test=args.test,
    )
    print(f"wrote {archive.relative_to(config.PROJECT_ROOT)} ({size_kb:,.0f} KB)")
    print(f"wrote {index_csv.relative_to(config.PROJECT_ROOT)}")
    if size_kb > 1024:
        print(
            "[warn] sample exceeds the 1 MB pre-commit limit - "
            "reduce --classes/--train/--test or keep it DVC-only"
        )


if __name__ == "__main__":
    main()