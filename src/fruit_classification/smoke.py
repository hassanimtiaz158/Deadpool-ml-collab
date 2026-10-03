"""End-to-end smoke run: prepare -> train -> evaluate on a small sample.

This is what CI executes to prove the pipeline runs on a teammate's machine
without downloading the full 847 MB dataset. It uses ``data/raw/sample.zip``.

    python -m fruit_classification.smoke
"""

from __future__ import annotations

import argparse
import json
import os

from fruit_classification import config, prepare
from fruit_classification.modeling.evaluate import evaluate
from fruit_classification.modeling.train import train

SMOKE_DEFAULTS = {
    "cache_tag": "smoke",
    "runtime": {
        "n_jobs": -1,
        "max_images_per_class": 12,
        "classes_limit": 8,
    },
    "train": {"model": "sgd_svm", "max_iter": 10},
}


def smoke_params(overrides: dict | None = None) -> dict:
    """Merge the smoke overrides into the repo params."""
    params = config.params_with_defaults(config.load_params())
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key] = {**params[key], **value}
        else:
            params[key] = value
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-run the fruit pipeline")
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--zip", default=None, help="dataset archive to use")
    args = parser.parse_args()

    zip_path = args.zip or (config.SAMPLE_ZIP if config.SAMPLE_ZIP.is_file() else None)
    if zip_path:
        os.environ[config.DATASET_ZIP_ENV] = str(zip_path)

    params = smoke_params(
        {
            "cache_tag": "smoke",
            "runtime": {
                "n_jobs": -1,
                "max_images_per_class": args.per_class,
                "classes_limit": args.classes,
            },
        }
    )

    print(f"[smoke] dataset archive: {config.dataset_zip(params)}")
    prepare.build(params, force=True)

    result = train(params, progress=False)
    metrics = evaluate(result, params, progress=False)

    smoke = {
        "n_train": metrics["test"]["n_samples"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_macro_f1": metrics["test"]["macro_f1"],
        "model": metrics["provenance"]["model"],
        "commit_sha": metrics["provenance"]["commit_sha"],
    }
    out = config.REPORTS_DIR / "smoke_metrics.json"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(smoke, handle, indent=2, sort_keys=True)
    print(json.dumps(smoke, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()