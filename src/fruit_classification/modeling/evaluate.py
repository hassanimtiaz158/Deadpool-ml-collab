"""Stage 3: evaluate the trained model and write reports/metrics.json.

Reports top-1 and top-5 accuracy (Fruits-360 is a 266-way problem where top-5 is
the realistic operating point), macro-averaged precision/recall/F1, per-class
metrics and the commit SHA of the code that produced the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from fruit_classification import config, dataset, prepare
from fruit_classification.compare import score
from fruit_classification.modeling.train import holdout_split, train


def evaluate(result: dict, params: dict, progress: bool = True) -> dict:
    cache_tag = result["run_info"].get("cache_tag", config.CANONICAL_TAG)
    reports_dir = config.report_dir(cache_tag)
    figures_dir = config.figures_dir(cache_tag)

    # Create the output tree up front - every artefact below lands under it.
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = result["model"]
    run_info = result["run_info"]

    # Prefer the class names the training run already resolved. Only fall back to
    # re-indexing the archive when they are absent, so evaluating a cached model
    # never requires the raw dataset to be present.
    classes = list(run_info.get("classes") or []) or _class_names()

    metrics: dict[str, object] = {}
    for split_name, X, y in (
        ("val", result["X_val"], result["y_val"]),
        ("test", result["X_test"], result["y_test"]),
    ):
        s = score(model, X, y)
        metrics[split_name] = {
            "n_samples": int(len(y)),
            "accuracy": round(s["accuracy"], 6),
            "top5_accuracy": round(s["top5_accuracy"], 6),
            "macro_f1": round(s["macro_f1"], 6),
            "macro_precision": round(s["macro_precision"], 6),
            "macro_recall": round(s["macro_recall"], 6),
        }

    # Per-class table + confusion matrix for the report.
    y_pred = model.predict(result["X_test"])
    labels = sorted(np.unique(np.concatenate([result["y_test"], y_pred])).tolist())
    label_names = (
        [classes[i] for i in labels] if classes and max(labels) < len(classes) else None
    )
    report = classification_report(
        result["y_test"],
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    # classification_report keys classes by column; transpose so each class is a
    # row, and keep only per-class rows (drop the scalar/aggregate rows).
    per_class = pd.DataFrame(report).T.rename_axis("class").reset_index()
    per_class = per_class[per_class["class"].isin(label_names or [])].copy()
    if "f1-score" in per_class:
        per_class = per_class.sort_values("f1-score")
    per_class.to_csv(reports_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(result["y_test"], y_pred, labels=labels)
    np.save(figures_dir / "confusion_matrix.npy", cm)
    with (figures_dir / "confusion_matrix_labels.json").open("w", encoding="utf-8") as h:
        json.dump(label_names or [str(i) for i in labels], h)

    metrics["provenance"] = {
        "commit_sha": run_info.get("commit_sha", "unknown"),
        "seed": params["seed"],
        "cache_tag": run_info.get("cache_tag", "unknown"),
        "model": run_info.get("model", "unknown"),
        "train_cfg": run_info.get("train_cfg", {}),
        "feature_cfg": params["features"],
        "split_cfg": params["split"],
        "data_md5": _data_hash(),
        "fit_seconds": run_info.get("fit_seconds"),
    }
    metrics["params"] = params

    metrics_path = reports_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if progress:
        print(f"[evaluate] wrote {metrics_path.relative_to(config.PROJECT_ROOT)}")
    return metrics


def _class_names() -> list[str]:
    """Sorted class names, or an empty list if the archive is not available."""
    try:
        return list(
            dataset.class_labels(
                dataset.build_index(splits=(dataset.SPLIT_TRAIN, dataset.SPLIT_TEST))
            )
        )
    except (FileNotFoundError, ValueError, KeyError):
        return []


def _data_hash() -> str:
    """md5 recorded in the ``.dvc`` pointer for the raw dataset."""
    pointer = config.RAW_DIR / "fruits-360_100x100.zip.dvc"
    if not pointer.is_file():
        return "unknown"
    for line in pointer.read_text(encoding="utf-8").splitlines():
        # DVC writes the hash as a YAML list item, so the line starts with
        # "- md5: <hash>" - not "md5:". Strip the dash before matching.
        entry = line.strip().removeprefix("-").strip()
        if entry.startswith("md5:"):
            return entry.split(":", 1)[1].strip()
    return "unknown"


def load_trained(params: dict) -> dict:
    """Rebuild the evaluation inputs from the artefacts ``train`` already wrote.

    Re-deriving the validation split is unavoidable and cheap; refitting the
    estimator is not. Loading ``models/model.joblib`` keeps ``dvc repro`` honest
    about which stage produced what, and halves the wall time of ``evaluate``.
    """
    model_path = config.model_path(params.get("cache_tag", config.CANONICAL_TAG))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_path}. Run `dvc repro train` first."
        )

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        model, run_info = bundle["model"], bundle.get("run_info", {})
    else:  # a bare estimator, from an older artefact
        model, run_info = bundle, {}

    X, y, X_test, y_test = prepare.load(params.get("cache_tag", "full"))
    # train_test_split returns INTERLEAVED - (X_tr, X_val, y_tr, y_val) - so the
    # two label arrays are dropped by name, not sliced off the end. Slicing [2:]
    # silently yields (y_tr, y_test) and hands the labels to the estimator as if
    # they were features.
    _, X_val, _, y_val = holdout_split(
        X,
        y,
        test_size=params["split"].get("test_size", 0.2),
        seed=params["seed"],
        stratify=params["split"].get("stratify", True),
    )

    run_info = dict(run_info)
    run_info.setdefault("model", params["train"].get("model", "unknown"))
    run_info.setdefault("train_cfg", params["train"])
    run_info.setdefault("cache_tag", params.get("cache_tag", "full"))
    run_info.setdefault("classes", _class_names())

    return {
        "model": model,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "run_info": run_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fruit classifier")
    parser.add_argument("--model", help="override train.model from params.yaml")
    parser.add_argument(
        "--refit",
        action="store_true",
        help="retrain instead of loading models/model.joblib (debugging only)",
    )
    args = parser.parse_args()

    params = config.params_with_defaults(config.load_params())
    if args.model:
        params["train"]["model"] = args.model

    result = train(params) if args.refit else load_trained(params)
    metrics = evaluate(result, params)
    print(json.dumps({k: metrics[k] for k in ("val", "test")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()