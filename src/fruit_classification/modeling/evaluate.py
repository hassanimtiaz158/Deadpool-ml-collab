"""Stage 3: evaluate the trained model and write reports/metrics.json.

Reports top-1 and top-5 accuracy (Fruits-360 is a 266-way problem where top-5 is
the realistic operating point), macro-averaged precision/recall/F1, per-class
metrics and the commit SHA of the code that produced the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from fruit_classification import config, dataset
from fruit_classification.compare import score
from fruit_classification.modeling.train import train


def evaluate(result: dict, params: dict, progress: bool = True) -> dict:
    model = result["model"]
    run_info = result["run_info"]
    classes = dataset.class_labels(
        dataset.build_index(splits=(dataset.SPLIT_TRAIN, dataset.SPLIT_TEST))
    )
    classes = list(run_info.get("classes") or classes)

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
    labels = sorted(np.unique(np.concatenate([result["y_test"], y_pred])))
    report = classification_report(
        result["y_test"],
        y_pred,
        labels=labels,
        target_names=[classes[i] for i in labels] if len(classes) >= max(labels) + 1 else None,
        output_dict=True,
        zero_division=0,
    )
    per_class = (
        pd.DataFrame(report)
        .rename_axis("class")
        .sort_values("f1-score")
        .reset_index()
    )
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(config.REPORTS_DIR / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(result["y_test"], y_pred, labels=labels)
    np.save(config.FIGURES_DIR / "confusion_matrix.npy", cm)
    with (config.FIGURES_DIR / "confusion_matrix_labels.json").open("w", encoding="utf-8") as h:
        json.dump([classes[i] for i in labels] if len(classes) >= max(labels) + 1 else labels, h)

    metrics["provenance"] = {
        "commit_sha": run_info["commit_sha"],
        "seed": params["seed"],
        "cache_tag": run_info["cache_tag"],
        "model": run_info["model"],
        "train_cfg": run_info["train_cfg"],
        "feature_cfg": params["features"],
        "split_cfg": params["split"],
        "data_md5": _data_hash(),
        "fit_seconds": run_info["fit_seconds"],
    }
    metrics["params"] = params

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = config.REPORTS_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if progress:
        print(f"[evaluate] wrote {metrics_path.relative_to(config.PROJECT_ROOT)}")
    return metrics


def _data_hash() -> str:
    """md5 recorded in the ``.dvc`` pointer for the raw dataset."""
    pointer = config.RAW_DIR / "fruits-360_100x100.zip.dvc"
    if not pointer.is_file():
        return "unknown"
    for line in pointer.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("md5:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fruit classifier")
    parser.add_argument("--model", help="override train.model from params.yaml")
    args = parser.parse_args()

    params = config.params_with_defaults(config.load_params())
    if args.model:
        params["train"]["model"] = args.model

    result = train(params)
    metrics = evaluate(result, params)
    print(json.dumps({k: metrics[k] for k in ("val", "test")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()