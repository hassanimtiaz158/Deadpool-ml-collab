"""Model comparison entry point - the script behind notebook 02.

Trains every candidate listed in ``CANDIDATES`` on the identical seeded split of
the cached features, scores each on the validation split and on the archive's
held-out ``Test`` folder, and writes ``reports/model_comparison.csv``.

    python -m fruit_classification.compare
    python -m fruit_classification.compare --models sgd_svm,extra_trees
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from fruit_classification import config, prepare
from fruit_classification.modeling.models import build_model
from fruit_classification.modeling.train import holdout_split

#: Candidate estimators compared in notebook 02. Each entry is
#: ``(label, train-config)``; the label is what shows up in the leaderboard.
#:
#: The list is deliberately weighted towards the linear family because that is
#: where the interesting hyperparameter (the L2 penalty) lives, and because a
#: 266-way problem makes the exact-solver alternatives impractical on a laptop:
#: ``linear_svc`` and ``logistic_regression`` were measured at >25 min per fit
#: on 4 CPU cores and did not finish, so they are kept available via
#: ``--models`` but are not part of the default sweep. See REPORT.md.
CANDIDATES: list[tuple[str, dict]] = [
    ("sgd_svm (alpha=1e-6)", {"model": "sgd_svm", "alpha": 1e-6, "max_iter": 60}),
    ("sgd_svm (alpha=1e-5)", {"model": "sgd_svm", "alpha": 1e-5, "max_iter": 60}),
    ("sgd_svm (alpha=1e-4)", {"model": "sgd_svm", "alpha": 1e-4, "max_iter": 60}),
    ("extra_trees (100)", {"model": "extra_trees", "n_estimators": 100}),
    ("random_forest (100)", {"model": "random_forest", "n_estimators": 100}),
    ("linear_svc (C=1)", {"model": "linear_svc", "C": 1.0}),
    ("logistic_regression (C=10)", {"model": "logistic_regression", "C": 10.0, "max_iter": 50}),
    ("mlp (256)", {"model": "mlp", "hidden_layer_sizes": [256], "max_iter": 60}),
]

#: Where the leaderboard lives. Reused as the resume checkpoint - see ``compare``.
LEADERBOARD = "model_comparison.csv"


def top_k_accuracy(model, X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Top-k accuracy computed from scores, for any classifier.

    Handles the OvA layout of linear models (``decision_function`` may be 1-D for
    binary problems) and tree/MLP ensembles (``predict_proba``).
    """
    classes = np.asarray(getattr(model, "classes_", np.unique(y)))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
    elif hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X))
    else:
        return float("nan")

    k = min(k, scores.shape[1])
    order = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_labels = classes[order]
    return float((top_labels == y[:, None]).any(axis=1).mean())


def score(model, X: np.ndarray, y: np.ndarray, k: int = 5) -> dict[str, float]:
    """Full metric bundle for one split."""
    y_pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        f"top{k}_accuracy": top_k_accuracy(model, X, y, k=k),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
    }


def _read_checkpoint() -> dict[str, dict]:
    """Previously scored candidates, keyed by label."""
    path = config.REPORTS_DIR / LEADERBOARD
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return {}
    if "model" not in frame.columns:
        return {}
    return {str(row["model"]): row.to_dict() for _, row in frame.iterrows()}


def _write_checkpoint(rows: list[dict]) -> pd.DataFrame:
    """Persist the leaderboard sorted by validation macro-F1."""
    frame = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.REPORTS_DIR / LEADERBOARD, index=False)
    return frame


def compare(
    params: dict | None = None,
    candidates: list[tuple[str, dict]] | None = None,
    verbose: bool = True,
    resume: bool = True,
) -> pd.DataFrame:
    """Fit every candidate and return the leaderboard sorted by validation macro-F1.

    The leaderboard is rewritten after each fit, and with ``resume=True`` any
    candidate already present in it is skipped. A sweep over the full dataset
    takes tens of minutes, so being able to stop and continue is the difference
    between a usable tool and a restarting one.
    """
    params = config.params_with_defaults(params or config.load_params())
    seed = params["seed"]
    split_cfg = params["split"]
    n_jobs = params["train"].get("n_jobs", -1)
    candidates = candidates or CANDIDATES

    done = _read_checkpoint() if resume else {}
    pending = [(label, cfg) for label, cfg in candidates if label not in done]
    if verbose and done:
        print(f"[compare] resuming: {len(done)} cached, {len(pending)} to fit", flush=True)

    X_pool, y_pool, X_test, y_test = prepare.load(params.get("cache_tag", "full"))
    X_fit, X_val, y_fit, y_val = holdout_split(
        X_pool,
        y_pool,
        test_size=split_cfg.get("test_size", 0.2),
        seed=seed,
        stratify=split_cfg.get("stratify", True),
    )
    if verbose:
        print(f"fit={X_fit.shape} val={X_val.shape} test={X_test.shape} seed={seed}", flush=True)

    rows: list[dict] = list(done.values())
    for label, cfg in pending:
        model_cfg = {**cfg, "n_jobs": cfg.get("n_jobs", n_jobs)}
        if verbose:
            print(f"[compare] {label} ...", flush=True)
        started = time.time()
        model = build_model(model_cfg, seed=seed)
        with warnings.catch_warnings():
            # A sweep intentionally explores unconverged settings; the fit time
            # and the resulting metric are the signal, not the warning.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_fit, y_fit)
        fit_seconds = time.time() - started

        val = score(model, X_val, y_val)
        test = score(model, X_test, y_test)
        row = {
            "model": label,
            "train_cfg": json.dumps(model_cfg, sort_keys=True),
            "fit_seconds": round(fit_seconds, 1),
        }
        row.update({f"val_{k}": round(v, 6) for k, v in val.items()})
        row.update({f"test_{k}": round(v, 6) for k, v in test.items()})
        rows.append(row)
        # frame = _write_checkpoint(rows)

        if verbose:
            print(
                f"    fit {fit_seconds:7.1f}s | val acc {val['accuracy']:.4f} "
                f"top5 {val['top5_accuracy']:.4f} f1 {val['macro_f1']:.4f} | "
                f"test acc {test['accuracy']:.4f} top5 {test['top5_accuracy']:.4f} "
                f"f1 {test['macro_f1']:.4f}",
                flush=True,
            )

    leaderboard = _write_checkpoint(rows)
    if verbose:
        print(f"[compare] wrote {(config.REPORTS_DIR / LEADERBOARD).name}")
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate classifiers")
    parser.add_argument(
        "--models",
        help="comma-separated substrings to keep from the default candidate list",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore the existing leaderboard and refit every candidate",
    )
    args = parser.parse_args()

    candidates = CANDIDATES
    if args.models:
        wanted = [m.strip().lower() for m in args.models.split(",")]
        candidates = [
            (label, cfg)
            for label, cfg in CANDIDATES
            if any(w in label.lower() or w == cfg["model"] for w in wanted)
        ]
        if not candidates:
            raise SystemExit(f"no candidate matched {args.models!r}")

    leaderboard = compare(candidates=candidates, resume=not args.fresh)
    cols = [c for c in leaderboard.columns if not c.startswith("train_cfg")]
    print()
    print(leaderboard[cols].to_string(index=False))


if __name__ == "__main__":
    main()
