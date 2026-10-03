# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Model selection: which classifier should we ship?
#
# **Question.** Fruits-360 at 100×100 is a **266-way** classification problem with
# a long-tailed label distribution. We need a model that is (a) accurate enough to
# be useful, (b) trainable on a laptop CPU in minutes, not hours, and
# (c) reproducible from `params.yaml` alone.
#
# **Candidates compared below**, all on identical cached features and an
# identical seeded split:
#
# | Model | Why it is a candidate |
# | --- | --- |
# | `sgd_svm` (linear SVM, one-vs-rest) | The classic strong baseline for HOG features. Fast, scales to 113k×272. |
# | `logistic_regression` | Probabilistic alternative to a linear SVM; useful when you need calibrated scores. |
# | `linear_svc` | Same hypothesis space, exact solver — the accuracy/speed reference point. |
# | `extra_trees` / `random_forest` | Axis-aligned trees on HOG cells; often competitive and quick to fit. |
# | `mlp` | One hidden layer — a cheap non-linear model that can carve up the HOG space. |
#
# **Selection rule (fixed before looking at any number, per the review
# checklist):** highest **validation macro-F1**, with top-5 accuracy as the
# tie-breaker. Macro-F1 weights every fruit class equally, and this dataset's
# rare classes are exactly where a useful model should be judged.
#
# **Reproduce.**
# ```bash
# dvc pull                            # dataset
# dvc repro                           # prepare -> train -> evaluate
# python -m fruit_classification.compare
# dvc exp run --set-param train.model=linear_svc train.C=0.5
# dvc exp show
# ```

# %% [markdown]
## 1. Setup

# %%

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from fruit_classification import config, dataset, plots, prepare  # noqa: E402
from fruit_classification.compare import compare, score, top_k_accuracy  # noqa: E402
from fruit_classification.modeling.models import build_model  # noqa: E402
from fruit_classification.modeling.train import holdout_split  # noqa: E402

plots.apply_style()
pd.set_option("display.width", 170)

params = config.params_with_defaults(config.load_params())
SEED = params["seed"]
print(f"seed from params.yaml : {SEED}")
print(f"split config          : {params['split']}")
print(f"feature config        : {params['features']}")
print(f"train config          : {params['train']}")

# %% [markdown]
## 2. Load the cached features
#
# `prepare` decodes all 189k JPEGs **once** and writes
# `data/interim/full_features.npz`. Everything below reuses that cache, so a
# model sweep costs only the fit time.

# %%

X_pool, y_pool, X_test, y_test = prepare.load(params["cache_tag"])
index = dataset.build_index(config.dataset_zip(params))
class_labels = dataset.class_labels(index)

print(f"train pool : {X_pool.shape}")
print(f"test set   : {X_test.shape}")
print(f"classes    : {len(class_labels)}")
print(f"feature dtype : {X_pool.dtype} ({X_pool.nbytes / 1e6:.1f} MB in memory)")

X_fit, X_val, y_fit, y_val = holdout_split(
    X_pool,
    y_pool,
    test_size=params["split"]["test_size"],
    seed=SEED,
    stratify=params["split"]["stratify"],
)
print(f"\nafter seeded split : fit={X_fit.shape}  val={X_val.shape}")

# %%

# Class balance in the fit split — this is why macro metrics matter here.
train_dist = pd.Series(y_fit).value_counts()
print(f"classes in fit split : {train_dist.size}")
print(f"smallest class       : {train_dist.min()} images")
print(f"largest class        : {train_dist.max()} images")
print(f"imbalance ratio      : {train_dist.max() / train_dist.min():.1f}x")
train_dist.sort_values().head(10).rename("images").to_frame()

# %% [markdown]
## 3. The metric
#
# `compare.score()` is the single scoring implementation, imported from the
# package so the notebook, `evaluate` and any CI job can never disagree about
# how a number was produced.

# %%

help(top_k_accuracy)

# %%

# Sanity-check the scorer on a deliberately weak model: it must sit near chance.
rng = np.random.default_rng(0)
chance = np.asarray(class_labels)[rng.integers(0, len(class_labels), size=len(y_test))]

class _Constant:
    """Trivial baseline that always predicts the majority class."""

    classes_ = np.array([int(np.bincount(y_fit).argmax())])

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


weak = score(_Constant(), X_val, y_val)
print(f"majority-class baseline : val acc {weak['accuracy']:.4f}  macro F1 {weak['macro_f1']:.4f}")
print(f"random-guess accuracy   : {np.mean(chance == y_val):.4f}")
print("\nAny candidate must clear these by a wide margin to be worth shipping.")

# %% [markdown]
## 4. Train and score every candidate
#
# Every model is fitted **once** on the training split, then scored on the seeded
# validation split and on the archive's held-out `Test` folder. Selection uses
# validation only; `Test` is read once at the end to report the final number of
# the already-chosen model, so it never influences the choice.

# %%

started = time.time()
leaderboard = compare(params=params)
print(f"\ntotal comparison time: {time.time() - started:.1f}s")

cols = [c for c in leaderboard.columns if c != "train_cfg"]
leaderboard[cols]

# %% [markdown]
## 5. Visual comparison

# %%

long = leaderboard.melt(
    id_vars=["model", "fit_seconds"],
    value_vars=["val_accuracy", "val_top5", "val_macro_f1", "test_macro_f1"],
    var_name="metric",
    value_name="score",
)

fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
sns.barplot(data=long, y="model", x="score", hue="metric", ax=axes[0], palette="Set2")
axes[0].set_title("Validation vs test metrics")
axes[0].set_xlabel("score")
axes[0].set_xlim(0, 1)
axes[0].tick_params(axis="y", labelsize=8)

sns.barplot(data=leaderboard, y="model", x="fit_seconds", ax=axes[1], color="#4C72B0")
axes[1].set_title("Fit time (lower is better)")
axes[1].set_xlabel("seconds")
axes[1].tick_params(axis="y", labelsize=8)
axes[1].set_yticklabels([])
fig.tight_layout()

# %%

# Accuracy vs speed trade-off: the decision frontier.
fig, ax = plt.subplots(figsize=(9, 6))
for _, row in leaderboard.iterrows():
    ax.scatter(row.fit_seconds, row.test_macro_f1, s=120)
    ax.annotate(
        row.model,
        (row.fit_seconds, row.test_macro_f1),
        textcoords="offset points",
        xytext=(6, 4),
        fontsize=8,
    )
ax.set_xscale("log")
ax.set_xlabel("fit time (s, log scale)")
ax.set_ylabel("test macro F1")
ax.set_title("Speed vs quality — the decision frontier")
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
## 6. Accuracy vs top-5: what does the model actually know?
#
# Top-1 on a 266-way problem is a harsh metric. Top-5 says whether the model at
# least narrows the fruit to the right *kind*, which is what a useful assistant
# would surface to a user.

# %%

winner_row = leaderboard.sort_values(["val_macro_f1", "val_top5"], ascending=False).iloc[0]
print(f"winner by validation macro-F1 : {winner_row.model}")
print(f"  test accuracy   : {winner_row.test_accuracy:.4f}")
print(f"  test top-5      : {winner_row.test_top5:.4f}")
print(f"  test macro F1   : {winner_row.test_macro_f1:.4f}")
print(f"  fit time        : {winner_row.fit_seconds:.1f}s")
print(f"  train cfg       : {winner_row.train_cfg}")

# %%

winning_cfg = json.loads(winner_row.train_cfg)
best = build_model({**winning_cfg, "n_jobs": -1}, seed=SEED)
best.fit(X_fit, y_fit)
y_pred = best.predict(X_test)

# Which classes get confused? Restrict to the most frequent classes so the
# figure stays readable.
cm = pd.crosstab(pd.Series(y_test, name="true"), pd.Series(y_pred, name="pred"))
row_totals = cm.sum(axis=1).sort_values(ascending=False)
top_labels = row_totals.head(20).index
cm_top = cm.loc[top_labels, top_labels]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    np.log1p(cm_top.values),
    cmap="mako",
    xticklabels=[class_labels[i] for i in cm_top.columns],
    yticklabels=[class_labels[i] for i in cm_top.index],
    ax=ax,
    cbar_kws={"label": "log(1 + test images)"},
)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.tick_params(axis="both", labelsize=8)
ax.set_title("Confusion matrix — 20 most frequent test classes")
fig.tight_layout()

# %%

# Per-class breakdown for the winner: where are we strong, where are we weak?
n_test = pd.Series(y_test).value_counts()
n_correct = pd.Series(y_test[y_test == y_pred]).value_counts()

per_class = pd.DataFrame(
    {
        "class": [class_labels[i] for i in range(len(class_labels))],
        "n_test": n_test.reindex(range(len(class_labels))).fillna(0).values,
    }
)
per_class["n_correct"] = (
    n_correct.reindex(range(len(class_labels))).fillna(0).values
)
per_class["recall"] = np.where(
    per_class.n_test > 0, per_class.n_correct / per_class.n_test.replace(0, np.nan), np.nan
)
per_class["n_predicted"] = pd.Series(y_pred).value_counts().reindex(
    range(len(class_labels))
).fillna(0).values
# Precision from the model's own perspective: of everything predicted as class i,
# how much really was class i?
tp = pd.Series((y_test == y_pred)[y_pred == pd.Series(y_pred).values].astype(int)) \
    .groupby(pd.Series(y_pred)[y_test == y_pred]).sum() \
    .reindex(range(len(class_labels))).fillna(0).values
per_class["precision"] = np.where(
    per_class.n_predicted > 0, tp / per_class.n_predicted.replace(0, np.nan), np.nan
)

per_class.sort_values("n_test", ascending=False).head(20)

# %%

solved = per_class[per_class.n_test >= 20]
print(f"classes with >=20 test images : {len(solved)}")
print(f"mean per-class recall         : {solved.recall.mean():.4f}")
print(f"median per-class recall       : {solved.recall.median():.4f}")
print(f"classes never recalled        : {(solved.recall == 0).sum()}")
print(f"classes never predicted       : {(per_class.n_predicted == 0).sum()}")
print("\nWorst 10 classes by recall (>=20 test images):")
solved.nsmallest(10, "recall")[["class", "n_test", "recall"]]

# %% [markdown]
### Rare classes are where accuracy hides the truth

# %%

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
solved.plot.scatter(x="n_test", y="recall", s=28, alpha=0.6, ax=axes[0], color="#4C72B0")
axes[0].set_xscale("log")
axes[0].axhline(solved.recall.mean(), ls="--", color="crimson", label="mean recall")
axes[0].set_title("Per-class recall vs class frequency")
axes[0].set_xlabel("test images for the class")
axes[0].set_ylabel("recall")
axes[0].legend()

axes[1].hist(solved.recall.dropna(), bins=25, color="#55A868", edgecolor="white")
axes[1].set_title("Distribution of per-class recall")
axes[1].set_xlabel("recall")
axes[1].set_ylabel("classes")
fig.tight_layout()

# %% [markdown]
## 7. Verdict — what we promote to `params.yaml`

# %%

verdict = pd.DataFrame(
    [
        ["Selection metric", "validation macro-F1 (top-5 as tie-break)"],
        ["Winner", winner_row.model],
        ["Train config", winner_row.train_cfg],
        ["Validation accuracy", f"{winner_row.val_accuracy:.4f}"],
        ["Validation macro-F1", f"{winner_row.val_macro_f1:.4f}"],
        ["Test accuracy", f"{winner_row.test_accuracy:.4f}"],
        ["Test top-5 accuracy", f"{winner_row.test_top5:.4f}"],
        ["Test macro-F1", f"{winner_row.test_macro_f1:.4f}"],
        ["Fit time", f"{winner_row.fit_seconds:.1f}s"],
        ["Seed", SEED],
        ["Feature dim", X_fit.shape[1]],
    ],
    columns=["Item", "Value"],
)
verdict

# %%

# The winning block to copy into params.yaml (or: dvc exp apply <winner>).
print("Copy this into params.yaml under `train:`")
print(json.dumps(winning_cfg, indent=2, sort_keys=True))

# %% [markdown]
## 8. Reproducibility check
#
# Same seed → same estimator → same predictions. This is the checkpoint a
# reviewer runs before approving the config.

# %%

again = build_model({**winning_cfg, "n_jobs": -1}, seed=SEED)
again.fit(X_fit, y_fit)
identical = np.array_equal(again.predict(X_val), best.predict(X_val))

print(f"re-fitting with seed={SEED} reproduces identical validation predictions: {identical}")
if not identical:
    raise SystemExit("Reproducibility check FAILED — do not promote this config")
print("\nReproducibility: OK")

# %%

# And the metric itself must be stable, not just the predictions.
rescored = score(best, X_val, y_val)
print(f"macro-F1 rescored : {rescored['macro_f1']:.6f}")
print(f"leaderboard value : {winner_row.val_macro_f1:.6f}")
assert abs(rescored["macro_f1"] - winner_row.val_macro_f1) < 1e-9, "metric drifted"
print("metric is stable: OK")

# %% [markdown]
## 9. Known limitations and next steps
#
# * **Ceiling.** HOG + colour histogram on 32×32 images is a strong *classical*
#   baseline, not a state-of-the-art result. A small CNN — or a frozen
#   CLIP/DINOv2 encoder plus a linear head — would beat it, at the cost of a GPU
#   dependency that would make CI and `dvc repro` far heavier. Documented
#   trade-off, not an oversight.
# * **Near-duplicate frames.** Notebook 01 §8 shows several capture frames per
#   object. A split grouped by capture id would be stricter than the random split
#   used here and is the obvious next correctness improvement — it may also
#   *lower* the reported numbers, which is exactly why it should be done before a
#   release, not after.
# * **Label typo.** `BlackBerry 4` / `Blackberry 4` means one fruit class can
#   never be scored correctly until a `data/` PR normalises the label.
# * **Hyper-parameter search.** `max_iter`, `alpha` and `C` were compared over a
#   handful of values; a wider sweep via `dvc exp run --set-param` is the natural
#   continuation.
# * **Feature cache size.** `full_features.npz` is ~180 MB compressed. It is a
#   pipeline output, not a Git artefact, and is rebuilt by `dvc repro prepare`.

# %%

print("Reproducibility table for the release -> REPORT.md")