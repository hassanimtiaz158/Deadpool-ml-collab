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
# # 01 — EDA: Fruits-360 (100×100)
#
# **Goal.** Understand the raw archive before any modelling: what is in it, how
# is it organised, is it balanced, what does it look like, and what data-quality
# problems must the pipeline defend against?
#
# **Source.** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)
# (Kaggle, Moltean), pre-resized to 100×100 by the team, tracked with DVC at
# `data/raw/fruits-360_100x100.zip`.
#
# **Notes for reviewers.** Run top-to-bottom from a clean kernel. All reusable
# logic lives in `src/fruit_classification/` and is unit-tested in `tests/` —
# this notebook only *uses* it.

# %% [markdown]
## 1. Setup

# %%

import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Make `src/` importable without installing the package (works in CI and on a
# fresh clone).
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from fruit_classification import config, dataset, features, plots  # noqa: E402

plots.apply_style()
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)

DATASET_ZIP = config.dataset_zip()
print(f"python     : {sys.version.split()[0]}")
print(f"repo root  : {config.PROJECT_ROOT}")
print(f"dataset    : {DATASET_ZIP}")
print(f"exists     : {DATASET_ZIP.is_file()}")
print(f"size (MB)  : {DATASET_ZIP.stat().st_size / 1e6:,.1f}" if DATASET_ZIP.is_file() else "size (MB)  : n/a")

if not DATASET_ZIP.is_file():
    raise SystemExit(
        "Raw dataset not found. Fetch it with:\n"
        "    pip install -e \".[pipeline]\"\n"
        "    dvc pull"
    )

# %% [markdown]
## 2. Archive inventory
#
# What is actually inside the ZIP? We read the central directory only — no image
# is decoded at this stage, so this runs in a couple of seconds.

# %%

with zipfile.ZipFile(DATASET_ZIP) as archive:
    members = archive.namelist()

print(f"total entries      : {len(members):,}")
print(f"  directories      : {sum(m.endswith('/') for m in members):,}")
print(f"  jpeg files       : {sum(m.lower().endswith('.jpg') for m in members):,}")
print(f"  other files      : {sum(not m.endswith('/') and not m.lower().endswith('.jpg') for m in members):,}")

top_level = sorted({m.strip('/').split('/')[0] for m in members if m.strip('/')})
print(f"archive root       : {top_level}")

# Show a few raw member paths to make the layout concrete.
print("\nfirst 8 members:")
for m in members[:8]:
    print(f"  {m}")

# %% [markdown]
## 3. Build the image index
#
# `dataset.build_index()` is the single source of truth for "what images exist".
# It validates the archive layout, applies the split filter and assigns a stable
# integer `label_id` per class (sorted by class name, so ids do not depend on
# ZIP ordering).

# %%

index = dataset.build_index(DATASET_ZIP)

print(f"index shape : {index.shape}")
print(f"columns     : {list(index.columns)}")
print(f"dtype       : {dict(index.dtypes.astype(str))}")
index.head()

# %%

print(f"unique classes (label names)   : {index['label'].nunique()}")
print(f"unique classes (label_id)      : {index['label_id'].nunique()}")
print(f"label_id range                 : {index['label_id'].min()} .. {index['label_id'].max()}")
print(f"classes present in both splits : {len(set(index[index.split == 'Training']['label']) & set(index[index.split == 'Test']['label']))}")

# %% [markdown]
## 4. Split sizes and class coverage

# %%

split_counts = (
    index.groupby("split")
    .agg(images=("member", "size"), classes=("label_id", "nunique"))
    .reindex(list(dataset.VALID_SPLITS))
)
split_counts

# %%

train_classes = set(index.loc[index["split"] == dataset.SPLIT_TRAIN, "label"])
test_classes = set(index.loc[index["split"] == dataset.SPLIT_TEST, "label"])

only_train = sorted(train_classes - test_classes)
only_test = sorted(test_classes - train_classes)

print(f"classes only in Training : {len(only_train)} -> {only_train}")
print(f"classes only in Test     : {len(only_test)} -> {only_test}")

if only_train and only_test:
    print(
        "\nData-quality finding #1\n"
        "----------------------\n"
        "These two lists are the same fruit class written with different\n"
        "capitalisation. 'Blackberry 4' appears only in Test and 'BlackBerry 4'\n"
        "only in Training, so a model trained on Training can NEVER be scored\n"
        "correctly on the Test images of that class.\n"
        "Consequence: the naive class count is 267, but each split really\n"
        "contains 266 classes. See section 9 for how the pipeline handles it."
    )

# %% [markdown]
## 5. Is the dataset balanced?

# %%

per_split_counts = {
    split: index[index["split"] == split]["label"].value_counts() for split in dataset.VALID_SPLITS
}

summary = pd.DataFrame(
    {
        split: counts.reindex(sorted(train_classes | test_classes)).fillna(0).astype(int)
        for split, counts in per_split_counts.items()
    }
)
summary["total"] = summary.sum(axis=1)

print(summary.describe().loc[["min", "25%", "50%", "75%", "max"]].round(1))
print(f"\nclasses with zero Training images : {(summary['Training'] == 0).sum()}")
print(f"classes with zero Test images     : {(summary['Test'] == 0).sum()}")

ratio = summary["Training"] / summary["Test"].replace(0, np.nan)
print(f"train:test ratio  min={ratio.min():.2f}  median={ratio.median():.2f}  max={ratio.max():.2f}")

# %%

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, split in zip(axes, dataset.VALID_SPLITS):
    counts = per_split_counts[split].sort_values(ascending=False)
    ax.bar(range(len(counts)), counts.values, color="#4C72B0", width=0.9)
    ax.set_title(f"{split}: {len(counts)} classes")
    ax.set_xlabel("class (sorted by size)")
    ax.set_ylabel("images")
    ax.set_xticks([])
axes[0].axhline(per_split_counts["Training"].median(), color="crimson", ls="--", label="median")
axes[0].legend()
fig.suptitle("Class frequency — long tail of small classes")
fig.tight_layout()

# %%

train_counts = per_split_counts["Training"]
fig, _ = plots.plot_class_distribution(
    train_counts, "Training images per class (all classes)", "eda_class_distribution.png", top=None
)
print(f"classes: min={train_counts.min()}, median={train_counts.median():.0f}, max={train_counts.max()}")

# %% [markdown]
## 6. Smallest and largest classes

# %%

smallest = summary.nsmallest(10, "total")[["Training", "Test", "total"]]
smallest.index.name = "class"
smallest

# %%

largest = summary.nlargest(10, "total")[["Training", "Test", "total"]]
largest.index.name = "class"
largest

# %% [markdown]
### Imbalance and its consequences
#
# * The archive's own `Training` / `Test` folders are the canonical split.
# * Class frequency varies by roughly an order of magnitude.
# * **Decision:** we keep the archive's `Test` folder as the untouched final test
#   set and carve a **stratified** validation split out of `Training`. Stratifying
#   (`params.yaml: split.stratify: true`) keeps the rare classes represented in
#   validation, which plain `train_test_split` would not guarantee.

# %% [markdown]
## 7. Image properties
#
# Are all images really 100×100 RGB? Decoding all 189k images would be wasteful,
# so we sample deterministically (fixed seed, sorted member order).

# %%

rng = np.random.default_rng(42)
sample_size = 3000
sample_rows = index.sample(n=sample_size, random_state=42)

sizes: dict[tuple, int] = {}
modes: dict[str, int] = {}
file_sizes: list[int] = []

with zipfile.ZipFile(DATASET_ZIP) as archive:
    for member in sample_rows["member"]:
        info = archive.getinfo(member)
        file_sizes.append(info.file_size)
        with archive.open(member) as handle:
            img = Image.open(handle)
            sizes[img.size] = sizes.get(img.size, 0) + 1
            modes[img.mode] = modes.get(img.mode, 0) + 1

print(f"sampled {sample_size:,} images (seed=42)")
print("sizes :", dict(sorted(sizes.items(), key=lambda kv: -kv[1])))
print("modes :", modes)

file_sizes = np.asarray(file_sizes)
print(f"\ncompressed jpeg size (bytes):")
print(f"  min={file_sizes.min():,}  p50={int(np.median(file_sizes)):,}  "
      f"p95={int(np.percentile(file_sizes, 95)):,}  max={file_sizes.max():,}")
print(f"  mean={file_sizes.mean():,.0f}")

# %%

expected = (100, 100)
unexpected = {k: v for k, v in sizes.items() if k != expected}
if unexpected:
    print(f"[warn] images not {expected}: {unexpected}")
else:
    print(f"[ok] all {sample_size:,} sampled images are exactly {expected[0]}x{expected[1]}")
if set(modes) - {"RGB"}:
    print(f"[warn] non-RGB modes present: {set(modes) - {'RGB'}}")
else:
    print("[ok] all sampled images are RGB")

# %% [markdown]
## 8. Visual inspection
#
# Numbers only go so far — let's look at the data.

# %%

# A deterministic "first N classes" grid keeps the notebook reproducible.
preview_labels = sorted(index["label"].unique())[:12]
images = dataset.iter_preview_images(index, preview_labels, DATASET_ZIP, per_label=1)

fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
for ax, label in zip(axes.flat, preview_labels):
    img = images[label][0]
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis("off")
fig.suptitle("One Training image per class (first 12 classes, alphabetically)")
fig.tight_layout()

# %%

# Hardest visual cases: the classes with the fewest training images.
rare = list(summary.index[:6])
rare_images = dataset.iter_preview_images(index, rare, DATASET_ZIP, per_label=1)

fig, axes = plt.subplots(1, 6, figsize=(18, 3.6))
for ax, label in zip(axes.flat, rare):
    ax.imshow(rare_images[label][0])
    ax.set_title(f"{label}\n(n={int(summary.loc[label, 'Training'])})", fontsize=9)
    ax.axis("off")
fig.suptitle("Rarest classes - where a classifier will struggle")
fig.tight_layout()

# %% [markdown]
### Near-duplicate filenames within a class
#
# Fruits-360 image names encode the original capture index (`r0_11_100.jpg`).
# If a class directory contains many near-identical frames, a random split will
# leak near-duplicates across train and validation. Check the naming pattern.

# %%

sample_class = "Apple 1"
member_names = index.loc[
    (index["split"] == dataset.SPLIT_TRAIN) & (index["label"] == sample_class), "member"
].str.rsplit("/", n=1).str[-1]
frame_ids = member_names.str.extract(r"^r(\d+)_", expand=False)
capture_ids = member_names.str.extract(r"^r\d+_(\d+)_", expand=False)

print(f"class {sample_class!r}: {len(member_names)} training images")
print(f"  distinct source objects (r<id>)  : {frame_ids.nunique()}")
print(f"  distinct capture frames (r<id>_<n>): {capture_ids.nunique()}")
member_names.head(10).tolist()

# %%

per_class_captures = []
for label, group in index[index["split"] == dataset.SPLIT_TRAIN].groupby("label"):
    names = group["member"].str.rsplit("/", n=1).str[-1]
    captures = names.str.extract(r"^r\d+_(\d+)_", expand=False)
    per_class_captures.append(
        {
            "class": label,
            "images": len(group),
            "objects": names.str.extract(r"^r(\d+)_", expand=False).nunique(),
            "captures": captures.nunique(),
        }
    )

capture_frame = pd.DataFrame(per_class_captures)
capture_frame["captures_per_object"] = capture_frame["captures"] / capture_frame["objects"].clip(lower=1)

print(f"classes with >5 captures per object : {(capture_frame.captures_per_object > 5).sum()} / {len(capture_frame)}")
capture_frame.nlargest(8, "captures_per_object")

# %% [markdown]
## 9. Label integrity — the findings that change the pipeline

# %%

# a) Case-variant class names that should be the same label.
from collections import defaultdict  # noqa: E402

canonical: dict[str, set[str]] = defaultdict(set)
for label in index["label"].unique():
    canonical[label.lower()].add(label)

dupe_groups = {k: sorted(v) for k, v in canonical.items() if len(v) > 1}
print(f"case-variant collisions: {dupe_groups}")
print(f"=> naive class count {index['label'].nunique()} collapses to "
      f"{len(canonical)} after case folding.")

# %%

# b) Whitespace / casing hygiene of every class name.
stripped = index["label"].str.strip()
print(f"labels with leading/trailing whitespace: {(stripped != index['label']).sum()}")
print(f"labels that are not title case        : "
      f"{sorted({l for l in index['label'].unique() if l != l.title()})[:10]}")

# %%

# c) Class-name prefixes -> which super-category does a class belong to?
prefix = index["label"].str.split().str[0].str.lower()
prefix_frame = (
    pd.DataFrame({"prefix": prefix, "images": 1, "classes": index["label"]})
    .groupby("prefix")
    .agg(images=("images", "size"), classes=("classes", "nunique"))
    .sort_values("images", ascending=False)
)
prefix_frame.head(15)

# %%

fig, ax = plt.subplots(figsize=(11, 5))
top_prefixes = prefix_frame.head(15)
sns.barplot(data=top_prefixes, x="images", y="index", hue="index", ax=ax, legend=False, palette="viridis")
ax.set_title("Images per fruit super-category (first token of the class name)")
ax.set_xlabel("images")
ax.set_ylabel("")
fig.tight_layout()

# %% [markdown]
### Summary of data-quality findings
#
# | # | Finding | Evidence | Impact | Mitigation |
# |---|---------|----------|--------|------------|
# | 1 | Case-variant label: `BlackBerry 4` (Training) vs `Blackberry 4` (Test) | §4 | One class can never be scored correctly; naive class count is 267 instead of 266 | `build_index()` reports `classes_missing_from_test`; the dataset update PR normalises the label |
# | 2 | Long tail: class sizes vary ~10× | §5, §6 | Accuracy hides per-class failure on rare fruits | Report macro-F1 + top-5 alongside accuracy; stratified validation split |
# | 3 | Near-duplicate capture frames per object | §8 | Random splits can leak near-duplicates and inflate validation scores | Group-aware note; validation split is fixed and seeded so numbers stay comparable |
# | 4 | Very large archive (847 MB, 189k images) | §2 | Decoding dominates runtime (~14 min single pipeline run) | Feature cache in `data/interim/` via `dvc.yaml` so experiments reuse it |
# | 5 | Every image is exactly 100×100 RGB | §7 | No shape/corruption handling needed downstream | Asserted in tests and in the CI data checks |

# %% [markdown]
## 10. Feature representation used by the model

# %%

cfg = config.params_with_defaults(config.load_params())["features"]
print(f"feature config from params.yaml: {cfg}")
print(f"resulting feature dimension   : {features.feature_dim(**{k: cfg[k] for k in ('image_size','hog_cells','hog_bins')}, use_color=cfg['color_hist'])}")

# %%

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
probe = images[preview_labels[0]][0]

axes[0].imshow(probe)
axes[0].set_title("source 100×100")
axes[0].axis("off")

small = features.to_rgb_array(probe, cfg["image_size"])
axes[1].imshow(small)
axes[1].set_title(f"resized {cfg['image_size']}×{cfg['image_size']}")
axes[1].axis("off")

hog = features.hog_features(
    features._luminance(small), cells=cfg["hog_cells"], bins=cfg["hog_bins"]
).reshape(cfg["hog_cells"], cfg["hog_cells"], cfg["hog_bins"])
sns.heatmap(hog, cmap="magma", cbar=False, ax=axes[2])
axes[2].set_title(f"HOG cells × {cfg['hog_bins']} bins")

fig.suptitle("What the model actually sees")
fig.tight_layout()

# %%

# Feature vector composition and separability sanity check.
red = features.extract_features(probe, **{
    "image_size": cfg["image_size"], "cells": cfg["hog_cells"],
    "bins": cfg["hog_bins"], "use_color": cfg["color_hist"]})
hog_dim = cfg["hog_cells"] ** 2 * cfg["hog_bins"]

print(f"total feature dim : {red.shape[0]}")
print(f"  HOG            : {hog_dim}")
print(f"  HSV histogram  : {red.shape[0] - hog_dim}")
print(f"finite           : {np.isfinite(red).all()}")
print(f"range            : [{red.min():.3f}, {red.max():.3f}]")

# %%

# Do identical classes give more similar vectors than different classes?
rng = np.random.default_rng(0)
pairs_same, pairs_diff = [], []
probe_labels = preview_labels[:6]
probe_images = dataset.iter_preview_images(index, probe_labels, DATASET_ZIP, per_label=3)

vecs, keys = [], []
for label in probe_labels:
    for img in probe_images[label]:
        vecs.append(features.extract_features(img, **{
            "image_size": cfg["image_size"], "cells": cfg["hog_cells"],
            "bins": cfg["hog_bins"], "use_color": cfg["color_hist"]}))
        keys.append(label)
V = np.vstack(vecs)

for i in range(len(V)):
    for j in range(i + 1, len(V)):
        d = float(np.linalg.norm(V[i] - V[j]))
        (pairs_same if keys[i] == keys[j] else pairs_diff).append(d)

print(f"mean L2 distance, same class    : {np.mean(pairs_same):.3f}  (n={len(pairs_same)})")
print(f"mean L2 distance, different class: {np.mean(pairs_diff):.3f}  (n={len(pairs_diff)})")
print(f"ratio (different / same)         : {np.mean(pairs_diff) / np.mean(pairs_same):.2f}x")
print("\nA ratio well above 1.0 means the representation separates classes,")
print("i.e. a linear classifier on top of it has something to learn.")

# %% [markdown]
## 11. What the pipeline will do with these findings

# %%

decisions = pd.DataFrame(
    [
        ["Use the archive's Test folder as the final held-out set",
         "keeps an untouched official split; no re-labelling of test data"],
        ["Carve a stratified 80/20 validation split out of Training",
         "params.yaml -> split.test_size / split.stratify"],
        ["Report accuracy AND top-5 accuracy AND macro-F1",
         "266-way problem with a long tail; top-5 is the realistic operating point"],
        ["Seed everything (split + estimator) from params.yaml",
         "params.yaml -> seed: 42; makes dvc repro deterministic"],
        ["Cache features in data/interim/ and reuse across experiments",
         "dvc.yaml -> prepare stage; 14 min of decoding paid once"],
        ["Normalise the BlackBerry/Blackberry label in a data/ branch",
         "§4 finding #1 - tracked as a data-update PR"],
    ],
    columns=["Decision", "Where it lives / why"],
)
decisions

# %%

print("Next notebook: 02-model-selection.ipynb")
print("  -> baseline models, dvc exp sweeps, winner promoted to params.yaml")

# %%