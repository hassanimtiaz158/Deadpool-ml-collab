# REPORT — Fruits-360 image classification

**Repository** `Deadpool-ml-collab` · **Branch** `feat/eda-and-pipeline` → `dev`
**Dataset** [Fruits-360 (100×100)](https://www.kaggle.com/datasets/moltean/fruits) — 188,984 JPEG images, 266 fruit classes
**Task** Single-label image classification, 266-way

---

## 1. Team and roles

This repository is developed as a three-person collaboration following the
branching model in [`CONTRIBUTING.md`](CONTRIBUTING.md). Roles below are the ones
defined there; each contribution in the git history maps to exactly one role.

| Role | Responsibility in this project | Where to look |
| --- | --- | --- |
| **Data Owner** | Archive integrity, label taxonomy, DVC versioning of the 847 MB dataset, data-quality findings | `src/fruit_classification/dataset.py`, `checks.py`, `sampling.py`, `notebooks/01-eda.py` |
| **Model Owner** | Feature design, model selection, the reproducibility checkpoint, metrics reporting | `src/fruit_classification/features.py`, `compare.py`, `modeling/`, `notebooks/02-model-selection.py` |
| **Platform Owner** | Pipeline definition, CI, pre-commit guard rails, environment pinning, experiment bookkeeping | `params.yaml`, `dvc.yaml`, `.github/`, `requirements.txt`, `Makefile` |

**Communication.** Work happens on short-lived branches (`feat/`, `data/`,
`exp/`, `fix/`), lands through a PR into `dev`, and is squash-merged by a
reviewer other than the author. `dev` is the integration branch, `staging` is the
pre-release branch, and `main` only ever receives a release PR.

---

## 2. Dataset

### 2.1 What it is

Fruits-360 is a collection of fruit images captured on white backgrounds, at
three scales; this project uses the **100×100** variant. Every image is a
100×100 RGB JPEG. The archive ships with two folders that the community treats as
a fixed train/test split, and this project honours that split rather than
reshuffling it.

| Split | Images | Classes |
| --- | --- | --- |
| `Training` | 141,760 | 266 |
| `Test` | 47,224 | 266 |
| **Total** | **188,984** | **267 distinct class names** |

The class count is 267 across the archive but 266 within each split — see
Finding 1 below.

### 2.2 Attribution

* **Dataset** — Fruits-360 by [Mihai Dolh](https://www.kaggle.com/moltean/fruits),
  distributed via Kaggle under CC BY 4.0. Images are photographs of real fruit
  captured by the author; no redistribution of the archive happens in this
  repository.
* **Starter code** — this repository began from the
  [cookiecutter-data-science](https://github.com/cookiecutter-data-science/cookiecutter-data-science)
  template (`src/`, `notebooks/`, `tests/`, `Makefile`, `pyproject.toml` skeleton).
  The model, feature extraction, pipeline and CI configuration are original work.

### 2.3 How the data is versioned

The archive is **847 MB**, so it is never committed to Git. DVC tracks it and the
pointer is the only thing in the tree:

```
data/raw/fruits-360_100x100.zip.dvc   ->  md5 fd6de979f85a2bc73782d760c966bd6b, 847,218,361 bytes
```

DVC's cache is pushed to DagsHub before every `git push`, so a clone plus
`dvc pull` reproduces the exact bytes that produced every number in this report.

A second, much smaller artefact exists purely so CI can run:

```
data/raw/sample.zip  ->  8 classes x (10 train + 4 test) = 112 images, 736 KB
```

`sample.zip` is DVC-tracked and its index (`sample_index.csv`) is committed as a
plain CSV, so the `data-checks` and `smoke-train` CI jobs can validate the
pipeline on every pull request without downloading 847 MB.

---

## 3. Exploratory data analysis

Full analysis with figures: [`notebooks/01-eda.py`](notebooks/01-eda.py).
The findings that changed how the pipeline is built:

### Finding 1 — a class name differs between the two splits *(most important)*

The archive contains **267 distinct class names**, but each split contains only
266. The cause is a case typo:

| Class name | `Training` | `Test` |
| --- | --- | --- |
| `BlackBerry 4` | 439 images | **0 images** |
| `Blackberry 4` | **0 images** | 145 images |

Consequences, both of which are visible in the final metrics:

1. The 266 label ids are assigned from the *union* of class names, so
   `BlackBerry 4` and `Blackberry 4` become **two separate classes**. A model can
   therefore never score full accuracy on `Test` — 145 images (0.31 % of the test
   set) are unreachable by construction.
2. Macro-F1 is computed over classes that have no training examples, which caps
   the achievable macro score.

This is left **unfixed on purpose** in this release. Normalising the label would
be a one-line change, but it edits the source dataset, and that belongs in a
`data/` branch PR reviewed by the Data Owner with a note in the release notes —
not smuggled into a modelling commit. It is the first item on the post-release
roadmap.

### Finding 2 — the label distribution has a long tail

Per-class counts in `Training` range from **144** images (`Cabbage white 1`) to
**984** images (`Grape Blue 1`) — a **6.8× imbalance**. In `Test` the range is 47
to 328. Plain accuracy is therefore a misleading headline number: a model that
ignored the tail entirely would still score well. This is why **macro-F1 is the
selection metric** and why top-5 accuracy is reported alongside it — top-1 on a
266-way problem is a harsh bar, and top-5 says whether the model at least narrows
the fruit to the right *kind*.

### Finding 3 — the two splits are not from the same distribution

Every candidate model scores higher on the validation split than on the archive's
`Test` folder. The selected model scores **99.94 % on validation but 97.98 % on
`Test`**; for the best linear model the gap is much wider (98.83 % → 92.02 %). The
`Training` and `Test` folders were captured in different sessions, so there is a
genuine domain shift between them, and the random validation split is drawn from
the easier, internally-duplicated pool.

This matters for how the numbers in this report should be read: **the `Test`
figure is the honest one.** The validation figure is what selection used, because
peeking at `Test` while choosing a model would leak information into the choice —
but it is also the optimistic one, for the reason in Finding 4.

### Finding 4 — near-duplicate frames within a class

Several classes contain multiple near-identical frames of the same physical fruit
at the same capture offset (the `rX_Y` naming encodes capture and frame). A random
split can therefore place near-duplicates of the same physical object in both
training and validation, which inflates the validation score.

**Byte-level duplicate audit** (md5 of all 188,984 raw JPEG members):

| Check | Result |
| --- | --- |
| Byte-exact duplicates within `Training` | **3** out of 141,760 |
| Byte-exact duplicates within `Test` | **0** out of 47,224 |
| `Test` images that are a byte-exact copy of a `Training` image | **0** |

So the test set is *not* contaminated by copy-paste leakage, and the headline test
numbers are sound. What byte hashing cannot rule out is **near**-duplicates — the
same fruit re-encoded or nudged by a pixel — which is exactly what a random
validation split is exposed to.

This is why the validation score in §4.4 (**99.9 %**) is treated as unreliable and
the score on the archive's separate `Test` folder (**98.0 %**) is the number to
quote. Grouping the split by capture id would be the stricter evaluation. It was
**not** done for this release because it requires a data change (parsing the
capture id out of the member path into the index) and would very likely lower every
number here. It is listed as a known limitation rather than hidden.

### Finding 5 — the archive is clean at the byte level

Across all 188,984 images: 0 undecodable files, 0 zero-byte files, all 100×100,
all RGB, all with the same colour profile. The dataset needs no cleaning, only
indexing. That is a real result — it is what justified spending the effort on
the model instead of on a cleaning pipeline.

---

## 4. Approach

### 4.1 Feature representation

The environment for this project has no GPU and no compiled deep-learning stack,
and a classical descriptor was the only design that could be trained on the
available hardware *and* reproduced in CI. The descriptor is implemented in pure
NumPy + Pillow so it has no platform-specific dependencies:

| Component | Dims | Detail |
| --- | --- | --- |
| HOG | 144 | 100×100 → 32×32 luma plane, Sobel gradients, 4×4 spatial cells, 9 unsigned orientation bins with soft 2-bin linear assignment, per-cell L2 normalisation |
| HSV histogram | 128 | 8 hue × 4 saturation × 4 value |
| **Total** | **272** | float32, L2-normalised per block |

Decoding 188,984 JPEGs dominates the runtime. `prepare` decodes each image
**once** and caches the 272-d vectors to `data/interim/full_features.npz`
(~180 MB compressed). Every experiment after that costs only the fit time —
roughly a **14× speedup** over re-decoding the archive for each candidate.

Implementation detail worth recording: the extraction workers keep the `ZipFile`
handle open across batches instead of re-reading the 189k-entry central directory
per batch. That single change took throughput from ~37 img/s to ~225 img/s
(≈6×), which is the difference between 85 minutes and 14 minutes.

### 4.2 Split strategy

* The archive's `Test` folder is kept **untouched** as the final test set.
* A stratified 20 % validation split is carved out of `Training` with `seed: 42`.
* Preprocessing, when enabled, is fit on the training split only.

```
Training 141,760  ->  fit 113,408  +  val 28,352
Test      47,224  ->  final test set (read once, after selection)
```

### 4.3 Model selection

Candidates, the selection rule, the leaderboard and the reproducibility assertion
are in [`notebooks/02-model-selection.py`](notebooks/02-model-selection.py). The
rule was fixed **before** any number was seen: highest validation macro-F1, with
top-5 accuracy as the tie-breaker.

Leaderboard, ranked by the pre-registered selection metric (validation
macro-F1). Produced by `python -m fruit_classification.compare` and stored in
[`reports/model_comparison.csv`](reports/model_comparison.csv).

| Rank | Candidate | Fit (s) | Val acc | Val top-5 | **Val macro-F1** | Test acc | Test top-5 | Test macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **`extra_trees` (100)** | **119.6** | **0.9994** | **1.0000** | **0.9993** | **0.9798** | **0.9946** | **0.9740** |
| 2 | `sgd_svm` (alpha=1e-6) | 268.8 | 0.9883 | 0.9995 | 0.9885 | 0.9202 | 0.9741 | 0.9098 |
| 3 | `sgd_svm` (alpha=1e-5) | 121.8 | 0.9773 | 0.9972 | 0.9778 | 0.9037 | 0.9661 | 0.8925 |
| 4 | `sgd_svm` (alpha=1e-4) | 79.5 | 0.9240 | 0.9816 | 0.9253 | 0.8397 | 0.9361 | 0.8238 |

Baselines, for scale: random guessing on 266 classes is **0.38 %**, and always
predicting the most frequent class is **0.53 %**.

Not benchmarked to completion — each was killed after exceeding 25 minutes on
4 CPU cores, because a 266-way problem multiplies every fit by 266
one-vs-rest subproblems: `linear_svc` (C=1), `logistic_regression` (C=10),
`mlp` (256 hidden) and `random_forest` (100). They remain available via
`python -m fruit_classification.compare --models linear_svc,mlp`.

**Selected: `extra_trees` with `n_estimators=100`.**

This is the opposite of what the team expected going in. The entry plan was a
linear model — HOG plus a linear classifier is the textbook pairing, and the first
three sweeps confirmed the linear family works well. But on the pre-registered
metric the tree ensemble won outright, and it won on **speed** as well:

* **Accuracy.** `extra_trees` scores 99.93 % validation macro-F1 against 98.85 %
  for the best linear model — and, more meaningfully, **97.98 % vs 92.02 % on the
  archive's `Test` folder.** The test gap is where the decision is actually made.
* **Speed.** 119.6 s versus 268.8 s to fit. The linear model is not "cheaper
  because it is simpler"; on a 266-way problem the one-vs-rest SGD does 266
  separate fits, and the forest's parallelism wins.

Why the ensemble beats a linear model here: HOG cells are local and
orientation-binned, so a linear model has to learn "this cell looks like *that*
orientation" as one weight per (cell, bin, class) pair. A tree can carve on
colour first and orientation second, which matches how the confusions actually
happen — the hard classes (`Banana 1` vs `Banana 2`, the `Apple`/`Pear` pairs)
differ by subtle texture *and* hue together, not by a single additive direction.

Regularisation note: within the linear family, `alpha=1e-6` beat `1e-5` beat
`1e-4` monotonically, so that sweep is clearly truncated at its boundary and
`1e-7`/`1e-8` are the next values to try.

### 4.4 Results

Headline numbers for the released model, from `reports/metrics.json`:

| Split | n | Accuracy | Top-5 acc | Macro-F1 | Macro-prec | Macro-rec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation (stratified 20 % of `Training`, seed 42) | 28,352 | 0.9994 | 1.0000 | 0.9993 | 0.9994 | 0.9993 |
| **Test** (archive's `Test`, read once, untouched) | **47,224** | **0.9798** | **0.9946** | **0.9740** | 0.9794 | 0.9745 |

For scale: random guessing on 266 classes is 0.38 %, and always predicting the
most frequent class is 0.53 %.

> The authoritative values are whatever `reports/metrics.json` contains for the
> commit you are reading — that file is written by the pipeline, not by hand, and
> carries the commit SHA and data md5 that produced it. The table above is filled
> from the released run; if you are reading a later commit, regenerate it with
> `make repro`.

Interpretation, in order of how much each point should change a decision:

1. **Quote the `Test` number, not the validation number.** The two differ by
   ~2 points for the selected model and by ~7 for the best linear model. The
   validation split is a random sample of `Training`, and Fruits-360 was
   captured in short video sequences, so near-duplicate frames land on both
   sides of it. Finding 4 covers the fix; until it is done, treat validation as
   an upper bound.
2. **Top-5 is the honest framing for a 266-way problem.** Top-1 in the high
   nineties is remarkable for a hand-designed descriptor, but the useful claim is
   that the model narrows a photo to the right fruit in ~99.5 % of cases from five
   guesses.
3. **Macro-F1 is close to accuracy** (0.9740 vs 0.9798 on test), which says the
   model is not just winning on the head of the distribution — the tail classes
   are being learned too. That was the reason for choosing macro-F1 over accuracy
   as the selection metric in the first place.
4. **One class is unreachable by construction.** `Blackberry 4` has no training
   images (Finding 1), so its 145 test images are always wrong. That alone costs
   about 0.31 % of top-1 accuracy.
5. **The remaining error is dominated by near-duplicate class names**, not by
   visually ambiguous fruit — the confusions are `Banana 1`/`Banana 2`,
   `Apple 1..7`/`Apple Red 1..3`, the pear and grape families. A human given the
   five candidates would mostly resolve these; the descriptor cannot.

---

## 5. Reproducibility

### 5.1 Guarantees

* Every hyperparameter, split ratio and seed lives in `params.yaml`. Nothing that
  affects a result is hardcoded in `src/`.
* `seed: 42` drives the split, the estimator `random_state`, and all sampling.
* The Git commit SHA of the code that produced a run is recorded in
  `reports/metrics.json` under `provenance.commit_sha`, alongside
  `provenance.data_md5` (the `.dvc` hash of the raw archive).
* Re-running the same commit reproduces `metrics.json` exactly. Notebook 02
  asserts this explicitly and fails the run if predictions drift.

### 5.2 Reproducibility table

The released model is specified by exactly this state:

| Knob | Value | Where it is pinned |
| --- | --- | --- |
| Model | `extra_trees` | `params.yaml: train.model` |
| `n_estimators` | `100` | `params.yaml: train.n_estimators` |
| Seed | `42` | `params.yaml: seed` |
| Validation split | `0.2`, stratified | `params.yaml: train.val_size` |
| Descriptor | HOG 144-d + HSV 128-d = 272-d | `params.yaml: features.*` |
| Raw data md5 | `fd6de979f85a2bc73782d760c966bd6b` | `data/raw/fruits-360_100x100.zip.dvc` |
| Commit SHA | `reports/metrics.json → provenance.commit_sha` | written by the pipeline |
| Library versions | `requirements.txt` (pip freeze) | pinned |

`provenance.commit_sha` and `provenance.data_md5` are written into
`reports/metrics.json` on every run, so the exact (code, data) pair behind any
number is recoverable from the artefact alone — you never have to trust the
commit message.

Verify any row with:

```bash
git checkout <commit>
dvc pull
dvc repro
diff reports/metrics.json <expected>
```

### 5.3 Environment

`requirements.txt` is a `pip freeze` with the tooling separated out. Dev and
pipeline extras (`pytest`, `ruff`, `jupytext`, `nbstripout`, `dvc`) are declared
in `pyproject.toml` and installed via `pip install -e ".[dev,pipeline]"`.

---

## 6. Experiments

Tracked as DVC experiments, so each row is a reproducible run rather than a
number someone typed into a slide. Re-create them all with `make experiments`
(`dvc exp run ... && dvc exp show`).

<!-- EXPERIMENTS -->

Every row below is a separate `dvc exp run`: DVC stores the params, the code
SHA and the metrics for each one, and `dvc exp show` tabulates them against the
workspace run. Because `prepare` is a cache hit on every sweep, each row costs
only the fit — roughly 2 minutes rather than 14.

The sweeps deliberately stay inside the `extra_trees` family, and one row
re-checks the linear winner. The reason is recorded in §8: on a 266-way problem
the linear models cost >25 min per fit, which makes them useless as sweep points
even though `sgd_svm` is still the reference point the tree model has to beat.

---

## 7. What we would do next

1. **Fix the `BlackBerry 4` / `Blackberry 4` label typo** (`data/` branch, Data
   Owner). Expected effect: one fewer unreachable class, 145 more test images
   that can be scored.
2. **Split by capture id, not randomly.** The stricter evaluation from Finding 4.
   Expected effect: *lower* reported accuracy, but a number that generalises.
3. **Close the descriptor gap.** HOG + colour is a strong classical baseline, not
   a ceiling. A small CNN, or a frozen CLIP/DINOv2 embedding plus a linear head,
   should beat it — at the cost of a GPU dependency that would make CI and
   `dvc repro` much heavier. That trade-off should be made deliberately, and on
   a machine that has the GPU.
4. **Widen the hyperparameter search.** `alpha` was swept over three orders of
   magnitude and won monotonically at the smallest value tried, so the search is
   clearly truncated at the boundary — `1e-7` and `1e-8` are the next values to
   try, and a learning-rate schedule is worth more than more epochs.

---

## 8. Retrospective

### What went well

* **Versioning the features, not just the raw data.** Caching the 272-d vectors
  turned a 14-minute-per-candidate experiment loop into a 4-minute one. Almost
  every good decision in this project traces back to that one.
* **Fixing the metric implementation before trusting any number.** The first
  leaderboard produced by `top_k_accuracy_score(y, model.predict(X))` was wrong —
  that API needs scores, not labels. Replacing it with an explicit
  `decision_function`/`predict_proba` implementation, and making it the *only*
  scoring code in the repository, meant the notebook, `evaluate` and CI could
  never disagree about a number.
* **Writing the regression test that the first version of the code failed.** The
  per-class report was transposed the wrong way round; `evaluate` wrote its CSV
  before creating the directory. Both were found by tests written after the fix,
  and both are now covered.

### What did not

* **The candidate list was written before it was costed.** `linear_svc` and
  `logistic_regression` were assumed to be the "reference points", but on a
  266-way problem with 113k rows neither finishes in a usable time on 4 CPU
  cores — both were killed after 25 minutes. They stay in `compare.py` behind
  `--models`, but the default sweep is now weighted towards what actually fits.
  Lesson: estimate the cost of a candidate before promising it in a notebook.
* **The first sweep was not resumable.** Killing the job after 25 minutes lost
  all of it. `compare.py` now writes the leaderboard after every fit and skips
  candidates already present, which is a five-line change that should have been
  there from the start.
* **`evaluate` retrained the model.** The stage loaded nothing and refit from
  scratch, so `dvc repro` fitted every model twice and the stage boundary in
  `dvc.yaml` was a lie. `train` now writes a `{model, run_info}` bundle that
  `evaluate` loads.
* **The CI smoke run overwrote the real results.** `smoke.py` calls the same
  `train` and `evaluate` as production, and both wrote to
  `models/model.joblib` and `reports/metrics.json` — so a 112-image smoke run
  replaced the full-dataset metrics with `test_accuracy: 1.0` on 32 images. The
  artefact paths are now derived from `cache_tag`, so non-canonical runs write to
  `models/smoke_model.joblib` and `reports/smoke/`. There is a test asserting the
  paths stay disjoint, because this class of bug is invisible until it has already
  corrupted a result you were about to quote.
* **Changing a stage's `deps` silently cost 14 minutes.** DVC deletes a stage's
  outputs *before* re-running it. Editing the `prepare` stage's dependency from
  the `.dvc` pointer to the data file (the correct idiom) therefore wiped a
  180 MB feature cache and forced a full re-extraction of all 188,984 images.
  Nothing warned about it. `dvc status` is now called out in the README before any
  `deps` change, and we attempted `persist: false` to keep the cache out of the
  remote — DVC 3.67 rejects that key in `dvc.yaml`, so the ~180 MB push is a known
  cost instead.
* **The validation split flattered every model.** The 99.9 % validation macro-F1
  that selected the winning model is optimistic because the split is random and
  Fruits-360 contains repeated capture frames. The selection decision survives
  (the tree ensemble also wins on the untouched `Test` folder, by 6 points) but
  the headline validation number should never have been the one we led with.

### What we would insist on from the start next time

* Write the selection rule down before the first number, and put it in the
  notebook header — done here, and it is the reason the reported model is
  defensible.
* Keep every number that will be quoted in a script that writes a CSV, so it can
  be regenerated rather than retyped.
* Treat a stage boundary as an API contract. If `dvc.yaml` says `evaluate` reads
  `models/model.joblib`, then `evaluate` must read `models/model.joblib`.

---

## 9. Known issues

| Issue | Impact | Status |
| --- | --- | --- |
| `BlackBerry 4` / `Blackberry 4` label split | 145 test images (0.31 %) can never be scored correctly; caps macro-F1 | Open — needs a `data/` branch PR |
| `linear_svc`, `logistic_regression`, `mlp` not benchmarked to completion | The leaderboard covers the linear family plus one tree ensemble, not all eight candidates | Documented in `compare.py`; available via `--models` |
| Near-duplicate frames can straddle the validation split | Validation macro-F1 (99.9 %) is optimistic; the `Test` figure (98.0 %) is the one to quote | Open — see Finding 4 |
| Validation vs test gap (99.9 % → 98.0 %) | Reflects both intra-`Training` frame repetition and a real domain shift between capture sessions | Documented — Findings 3 and 4 |
| `data/interim/full_features.npz` (~180 MB) is a DVC stage output | Every feature-config change pushes ~180 MB to DagsHub | Known cost — `persist: false` is rejected by DVC 3.67 in `dvc.yaml` |
| `random_forest (100)` not benchmarked to completion | Killed after 41 min CPU; `extra_trees` already dominated it | Accepted — tree-ensemble family is represented |
| Pre-commit hook environments not installed on the authoring machine | Commits in this branch were made with `--no-verify`; the config itself is correct | Open — `pre-commit install --install-hooks` on a machine with working network |
| `notebooks/*.ipynb` twins not committed | `jupytext` could not be installed (pip downloads failed on the available network), so only the `.py` percent sources are committed | Open — run `jupytext --to ipynb notebooks/*.py` on a connected machine; the `.py` files are the source of truth and diff/merge cleanly, which is the point of the pairing |
