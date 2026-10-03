# Contributing Guide

## 1. Overview

This repository is a collaborative Machine Learning project. All team members are expected to contribute code, create pull requests, review teammates' work, and maintain reproducibility throughout the project.

The project follows a Git-based workflow with:

* `main` — production and released model
* `staging` — release candidate and validation
* `dev` — integration branch
* `feat/<name>` — feature and production-code changes
* `data/<name>` — dataset changes tracked with DVC
* `exp/<member>-<idea>` — experimental work
* `fix/<name>` — urgent production fixes

All normal changes must go through pull requests and code review.

---

## 2. Branching Strategy

### `main`

`main` represents the production version of the project.

Rules:

* No direct pushes are allowed after the initial repository setup.
* Changes must come from `staging` through a pull request.
* At least one teammate must approve the pull request.
* All required CI checks must pass.
* Production releases are tagged on this branch.

Example release tag:

```text
model-v1.0
```

---

### `staging`

`staging` represents the release candidate.

Rules:

* Changes come from `dev` through a pull request.
* At least one teammate must approve the pull request.
* All required CI checks must pass.
* The final model must be independently reproduced before promotion to `main`.

The release candidate must be tested from a fresh clone using:

```bash
dvc pull
dvc repro
```

---

### `dev`

`dev` is the integration branch where completed work is combined.

Rules:

* No direct pushes are allowed.
* Changes must come through pull requests.
* Feature and data branches are created from `dev`.
* At least one teammate must review each pull request.
* CI checks must pass before merging.

---

## 3. Short-Lived Branches

### Feature Branches

Use feature branches for production code, pipeline changes, refactoring, and other project improvements.

Naming convention:

```text
feat/<name>
```

Examples:

```text
feat/preprocessing
feat/model-training
feat/evaluation
feat/ci
feat/eda-notebook
```

Create a feature branch from `dev`:

```bash
git switch dev
git pull
git switch -c feat/<name>
```

After completing the work:

```bash
git push -u origin feat/<name>
```

Open a pull request into `dev`.

Feature branches should be deleted after merging.

---

### Data Branches

Use data branches for dataset modifications.

Naming convention:

```text
data/<name>
```

Examples:

```text
data/initial-dataset
data/remove-duplicates
data/fix-labels
data/update-dataset
```

Dataset changes must be tracked using DVC.

Before pushing a data change:

```bash
dvc add data/raw/<dataset>.csv
dvc push
git add .
git commit -m "data: update dataset"
git push
```

**Always run `dvc push` before `git push` when data or DVC-tracked files change.**

Data branches must be merged into `dev` through a pull request.

---

### Experiment Branches

Use experiment branches for testing new ideas, hyperparameters, models, or approaches.

Naming convention:

```text
exp/<member>-<idea>
```

Examples:

```text
exp/hasan-max-depth
exp/ali-logistic-regression
exp/ahmed-random-forest
```

Experiment branches:

* Start from `dev`.
* Should be short-lived.
* Must be kept reproducible.
* Should use committed code before running experiments.
* Are not merged directly into `dev`.

If an experiment is successful, promote the change to a new `feat/` branch and create a pull request.

Example:

```bash
dvc exp show
dvc exp apply <experiment-name>
```

Then create a feature branch:

```bash
git switch dev
git switch -c feat/promote-best-model
```

At least one experiment branch may be intentionally abandoned and documented in `REPORT.md`.

---

### Fix Branches

Use fix branches for urgent production fixes.

Naming convention:

```text
fix/<name>
```

Example:

```text
fix/data-validation
fix/model-loading
fix/production-bug
```

Fix branches are created from `main`.

After fixing the issue:

1. Open a pull request into `main`.
2. Obtain the required approval.
3. Ensure CI passes.
4. Merge the fix.
5. Create a patch release tag if required.
6. Merge the updated `main` back into `dev`.

Example patch tag:

```text
model-v1.0.1
```

---

## 4. Development Workflow

The normal development flow is:

```text
dev
 ↓
feat/<name> or data/<name>
 ↓
Pull Request
 ↓
Code Review
 ↓
CI Checks
 ↓
dev
 ↓
staging
 ↓
Reproducibility Test
 ↓
main
 ↓
model-v1.0
```

No one should push directly to `dev`, `staging`, or `main`.

---

## 5. Creating a New Branch

Always update your local `dev` branch before starting new work:

```bash
git switch dev
git pull origin dev
```

Create a new branch:

```bash
git switch -c feat/<name>
```

or:

```bash
git switch -c data/<name>
```

or:

```bash
git switch -c exp/<member>-<idea>
```

---

## 6. Keeping Branches Updated

Before opening a pull request, update your branch with the latest `dev` changes:

```bash
git fetch origin
git rebase origin/dev
```

If conflicts occur, resolve them carefully and test the project before pushing again.

---

## 7. Commit Message Convention

This project uses Conventional Commits.

Use the following format:

```text
type: short description
```

### Common Types

| Type       | Purpose                       |
| ---------- | ----------------------------- |
| `feat`     | Add a new feature             |
| `fix`      | Fix a bug                     |
| `data`     | Add or modify dataset         |
| `exp`      | Run or document an experiment |
| `test`     | Add or modify tests           |
| `docs`     | Documentation changes         |
| `refactor` | Code restructuring            |
| `ci`       | CI/CD changes                 |
| `chore`    | Maintenance/configuration     |

### Examples

```text
feat: add preprocessing pipeline
feat: add random forest training
fix: handle missing values
data: update training dataset
exp: test max_depth=10
test: add preprocessing tests
docs: update README
ci: add GitHub Actions workflow
chore: update dependencies
```

Keep commits small and focused.

---

## 8. Pull Request Rules

Every production change must be submitted through a pull request.

A pull request should:

* Have a clear title.
* Explain what changed.
* Explain why the change was made.
* Include relevant metrics.
* Pass CI checks.
* Have at least one teammate review it.
* Address all requested changes before merging.

### Pull Request Title Examples

```text
feat: add preprocessing pipeline
data: update training dataset
feat: add model evaluation
ci: add pull request checks
release: v1.0
```

---

## 9. Pull Request Description

Use the following structure for pull requests:

```markdown
## What changed and why

Describe the changes and explain why they were needed.

## Metrics

| Metric | Before | After |
|---|---:|---:|
| Accuracy | - | - |
| Precision | - | - |
| Recall | - | - |
| F1 Score | - | - |

## Review checklist

- [ ] No data leakage
- [ ] Splits are fixed
- [ ] Preprocessing is fitted on training data only
- [ ] No hardcoded paths
- [ ] Seeds are set
- [ ] Metrics are calculated correctly
- [ ] `dvc push` completed if data/model changed
- [ ] Notebook restarted and run top-to-bottom if applicable
- [ ] Code style/linter passes
- [ ] Tests pass
```

This checklist follows the required review checklist in the assignment.

---

## 10. Code Review Rules

Every team member must participate in code reviews.

For this assignment:

* Each member must author at least **2 merged pull requests**.
* Each member must review at least **2 pull requests**.
* At least one pull request must receive a **"Changes requested"** review.
* Reviewers should actually check out and test pipeline-related changes when appropriate.

Do not approve a pull request without reviewing the changes.

---

## 11. Data and DVC Rules

Datasets and large model files must not be committed directly to Git.

Use DVC for dataset and model versioning.

Check the status of DVC:

```bash
dvc status
```

Track a dataset:

```bash
dvc add data/raw/<dataset>.csv
```

Upload DVC data:

```bash
dvc push
```

Download DVC data:

```bash
dvc pull
```

Reproduce the pipeline:

```bash
dvc repro
```

When data changes:

```text
Modify data
    ↓
dvc add
    ↓
dvc push
    ↓
git commit
    ↓
git push
```

Never run:

```text
git push
```

before:

```text
dvc push
```

when DVC-tracked data has changed.

---

## 12. Reproducibility Rules

Every experiment and model training run should be reproducible.

The following must be controlled or recorded:

* Git commit SHA
* Dataset DVC version
* `params.yaml`
* `dvc.lock`
* Random seed
* Python/package environment
* Model parameters
* Evaluation metrics

Do not run experiments on uncommitted code.

Before running an experiment:

```bash
git status
```

Make sure the working tree is clean.

Experiments should use fixed seeds whenever randomness is involved.

---

## 13. Data Leakage Prevention

All team members must ensure that there is no data leakage.

In particular:

* Do not use target information as a feature.
* Do not use future information to create features.
* Split the dataset before fitting preprocessing when appropriate.
* Fit scalers, encoders, and imputers using training data only.
* Keep train/test splits fixed and reproducible.

---

## 14. Notebook Rules

Notebooks are primarily used for exploration and analysis.

Each notebook should:

* Have a clear purpose.
* Run from top to bottom.
* Avoid hardcoded machine-specific paths.
* Have outputs stripped before committing.
* Have a corresponding Jupytext file where required.

Example:

```text
notebooks/
├── 01-eda.ipynb
└── 01-eda.py
```

Before opening a pull request:

1. Restart the notebook kernel.
2. Run all cells from top to bottom.
3. Confirm that the notebook runs successfully.
4. Confirm that execution counts and outputs are stripped where required.

---

## 15. Testing

Tests are stored in:

```text
tests/
```

Run tests using:

```bash
pytest tests/
```

All tests should pass before opening a pull request.

New reusable functionality should include appropriate unit tests.

---

## 16. Code Quality

Run the linter before creating a pull request:

```bash
ruff check .
```

Check formatting:

```bash
ruff format --check .
```

Fix formatting when necessary:

```bash
ruff format .
```

Pre-commit hooks should also be installed:

```bash
pre-commit install
```

Run all hooks manually:

```bash
pre-commit run --all-files
```

---

## 17. Secrets and Sensitive Information

Never commit:

* API keys
* Passwords
* Tokens
* Private keys
* `.env` files
* Cloud credentials
* Database credentials

Use environment variables or local configuration files for secrets.

Before committing:

```bash
git status
```

Review changed files carefully.

If a secret is accidentally committed, notify the team immediately and rotate/revoke the exposed credential.

---

## 18. CI Requirements

Every pull request into `dev`, `staging`, or `main` must pass the required CI checks.

CI should verify:

1. Ruff linting
2. Ruff formatting
3. Unit tests
4. Data checks
5. Smoke training

A pull request should not be merged while required CI checks are failing.

---

## 19. Merge Conflict Resolution

If two branches modify the same part of a file, a merge conflict may occur.

First update your branch:

```bash
git fetch origin
git rebase origin/dev
```

Resolve the conflict manually.

Then:

```bash
git add <resolved-file>
git rebase --continue
```

Run the tests again:

```bash
pytest tests/
```

Run the pipeline if required:

```bash
dvc repro
```

Then push the updated branch.

The conflict resolution should be documented in the relevant pull request.

---

## 20. Merge Strategy

The team will use **squash merging** for pull requests into `dev`.

This keeps the integration branch history clean by combining the commits from a pull request into a single commit.

Before merging, the reviewer must confirm:

* The change is understood.
* CI passes.
* Tests pass.
* The PR checklist is complete.
* Required documentation is included.
* No secrets or large files were added.
* DVC changes were pushed when applicable.

---

## 21. Release Process

A release follows:

```text
dev
 ↓
staging
 ↓
main
```

### Step 1 — Release PR

Create a pull request:

```text
dev → staging
```

Use the title:

```text
release: v1.0
```

Include:

* Included changes
* Final model metrics
* Experiment comparison
* Relevant PRs

### Step 2 — Reproducibility Test

A team member who did not train the final model should perform a fresh-clone test:

```bash
git clone <repository-url>
cd <repository>
git checkout staging
```

Then install dependencies:

```bash
uv sync
```

or:

```bash
pip install -r requirements.txt
```

Retrieve the data:

```bash
dvc pull
```

Reproduce the pipeline:

```bash
dvc repro
```

The resulting metrics must match the reported metrics.

### Step 3 — Production Release

After staging has been validated, create:

```text
staging → main
```

After approval and successful CI:

```bash
git checkout main
git pull
```

Create the release tag:

```bash
git tag -a model-v1.0 -m "First production model"
```

Push the tag:

```bash
git push origin model-v1.0
```

---

## 22. Hotfix Process

For urgent production issues:

```bash
git checkout main
git pull
git switch -c fix/<name>
```

After fixing:

```text
fix/<name>
      ↓
     main
      ↓
model-v1.0.1
      ↓
     dev
```

The fix must also be merged back into `dev` so that the production fix is not lost in future releases.

---

## 23. Before Opening a Pull Request

Run the following checks:

```bash
git status
```

Make sure there are no unintended files.

Run:

```bash
pre-commit run --all-files
```

Run tests:

```bash
pytest tests/
```

Check DVC:

```bash
dvc status
```

If data/model files changed:

```bash
dvc push
```

Then commit and push:

```bash
git add .
git commit -m "type: description"
git push
```

---

## 24. Before Merging a Pull Request

The reviewer should verify:

* [ ] The change matches the PR description.
* [ ] Code is understandable.
* [ ] No hardcoded absolute paths exist.
* [ ] No data leakage exists.
* [ ] Tests pass.
* [ ] CI passes.
* [ ] No secrets are committed.
* [ ] No large files are committed directly.
* [ ] DVC is correctly updated if required.
* [ ] Seeds are set where randomness is used.
* [ ] Metrics are calculated correctly.
* [ ] Notebook changes are reproducible.
* [ ] Documentation is updated where necessary.

---

## 25. Team Responsibilities

Although each member has an assigned ownership area, everyone is expected to contribute to the project.

### Data Owner

Responsible for:

* Dataset management
* DVC
* Data validation
* Dataset updates
* Data versioning

### Model Owner

Responsible for:

* Training pipeline
* Model configuration
* Experiments
* Model evaluation
* Reproducibility

### Platform Owner

Responsible for:

* CI
* Pre-commit
* Environment configuration
* GitHub workflows
* Release process

For teams of two, platform responsibilities should be shared.

---

## 26. Golden Rules

1. Never push directly to `dev`, `staging`, or `main`.
2. Always use pull requests.
3. Always review teammates' pull requests.
4. Never commit datasets directly to Git.
5. Use DVC for dataset versioning.
6. Run `dvc push` before `git push` when DVC-tracked data changes.
7. Never commit secrets.
8. Do not run experiments on uncommitted code.
9. Keep experiments reproducible.
10. Use meaningful Conventional Commit messages.
11. Run tests before opening a PR.
12. Make sure CI passes before merging.
13. Keep branches short-lived.
14. Document important conflicts and decisions.
15. Preserve reproducibility from data to final model.

---

## 27. Project Workflow Summary

```text
                    ┌──────────────┐
                    │     main     │
                    │  Production  │
                    └──────▲───────┘
                           │
                         PR + CI
                           │
                    ┌──────┴───────┐
                    │   staging    │
                    │Release Candidate│
                    └──────▲───────┘
                           │
                         PR + CI
                           │
                    ┌──────┴───────┐
                    │     dev      │
                    │ Integration  │
                    └──────▲───────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
       feat/<name>    data/<name>   promoted experiment
             │             │             │
             └─────────────┴─────────────┘
                           │
                         PR + Review
                           │
                          dev

Experiments:

dev
 │
 └── exp/<member>-<idea>
          │
          ├── successful → feat/<name> → dev
          │
          └── unsuccessful → abandoned
```

This workflow is designed to satisfy the collaboration, branching, DVC, experimentation, review, CI, and release requirements of the assignment.
