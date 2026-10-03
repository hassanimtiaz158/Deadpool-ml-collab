.PHONY: help requirements create_environment clean lint format test \
        data data-push index check prepare train evaluate repro smoke compare \
        notebooks dvc-status dvc-pull exp-show experiments verify release-check

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME  = fruit_classification
PYTHON_VERSION = 3.10
VENV          = venv
PY            = $(VENV)/Scripts/python.exe
PIP           = $(VENV)/Scripts/pip.exe

# On non-Windows shells fall back to the ambient interpreter.
ifeq ($(OS),Windows_Nt)
    PY  = $(VENV)/Scripts/python.exe
    PIP = $(VENV)/Scripts/pip.exe
else
    PY  = python3
    PIP = $(VENV)/bin/pip
endif

export PYTHONPATH := src


#################################################################################
# ENVIRONMENT                                                                    #
#################################################################################

.PHONY: requirements
requirements:
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev,pipeline]"

.PHONY: create_environment
create_environment:
	python -m venv $(VENV)
	$(MAKE) requirements

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -not -path "./$(VENV)/*" -delete
	find . -type d -name "__pycache__" -not -path "./$(VENV)/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache reports/data_checks.json reports/smoke

.PHONY: lint
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

.PHONY: format
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

.PHONY: test
test:
	$(PY) -m pytest tests/ -v


#################################################################################
# DATA (DVC)                                                                     #
#################################################################################

## Pull the dataset from the DagsHub DVC remote (always run before git pull).
.PHONY: data-pull
data-pull:
	dvc pull

## Push the DVC cache to DagsHub, then push the Git pointers.
## ORDER MATTERS: dvc push MUST complete before git push, otherwise the remote
## holds pointers to objects nobody can pull.
##
## DagsHub needs a token. Either export it once per shell:
##     set DAGSHUB_TOKEN=<your token>        # Windows
##     export DAGSHUB_TOKEN=<your token>     # bash/zsh
## or put it in the remote URL / ~/.config/dvc/config:
##     url = https://<user>:<token>@dagshub.com/hassanimtiaz158/Deadpool-ml-collab.dvc
.PHONY: data-push
data-push:
	dvc push
	git push

.PHONY: dvc-push
dvc-push:
	dvc push

## Rebuild the committed dataset index used by the CI data checks.
.PHONY: index
index:
	$(PY) -m fruit_classification.checks

## Extract and cache image features from the DVC-tracked archive.
.PHONY: prepare
prepare:
	$(PY) -m fruit_classification.prepare

.PHONY: check
check:
	$(PY) -m fruit_classification.checks


#################################################################################
# PIPELINE                                                                       #
#################################################################################

.PHONY: train
train:
	$(PY) -m fruit_classification.modeling.train

.PHONY: evaluate
evaluate:
	$(PY) -m fruit_classification.modeling.evaluate

## Reproduce the full pipeline defined in dvc.yaml.
.PHONY: repro
repro:
	dvc repro

## Fast end-to-end run on the small committed sample (what CI executes).
.PHONY: smoke
smoke:
	$(PY) -m fruit_classification.smoke

.PHONY: dvc-status
dvc-status:
	dvc status

.PHONY: dvc-pull
dvc-pull:
	dvc pull


#################################################################################
# EXPERIMENTS                                                                    #
#################################################################################

## Experiment sweeps. Each `dvc exp run` is a self-contained, reproducible run;
## `dvc exp show` tabulates them against the workspace run. See REPORT.md §6.
##
## These are deliberately cheap: the feature cache means every run here costs
## only the fit. Do NOT add linear_svc / logistic_regression / mlp here - on a
## 266-way problem they take >25 min each on 4 CPU cores.
.PHONY: experiments
experiments:
	dvc exp run --set-param train.n_estimators=300
	dvc exp run --set-param train.max_depth=40
	dvc exp run --set-param train.model=sgd_svm train.alpha=1.0e-6
	dvc exp run --set-param features.hog_cells=6
	dvc exp show

## Train every candidate in the comparison and write reports/model_comparison.csv.
.PHONY: compare
compare:
	$(PY) -m fruit_classification.compare

.PHONY: exp-show
exp-show:
	dvc exp show

## Re-run the winner and assert the metrics are bit-identical to the committed
## reports/metrics.json. This is the check a reviewer runs before a release.
.PHONY: verify
verify:
	dvc repro
	$(PY) -c "import json,pathlib;p=pathlib.Path('reports/metrics.json');print(json.dumps(json.loads(p.read_text())['test'],indent=2))"


#################################################################################
# NOTEBOOKS                                                                      #
#################################################################################

.PHONY: notebooks
notebooks:
	$(PY) -m jupyter lab notebooks/


#################################################################################
# SELF-DOCUMENTING COMMANDS                                                      #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys;
lines = '\n'.join([line for line in sys.stdin]);
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines);
print('Available rules:\n');
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]));
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PY) -c "$${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)