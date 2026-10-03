.PHONY: help requirements create_environment clean lint format test \
        data data-push index check prepare train evaluate repro smoke \
        notebooks dvc-status dvc-pull dvc-push exp-show experiments release-check

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
	rm -rf .pytest_cache .ruff_cache reports/data_checks.json reports/smoke_metrics.json

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

## Commit the current data state to Git (pointer only) and push it to the remote.
## Order matters: dvc push MUST come before git push.
.PHONY: data-push
data-push:
	dvc push
	git push

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

.PHONY: experiments
experiments:
	dvc exp run --set-param train.max_iter=40
	dvc exp run --set-param train.model=logistic_regression train.C=10
	dvc exp run --set-param train.model=linear_svc train.C=0.5
	dvc exp show

.PHONY: exp-show
exp-show:
	dvc exp show


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