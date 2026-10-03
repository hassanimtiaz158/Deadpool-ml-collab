"""Tests for params loading, seeding and metric reporting."""

from __future__ import annotations

import numpy as np
import pytest

from fruit_classification import config
from fruit_classification.modeling.models import MODELS, build_model
from fruit_classification.modeling.train import holdout_split


class TestLoadParams:
    def test_reads_params_yaml(self):
        params = config.load_params()
        assert "seed" in params
        assert "train" in params
        assert "features" in params

    def test_explicit_path_overrides(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text("seed: 7\n", encoding="utf-8")
        assert config.load_params(path)["seed"] == 7

    def test_missing_file_returns_empty(self, tmp_path):
        assert config.load_params(tmp_path / "absent.yaml") == {}

    def test_defaults_merge_without_mutating_file(self):
        params = config.params_with_defaults(config.load_params())
        assert params["seed"] == config.DEFAULT_SEED
        assert "image_size" in params["features"]
        assert "n_estimators" in params["train"]

    def test_user_values_win_over_defaults(self):
        params = config.params_with_defaults({"seed": 123, "train": {"model": "mlp"}})
        assert params["seed"] == 123
        assert params["train"]["model"] == "mlp"
        # untouched defaults survive the merge
        assert params["train"]["n_estimators"] == 200

    def test_committed_params_are_valid_for_the_pipeline(self):
        params = config.params_with_defaults(config.load_params())
        assert params["train"]["model"] in MODELS
        assert params["features"]["hog_bins"] > 0
        assert 0.0 < params["split"]["test_size"] < 1.0


class TestHoldoutSplit:
    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6)).astype(np.float32)
        y = np.repeat(np.arange(4), 50)
        return X, y

    def test_split_shapes(self, data):
        X, y = data
        X_tr, X_val, y_tr, y_val = holdout_split(X, y, 0.2, seed=42)
        assert X_tr.shape[0] == 160
        assert X_val.shape[0] == 40
        assert y_tr.shape[0] == 160 and y_val.shape[0] == 40

    def test_is_deterministic_for_a_fixed_seed(self, data):
        X, y = data
        a = holdout_split(X, y, 0.25, seed=42)
        b = holdout_split(X, y, 0.25, seed=42)
        assert np.array_equal(a[3], b[3])

    def test_different_seed_changes_membership(self, data):
        X, y = data
        a = holdout_split(X, y, 0.25, seed=42)
        b = holdout_split(X, y, 0.25, seed=7)
        assert not np.array_equal(a[3], b[3])

    def test_no_overlap_between_splits(self, data):
        X, y = data
        X_tr, X_val, _, _ = holdout_split(X, y, 0.2, seed=42)
        assert not np.array_equal(X_tr, X_val)
        assert len(X_tr) + len(X_val) == len(X)

    def test_stratified_split_keeps_class_balance(self, data):
        _, y = data
        _, _, y_tr, y_val = holdout_split(*data, 0.2, seed=42, stratify=True)
        assert np.bincount(y_tr).tolist() == [40, 40, 40, 40]
        assert np.bincount(y_val).tolist() == [10, 10, 10, 10]

    def test_handles_single_class(self):
        X = np.zeros((10, 3), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        _, _, _, y_val = holdout_split(X, y, 0.2, seed=42)
        assert len(y_val) == 2


class TestBuildModel:
    @pytest.mark.parametrize("name", MODELS)
    def test_builds_every_supported_model(self, name):
        cfg = {"model": name}
        if name in {"sgd_svm", "mlp"}:
            cfg["max_iter"] = 2
        if name in {"random_forest", "extra_trees"}:
            cfg["n_estimators"] = 3
        model = build_model(cfg, seed=42)
        assert hasattr(model, "fit") and hasattr(model, "predict")

    def test_unknown_model_raises_with_help(self):
        with pytest.raises(ValueError, match="expected one of"):
            build_model({"model": "not_a_model"}, seed=42)

    def test_seed_is_propagated(self):
        a = build_model({"model": "sgd_svm", "max_iter": 2}, seed=42)
        b = build_model({"model": "sgd_svm", "max_iter": 2}, seed=42)
        assert a.random_state == b.random_state == 42

    def test_different_seeds_give_different_estimators(self):
        a = build_model({"model": "random_forest", "n_estimators": 3}, seed=1)
        b = build_model({"model": "random_forest", "n_estimators": 3}, seed=2)
        assert a.random_state != b.random_state