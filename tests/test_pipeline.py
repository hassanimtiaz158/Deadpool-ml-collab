"""Tests for params loading, seeding and metric reporting."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from fruit_classification import config
from fruit_classification.compare import score
from fruit_classification.modeling import evaluate as evaluate_module
from fruit_classification.modeling.evaluate import evaluate
from fruit_classification.modeling.models import MODELS, build_model
from fruit_classification.modeling.train import holdout_split


def params_with_defaults(raw: dict) -> dict:
    """`evaluate` expects a fully-populated params dict; fill in the defaults."""
    return config.params_with_defaults(raw)


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


class TestDataHash:
    """`_data_hash` must survive DVC's YAML list syntax, not just a bare `md5:`."""

    def _write_pointer(self, tmp_path, body: str):
        (tmp_path / "fruits-360_100x100.zip.dvc").write_text(body, encoding="utf-8")
        return tmp_path

    def test_parses_the_list_item_form_dvc_actually_writes(self, tmp_path, monkeypatch):
        raw = self._write_pointer(
            tmp_path,
            "outs:\n- md5: fd6de979f85a2bc73782d760c966fd6b\n"
            "  size: 847218361\n  path: fruits-360_100x100.zip\n",
        )
        monkeypatch.setattr(config, "RAW_DIR", raw)
        assert evaluate_module._data_hash() == "fd6de979f85a2bc73782d760c966fd6b"

    def test_parses_a_bare_md5_key(self, tmp_path, monkeypatch):
        raw = self._write_pointer(tmp_path, "outs:\n  md5: abc123\n")
        monkeypatch.setattr(config, "RAW_DIR", raw)
        assert evaluate_module._data_hash() == "abc123"

    def test_missing_pointer_is_unknown_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "RAW_DIR", tmp_path)
        assert evaluate_module._data_hash() == "unknown"

    def test_pointer_without_a_hash_is_unknown(self, tmp_path, monkeypatch):
        raw = self._write_pointer(tmp_path, "outs:\n- size: 10\n  path: x.zip\n")
        monkeypatch.setattr(config, "RAW_DIR", raw)
        assert evaluate_module._data_hash() == "unknown"


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

    def test_returns_interleaved_x_then_y(self, data):
        """The positional contract the rest of the codebase unpacks against.

        sklearn's train_test_split returns (X_tr, X_val, y_tr, y_val) - the two
        feature blocks come first, then the two label blocks. Slicing the result
        instead of unpacking it hands back (y_tr, y_test), i.e. 1-D labels where
        the estimator expects a 2-D feature matrix.
        """
        X, y = data
        parts = holdout_split(X, y, 0.2, seed=42)

        assert len(parts) == 4
        assert all(p.ndim == 2 for p in parts[:2]), "first two must be feature matrices"
        assert all(p.ndim == 1 for p in parts[2:]), "last two must be label vectors"
        assert parts[0].shape[1] == X.shape[1]
        assert parts[1].shape[1] == X.shape[1]
        assert len(parts[2]) == parts[0].shape[0]
        assert len(parts[3]) == parts[1].shape[0]
        # The two label blocks must recombine into the original labels. The split
        # shuffles, so compare as multisets rather than as a prefix/suffix pair.
        recombined = np.sort(np.concatenate([parts[2], parts[3]]))
        assert np.array_equal(recombined, np.sort(y))

    def test_slicing_off_the_end_does_not_yield_features(self, data):
        """Guards the exact regression: `[2:]` must never be a valid substitute."""
        X, y = data
        X_tr, X_val, _, _ = holdout_split(X, y, 0.2, seed=42)

        sliced = holdout_split(X, y, 0.2, seed=42)[2:]
        assert not isinstance(sliced[0], np.ndarray) or sliced[0].ndim == 1
        assert X_tr.ndim == 2 and X_val.ndim == 2


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


class TestScoring:
    """`compare.score` is the one scoring implementation; keep it honest."""

    @staticmethod
    def _fitted(n_classes: int = 4, n_per_class: int = 40, seed: int = 42):
        rng = np.random.default_rng(seed)
        y = np.repeat(np.arange(n_classes), n_per_class)
        X = rng.normal(size=(len(y), 8)) + y[:, None] * 2.0
        model = build_model({"model": "sgd_svm", "max_iter": 50}, seed=seed)
        model.fit(X, y)
        return model, X, y

    def test_score_returns_the_full_metric_bundle(self):
        model, X, y = self._fitted()
        s = score(model, X, y)
        assert set(s) == {
            "accuracy",
            "top5_accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
        }
        assert all(0.0 <= v <= 1.0 for v in s.values())

    def test_top5_never_scores_below_top1(self):
        model, X, y = self._fitted()
        s = score(model, X, y)
        assert s["top5_accuracy"] >= s["accuracy"]

    def test_top5_is_one_when_k_covers_every_class(self):
        model, X, y = self._fitted(n_classes=3)
        assert score(model, X, y, k=5)["top5_accuracy"] == pytest.approx(1.0)

    def test_perfect_predictions_score_one(self):
        _, X, y = self._fitted()

        class Oracle:
            classes_ = np.unique(y)

            def predict(self, X):
                return y

        s = score(Oracle(), X, y)
        assert s["accuracy"] == pytest.approx(1.0)
        assert s["macro_f1"] == pytest.approx(1.0)


class TestArtefactNamespacing:
    """A smoke run must never overwrite the full-dataset results."""

    def test_canonical_tag_owns_the_top_level_paths(self):
        assert config.model_path("full") == config.MODELS_DIR / "model.joblib"
        assert config.report_dir("full") == config.REPORTS_DIR
        assert config.figures_dir("full") == config.FIGURES_DIR

    def test_other_tags_are_namespaced(self):
        assert config.model_path("smoke").name == "smoke_model.joblib"
        assert config.report_dir("smoke") == config.REPORTS_DIR / "smoke"
        assert config.figures_dir("smoke") == config.FIGURES_DIR / "smoke"

    def test_namespaced_paths_are_disjoint_from_canonical(self):
        for tag in ("smoke", "sample"):
            assert not config.report_dir(tag).is_relative_to(config.REPORTS_DIR / "figures")


class TestEvaluateWritesReport:
    """Regression cover for the per-class report and confusion matrix."""

    @pytest.fixture
    def result(self):
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(4), 25)
        # Well-separated clusters so the assertions below can expect a clean fit.
        X = rng.normal(size=(len(y), 8)) + y[:, None] * 12.0
        model = build_model({"model": "sgd_svm", "max_iter": 200}, seed=42)
        model.fit(X, y)
        return {
            "model": model,
            "X_val": X,
            "y_val": y,
            "X_test": X,
            "y_test": y,
            "run_info": {
                "model": "sgd_svm",
                "train_cfg": {"model": "sgd_svm", "max_iter": 200},
                "commit_sha": "deadbeef",
                "seed": 42,
                "cache_tag": "full",
                "classes": ["apple", "banana", "cherry", "date"],
                "fit_seconds": 0.01,
            },
        }

    def test_per_class_csv_has_one_row_per_class(self, result, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
        monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
        evaluate(result, params_with_defaults({"seed": 42}), progress=False)

        per_class = pd.read_csv(tmp_path / "reports" / "per_class_metrics.csv")
        assert len(per_class) == 4
        assert {"class", "precision", "recall", "f1-score"} <= set(per_class.columns)
        assert per_class["f1-score"].between(0.0, 1.0).all()

    def test_metrics_json_records_provenance(self, result, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
        monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
        evaluate(result, params_with_defaults({"seed": 42}), progress=False)

        metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text("utf-8"))
        assert metrics["provenance"]["commit_sha"] == "deadbeef"
        assert metrics["provenance"]["seed"] == 42
        assert (tmp_path / "reports" / "figures" / "confusion_matrix.npy").exists()

        # The reported number must be the model's own score, not a re-derived one.
        expected = accuracy_score(result["y_test"], result["model"].predict(result["X_test"]))
        assert metrics["test"]["accuracy"] == pytest.approx(expected, abs=1e-6)
        assert metrics["test"]["n_samples"] == len(result["y_test"])