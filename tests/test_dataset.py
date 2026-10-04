"""Tests for dataset indexing and split helpers (no dataset download required)."""

from __future__ import annotations

import pandas as pd
import pytest

from fruit_classification import config, dataset


class TestParseMember:
    @pytest.mark.parametrize(
        "member,expected",
        [
            ("fruits-360/Training/Apple 1/r0_1_100.jpg", ("Training", "Apple 1")),
            ("fruits-360/Test/Apple 1/r0_1_100.jpg", ("Test", "Apple 1")),
        ],
    )
    def test_valid_members(self, member, expected):
        assert dataset.parse_member(member)[:2] == expected

    @pytest.mark.parametrize(
        "member",
        [
            "fruits-360/",
            "fruits-360/Training/Apple 1/",
            "fruits-360/Training/Apple 1/r0_1_100.png",
            "fruits-360/Validation/Apple 1/r0_1_100.jpg",
            "fruits-360/Apple 1/r0_1_100.jpg",
        ],
    )
    def test_rejected_members(self, member):
        assert dataset.parse_member(member) is None

    def test_keeps_full_member_path(self):
        member = "fruits-360/Test/Apple 1/r0_1_100.jpg"
        assert dataset.parse_member(member)[2] == member


class TestBuildIndex:
    def test_missing_dataset_raises_with_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dvc pull"):
            dataset.build_index(tmp_path / "nope.zip")

    def test_empty_archive_raises(self, tmp_path):
        import zipfile

        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("fruits-360/", "")
        with pytest.raises(ValueError, match="no images matched"):
            dataset.build_index(path)

    def test_builds_stable_label_ids(self, tmp_path):
        index = dataset.build_index(_tiny_archive(tmp_path))
        assert list(index.columns) == ["split", "label", "member", "label_id"]
        assert set(index["split"]) == {"Training", "Test"}
        # Label ids are sorted by class name, not ZIP member order.
        assert index.sort_values("label_id")["label"].iloc[0] == "Apple 1"
        assert dataset.INDEX_COLUMNS  # documented column contract

    def test_reports_class_missing_from_test(self, tmp_path):
        index = dataset.build_index(_tiny_archive(tmp_path))
        assert index.attrs["classes_missing_from_test"] == ["BlackBerry 4"]

    def test_class_labels_round_trips(self, tmp_path):
        index = dataset.build_index(_tiny_archive(tmp_path))
        labels = dataset.class_labels(index)
        assert len(labels) == index["label"].nunique()
        assert labels[index["label_id"].iloc[0]] == index["label"].iloc[0]

    def test_split_filter_is_respected(self, tmp_path):
        index = dataset.build_index(_tiny_archive(tmp_path), splits=("Training",))
        assert set(index["split"]) == {"Training"}


class TestSelectSubset:
    @pytest.fixture()
    def index(self) -> pd.DataFrame:
        rows = []
        for label_id in range(5):
            for i in range(10):
                rows.append(
                    {
                        "split": "Training",
                        "label": f"class_{label_id}",
                        "member": f"fruits-360/Training/class_{label_id}/img_{i}.jpg",
                        "label_id": label_id,
                    }
                )
        return pd.DataFrame(rows)

    def test_caps_images_per_class(self, index):
        sub = dataset.select_subset(index, "Training", max_per_class=3)
        assert len(sub) == 15
        assert sub.groupby("label_id").size().eq(3).all()

    def test_limits_class_count(self, index):
        sub = dataset.select_subset(index, "Training", classes_limit=2)
        assert sub["label_id"].nunique() == 2
        assert sub["label_id"].unique().tolist() == [0, 1]

    def test_is_deterministic(self, index):
        a = dataset.select_subset(index, "Training", max_per_class=4)
        b = dataset.select_subset(index, "Training", max_per_class=4)
        assert a["member"].tolist() == b["member"].tolist()

    def test_filters_split(self, index):
        assert dataset.select_subset(index, "Test").empty

    def test_no_filters_returns_all(self, index):
        assert len(dataset.select_subset(index, "Training")) == len(index)


class TestPaths:
    def test_paths_are_repo_relative(self):
        for path in (
            config.DATA_DIR,
            config.RAW_DIR,
            config.INTERIM_DIR,
            config.MODELS_DIR,
            config.REPORTS_DIR,
        ):
            assert path.is_absolute()
            assert "Deadpool-ml-collab" in str(path)

    def test_zip_path_lives_in_raw(self):
        assert config.DATASET_ZIP.parent == config.RAW_DIR
        assert config.DATASET_ZIP.suffix == ".zip"


def _tiny_archive(tmp_path):
    """Build a 3-class archive that reproduces the real dataset's label typo."""
    import zipfile

    path = tmp_path / "tiny.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for split in ("Training", "Test"):
            archive.writestr(f"fruits-360/{split}/", "")
            names = (
                ["Apple 1", "Banana", "BlackBerry 4"]
                if split == "Training"
                else ["Apple 1", "Banana", "Blackberry 4"]
            )
            for name in names:
                archive.writestr(f"fruits-360/{split}/{name}/", "")
                for i in range(3):
                    archive.writestr(
                        f"fruits-360/{split}/{name}/r0_{i}_100.jpg", b"\xff\xd8\xff\xd9"
                    )
    return path
