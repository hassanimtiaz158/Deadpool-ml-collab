"""Inference helper: load the serialised model and classify images."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from fruit_classification import config, dataset
from fruit_classification.features import extract_features

DEFAULT_MODEL = config.model_path()


def load_model(path: str | Path = DEFAULT_MODEL):
    """Load the trained estimator from a joblib artefact.

    ``train`` writes a ``{"model": ..., "run_info": ...}`` bundle so that
    ``evaluate`` can report provenance without refitting. A bare estimator is
    also accepted, which keeps older artefacts loadable.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no trained model at {path}. Run `dvc repro` first.")
    bundle = joblib.load(path)
    return bundle["model"] if isinstance(bundle, dict) else bundle


def predict_images(
    images: list[Image.Image],
    model=None,
    feature_cfg: dict | None = None,
) -> list[str]:
    """Return the predicted class name for each PIL image."""
    model = model if model is not None else load_model()
    feature_cfg = config.params_with_defaults().get("features") if feature_cfg is None else feature_cfg
    labels = dataset.class_labels(dataset.build_index())
    X = np.vstack(
        [
            extract_features(
                img,
                image_size=feature_cfg["image_size"],
                cells=feature_cfg["hog_cells"],
                bins=feature_cfg["hog_bins"],
                use_color=feature_cfg.get("color_hist", True),
            )
            for img in images
        ]
    )
    return [labels[i] for i in model.predict(X)]


def predict_from_zip(members: list[str], model=None, feature_cfg: dict | None = None) -> list[str]:
    """Classify images identified by their path inside the raw dataset ZIP."""
    with zipfile.ZipFile(config.DATASET_ZIP) as archive:
        images = []
        for member in members:
            with archive.open(member) as handle:
                img = Image.open(handle)
                img.load()
            images.append(img.convert("RGB"))
    return predict_images(images, model=model, feature_cfg=feature_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a few dataset images")
    parser.add_argument("--n", type=int, default=5, help="how many sample images to score")
    args = parser.parse_args()

    index = dataset.build_index()
    sample = index[index["split"] == dataset.SPLIT_TEST].head(args.n)
    preds = predict_from_zip(sample["member"].tolist())
    rows = [
        {"member": m, "true": t, "predicted": p, "correct": t == p}
        for m, t, p in zip(sample["member"], sample["label"], preds)
    ]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()