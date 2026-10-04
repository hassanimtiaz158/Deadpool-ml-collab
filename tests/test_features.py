"""Unit tests for the reusable feature functions in ``src/fruit_classification``."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fruit_classification.features import (
    color_histogram,
    extract_features,
    feature_dim,
    hog_features,
    to_rgb_array,
)


def _solid(colour: tuple[int, int, int], size: int = 100) -> Image.Image:
    return Image.new("RGB", (size, size), colour)


def _gradient(size: int = 32) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.tile(ramp, (size, 1))


class TestToRgbArray:
    def test_converts_to_rgb(self):
        grey = Image.new("L", (100, 100), 128)
        arr = to_rgb_array(grey, size=32)
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.float32
        assert arr.max() <= 1.0

    def test_resizes_to_requested_size(self):
        arr = to_rgb_array(_solid((10, 200, 30)), size=64)
        assert arr.shape == (64, 64, 3)

    def test_normalises_pixel_range(self):
        arr = to_rgb_array(_solid((255, 255, 255)), size=16)
        assert np.allclose(arr, 1.0)


class TestHogFeatures:
    def test_shape_matches_grid(self):
        vec = hog_features(_gradient(), cells=4, bins=9)
        assert vec.shape == (4 * 4 * 9,)

    def test_values_are_finite_and_non_negative(self):
        vec = hog_features(_gradient(), cells=4, bins=9)
        assert np.isfinite(vec).all()
        assert (vec >= 0).all()

    def test_flat_image_has_no_edge_energy(self):
        flat = np.full((32, 32), 0.5, dtype=np.float32)
        vec = hog_features(flat, cells=4, bins=9)
        assert np.allclose(vec, 0.0, atol=1e-6)

    def test_vertical_edge_populates_gradient_orientations(self):
        img = np.zeros((32, 32), dtype=np.float32)
        img[:, 16:] = 1.0
        vec = hog_features(img, cells=4, bins=9).reshape(4, 4, 9)
        # A vertical edge is a horizontal gradient (angle 0) -> bins around the
        # centre of the unsigned 0..pi range.
        assert vec.sum() > 0
        assert vec[:, :, 3:6].sum() > 0.9 * vec.sum()

    def test_horizontal_edge_uses_different_bins_than_vertical(self):
        vertical = np.zeros((32, 32), dtype=np.float32)
        vertical[:, 16:] = 1.0
        horizontal = np.zeros((32, 32), dtype=np.float32)
        horizontal[16:, :] = 1.0
        v = hog_features(vertical, 4, 9)
        h = hog_features(horizontal, 4, 9)
        assert not np.allclose(v, h)

    def test_per_cell_normalisation(self):
        vec = hog_features(_gradient(), cells=2, bins=9).reshape(2, 2, 9)
        norms = np.linalg.norm(vec, axis=-1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            hog_features(np.zeros((8, 8, 3), dtype=np.float32))

    def test_rejects_indivisible_shape(self):
        with pytest.raises(ValueError):
            hog_features(np.zeros((30, 30), dtype=np.float32), cells=4)

    def test_is_deterministic(self):
        g = _gradient()
        assert np.array_equal(hog_features(g, 4, 9), hog_features(g, 4, 9))


class TestColorHistogram:
    def test_shape(self):
        hist = color_histogram(to_rgb_array(_solid((200, 10, 10)), size=32))
        assert hist.shape == (8 * 4 * 4,)

    def test_sums_to_one(self):
        hist = color_histogram(to_rgb_array(_solid((200, 10, 10)), size=32))
        assert hist.sum() == pytest.approx(1.0, abs=1e-5)

    def test_red_image_lands_in_red_hue_bin(self):
        hist = color_histogram(to_rgb_array(_solid((255, 0, 0)), size=32)).reshape(8, 4, 4)
        assert hist[0].sum() == pytest.approx(1.0, abs=1e-5)

    def test_greyscale_has_low_saturation(self):
        hist = color_histogram(to_rgb_array(_solid((128, 128, 128)), size=32)).reshape(8, 4, 4)
        assert hist[:, 0, :].sum() == pytest.approx(1.0, abs=1e-5)


class TestExtractFeatures:
    def test_dimension_matches_declared(self):
        vec = extract_features(_solid((120, 30, 200)), image_size=32, cells=4, bins=9)
        assert vec.shape == (feature_dim(image_size=32, cells=4, bins=9),)

    def test_color_histogram_can_be_disabled(self):
        vec = extract_features(_solid((120, 30, 200)), use_color=False)
        assert vec.shape == (4 * 4 * 9,)

    def test_returns_float32(self):
        assert extract_features(_solid((1, 2, 3))).dtype == np.float32

    def test_accepts_palette_and_greyscale_images(self):
        for img in (
            Image.new("L", (100, 100), 200),
            Image.new("P", (100, 100)),
            Image.new("RGBA", (100, 100), (10, 20, 30, 255)),
        ):
            assert np.isfinite(extract_features(img)).all()

    def test_deterministic_across_calls(self):
        img = _solid((77, 88, 99))
        assert np.array_equal(extract_features(img), extract_features(img))

    def test_distinguishes_red_from_blue(self):
        red = extract_features(_solid((255, 0, 0)))
        blue = extract_features(_solid((0, 0, 255)))
        assert not np.allclose(red, blue)
