"""PSNR measurement using both scikit-image and the plan formula."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from metrics.pairs import ImagePair, load_rgb_arrays


PSNR_PARAMETERS = {
    "resize": "lanczos",
    "dtype": "uint8",
    "data_range": 255,
}


def psnr_formula(original: np.ndarray, generated: np.ndarray) -> float:
    if original.shape != generated.shape:
        raise ValueError(f"PSNR array shapes differ: {original.shape} != {generated.shape}")
    difference = original.astype(np.float64) - generated.astype(np.float64)
    mean_squared_error = float(np.mean(np.square(difference), dtype=np.float64))
    if mean_squared_error == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mean_squared_error)


def compute_psnr(pairs: Sequence[ImagePair]) -> dict[str, float | list[float]]:
    if not pairs:
        raise ValueError("PSNR requires at least one image pair")
    values: list[float] = []
    for pair in pairs:
        original, generated = load_rgb_arrays(pair)
        library_value = float(
            peak_signal_noise_ratio(original, generated, data_range=255)
        )
        direct_value = psnr_formula(original, generated)
        if not (
            math.isinf(library_value)
            and math.isinf(direct_value)
            or abs(library_value - direct_value) <= 1e-6
        ):
            raise RuntimeError(
                "PSNR implementations disagree for "
                f"{pair.item_id}: skimage={library_value}, formula={direct_value}"
            )
        values.append(library_value)
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value_array.mean()),
        "std": float(value_array.std()),
        "per_image": values,
    }
