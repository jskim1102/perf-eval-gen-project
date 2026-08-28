"""Wang et al. SSIM measured independently per RGB channel."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from skimage.metrics import structural_similarity

from metrics.pairs import ImagePair, load_rgb_arrays


SSIM_PARAMETERS = {
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
    "data_range": 255,
    "channel": "per-channel mean",
}


def compute_ssim(pairs: Sequence[ImagePair]) -> dict[str, float | list[float]]:
    if not pairs:
        raise ValueError("SSIM requires at least one image pair")
    values: list[float] = []
    for pair in pairs:
        original, generated = load_rgb_arrays(pair)
        channel_values = [
            structural_similarity(
                original[:, :, channel],
                generated[:, :, channel],
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=255,
            )
            for channel in range(3)
        ]
        values.append(float(np.mean(channel_values, dtype=np.float64)))
    value_array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value_array.mean()),
        "std": float(value_array.std()),
        "per_image": values,
    }
