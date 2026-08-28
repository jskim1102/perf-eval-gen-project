"""Clean-FID wrapper with the evaluation protocol fixed in code."""

from __future__ import annotations

from pathlib import Path

import torch
from cleanfid import fid as clean_fid


FID_PARAMETERS = {
    "impl": "clean-fid",
    "mode": "clean",
    "feature": "inception_v3",
    "input": "299x299 RGB",
    "resize": "bicubic+antialias",
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def configure_measurement_determinism(*, seed: int = 0) -> None:
    """Fix every torch switch that can change the measurement result."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def measurement_determinism_state() -> dict[str, bool | int]:
    """Read back the active measurement settings from torch itself."""

    return {
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_seed": int(torch.initial_seed()),
    }


def _image_count(directory: Path) -> int:
    if not directory.is_dir():
        raise ValueError(f"FID image directory does not exist: {directory}")
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def compute_fid(
    real_directory: Path, generated_directory: Path
) -> tuple[float, dict[str, str]]:
    """Compute FID using clean-fid's immutable clean Inception-v3 protocol."""

    configure_measurement_determinism(seed=0)

    real_directory = real_directory.expanduser().absolute()
    generated_directory = generated_directory.expanduser().absolute()
    real_count = _image_count(real_directory)
    generated_count = _image_count(generated_directory)
    if real_count < 2:
        raise ValueError(f"FID requires at least two images, got {real_count}")
    if real_count != generated_count:
        raise ValueError(
            "FID reconstruction sets must have equal size: "
            f"input={real_count}, generated={generated_count}"
        )

    value = clean_fid.compute_fid(
        str(real_directory),
        str(generated_directory),
        mode="clean",
        model_name="inception_v3",
        num_workers=0,
        batch_size=32,
        device="cuda",
        verbose=False,
        use_dataparallel=False,
    )
    parameters = {
        **FID_PARAMETERS,
        "real_set": f"input x{real_count}",
        "gen_set": f"reconstructed y{generated_count}",
    }
    return float(value), parameters
