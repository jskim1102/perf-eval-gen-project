from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from metrics.pairs import load_image_pairs, load_rgb_arrays
from metrics.psnr import PSNR_PARAMETERS, compute_psnr, psnr_formula
from metrics.ssim import SSIM_PARAMETERS, compute_ssim


PAIR_FIELDS = [
    "split",
    "group",
    "product_type",
    "item_id",
    "image_id",
    "width",
    "height",
    "sha256",
    "source_path",
    "selected_path",
]
GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_pair_manifests(tmp_path: Path) -> tuple[Path, Path]:
    eval_root = tmp_path / "eval500"
    run_root = tmp_path / "runs" / "metric"
    pair_manifest = eval_root / "manifests" / "psnr_ssim_100.csv"
    generated_manifest = run_root / "generated.csv"
    pair_manifest.parent.mkdir(parents=True)
    (eval_root / "input" / "가구").mkdir(parents=True)
    (run_root / "images" / "가구").mkdir(parents=True)

    pair_rows: list[dict[str, str]] = []
    generated_rows: list[dict[str, str]] = []
    for index, delta in enumerate((3, 12, 24)):
        filename = f"PRODUCT__image-{index}.jpg"
        input_path = eval_root / "input" / "가구" / filename
        output_path = run_root / "images" / "가구" / filename.replace(".jpg", ".png")
        y, x = np.mgrid[:48, :52]
        source = np.stack(
            (
                (x * 3 + index * 15) % 220,
                (y * 4 + index * 9) % 220,
                ((x + y) * 2 + index * 6) % 220,
            ),
            axis=2,
        ).astype(np.uint8)
        Image.fromarray(source, mode="RGB").save(input_path, quality=100)
        with Image.open(input_path) as decoded:
            resized = np.asarray(
                decoded.convert("RGB").resize((40, 40), Image.Resampling.LANCZOS),
                dtype=np.int16,
            )
        generated = np.clip(resized + delta, 0, 255).astype(np.uint8)
        Image.fromarray(generated, mode="RGB").save(output_path, format="PNG")
        pair_rows.append(
            {
                "split": "input",
                "group": "가구",
                "product_type": "PRODUCT",
                "item_id": f"item-{index}",
                "image_id": f"image-{index}",
                "width": "52",
                "height": "48",
                "sha256": _sha256(input_path),
                "source_path": str(input_path),
                "selected_path": f"input/가구/{filename}",
            }
        )
        generated_rows.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "seed": "0",
                "strength": "0.25",
            }
        )

    with pair_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pair_rows)
    # Reverse generated.csv deliberately: pair order must come from the fixed
    # pair manifest, never from a directory or generated-manifest sort.
    with generated_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
        writer.writeheader()
        writer.writerows(reversed(generated_rows))
    return pair_manifest, generated_manifest


def test_direct_psnr_formula_matches_skimage_within_one_e_minus_six() -> None:
    rng = np.random.default_rng(42)
    original = rng.integers(0, 256, size=(32, 36, 3), dtype=np.uint8)
    reconstructed = np.clip(
        original.astype(np.int16) + rng.integers(-10, 11, original.shape),
        0,
        255,
    ).astype(np.uint8)

    direct = psnr_formula(original, reconstructed)
    library = float(
        peak_signal_noise_ratio(original, reconstructed, data_range=255)
    )

    assert abs(direct - library) <= 1e-6


def test_pair_loading_and_metrics_keep_fixed_manifest_order(tmp_path: Path) -> None:
    pair_manifest, generated_manifest = _build_pair_manifests(tmp_path)

    pairs = load_image_pairs(pair_manifest, generated_manifest)
    psnr_result = compute_psnr(pairs)
    ssim_result = compute_ssim(pairs)

    assert [pair.item_id for pair in pairs] == ["item-0", "item-1", "item-2"]
    assert list(psnr_result) == ["mean", "std", "per_image"]
    assert list(ssim_result) == ["mean", "std", "per_image"]
    assert len(psnr_result["per_image"]) == 3
    assert len(ssim_result["per_image"]) == 3
    assert psnr_result["per_image"] == sorted(
        psnr_result["per_image"], reverse=True
    )
    assert ssim_result["per_image"] == sorted(
        ssim_result["per_image"], reverse=True
    )
    assert PSNR_PARAMETERS == {
        "resize": "lanczos",
        "dtype": "uint8",
        "data_range": 255,
    }
    assert SSIM_PARAMETERS == {
        "gaussian_weights": True,
        "sigma": 1.5,
        "use_sample_covariance": False,
        "data_range": 255,
        "channel": "per-channel mean",
    }


def test_original_is_resized_to_generated_size_as_rgb_uint8(tmp_path: Path) -> None:
    pair_manifest, generated_manifest = _build_pair_manifests(tmp_path)
    pair = load_image_pairs(pair_manifest, generated_manifest)[0]

    original, generated = load_rgb_arrays(pair)

    assert original.shape == generated.shape == (40, 40, 3)
    assert original.dtype == generated.dtype == np.uint8
