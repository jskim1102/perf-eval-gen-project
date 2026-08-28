"""Measure one generated run and write the canonical results.json."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from metrics.fid import (
    configure_measurement_determinism,
    measurement_determinism_state,
)
from metrics.pairs import GENERATED_FIELDS, PROMPT_FIELD, load_image_pairs
from metrics.psnr import PSNR_PARAMETERS, compute_psnr
from metrics.schema import validate_results
from metrics.ssim import SSIM_PARAMETERS, compute_ssim


TARGETS = {"psnr": 25.0, "ssim": 0.9}
STEPS = 30
GUIDANCE = 5.0
SCHEDULER = "EulerDiscreteScheduler"
OUTPUT_DESCRIPTION = "1024x1024 PNG"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_generated_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"generated manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if fields not in (GENERATED_FIELDS, [*GENERATED_FIELDS, PROMPT_FIELD]):
            raise ValueError(f"generated.csv fields differ: {fields}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"measurement requires at least two generated images: {len(rows)}")
    return rows


def _package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def _driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    driver = completed.stdout.splitlines()[0].strip()
    if not driver:
        raise RuntimeError("nvidia-smi returned an empty driver version")
    return driver


def runtime_environment() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to record the measurement environment")
    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("torch did not report a CUDA runtime version")
    return {
        "python": platform.python_version(),
        "torch": _package_version("torch"),
        "diffusers": _package_version("diffusers"),
        "cleanfid": _package_version("clean-fid"),
        "skimage": _package_version("scikit-image"),
        "gpu": torch.cuda.get_device_name(0),
        "driver": _driver_version(),
        "cuda": cuda_version,
        "determinism": measurement_determinism_state(),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".results.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def measure_run(
    *,
    run_id: str,
    runs_root: Path,
    input_manifest: Path,
    pair_manifest: Path,
    strength: float,
    seed: int,
    model: str,
    baseline_ref: str | None = None,
    note: str | None = None,
    prompt_protocol: dict[str, Any] | None = None,
    require_all_pairs: bool = True,
    created_at: str | None = None,
) -> dict[str, Any]:
    configure_measurement_determinism(seed=0)
    runs_root = runs_root.expanduser().absolute()
    run_root = runs_root / run_id
    generated_manifest = run_root / "generated.csv"
    generated_rows = _read_generated_manifest(generated_manifest)
    pairs = load_image_pairs(
        pair_manifest,
        generated_manifest,
        require_all=require_all_pairs,
    )

    psnr_result = compute_psnr(pairs)
    ssim_result = compute_ssim(pairs)
    protocol: dict[str, Any] = {
        "seed": seed,
        "n_input": len(generated_rows),
        "n_pairs": len(pairs),
        "strength": strength,
        "model": model,
        "steps": STEPS,
        "guidance": GUIDANCE,
        "scheduler": SCHEDULER,
        "output": OUTPUT_DESCRIPTION,
        "psnr": dict(PSNR_PARAMETERS),
        "ssim": dict(SSIM_PARAMETERS),
    }
    if note is not None:
        protocol["note"] = note
    if prompt_protocol is not None:
        protocol["prompt_protocol"] = dict(prompt_protocol)

    results: dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "protocol": protocol,
        "env": runtime_environment(),
        "inputs": {
            "input_manifest_sha256": _sha256(input_manifest),
            "generated_manifest_sha256": _sha256(generated_manifest),
        },
        "metrics": {
            "psnr": psnr_result,
            "ssim": ssim_result,
        },
        "pairs": [
            {
                "item_id": pair.item_id,
                "group": pair.group,
                "input_path": str(pair.input_path),
                "output_path": str(pair.output_path),
            }
            for pair in pairs
        ],
        "targets": dict(TARGETS),
        "baseline_ref": baseline_ref,
    }
    validate_results(
        results,
        eval_root=input_manifest.expanduser().absolute().parent.parent,
        runs_root=runs_root,
        require_determinism=True,
    )
    _atomic_write_json(run_root / "results.json", results)
    return results
