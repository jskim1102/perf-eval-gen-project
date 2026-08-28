#!/usr/bin/env python3
"""Re-run an evaluation and require bit-for-bit identical metric values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_eval import run_evaluation


class ReproductionMismatch(RuntimeError):
    """A fresh run did not reproduce the canonical metrics exactly."""


def _read_result(runs_root: Path, run_id: str) -> dict[str, Any]:
    path = runs_root / run_id / "results.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read results for {run_id}: {exc}") from exc


def _metric_scalars(result: dict[str, Any]) -> dict[str, float]:
    return {
        "psnr_mean": result["metrics"]["psnr"]["mean"],
        "ssim_mean": result["metrics"]["ssim"]["mean"],
    }


def _reproduction_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "psnr": result["metrics"]["psnr"],
        "ssim": result["metrics"]["ssim"],
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compare_runs(*, against: str, run_id: str, runs_root: Path) -> dict[str, Any]:
    runs_root = runs_root.expanduser().absolute()
    expected = _read_result(runs_root, against)
    actual = _read_result(runs_root, run_id)
    expected_scalars = _metric_scalars(expected)
    actual_scalars = _metric_scalars(actual)
    differences = {
        key: {"expected": expected_scalars[key], "actual": actual_scalars[key]}
        for key in expected_scalars
        if expected_scalars[key] != actual_scalars[key]
    }
    expected_metrics_sha256 = _payload_sha256(_reproduction_metrics(expected))
    actual_metrics_sha256 = _payload_sha256(_reproduction_metrics(actual))
    full_metrics_identical = expected_metrics_sha256 == actual_metrics_sha256
    if not full_metrics_identical:
        differences["metrics_payload"] = {
            "expected_sha256": expected_metrics_sha256,
            "actual_sha256": actual_metrics_sha256,
        }
    report: dict[str, Any] = {
        "against": against,
        "run_id": run_id,
        "identical": full_metrics_identical,
        "differences": differences,
        "full_metrics_identical": full_metrics_identical,
        "expected_metrics_sha256": expected_metrics_sha256,
        "actual_metrics_sha256": actual_metrics_sha256,
        "possible_causes": [
            "seed",
            "cuDNN deterministic settings",
            "TF32 settings",
            "GPU or driver",
            "torch/diffusers/scikit-image versions",
        ],
    }
    diff_path = runs_root / run_id / "reproduction-diff.json"
    diff_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report["identical"]:
        raise ReproductionMismatch(
            f"metrics differ; inspect {diff_path} and the listed determinism controls"
        )
    return report


def reproduce(
    *,
    against: str,
    run_id: str,
    runs_root: Path,
    run_runner: Callable[..., Any] = run_evaluation,
) -> dict[str, Any]:
    runs_root = runs_root.expanduser().absolute()
    original = _read_result(runs_root, against)
    run_kwargs: dict[str, Any] = {
        "run_id": run_id,
        "strength": float(original["protocol"]["strength"]),
        "model": str(original["protocol"]["model"]),
        "limit": None,
        "runs_root": runs_root,
    }
    prompt_protocol = original["protocol"].get("prompt_protocol")
    if prompt_protocol is not None:
        run_kwargs["prompt_manifest"] = Path(
            prompt_protocol["manifest_path"]
        ).expanduser().absolute()
    run_runner(
        **run_kwargs,
    )
    return compare_runs(against=against, run_id=run_id, runs_root=runs_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--against", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    runs_root = Path(runs_root_value)
    if args.verify_only:
        report = compare_runs(
            against=args.against,
            run_id=args.run_id,
            runs_root=runs_root,
        )
    else:
        report = reproduce(
            against=args.against,
            run_id=args.run_id,
            runs_root=runs_root,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
