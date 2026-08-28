#!/usr/bin/env python3
"""Verify generated images and reject copy/re-encoded pass-through outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generate import (
    GENERATED_FIELDS,
    IMAGE_SIZE,
    PROMPT_FIELD,
    file_sha256,
    validate_run_id,
)


FOREGROUND_WHITE_CUTOFF = 245
FOREGROUND_CHANGE_DELTA = 3.0
MIN_FOREGROUND_PIXELS = 1024
MIN_FOREGROUND_MEAN_ABS_DIFF = 3.0
MIN_FOREGROUND_CHANGED_FRACTION = 0.25
GENERATION_SOURCE = Path(__file__).with_name("generate.py")


class GeneratedVerificationError(ValueError):
    """A generated run violates its structural or anti-pass-through contract."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedVerificationError(message)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_manifest(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            _check(
                fields in (GENERATED_FIELDS, [*GENERATED_FIELDS, PROMPT_FIELD]),
                f"generated.csv fields differ: {fields}",
            )
            rows = list(reader)
    except OSError as exc:
        raise GeneratedVerificationError(f"cannot read {path}: {exc}") from exc
    _check(bool(rows), f"generated.csv has no rows: {path}")
    return rows


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def verify_generation_source_has_no_copy_or_link_path(source_path: Path) -> bool:
    """Reject explicit filesystem copy, hard-link, and symbolic-link routes."""

    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as exc:
        raise GeneratedVerificationError(
            f"cannot inspect generation source {source_path}: {exc}"
        ) from exc
    forbidden_names = {
        "copy",
        "copy2",
        "copyfile",
        "link",
        "symlink",
        "hardlink_to",
        "link_to",
        "symlink_to",
    }
    calls = {
        _call_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    forbidden_calls = sorted(
        name for name in calls if name and name.rsplit(".", 1)[-1] in forbidden_names
    )
    _check(
        not forbidden_calls,
        f"generation source contains a copy/link path: {forbidden_calls}",
    )
    return True


def _pixel_report(input_path: Path, output_path: Path) -> dict[str, float | int]:
    try:
        with Image.open(input_path) as source:
            input_image = source.convert("RGB").resize(
                IMAGE_SIZE, Image.Resampling.LANCZOS
            )
        with Image.open(output_path) as generated:
            output_image = generated.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise GeneratedVerificationError(f"cannot compare image pixels: {exc}") from exc

    input_pixels = np.asarray(input_image, dtype=np.int16)
    output_pixels = np.asarray(output_image, dtype=np.int16)
    absolute_difference = np.abs(output_pixels - input_pixels).astype(np.float32)
    per_pixel_difference = absolute_difference.mean(axis=2)

    # White product-shot backgrounds dominate the frame, so a global equal-pixel
    # fraction would hide a copied foreground. The acceptance threshold is applied
    # only to pixels whose input RGB has at least one channel below 245; background
    # statistics are reported separately as supporting evidence.
    foreground_mask = np.any(input_pixels < FOREGROUND_WHITE_CUTOFF, axis=2)
    background_mask = ~foreground_mask
    foreground_pixels = int(foreground_mask.sum())
    _check(
        foreground_pixels >= MIN_FOREGROUND_PIXELS,
        f"pass-through check has too little foreground: {foreground_pixels} pixels",
    )

    foreground_diff = per_pixel_difference[foreground_mask]
    foreground_mean = float(foreground_diff.mean())
    foreground_changed = float(
        np.mean(foreground_diff >= FOREGROUND_CHANGE_DELTA)
    )
    background_mean = (
        float(per_pixel_difference[background_mask].mean())
        if np.any(background_mask)
        else 0.0
    )

    _check(
        foreground_mean > 0.0,
        "pass-through suspected: foreground mean absolute difference is zero",
    )

    return {
        "global_mean_abs_diff": float(per_pixel_difference.mean()),
        "foreground_pixels": foreground_pixels,
        "foreground_mean_abs_diff": foreground_mean,
        "foreground_changed_fraction": foreground_changed,
        "background_mean_abs_diff": background_mean,
    }


def _distribution(values: list[float], *, threshold: float) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    below_count = int(np.sum(array < threshold))
    return {
        "observation_threshold": threshold,
        "minimum": float(np.min(array)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
        "below_threshold_count": below_count,
        "below_threshold_ratio": float(below_count / len(array)),
    }


def _validation_summary(image_reports: list[dict[str, Any]]) -> dict[str, Any]:
    foreground_mae = [
        float(report["foreground_mean_abs_diff"]) for report in image_reports
    ]
    changed_fraction = [
        float(report["foreground_changed_fraction"]) for report in image_reports
    ]
    return {
        "image_count": len(image_reports),
        "all_output_sha256_differ_from_input": True,
        "all_foreground_mae_positive": all(value > 0.0 for value in foreground_mae),
        "copy_or_link_path_absent": True,
        "foreground_definition": "any input RGB channel < 245",
        "changed_pixel_delta": FOREGROUND_CHANGE_DELTA,
        "foreground_mae": _distribution(
            foreground_mae,
            threshold=MIN_FOREGROUND_MEAN_ABS_DIFF,
        ),
        "foreground_changed_fraction": _distribution(
            changed_fraction,
            threshold=MIN_FOREGROUND_CHANGED_FRACTION,
        ),
    }


def verify_run(*, runs_root: Path, run_id: str, strict: bool) -> dict[str, Any]:
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise GeneratedVerificationError(str(exc)) from exc

    runs_root = runs_root.expanduser().absolute()
    _check(runs_root.is_dir(), f"RUNS_ROOT does not exist: {runs_root}")
    _check(not runs_root.is_symlink(), f"RUNS_ROOT cannot be a symlink: {runs_root}")
    run_root = runs_root / run_id
    images_root = run_root / "images"
    _check(run_root.is_dir(), f"run directory does not exist: {run_root}")
    _check(not run_root.is_symlink(), f"run directory cannot be a symlink: {run_root}")
    _check(images_root.is_dir(), f"images directory does not exist: {images_root}")
    _check(not images_root.is_symlink(), f"images directory cannot be a symlink: {images_root}")

    rows = _read_manifest(run_root / "generated.csv")
    resolved_images_root = images_root.resolve()
    seen_inputs: set[Path] = set()
    seen_outputs: set[Path] = set()
    image_reports: list[dict[str, Any]] = []
    if strict:
        verify_generation_source_has_no_copy_or_link_path(GENERATION_SOURCE)

    for row_number, row in enumerate(rows, start=2):
        label = f"generated.csv:{row_number}"
        input_path = Path(row["input_path"])
        output_path = Path(row["output_path"])
        _check(input_path.is_absolute(), f"{label} input_path must be absolute")
        _check(output_path.is_absolute(), f"{label} output_path must be absolute")
        _check(input_path.is_file(), f"{label} input is missing: {input_path}")
        _check(output_path.is_file(), f"{label} output is missing: {output_path}")
        _check(not output_path.is_symlink(), f"{label} output cannot be a symlink")
        _check(
            _is_relative_to(output_path.resolve(), resolved_images_root),
            f"{label} output is outside the run images directory",
        )
        _check(input_path not in seen_inputs, f"{label} duplicates an input_path")
        _check(output_path not in seen_outputs, f"{label} duplicates an output_path")
        seen_inputs.add(input_path)
        seen_outputs.add(output_path)

        try:
            seed = int(row["seed"])
            strength = float(row["strength"])
        except ValueError as exc:
            raise GeneratedVerificationError(
                f"{label} seed/strength is not numeric"
            ) from exc
        _check(seed >= 0, f"{label} seed must be non-negative")
        _check(0.0 < strength <= 1.0, f"{label} strength is out of range")
        if PROMPT_FIELD in row:
            _check(bool(row[PROMPT_FIELD].strip()), f"{label} prompt must be non-empty")

        actual_hash = file_sha256(output_path)
        _check(
            actual_hash == row["sha256"],
            f"{label} sha256 mismatch: expected {row['sha256']}, got {actual_hash}",
        )
        _check(
            actual_hash != file_sha256(input_path),
            f"{label} output sha256 is identical to input (pass-through)",
        )

        try:
            with Image.open(output_path) as output:
                output_format = output.format
                output_size = output.size
                output_mode = output.mode
                output.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise GeneratedVerificationError(f"{label} output is unreadable: {exc}") from exc
        _check(output_format == "PNG", f"{label} output format must be PNG")
        _check(output_size == IMAGE_SIZE, f"{label} output must be 1024x1024")
        _check(output_mode == "RGB", f"{label} output mode must be RGB")

        report: dict[str, Any] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sha256": actual_hash,
            "seed": seed,
            "strength": strength,
        }
        if strict:
            report.update(_pixel_report(input_path, output_path))
        image_reports.append(report)

    actual_pngs = {path.resolve() for path in images_root.rglob("*.png")}
    expected_pngs = {path.resolve() for path in seen_outputs}
    _check(
        actual_pngs == expected_pngs,
        "images directory PNG set differs from generated.csv",
    )

    result = {
        "run_id": run_id,
        "count": len(rows),
        "strict": strict,
        "anti_pass_through": {
            "foreground_definition": "any input RGB channel < 245",
            "foreground_mae_observation_threshold": MIN_FOREGROUND_MEAN_ABS_DIFF,
            "changed_pixel_delta": FOREGROUND_CHANGE_DELTA,
            "foreground_changed_fraction_observation_threshold": MIN_FOREGROUND_CHANGED_FRACTION,
        },
        "all_model_output_checks_passed": True,
        "images": image_reports,
    }
    if strict:
        result["generation_validation"] = _validation_summary(image_reports)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    report = verify_run(
        runs_root=Path(runs_root_value),
        run_id=args.run_id,
        strict=args.strict,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
