#!/usr/bin/env python3
"""Generate selected EVAL500 items and record per-image PSNR and SSIM."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.pairs import ImagePair
from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim
from scripts.generate import GenerationConfig, run_generation, validate_run_id
from scripts.run_eval import load_prompt_manifest_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST = PROJECT_ROOT / "dataset" / "psnr_ssim" / "manifests" / "input.csv"
TRY_TARGETS = {"psnr": 25.0, "ssim": 0.9}
MAX_TRY_ITEMS = 100


def _read_input_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "item_id",
            "group",
            "product_type",
            "image_id",
            "source_path",
            "selected_path",
        }
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            raise ValueError(f"input.csv lacks fields: {sorted(required - fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("input.csv is empty")
    return rows


def _read_generated_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("generated.csv is empty")
    return rows


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".try.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_try(
    *,
    run_id: str,
    item_ids: Sequence[str],
    strength: float,
    runs_root: Path,
    input_manifest: Path = INPUT_MANIFEST,
    prompt_manifest: Path | None = None,
    generation_runner: Callable[..., Any] = run_generation,
) -> dict[str, Any]:
    validate_run_id(run_id)
    if not run_id.startswith("try-"):
        raise ValueError("trial run_id must start with literal 'try-'")
    requested = list(item_ids)
    if not 1 <= len(requested) <= MAX_TRY_ITEMS:
        raise ValueError(f"item_ids must contain 1 to {MAX_TRY_ITEMS} entries")
    if len(requested) != len(set(requested)):
        raise ValueError("item_ids must be unique")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be greater than 0 and at most 1")

    input_manifest = input_manifest.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    rows = _read_input_rows(input_manifest)
    rows_by_id = {row["item_id"]: row for row in rows}
    missing = [item_id for item_id in requested if item_id not in rows_by_id]
    if missing:
        raise ValueError(f"item_ids are not in input.csv: {missing}")

    generation_kwargs: dict[str, Any] = {
        "manifest_path": input_manifest,
        "runs_root": runs_root,
        "config": GenerationConfig(strength=strength, seed=0, run_id=run_id),
        "item_ids": requested,
    }
    if prompt_manifest is not None:
        prompt_contract = load_prompt_manifest_contract(
            input_manifest=input_manifest,
            prompt_manifest=prompt_manifest,
        )
        generation_kwargs["prompt_resolver"] = prompt_contract.resolver
    report = generation_runner(**generation_kwargs)
    generated_by_input = {
        Path(row["input_path"]): Path(row["output_path"])
        for row in _read_generated_rows(report.generated_manifest)
    }
    eval_root = input_manifest.parent.parent
    items: list[dict[str, Any]] = []
    for item_id in requested:
        row = rows_by_id[item_id]
        generated_input = (eval_root / row["selected_path"]).absolute()
        output_path = generated_by_input.get(generated_input)
        if output_path is None:
            raise ValueError(f"generated.csv has no selected item: {item_id}")
        pair = ImagePair(
            item_id=item_id,
            group=row["group"],
            input_path=generated_input,
            output_path=output_path,
        )
        psnr = float(compute_psnr([pair])["per_image"][0])
        ssim = float(compute_ssim([pair])["per_image"][0])
        source_path = Path(row["source_path"])
        if not source_path.is_absolute():
            raise ValueError(f"input.csv source_path must be absolute: {source_path}")
        items.append(
            {
                "item_id": item_id,
                "group": row["group"],
                "product_type": row["product_type"],
                "image_id": row["image_id"],
                "input_path": row["source_path"],
                "output_path": str(output_path),
                "psnr": psnr,
                "ssim": ssim,
            }
        )

    payload = {
        "run_id": run_id,
        "strength": strength,
        "items": items,
        "metrics": {
            "psnr": {"mean": fmean(item["psnr"] for item in items)},
            "ssim": {"mean": fmean(item["ssim"] for item in items)},
        },
        "targets": dict(TRY_TARGETS),
    }
    _atomic_write(runs_root / run_id / "try.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--item-id", action="append", dest="item_ids", required=True)
    parser.add_argument("--manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--prompt-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    payload = run_try(
        run_id=args.run_id,
        item_ids=args.item_ids,
        strength=args.strength,
        runs_root=Path(runs_root_value),
        input_manifest=args.manifest,
        prompt_manifest=args.prompt_manifest,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
