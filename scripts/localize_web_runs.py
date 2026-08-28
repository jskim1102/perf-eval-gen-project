#!/usr/bin/env python3
"""Copy the canonical V2 web runs below the project dataset root."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_web_datasets import (
    _atomic_copy,
    _atomic_write_csv,
    _atomic_write_json,
    _prepare_destination,
    _read_csv,
    _read_json,
    _safe_relative,
    _sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUNS_ROOT = Path("/home/kim_3090/datasets/abo/perf-eval-gen-runs")
DEFAULT_DESTINATION_RUNS_ROOT = PROJECT_ROOT / "dataset" / "runs"
DEFAULT_FID_DATASET_ROOT = PROJECT_ROOT / "dataset" / "fid"
DEFAULT_PSNR_SSIM_DATASET_ROOT = PROJECT_ROOT / "dataset" / "psnr_ssim"
GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength"]
PROMPT_FIELD = "prompt"


@dataclass(frozen=True)
class LocalizedRunReport:
    run_root: Path
    count: int
    generated_manifest_sha256: str


def _generated_by_name(path: Path) -> dict[str, dict[str, str]]:
    fields, rows = _read_csv(path)
    if fields not in (GENERATED_FIELDS, [*GENERATED_FIELDS, PROMPT_FIELD]):
        raise ValueError(f"generated manifest fields differ: {fields}")
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        name = Path(row["input_path"]).name
        if name in by_name:
            raise ValueError(f"duplicate generated input filename: {name}")
        by_name[name] = row
    return by_name


def _localized_generated_row(
    *,
    source_row: dict[str, str],
    input_path: Path,
    output_path: Path,
    expected_prompt: str | None,
) -> dict[str, str]:
    source_output = Path(source_row["output_path"])
    if _sha256(source_output) != source_row["sha256"]:
        raise ValueError(f"source generated output SHA-256 differs: {source_output}")
    _atomic_copy(source_output, output_path)
    if _sha256(output_path) != source_row["sha256"]:
        raise ValueError(f"localized generated output SHA-256 differs: {output_path}")
    localized = {
        "input_path": str(input_path.absolute()),
        "output_path": str(output_path.absolute()),
        "sha256": source_row["sha256"],
        "seed": source_row["seed"],
        "strength": source_row["strength"],
    }
    if expected_prompt is not None:
        if source_row.get(PROMPT_FIELD) != expected_prompt:
            raise ValueError(f"source generated prompt differs for {input_path.name}")
        localized[PROMPT_FIELD] = expected_prompt
    return localized


def localize_fid_run(
    *,
    source_runs_root: Path = DEFAULT_SOURCE_RUNS_ROOT,
    destination_runs_root: Path = DEFAULT_DESTINATION_RUNS_ROOT,
    dataset_root: Path = DEFAULT_FID_DATASET_ROOT,
    run_id: str = "fid500-v2",
    expected_count: int = 500,
) -> LocalizedRunReport:
    source_runs_root = source_runs_root.expanduser().absolute()
    destination_runs_root = _prepare_destination(destination_runs_root)
    dataset_root = dataset_root.expanduser().absolute()
    destination_run = _prepare_destination(destination_runs_root / run_id)
    _, input_rows = _read_csv(dataset_root / "manifests" / "input.csv")
    if len(input_rows) != expected_count:
        raise ValueError(f"localized FID input count differs from {expected_count}")
    source_generated = _generated_by_name(
        source_runs_root / run_id / "generated.csv"
    )
    localized_rows: list[dict[str, str]] = []
    for row in input_rows:
        relative = _safe_relative(row["selected_path"], label="FID selected_path")
        input_path = dataset_root / relative
        try:
            source_row = source_generated[input_path.name]
        except KeyError as exc:
            raise ValueError(f"FID source run is missing {input_path.name}") from exc
        output_relative = Path(*relative.parts[1:]).with_suffix(".png")
        localized_rows.append(
            _localized_generated_row(
                source_row=source_row,
                input_path=input_path,
                output_path=destination_run / "images" / output_relative,
                expected_prompt=source_row.get(PROMPT_FIELD),
            )
        )
    generated_path = destination_run / "generated.csv"
    _atomic_write_csv(
        generated_path,
        fields=[*GENERATED_FIELDS, PROMPT_FIELD],
        rows=localized_rows,
    )

    source_result = _read_json(source_runs_root / run_id / "fid500.json")
    dataset = source_result.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("FID result lacks dataset contract")
    generation_manifest = dataset_root / "manifests" / "input.csv"
    dataset.update(
        {
            "root": str(dataset_root),
            "input_directory": str((dataset_root / "input").absolute()),
            "selection_manifest": str((dataset_root / "manifest.csv").absolute()),
            "generation_manifest": str(generation_manifest.absolute()),
            "generation_manifest_sha256": _sha256(generation_manifest),
            "generated_manifest_sha256": _sha256(generated_path),
        }
    )
    method_details = dataset.get("selection_method_details")
    if isinstance(method_details, dict) and "feature_cache" in method_details:
        method_details["feature_cache"] = "not materialized; frozen selection provenance only"
    _atomic_write_json(destination_run / "fid500.json", source_result)
    return LocalizedRunReport(
        run_root=destination_run,
        count=len(localized_rows),
        generated_manifest_sha256=_sha256(generated_path),
    )


def _prompt_map(path: Path) -> dict[str, str]:
    fields, rows = _read_csv(path)
    if fields != ["item_id", "prompt", "name_source"]:
        raise ValueError(f"prompt manifest fields differ: {fields}")
    prompts: dict[str, str] = {}
    for row in rows:
        if row["item_id"] in prompts:
            raise ValueError(f"duplicate prompt item_id: {row['item_id']}")
        prompts[row["item_id"]] = row["prompt"]
    return prompts


def localize_psnr_ssim_run(
    *,
    source_runs_root: Path = DEFAULT_SOURCE_RUNS_ROOT,
    destination_runs_root: Path = DEFAULT_DESTINATION_RUNS_ROOT,
    dataset_root: Path = DEFAULT_PSNR_SSIM_DATASET_ROOT,
    source_run_id: str = "main-v2",
    run_id: str = "main-v2-100",
    expected_count: int = 100,
) -> LocalizedRunReport:
    source_runs_root = source_runs_root.expanduser().absolute()
    destination_runs_root = _prepare_destination(destination_runs_root)
    dataset_root = dataset_root.expanduser().absolute()
    destination_run = _prepare_destination(destination_runs_root / run_id)
    _, input_rows = _read_csv(dataset_root / "manifests" / "input.csv")
    if len(input_rows) != expected_count:
        raise ValueError(f"localized PSNR/SSIM input count differs from {expected_count}")
    prompts = _prompt_map(dataset_root / "manifests" / "prompts.csv")
    source_generated = _generated_by_name(
        source_runs_root / source_run_id / "generated.csv"
    )
    localized_rows: list[dict[str, str]] = []
    for row in input_rows:
        relative = _safe_relative(
            row["selected_path"], label="PSNR/SSIM selected_path"
        )
        input_path = dataset_root / relative
        try:
            source_row = source_generated[input_path.name]
            prompt = prompts[row["item_id"]]
        except KeyError as exc:
            raise ValueError(
                f"PSNR/SSIM source run or prompt is missing {input_path.name}"
            ) from exc
        output_relative = Path(*relative.parts[1:]).with_suffix(".png")
        localized_rows.append(
            _localized_generated_row(
                source_row=source_row,
                input_path=input_path,
                output_path=destination_run / "images" / output_relative,
                expected_prompt=prompt,
            )
        )
    generated_path = destination_run / "generated.csv"
    _atomic_write_csv(
        generated_path,
        fields=[*GENERATED_FIELDS, PROMPT_FIELD],
        rows=localized_rows,
    )
    return LocalizedRunReport(
        run_root=destination_run,
        count=len(localized_rows),
        generated_manifest_sha256=_sha256(generated_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-runs-root", type=Path, default=DEFAULT_SOURCE_RUNS_ROOT)
    parser.add_argument(
        "--destination-runs-root", type=Path, default=DEFAULT_DESTINATION_RUNS_ROOT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: dict[str, dict[str, Any]] = {
        "fid": asdict(
            localize_fid_run(
                source_runs_root=args.source_runs_root,
                destination_runs_root=args.destination_runs_root,
            )
        ),
        "psnr_ssim": asdict(
            localize_psnr_ssim_run(
                source_runs_root=args.source_runs_root,
                destination_runs_root=args.destination_runs_root,
            )
        ),
    }
    for report in reports.values():
        report["run_root"] = str(report["run_root"])
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
