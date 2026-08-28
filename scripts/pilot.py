#!/usr/bin/env python3
"""Sweep the fixed six SDXL strengths over the first 50 fixed metric pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.runner import TARGETS, measure_run
from scripts.generate import MODEL_ID, GenerationConfig, run_generation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL500_ROOT = Path("/home/kim_3090/datasets/abo/curated/eval500")
PAIR_MANIFEST = EVAL500_ROOT / "manifests" / "psnr_ssim_100.csv"
DEFAULT_SUMMARY_ROOT = PROJECT_ROOT / "runs" / "pilot"
DEFAULT_STRENGTHS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SEED = 0
SELECTION_RULE = "maximum PSNR among strengths meeting PSNR and SSIM targets"
FID_NOTE = (
    "FID is excluded from the EVAL500 measurement axis and transferred "
    "to a separate dataset pending definition."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meets_selection_gate(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["psnr"]["mean"]) >= TARGETS["psnr"]
        and float(metrics["ssim"]["mean"]) >= TARGETS["ssim"]
    )


def _gate_deficit(entry: dict[str, Any]) -> float:
    return (
        max(TARGETS["psnr"] / max(entry["psnr"], 1e-12) - 1.0, 0.0)
        + max(TARGETS["ssim"] / max(entry["ssim"], 1e-12) - 1.0, 0.0)
    )


def _apply_selection(summary: dict[str, Any]) -> dict[str, Any]:
    entries = summary.get("results")
    if not isinstance(entries, list) or not entries:
        raise ValueError("pilot results must be a non-empty list")
    for entry in entries:
        metrics = {
            "psnr": {"mean": entry["psnr"]},
            "ssim": {"mean": entry["ssim"]},
        }
        entry["meets_psnr_ssim_gate"] = _meets_selection_gate(metrics)
        entry.pop("meets_all_targets", None)

    passing = [entry for entry in entries if entry["meets_psnr_ssim_gate"]]
    if passing:
        best = max(passing, key=lambda entry: (entry["psnr"], -entry["strength"]))
        selected: float | None = best["strength"]
    else:
        best = min(entries, key=lambda entry: (_gate_deficit(entry), entry["strength"]))
        selected = None

    summary["selection_rule"] = SELECTION_RULE
    summary["fid_note"] = FID_NOTE
    summary["selected"] = selected
    summary["best"] = {
        "strength": best["strength"],
        "psnr": best["psnr"],
        "ssim": best["ssim"],
        "psnr_ssim_target_deficit": _gate_deficit(best),
    }
    return summary


def _atomic_write(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _render_markdown(summary: dict[str, Any]) -> str:
    includes_historical_fid = all("fid" in entry for entry in summary["results"])
    lines = [
        "# Strength pilot",
        "",
        f"- N: {summary['n']}",
        f"- Seed: {summary['seed']}",
        f"- Model: `{summary['model']}`",
        f"- Selection: {summary['selection_rule']}",
        f"- FID note: {summary['fid_note']}",
        "",
    ]
    if includes_historical_fid:
        lines.extend(
            [
                "| Strength | FID (historical) | PSNR | SSIM | Result |",
                "| ---: | ---: | ---: | ---: | :---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Strength | PSNR | SSIM | Result |",
                "| ---: | ---: | ---: | :---: |",
            ]
        )
    for entry in summary["results"]:
        verdict = "PASS" if entry["meets_psnr_ssim_gate"] else "FAIL"
        if includes_historical_fid:
            lines.append(
                f"| {entry['strength']:.2f} | {entry['fid']:.6f} | "
                f"{entry['psnr']:.6f} | {entry['ssim']:.6f} | {verdict} |"
            )
        else:
            lines.append(
                f"| {entry['strength']:.2f} | {entry['psnr']:.6f} | "
                f"{entry['ssim']:.6f} | {verdict} |"
            )
    lines.extend(["", f"Selected: `{summary['selected']}`", ""])
    if summary["selected"] is None:
        lines.extend(
            [
                "No strength met the PSNR and SSIM gate; selected remains null.",
                f"Best observed strength by PSNR/SSIM target deficit: `{summary['best']['strength']}`.",
                "",
            ]
        )
    return "\n".join(lines)


def run_pilot(
    *,
    n: int,
    strengths: Sequence[float],
    runs_root: Path,
    pair_manifest: Path = PAIR_MANIFEST,
    summary_root: Path = DEFAULT_SUMMARY_ROOT,
    generation_runner: Callable[..., Any] = run_generation,
    measurement_runner: Callable[..., dict[str, Any]] = measure_run,
) -> dict[str, Any]:
    if n <= 1:
        raise ValueError("pilot n must be at least two")
    if n > 100:
        raise ValueError("pilot n cannot exceed the fixed 100-pair manifest")
    normalized_strengths = [float(value) for value in strengths]
    if normalized_strengths != sorted(set(normalized_strengths)):
        raise ValueError("strengths must be unique and ascending")
    if any(not 0.0 < value <= 1.0 for value in normalized_strengths):
        raise ValueError("all strengths must be in (0, 1]")
    pair_manifest = pair_manifest.expanduser().absolute()
    if not pair_manifest.is_file():
        raise FileNotFoundError(f"fixed pair manifest does not exist: {pair_manifest}")
    runs_root = runs_root.expanduser().absolute()

    entries: list[dict[str, Any]] = []
    for strength in normalized_strengths:
        run_id = f"pilot-s{round(strength * 100):03d}"
        generation_report = generation_runner(
            manifest_path=pair_manifest,
            runs_root=runs_root,
            config=GenerationConfig(
                strength=strength,
                seed=SEED,
                run_id=run_id,
                limit=n,
                model=MODEL_ID,
            ),
        )
        measured = measurement_runner(
            run_id=run_id,
            runs_root=runs_root,
            input_manifest=pair_manifest,
            pair_manifest=pair_manifest,
            strength=strength,
            seed=SEED,
            model=MODEL_ID,
            require_all_pairs=False,
        )
        metrics = measured["metrics"]
        entry = {
            "run_id": run_id,
            "strength": strength,
            "psnr": float(metrics["psnr"]["mean"]),
            "ssim": float(metrics["ssim"]["mean"]),
            "meets_psnr_ssim_gate": _meets_selection_gate(metrics),
            "generation_count": int(getattr(generation_report, "count", n)),
            "elapsed_seconds": float(generation_report.elapsed_seconds),
            "peak_vram_gib": float(generation_report.peak_vram_gib),
        }
        entries.append(entry)
        print(
            f"pilot strength={strength:.2f} psnr={entry['psnr']:.6f} "
            f"ssim={entry['ssim']:.6f} "
            f"result={'PASS' if entry['meets_psnr_ssim_gate'] else 'FAIL'}",
            flush=True,
        )

    summary: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "n": n,
        "seed": SEED,
        "model": MODEL_ID,
        "strengths": normalized_strengths,
        "targets": dict(TARGETS),
        "selection_rule": SELECTION_RULE,
        "fid_note": FID_NOTE,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": _sha256(pair_manifest),
        "runs_root": str(runs_root),
        "results": entries,
    }
    _apply_selection(summary)
    summary_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        summary_root / "pilot.json",
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(summary_root / "pilot.md", _render_markdown(summary))
    return summary


def reselect_pilot(summary_path: Path) -> dict[str, Any]:
    """Reapply the fixed gate to existing measurements without GPU generation."""

    summary_path = summary_path.expanduser().absolute()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read pilot summary {summary_path}: {exc}") from exc
    original_results = deepcopy(summary.get("results", []))
    _apply_selection(summary)
    for original, reselected in zip(
        original_results, summary["results"], strict=True
    ):
        for key, value in original.items():
            if key not in {"meets_all_targets", "meets_psnr_ssim_gate"} and (
                reselected.get(key) != value
            ):
                raise ValueError("reselection changed measured pilot values")
    summary["reselected_at"] = datetime.now().astimezone().isoformat()
    _atomic_write(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(summary_path.with_suffix(".md"), _render_markdown(summary))
    return summary


def verify_pilot(summary_path: Path, *, check_runs: bool = False) -> dict[str, Any]:
    summary_path = summary_path.expanduser().absolute()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("pilot results must be a non-empty list")
    strengths = [entry["strength"] for entry in results]
    if strengths != summary["strengths"]:
        raise ValueError("pilot result strengths differ from the fixed sweep")
    if summary.get("selection_rule") != SELECTION_RULE:
        raise ValueError("pilot selection rule differs from the fixed rule")
    if summary.get("fid_note") != FID_NOTE:
        raise ValueError("pilot FID comparability note is missing or changed")
    expected_passes: list[dict[str, Any]] = []
    for entry in results:
        metrics = {
            "psnr": {"mean": entry["psnr"]},
            "ssim": {"mean": entry["ssim"]},
        }
        expected = _meets_selection_gate(metrics)
        if entry.get("meets_psnr_ssim_gate") != expected:
            raise ValueError(f"pilot PASS/FAIL flag differs at strength {entry['strength']}")
        if "meets_all_targets" in entry:
            raise ValueError("obsolete all-target pilot gate remains in the summary")
        if expected:
            expected_passes.append(entry)
    expected_selected = (
        max(expected_passes, key=lambda entry: (entry["psnr"], -entry["strength"]))[
            "strength"
        ]
        if expected_passes
        else None
    )
    if summary["selected"] != expected_selected:
        raise ValueError("pilot selected does not follow the fixed maximum-PSNR rule")
    markdown_path = summary_path.with_suffix(".md")
    if not markdown_path.is_file():
        raise FileNotFoundError("pilot.md is missing")
    markdown = markdown_path.read_text(encoding="utf-8")
    if FID_NOTE not in markdown or SELECTION_RULE not in markdown:
        raise ValueError("pilot.md is missing the fixed selection rule or FID note")

    if check_runs:
        runs_root = Path(summary["runs_root"])
        for entry in results:
            result_path = runs_root / entry["run_id"] / "results.json"
            measured = json.loads(result_path.read_text(encoding="utf-8"))
            observed = measured["metrics"]
            if (
                observed["psnr"]["mean"] != entry["psnr"]
                or observed["ssim"]["mean"] != entry["ssim"]
            ):
                raise ValueError(f"pilot summary metrics drifted for {entry['run_id']}")
            generated_path = runs_root / entry["run_id"] / "generated.csv"
            with generated_path.open(encoding="utf-8", newline="") as handle:
                generated_count = sum(1 for _ in csv.DictReader(handle))
            if generated_count != summary["n"]:
                raise ValueError(f"pilot generated count differs for {entry['run_id']}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--strengths",
        default=",".join(f"{value:.2f}" for value in DEFAULT_STRENGTHS),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--reselect", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = DEFAULT_SUMMARY_ROOT / "pilot.json"
    if args.reselect:
        summary = reselect_pilot(summary_path)
        print(
            f"pilot_reselected=true selected={summary['selected']} "
            f"results={len(summary['results'])}"
        )
        return
    if args.verify_only:
        summary = verify_pilot(summary_path, check_runs=True)
        print(
            f"pilot_verified=true selected={summary['selected']} "
            f"results={len(summary['results'])}"
        )
        return
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    strengths = [float(value) for value in args.strengths.split(",") if value]
    summary = run_pilot(
        n=args.n,
        strengths=strengths,
        runs_root=Path(runs_root_value),
    )
    print(f"pilot_json={DEFAULT_SUMMARY_ROOT / 'pilot.json'}")
    print(f"pilot_md={DEFAULT_SUMMARY_ROOT / 'pilot.md'}")
    print(f"selected={summary['selected']}")


if __name__ == "__main__":
    main()
