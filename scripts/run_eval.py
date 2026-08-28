#!/usr/bin/env python3
"""Run generation and the fixed EVAL500 PSNR/SSIM measurements."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.fid import configure_measurement_determinism, measurement_determinism_state
from metrics.runner import _atomic_write_json, measure_run
from metrics.schema import validate_results
from scripts.build_eval500_prompts import (
    NAME_MAX_WORDS,
    PROMPT_FIELDS,
    PROMPT_TEMPLATE,
)
from scripts.generate import (
    MODEL_ID,
    SD15_MODEL_ID,
    GenerationConfig,
    InputRecord,
    file_sha256,
    run_generation,
)
from scripts.verify_generated import verify_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL500_ROOT = PROJECT_ROOT / "dataset" / "psnr_ssim"
INPUT_MANIFEST = EVAL500_ROOT / "manifests" / "input.csv"
PAIR_MANIFEST = EVAL500_ROOT / "manifests" / "psnr_ssim_100.csv"
DEFAULT_PROMPT_MANIFEST = EVAL500_ROOT / "manifests" / "prompts.csv"
PROMPT_PROTOCOL_TEMPLATE = PROMPT_TEMPLATE
SEED = 0
BASELINE_FALLBACK_INCREMENT = 0.10
DEFAULT_PILOT_PATH = PROJECT_ROOT / "runs" / "pilot" / "pilot.json"


class NoSelectedStrengthError(RuntimeError):
    """The fixed pilot honestly found no strength satisfying every target."""


@dataclass(frozen=True)
class EvaluationOutcome:
    generation: Any
    results: dict[str, Any]
    strength: float
    model: str
    note: str | None


@dataclass(frozen=True)
class PromptManifestContract:
    resolver: Callable[[Any], str]
    protocol: dict[str, Any]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def load_prompt_manifest_contract(
    *, input_manifest: Path, prompt_manifest: Path
) -> PromptManifestContract:
    """Validate the full prompt join and resolve prompts by absolute input path."""

    input_manifest = input_manifest.expanduser().absolute()
    prompt_manifest = prompt_manifest.expanduser().absolute()
    input_fields, input_rows = _read_csv(input_manifest)
    required_input_fields = {"item_id", "selected_path"}
    if not required_input_fields.issubset(input_fields):
        raise ValueError(
            "input manifest lacks prompt join fields: "
            f"{sorted(required_input_fields - set(input_fields))}"
        )
    prompt_fields, prompt_rows = _read_csv(prompt_manifest)
    if prompt_fields != PROMPT_FIELDS:
        raise ValueError(f"prompt manifest fields differ: {prompt_fields}")

    prompts_by_item: dict[str, str] = {}
    for row_number, row in enumerate(prompt_rows, start=2):
        item_id = row.get("item_id", "")
        prompt = row.get("prompt", "")
        source = row.get("name_source", "")
        if not item_id or not prompt:
            raise ValueError(f"prompt manifest row {row_number} is incomplete")
        if item_id in prompts_by_item:
            raise ValueError(f"duplicate prompt manifest item_id: {item_id}")
        if source not in {"en", "fallback"}:
            raise ValueError(
                f"prompt manifest row {row_number} has invalid name_source: {source}"
            )
        prompts_by_item[item_id] = prompt

    eval_root = input_manifest.parent.parent
    prompts_by_input: dict[Path, str] = {}
    seen_item_ids: set[str] = set()
    for row_number, row in enumerate(input_rows, start=2):
        item_id = row.get("item_id", "")
        if not item_id:
            raise ValueError(f"input manifest row {row_number} has an empty item_id")
        if item_id in seen_item_ids:
            raise ValueError(f"duplicate input manifest item_id: {item_id}")
        seen_item_ids.add(item_id)
        selected_path = Path(row.get("selected_path", ""))
        if (
            selected_path.is_absolute()
            or ".." in selected_path.parts
            or not selected_path.parts
            or selected_path.parts[0] != "input"
        ):
            raise ValueError(
                f"input manifest row {row_number} has unsafe selected_path: {selected_path}"
            )
        input_path = (eval_root / selected_path).absolute()
        if input_path in prompts_by_input:
            raise ValueError(f"duplicate prompt input path: {input_path}")
        try:
            prompts_by_input[input_path] = prompts_by_item[item_id]
        except KeyError as exc:
            raise ValueError(f"prompt manifest is missing item_id: {item_id}") from exc

    extra_item_ids = sorted(set(prompts_by_item) - seen_item_ids)
    if extra_item_ids:
        raise ValueError(
            "prompt manifest contains item_id(s) outside input manifest: "
            + ", ".join(extra_item_ids[:10])
        )

    def resolve(record: Any) -> str:
        input_path = Path(record.input_path).expanduser().absolute()
        try:
            return prompts_by_input[input_path]
        except KeyError as exc:
            raise ValueError(f"no per-image prompt for input: {input_path}") from exc

    protocol = {
        "mode": "per-image",
        "template": PROMPT_PROTOCOL_TEMPLATE,
        "name_source": (
            "ABO item_name language_tag starts with en; "
            "fallback to manifest product_type"
        ),
        "name_max_words": NAME_MAX_WORDS,
        "truncation": "first 15 whitespace-delimited words",
        "whitespace_normalization": "strip and collapse consecutive whitespace",
        "manifest_path": str(prompt_manifest),
        "manifest_sha256": file_sha256(prompt_manifest),
    }
    return PromptManifestContract(resolver=resolve, protocol=protocol)


def resolve_strength(*, explicit: float | None, strength_from: Path | None) -> float:
    if (explicit is None) == (strength_from is None):
        raise ValueError("provide exactly one of --strength or --strength-from")
    if explicit is not None:
        value = float(explicit)
    else:
        assert strength_from is not None
        source = strength_from.expanduser().absolute()
        try:
            pilot = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read strength source {source}: {exc}") from exc
        selected = pilot.get("selected")
        if selected is None:
            raise NoSelectedStrengthError(
                f"pilot selected is null in {source}; no strength met all three targets"
            )
        value = float(selected)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {value}")
    return value


def verification_defaults(
    *,
    run_id: str,
    limit: int | None,
    explicit_strength: float | None,
    strength_from: Path | None,
    requested_model: str,
    pilot_path: Path = DEFAULT_PILOT_PATH,
) -> tuple[float | None, str | None, int | None, int | None]:
    """Resolve the contract that the task's terse --verify-only CLI implies."""

    if explicit_strength is not None or strength_from is not None:
        expected_strength = resolve_strength(
            explicit=explicit_strength,
            strength_from=strength_from,
        )
    elif run_id in {"main", "baseline", "main-v2", "baseline-v2", "main-v2-repro"}:
        expected_strength = resolve_strength(explicit=None, strength_from=pilot_path)
    else:
        expected_strength = None

    if limit is not None:
        return expected_strength, requested_model, limit, limit
    if run_id in {"main", "main-v2", "main-v2-repro"}:
        return expected_strength, MODEL_ID, 500, 100
    if run_id in {"baseline", "baseline-v2"}:
        baseline_model = requested_model if requested_model != MODEL_ID else SD15_MODEL_ID
        return expected_strength, baseline_model, 500, 100
    return expected_strength, requested_model, None, None


def run_evaluation(
    *,
    run_id: str,
    strength: float,
    model: str,
    limit: int | None,
    runs_root: Path,
    input_manifest: Path = INPUT_MANIFEST,
    pair_manifest: Path = PAIR_MANIFEST,
    prompt_manifest: Path | None = None,
    generation_runner: Callable[..., Any] = run_generation,
    measurement_runner: Callable[..., dict[str, Any]] = measure_run,
    generated_verifier: Callable[..., dict[str, Any]] = verify_run,
    allow_baseline_fallback: bool = False,
) -> EvaluationOutcome:
    if limit is not None and not 2 <= limit <= 100:
        raise ValueError("--limit must be between 2 and 100 fixed pairs")
    input_manifest = input_manifest.expanduser().absolute()
    pair_manifest = pair_manifest.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    prompt_contract = (
        load_prompt_manifest_contract(
            input_manifest=input_manifest,
            prompt_manifest=prompt_manifest,
        )
        if prompt_manifest is not None
        else None
    )
    generation_manifest = pair_manifest if limit is not None else input_manifest
    actual_strength = strength
    actual_model = model
    note: str | None = None

    config = GenerationConfig(
        strength=actual_strength,
        seed=SEED,
        run_id=run_id,
        limit=limit,
        model=actual_model,
    )
    try:
        generation_kwargs: dict[str, Any] = {
            "manifest_path": generation_manifest,
            "runs_root": runs_root,
            "config": config,
        }
        if prompt_contract is not None:
            generation_kwargs["prompt_resolver"] = prompt_contract.resolver
        generation = generation_runner(
            **generation_kwargs,
        )
    except (OSError, ValueError) as exc:
        if not allow_baseline_fallback or model == MODEL_ID:
            raise
        actual_strength = min(1.0, strength + BASELINE_FALLBACK_INCREMENT)
        actual_model = MODEL_ID
        note = (
            f"requested baseline model {model} was inaccessible ({type(exc).__name__}); "
            f"used SDXL with elevated strength {actual_strength:.2f}"
        )
        print(f"baseline_fallback={note}", flush=True)
        fallback_generation_kwargs: dict[str, Any] = {
            "manifest_path": generation_manifest,
            "runs_root": runs_root,
            "config": GenerationConfig(
                strength=actual_strength,
                seed=SEED,
                run_id=run_id,
                limit=limit,
                model=actual_model,
            ),
        }
        if prompt_contract is not None:
            fallback_generation_kwargs["prompt_resolver"] = prompt_contract.resolver
        generation = generation_runner(**fallback_generation_kwargs)

    print(
        f"generation_complete={generation.count}/{generation.count} "
        f"elapsed_seconds={generation.elapsed_seconds:.3f} "
        f"peak_vram_gib={generation.peak_vram_gib:.3f}",
        flush=True,
    )
    generated_validation = generated_verifier(
        runs_root=runs_root,
        run_id=run_id,
        strict=True,
    )["generation_validation"]
    print(f"measuring run_id={run_id}", flush=True)
    measurement_kwargs: dict[str, Any] = {
        "run_id": run_id,
        "runs_root": runs_root,
        "input_manifest": input_manifest,
        "pair_manifest": pair_manifest,
        "strength": actual_strength,
        "seed": SEED,
        "model": actual_model,
        "note": note,
        "require_all_pairs": limit is None,
    }
    if prompt_contract is not None:
        measurement_kwargs["prompt_protocol"] = prompt_contract.protocol
    results = measurement_runner(**measurement_kwargs)
    results["generation_validation"] = generated_validation
    result_path = runs_root / run_id / "results.json"
    if result_path.is_file():
        validate_results(
            results,
            eval_root=input_manifest.parent.parent,
            runs_root=runs_root,
            require_determinism=True,
            require_generation_validation=True,
        )
        _atomic_write_json(result_path, results)
    print(
        f"measured run_id={run_id} psnr={results['metrics']['psnr']['mean']:.6f} "
        f"ssim={results['metrics']['ssim']['mean']:.6f}",
        flush=True,
    )
    return EvaluationOutcome(
        generation=generation,
        results=results,
        strength=actual_strength,
        model=actual_model,
        note=note,
    )


def _generated_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_evaluation(
    *,
    run_id: str,
    runs_root: Path,
    input_manifest: Path = INPUT_MANIFEST,
    prompt_manifest: Path | None = None,
    expected_strength: float | None = None,
    expected_model: str | None = None,
    expected_count: int | None = None,
    expected_pair_count: int | None = None,
    generated_verifier: Callable[..., dict[str, Any]] = verify_run,
) -> dict[str, Any]:
    runs_root = runs_root.expanduser().absolute()
    result_path = runs_root / run_id / "results.json"
    try:
        results = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation result {result_path}: {exc}") from exc
    validate_results(
        results,
        eval_root=input_manifest.expanduser().absolute().parent.parent,
        runs_root=runs_root,
    )
    generated_manifest = runs_root / run_id / "generated.csv"
    actual_input_hash = file_sha256(input_manifest.expanduser().absolute())
    actual_generated_hash = file_sha256(generated_manifest)
    if results["inputs"]["input_manifest_sha256"] != actual_input_hash:
        raise ValueError("results input manifest hash differs from the actual file")
    if results["inputs"]["generated_manifest_sha256"] != actual_generated_hash:
        raise ValueError("results generated manifest hash differs from the actual file")

    generated = generated_verifier(runs_root=runs_root, run_id=run_id, strict=True)
    count = int(results["protocol"]["n_input"])
    if generated["count"] != count:
        raise ValueError("generated.csv count differs from protocol.n_input")
    if expected_count is not None and count != expected_count:
        raise ValueError(f"expected {expected_count} generated images, got {count}")
    if (
        expected_pair_count is not None
        and results["protocol"]["n_pairs"] != expected_pair_count
    ):
        raise ValueError(
            f"expected {expected_pair_count} fixed metric pairs, "
            f"got {results['protocol']['n_pairs']}"
        )

    protocol = results["protocol"]
    recorded_prompt_protocol = protocol.get("prompt_protocol")
    effective_prompt_manifest = prompt_manifest
    if effective_prompt_manifest is None and recorded_prompt_protocol is not None:
        effective_prompt_manifest = Path(recorded_prompt_protocol["manifest_path"])
    if effective_prompt_manifest is not None:
        prompt_contract = load_prompt_manifest_contract(
            input_manifest=input_manifest,
            prompt_manifest=effective_prompt_manifest,
        )
        if recorded_prompt_protocol != prompt_contract.protocol:
            raise ValueError(
                "results prompt protocol differs from the current prompt manifest contract"
            )
    elif recorded_prompt_protocol is not None:
        raise ValueError("results prompt protocol cannot be verified")
    actual_model = protocol["model"]
    fallback_used = (
        expected_model is not None
        and expected_model != MODEL_ID
        and actual_model == MODEL_ID
        and isinstance(protocol.get("note"), str)
        and expected_model in protocol["note"]
    )
    if expected_model is not None and actual_model != expected_model and not fallback_used:
        raise ValueError("results protocol model differs from the requested model")
    if expected_strength is not None:
        accepted_strength = (
            min(1.0, expected_strength + BASELINE_FALLBACK_INCREMENT)
            if fallback_used
            else expected_strength
        )
        if protocol["strength"] != accepted_strength:
            raise ValueError(
                "results protocol strength differs from the fixed requested strength"
            )

    rows = _generated_rows(generated_manifest)
    if effective_prompt_manifest is not None:
        for row_number, row in enumerate(rows, start=2):
            if "prompt" not in row:
                raise ValueError("generated.csv lacks the per-image prompt field")
            expected_prompt = prompt_contract.resolver(
                InputRecord(
                    input_path=Path(row["input_path"]),
                    output_relative_path=Path("unused.png"),
                )
            )
            if row["prompt"] != expected_prompt:
                raise ValueError(
                    f"generated.csv row {row_number} prompt differs from prompts.csv"
                )
    if any(int(row["seed"]) != results["protocol"]["seed"] for row in rows):
        raise ValueError("generated.csv seed differs from results protocol")
    if any(float(row["strength"]) != results["protocol"]["strength"] for row in rows):
        raise ValueError("generated.csv strength differs from results protocol")
    configure_measurement_determinism(seed=0)
    runtime_determinism = measurement_determinism_state()
    if results["env"].get("determinism") != runtime_determinism:
        raise ValueError("recorded env.determinism differs from the active torch runtime")
    if results.get("generation_validation") != generated["generation_validation"]:
        raise ValueError("recorded generation validation differs from strict verification")
    validate_results(
        results,
        eval_root=input_manifest.expanduser().absolute().parent.parent,
        runs_root=runs_root,
        require_determinism=True,
        require_generation_validation=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--strength", type=float)
    group.add_argument("--strength-from", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-manifest", type=Path, default=INPUT_MANIFEST)
    parser.add_argument("--pair-manifest", type=Path, default=PAIR_MANIFEST)
    parser.add_argument("--prompt-manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    runs_root = Path(runs_root_value)
    if args.verify_only:
        prompt_manifest = args.prompt_manifest
        if prompt_manifest is None and args.run_id in {
            "main-v2",
            "baseline-v2",
            "main-v2-repro",
        }:
            prompt_manifest = DEFAULT_PROMPT_MANIFEST
        expected_strength, expected_model, expected_count, expected_pair_count = (
            verification_defaults(
                run_id=args.run_id,
                limit=args.limit,
                explicit_strength=args.strength,
                strength_from=args.strength_from,
                requested_model=args.model,
            )
        )
        results = verify_evaluation(
            run_id=args.run_id,
            runs_root=runs_root,
            input_manifest=args.input_manifest,
            prompt_manifest=prompt_manifest,
            expected_strength=expected_strength,
            expected_model=expected_model,
            expected_count=expected_count,
            expected_pair_count=expected_pair_count,
        )
        print(
            f"evaluation_verified=true run_id={args.run_id} "
            f"n_input={results['protocol']['n_input']} "
            f"n_pairs={results['protocol']['n_pairs']}"
        )
        return

    strength = resolve_strength(
        explicit=args.strength,
        strength_from=args.strength_from,
    )
    outcome = run_evaluation(
        run_id=args.run_id,
        strength=strength,
        model=args.model,
        limit=args.limit,
        runs_root=runs_root,
        input_manifest=args.input_manifest,
        pair_manifest=args.pair_manifest,
        prompt_manifest=args.prompt_manifest,
        allow_baseline_fallback=args.model != MODEL_ID,
    )
    print(f"results={runs_root.expanduser().absolute() / args.run_id / 'results.json'}")
    print(f"actual_model={outcome.model}")
    print(f"actual_strength={outcome.strength}")


if __name__ == "__main__":
    main()
