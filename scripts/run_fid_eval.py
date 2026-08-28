#!/usr/bin/env python3
"""Generate and measure the frozen AI Hub FID500 reconstruction set."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.fid import (
    compute_fid,
    configure_measurement_determinism,
    measurement_determinism_state,
)
from scripts.generate import (
    GUIDANCE_SCALE,
    IMAGE_SIZE,
    MODEL_ID,
    NUM_INFERENCE_STEPS,
    PROMPT,
    GenerationConfig,
    InputRecord,
    file_sha256,
    run_generation,
    validate_run_id,
)
from scripts.run_eval import DEFAULT_PILOT_PATH, SEED, resolve_strength
from scripts.verify_generated import verify_run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FID500_ROOT = PROJECT_ROOT / "dataset" / "fid"
TARGET_FID = 10.0
GENERATION_MANIFEST_FIELDS = [
    "item_id",
    "group",
    "width",
    "height",
    "sha256",
    "source_path",
    "selected_path",
]
SOURCE_MANIFEST_FIELDS = {
    "item_no",
    "대분류",
    "zip_member",
    "width",
    "height",
    "sha256",
}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _safe_component(value: str, *, label: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError(f"{label} must be one safe path component: {value!r}")
    return value


def _dataset_contract(
    dataset_root: Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, str]]]:
    dataset_root = dataset_root.expanduser().absolute()
    source_manifest = dataset_root / "manifest.csv"
    selection_path = dataset_root / "selection.json"
    selection = _load_json(selection_path, label="FID500 selection")
    try:
        with source_manifest.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read FID500 manifest {source_manifest}: {exc}") from exc
    missing = sorted(SOURCE_MANIFEST_FIELDS - fields)
    if missing:
        raise ValueError(f"FID500 manifest is missing fields: {missing}")
    if not rows:
        raise ValueError(f"FID500 manifest is empty: {source_manifest}")

    actual_manifest_hash = file_sha256(source_manifest)
    if selection.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("selection.json manifest_sha256 differs from manifest.csv")
    try:
        expected_count = int(selection["counts"]["final_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selection.json counts.final_count is missing or invalid") from exc
    if len(rows) != expected_count:
        raise ValueError(
            f"FID500 manifest count differs from selection: {len(rows)} != {expected_count}"
        )
    return source_manifest, selection_path, selection, rows


def _derived_manifest_rows(
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    _, _, selection, source_rows = _dataset_contract(dataset_root)
    try:
        category_directories = selection["rules"]["category_directories"]
    except (KeyError, TypeError) as exc:
        raise ValueError("selection.json rules.category_directories is missing") from exc
    if not isinstance(category_directories, dict):
        raise ValueError("selection.json rules.category_directories must be an object")

    dataset_root = dataset_root.expanduser().absolute()
    derived: list[dict[str, str]] = []
    seen_items: set[str] = set()
    seen_paths: set[str] = set()
    for row_number, row in enumerate(source_rows, start=2):
        item_no = _safe_component(row["item_no"], label=f"manifest.csv:{row_number} item_no")
        category = row["대분류"]
        try:
            category_directory = _safe_component(
                str(category_directories[category]),
                label=f"selection category directory for {category}",
            )
        except KeyError as exc:
            raise ValueError(
                f"manifest.csv:{row_number} category has no frozen directory mapping: {category}"
            ) from exc
        selected_path = f"input/{category_directory}/{item_no}.jpg"
        if item_no in seen_items:
            raise ValueError(f"duplicate item_no in FID500 manifest: {item_no}")
        if selected_path in seen_paths:
            raise ValueError(f"duplicate selected_path in FID500 manifest: {selected_path}")
        selected_image = dataset_root / selected_path
        if not selected_image.is_file():
            raise FileNotFoundError(f"frozen FID500 input is missing: {selected_image}")
        seen_items.add(item_no)
        seen_paths.add(selected_path)
        derived.append(
            {
                "item_id": item_no,
                "group": category,
                "width": row["width"],
                "height": row["height"],
                "sha256": row["sha256"],
                "source_path": row["zip_member"],
                "selected_path": selected_path,
            }
        )
    return selection, derived


def _manifest_bytes(rows: Sequence[dict[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=GENERATION_MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def build_generation_manifest(*, dataset_root: Path = DEFAULT_FID500_ROOT) -> Path:
    """Create the generate.py view using frozen CSV values only."""

    _, rows = _derived_manifest_rows(dataset_root)
    output_path = dataset_root.expanduser().absolute() / "manifests" / "input.csv"
    content = _manifest_bytes(rows)
    if output_path.is_file() and output_path.read_bytes() == content:
        return output_path
    if output_path.is_symlink():
        raise ValueError(f"generation manifest cannot be a symlink: {output_path}")
    _atomic_write_bytes(output_path, content)
    return output_path


def verify_generation_manifest(*, dataset_root: Path = DEFAULT_FID500_ROOT) -> Path:
    """Verify the derived view without creating or repairing it."""

    _, rows = _derived_manifest_rows(dataset_root)
    output_path = dataset_root.expanduser().absolute() / "manifests" / "input.csv"
    try:
        actual = output_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read generation manifest {output_path}: {exc}") from exc
    if actual != _manifest_bytes(rows):
        raise ValueError("generation manifest differs from frozen manifest.csv metadata")
    return output_path


def resolve_fid_strength(
    *,
    explicit: float | None,
    strength_from: Path | None,
    default_pilot_path: Path = DEFAULT_PILOT_PATH,
) -> tuple[float, Path]:
    """Resolve the fixed strength and retain an auditable source file path."""

    if explicit is None:
        if strength_from is None:
            raise ValueError("provide exactly one of --strength or --strength-from")
        source = strength_from.expanduser().absolute()
        return resolve_strength(explicit=None, strength_from=source), source
    if strength_from is not None:
        raise ValueError("provide exactly one of --strength or --strength-from")

    value = resolve_strength(explicit=explicit, strength_from=None)
    source = default_pilot_path.expanduser().absolute()
    selected = resolve_strength(explicit=None, strength_from=source)
    if value != selected:
        raise ValueError(
            f"explicit strength {value} does not match pilot selected {selected} in {source}"
        )
    return value, source


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    content = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, content)


@contextmanager
def _real_directory(
    *,
    dataset_root: Path,
    run_root: Path,
    verification: dict[str, Any],
    smoke: bool,
) -> Iterator[Path]:
    input_root = (dataset_root.expanduser().absolute() / "input").absolute()
    if not smoke:
        yield input_root
        return

    images = verification.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("strict generation verification did not return smoke image rows")
    with tempfile.TemporaryDirectory(prefix=".fid-real-subset-", dir=run_root) as raw:
        subset_root = Path(raw).absolute()
        seen_inputs: set[Path] = set()
        for index, image in enumerate(images):
            try:
                input_path = Path(image["input_path"]).absolute()
            except (KeyError, TypeError) as exc:
                raise ValueError("generation verification image lacks input_path") from exc
            try:
                input_path.resolve().relative_to(input_root.resolve())
            except ValueError as exc:
                raise ValueError(f"smoke input is outside frozen FID500 input: {input_path}") from exc
            if input_path in seen_inputs:
                raise ValueError(f"smoke verification duplicates input: {input_path}")
            seen_inputs.add(input_path)
            target = subset_root / f"{index:04d}{input_path.suffix.lower()}"
            target.symlink_to(input_path)
        yield subset_root


def _generated_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read generated manifest {path}: {exc}") from exc


def _dataset_prompt_resolver(
    *, selection: dict[str, Any], source_rows: Sequence[dict[str, str]]
) -> Callable[[InputRecord], str] | None:
    if "prompt_protocol" not in selection:
        return None
    prompts = {row["item_no"]: row.get("prompt", "") for row in source_rows}
    if not prompts or any(not prompt.strip() for prompt in prompts.values()):
        raise ValueError("prompt-aware dataset manifest has missing/empty prompt values")

    def resolve(record: InputRecord) -> str:
        try:
            return prompts[record.input_path.stem]
        except KeyError as exc:
            raise ValueError(f"generated input has no dataset prompt: {record.input_path}") from exc

    return resolve


def run_fid_evaluation(
    *,
    run_id: str,
    strength: float,
    strength_source: Path,
    limit: int | None,
    item_ids: Sequence[str] | None = None,
    runs_root: Path,
    dataset_root: Path = DEFAULT_FID500_ROOT,
    generation_runner: Callable[..., Any] = run_generation,
    generated_verifier: Callable[..., dict[str, Any]] = verify_run,
    fid_runner: Callable[[Path, Path], tuple[float, dict[str, str]]] = compute_fid,
) -> dict[str, Any]:
    validate_run_id(run_id)
    dataset_root = dataset_root.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    _, _, selection, source_rows = _dataset_contract(dataset_root)
    expected_count = int(selection["counts"]["final_count"])
    requested_ids = list(item_ids) if item_ids is not None else None
    if limit is not None and requested_ids is not None:
        raise ValueError("--limit and --item-id cannot be used together")
    if limit is not None and not 2 <= limit <= expected_count:
        raise ValueError(f"--limit must be between 2 and {expected_count}")
    if requested_ids is not None:
        if not 2 <= len(requested_ids) <= expected_count:
            raise ValueError(
                f"item_ids must contain between 2 and {expected_count} entries"
            )
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("item_ids must be unique")
        available_ids = {row["item_no"] for row in source_rows}
        missing = [item_id for item_id in requested_ids if item_id not in available_ids]
        if missing:
            raise ValueError(f"item_ids are not in FID500 manifest: {missing}")
    generation_manifest = build_generation_manifest(dataset_root=dataset_root)
    config = GenerationConfig(
        strength=strength,
        seed=SEED,
        run_id=run_id,
        limit=limit,
        model=MODEL_ID,
    )
    generation_kwargs: dict[str, Any] = {
        "manifest_path": generation_manifest,
        "runs_root": runs_root,
        "config": config,
    }
    if requested_ids is not None:
        generation_kwargs["item_ids"] = requested_ids
    prompt_resolver = _dataset_prompt_resolver(
        selection=selection, source_rows=source_rows
    )
    if prompt_resolver is not None:
        generation_kwargs["prompt_resolver"] = prompt_resolver
    generation = generation_runner(**generation_kwargs)
    verification = generated_verifier(runs_root=runs_root, run_id=run_id, strict=True)
    count = int(verification["count"])
    expected_generated = (
        len(requested_ids)
        if requested_ids is not None
        else limit if limit is not None else expected_count
    )
    if count != expected_generated or int(generation.count) != expected_generated:
        raise ValueError(
            f"generated count differs from protocol: {count}/{generation.count} "
            f"!= {expected_generated}"
        )

    run_root = (runs_root / run_id).absolute()
    generated_directory = (run_root / "images").absolute()
    configure_measurement_determinism(seed=SEED)
    print(f"measuring FID input_count={count}", flush=True)
    with _real_directory(
        dataset_root=dataset_root,
        run_root=run_root,
        verification=verification,
        smoke=limit is not None or requested_ids is not None,
    ) as real_directory:
        fid_value, fid_parameters = fid_runner(real_directory, generated_directory)
    determinism = measurement_determinism_state()

    source_manifest = dataset_root / "manifest.csv"
    generated_manifest = run_root / "generated.csv"
    dataset_record = {
        "root": str(dataset_root),
        "input_directory": str((dataset_root / "input").absolute()),
        "source_dataset": selection["source_dataset"],
        "selection_rule_version": selection["selection_rule_version"],
        "selection_seed": selection["seed"],
        "selection_manifest": str(source_manifest),
        "manifest_sha256": selection["manifest_sha256"],
        "generation_manifest": str(generation_manifest),
        "generation_manifest_sha256": file_sha256(generation_manifest),
        "generated_manifest_sha256": file_sha256(generated_manifest),
    }
    if "selection_method" in selection:
        dataset_record["selection_method"] = selection["selection_method"]
        dataset_record["selection_method_details"] = selection.get("method_details", {})
    if "prompt_protocol" in selection:
        dataset_record["prompt_protocol"] = selection["prompt_protocol"]
    measurement: dict[str, Any] = {
        "count": count,
        "smoke": limit is not None,
        "limit": limit,
        "fid": float(fid_value),
        "target": {"operator": "<=", "value": TARGET_FID},
        "verdict": "PASS" if fid_value <= TARGET_FID else "FAIL",
    }
    if requested_ids is not None:
        measurement["selection_mode"] = "selected"
        measurement["item_ids"] = requested_ids
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_record,
        "protocol": {
            "seed": SEED,
            "strength": strength,
            "strength_source": str(strength_source.expanduser().absolute()),
            "model": MODEL_ID,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "scheduler": "EulerDiscreteScheduler",
            "prompt": "per-image dataset prompt" if prompt_resolver is not None else PROMPT,
            "input_preprocess": (
                f"RGB {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} LANCZOS"
            ),
            "output": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB PNG lossless",
            "clean_fid": fid_parameters,
            "determinism": determinism,
        },
        "measurement": measurement,
        "generation_validation": verification["generation_validation"],
    }
    if prompt_resolver is not None:
        result["protocol"]["prompt_protocol"] = selection["prompt_protocol"]
    _atomic_write_json(run_root / "fid500.json", result)
    print(
        f"FID={float(fid_value):.12f} count={count} strength={strength} "
        f"manifest_sha256={selection['manifest_sha256']} "
        f"verdict={result['measurement']['verdict']}",
        flush=True,
    )
    return result


def _validate_recorded_contract(
    *,
    result: dict[str, Any],
    run_id: str,
    dataset_root: Path,
    runs_root: Path,
    generation_manifest: Path,
    verification: dict[str, Any],
) -> None:
    try:
        dataset = result["dataset"]
        protocol = result["protocol"]
        measurement = result["measurement"]
    except (KeyError, TypeError) as exc:
        raise ValueError("fid500.json is missing dataset/protocol/measurement") from exc
    if result.get("run_id") != run_id:
        raise ValueError("fid500.json run_id differs from requested run")

    _, _, selection, source_rows = _dataset_contract(dataset_root)
    generated_manifest = runs_root / run_id / "generated.csv"
    expected_dataset_values = {
        "manifest_sha256": selection["manifest_sha256"],
        "generation_manifest_sha256": file_sha256(generation_manifest),
        "generated_manifest_sha256": file_sha256(generated_manifest),
    }
    for key, expected in expected_dataset_values.items():
        if dataset.get(key) != expected:
            raise ValueError(f"fid500.json dataset.{key} differs from current files")
    if "selection_method" in selection:
        if dataset.get("selection_method") != selection["selection_method"]:
            raise ValueError("fid500.json selection method differs from selection.json")
        if dataset.get("selection_method_details") != selection.get("method_details", {}):
            raise ValueError("fid500.json selection method details differ from selection.json")
    if "prompt_protocol" in selection:
        if dataset.get("prompt_protocol") != selection["prompt_protocol"]:
            raise ValueError("fid500.json dataset prompt protocol differs from selection.json")
        if protocol.get("prompt") != "per-image dataset prompt":
            raise ValueError("fid500.json does not identify per-image dataset prompts")
        if protocol.get("prompt_protocol") != selection["prompt_protocol"]:
            raise ValueError("fid500.json protocol prompt details differ from selection.json")
    if protocol.get("model") != MODEL_ID or protocol.get("seed") != SEED:
        raise ValueError("fid500.json model/seed differs from the fixed generation protocol")
    count = int(verification["count"])
    if measurement.get("count") != count:
        raise ValueError("fid500.json count differs from strict generation verification")
    if result.get("generation_validation") != verification.get("generation_validation"):
        raise ValueError("fid500.json generation validation differs from strict verification")

    rows = _generated_rows(generated_manifest)
    try:
        strengths = {float(row["strength"]) for row in rows}
        seeds = {int(row["seed"]) for row in rows}
    except (KeyError, ValueError) as exc:
        raise ValueError("generated.csv contains invalid seed/strength values") from exc
    if strengths != {float(protocol["strength"])} or seeds != {SEED}:
        raise ValueError("generated.csv seed/strength differs from fid500.json protocol")
    if "prompt_protocol" in selection:
        expected_prompts = {row["item_no"]: row["prompt"] for row in source_rows}
        if any(
            row.get("prompt") != expected_prompts.get(Path(row["input_path"]).stem)
            for row in rows
        ):
            raise ValueError("generated.csv prompts differ from dataset manifest")


def verify_fid_evaluation(
    *,
    run_id: str,
    runs_root: Path,
    dataset_root: Path = DEFAULT_FID500_ROOT,
    expected_strength: float | None = None,
    expected_strength_source: Path | None = None,
    expected_limit: int | None = None,
    generated_verifier: Callable[..., dict[str, Any]] = verify_run,
    fid_runner: Callable[[Path, Path], tuple[float, dict[str, str]]] = compute_fid,
) -> dict[str, Any]:
    validate_run_id(run_id)
    dataset_root = dataset_root.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    result_path = runs_root / run_id / "fid500.json"
    result = _load_json(result_path, label="FID500 result")
    generation_manifest = verify_generation_manifest(dataset_root=dataset_root)
    verification = generated_verifier(runs_root=runs_root, run_id=run_id, strict=True)
    _validate_recorded_contract(
        result=result,
        run_id=run_id,
        dataset_root=dataset_root,
        runs_root=runs_root,
        generation_manifest=generation_manifest,
        verification=verification,
    )

    protocol = result["protocol"]
    measurement = result["measurement"]
    source = Path(protocol["strength_source"]).expanduser().absolute()
    current_strength = resolve_strength(explicit=None, strength_from=source)
    if current_strength != protocol["strength"]:
        raise ValueError("recorded strength differs from current pilot selected")
    if expected_strength is not None and protocol["strength"] != expected_strength:
        raise ValueError("recorded strength differs from requested verification strength")
    if expected_strength_source is not None and source != expected_strength_source.absolute():
        raise ValueError("recorded strength source differs from requested source")
    if expected_limit is not None and measurement.get("limit") != expected_limit:
        raise ValueError("recorded smoke limit differs from requested verification limit")

    smoke = measurement.get("smoke")
    if not isinstance(smoke, bool):
        raise ValueError("fid500.json measurement.smoke must be boolean")
    count = int(measurement["count"])
    selected = measurement.get("selection_mode") == "selected"
    if selected:
        item_ids = measurement.get("item_ids")
        if (
            not isinstance(item_ids, list)
            or len(item_ids) != count
            or len(item_ids) != len(set(item_ids))
        ):
            raise ValueError("selected FID measurement item_ids are invalid")
    elif smoke:
        if measurement.get("limit") != count:
            raise ValueError("smoke measurement limit must equal generated count")
    else:
        _, _, selection, _ = _dataset_contract(dataset_root)
        if measurement.get("limit") is not None or count != int(
            selection["counts"]["final_count"]
        ):
            raise ValueError("full FID measurement must cover the frozen dataset")

    configure_measurement_determinism(seed=SEED)
    generated_directory = (runs_root / run_id / "images").absolute()
    with _real_directory(
        dataset_root=dataset_root,
        run_root=(runs_root / run_id).absolute(),
        verification=verification,
        smoke=smoke or selected,
    ) as real_directory:
        actual_fid, actual_parameters = fid_runner(real_directory, generated_directory)
    if float(actual_fid) != measurement["fid"]:
        raise ValueError(
            f"recorded FID differs from recomputation: {measurement['fid']} != {actual_fid}"
        )
    if actual_parameters != protocol["clean_fid"]:
        raise ValueError("recorded clean-fid parameters differ from recomputation")
    if measurement["verdict"] != (
        "PASS" if actual_fid <= TARGET_FID else "FAIL"
    ):
        raise ValueError("recorded target verdict differs from the exact FID value")
    if protocol["determinism"] != measurement_determinism_state():
        raise ValueError("recorded determinism differs from the active torch runtime")
    print(
        f"FID500_VERIFIED run_id={run_id} FID={float(actual_fid):.12f} "
        f"count={count} smoke={str(smoke).lower()}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--strength", type=float)
    group.add_argument("--strength-from", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-id", action="append", dest="item_ids")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_FID500_ROOT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    runs_root = Path(runs_root_value)
    if args.verify_only:
        expected_strength: float | None = None
        expected_source: Path | None = None
        if args.strength is not None or args.strength_from is not None:
            expected_strength, expected_source = resolve_fid_strength(
                explicit=args.strength,
                strength_from=args.strength_from,
            )
        verify_fid_evaluation(
            run_id=args.run_id,
            runs_root=runs_root,
            dataset_root=args.dataset_root,
            expected_strength=expected_strength,
            expected_strength_source=expected_source,
            expected_limit=args.limit,
        )
        return

    strength, source = resolve_fid_strength(
        explicit=args.strength,
        strength_from=args.strength_from,
    )
    result = run_fid_evaluation(
        run_id=args.run_id,
        strength=strength,
        strength_source=source,
        limit=args.limit,
        item_ids=args.item_ids,
        runs_root=runs_root,
        dataset_root=args.dataset_root,
    )
    print(f"result={runs_root.expanduser().absolute() / args.run_id / 'fid500.json'}")
    print(f"FID={result['measurement']['fid']}")


if __name__ == "__main__":
    main()
