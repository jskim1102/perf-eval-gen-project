#!/usr/bin/env python3
"""Generate FLUX img2img reconstructions for FID_v2 and measure selected-set FID.

The input and real-reference image for each item is the deterministic thumbnail
recorded by ``manifest.csv``.  The generation protocol is intentionally fixed;
CLI strength is accepted only so the web wrapper can use the same subprocess
contract as the existing FID runner, and any value other than 0.15 is rejected.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

# FLUX sequential CPU offload is sensitive to CUDA allocator fragmentation.  It
# must be set before importing torch or diffusers.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image

from metrics.fid import compute_fid, measurement_determinism_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset" / "fid_v2"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
STRENGTH = 0.15
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5
SEED = 0
IMAGE_SIZE = 1024
TARGET_FID = 10.0
SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
LICENSE_NOTE = (
    "FLUX.1-dev is used for non-commercial benchmark/evaluation only; "
    "commercial adoption requires a license review, with SDXL retained as fallback."
)
GENERATED_FIELDS = [
    "item_id",
    "group",
    "input_path",
    "output_path",
    "sha256",
    "seed",
    "strength",
    "model",
    "prompt",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class InputRecord:
    item_id: str
    group: str
    product_type: str
    input_path: Path
    output_relative_path: Path
    prompt: str


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )


def _safe_relative_path(raw: str, *, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a safe relative path: {raw}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def load_records(dataset_root: Path) -> tuple[list[InputRecord], dict[str, Any]]:
    dataset_root = dataset_root.expanduser().absolute()
    manifest_path = dataset_root / "manifest.csv"
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"item_no", "대분류", "상품명", "thumbnail", "prompt"}
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                raise ValueError(
                    f"manifest.csv lacks fields: {sorted(required - fields)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read FID_v2 manifest {manifest_path}: {exc}") from exc
    if len(rows) < 2:
        raise ValueError("FID_v2 manifest must contain at least two rows")

    records: list[InputRecord] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        item_id = row["item_no"].strip()
        group = row["대분류"].strip()
        product_type = row["상품명"].strip()
        prompt = row["prompt"].strip()
        if not item_id or not group or not product_type or not prompt:
            raise ValueError(f"manifest.csv:{row_number} has an empty required value")
        if item_id in seen_ids:
            raise ValueError(f"manifest.csv:{row_number} duplicates item_no {item_id}")
        seen_ids.add(item_id)

        thumbnail = _safe_relative_path(
            row["thumbnail"], label=f"manifest.csv:{row_number} thumbnail"
        )
        if thumbnail.parts[0] != "input" or len(thumbnail.parts) < 2:
            raise ValueError(
                f"manifest.csv:{row_number} thumbnail must be below input/: {thumbnail}"
            )
        input_path = (dataset_root / thumbnail).absolute()
        if not input_path.is_file() or input_path.is_symlink():
            raise FileNotFoundError(f"FID_v2 thumbnail is missing or unsafe: {input_path}")
        output_relative_path = Path(*thumbnail.parts[1:]).with_suffix(".png")
        if output_relative_path in seen_outputs:
            raise ValueError(
                f"manifest.csv:{row_number} duplicates output path {output_relative_path}"
            )
        seen_outputs.add(output_relative_path)
        records.append(
            InputRecord(
                item_id=item_id,
                group=group,
                product_type=product_type,
                input_path=input_path,
                output_relative_path=output_relative_path,
                prompt=prompt,
            )
        )

    selection = _load_json(dataset_root / "selection.json", label="FID_v2 selection")
    return records, selection


def _select_records(
    records: Sequence[InputRecord],
    *,
    limit: int | None,
    item_ids: Sequence[str] | None,
) -> list[InputRecord]:
    requested = list(item_ids) if item_ids is not None else None
    if limit is not None and requested is not None:
        raise ValueError("--limit and --item-id cannot be used together")
    if limit is not None:
        if not 2 <= limit <= len(records):
            raise ValueError(f"--limit must be between 2 and {len(records)}")
        return list(records[:limit])
    if requested is None:
        return list(records)
    if not 2 <= len(requested) <= len(records):
        raise ValueError(f"item_ids must contain between 2 and {len(records)} entries")
    if len(requested) != len(set(requested)) or any(not item_id for item_id in requested):
        raise ValueError("item_ids must be non-empty and unique")
    by_id = {record.item_id: record for record in records}
    missing = [item_id for item_id in requested if item_id not in by_id]
    if missing:
        raise ValueError(f"item_ids are not in FID_v2 manifest: {missing}")
    return [by_id[item_id] for item_id in requested]


def load_flux_img2img_pipeline(model_id: str) -> FluxImg2ImgPipeline:
    pipe = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # The full FLUX model does not fit in the RTX 3090's 24 GiB VRAM.
    pipe.enable_sequential_cpu_offload()
    return pipe


def _atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError(f"refusing to replace output symlink: {output_path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        image.save(temporary, format="PNG", compress_level=9)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=".generated.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
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


@contextmanager
def _exclusive_gpu_lock(runs_root: Path) -> Iterator[None]:
    lock_path = runs_root / ".sdxl-generation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another generation process holds {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} mode=flux-img2img\n".encode())
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_existing_png(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"generated output is missing or unsafe: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"generated output must be {IMAGE_SIZE}x{IMAGE_SIZE} PNG: {path}"
                )
    except OSError as exc:
        raise ValueError(f"generated output is unreadable: {path}: {exc}") from exc


def _expected_row(
    record: InputRecord,
    *,
    output_path: Path,
    elapsed_seconds: float | None,
) -> dict[str, str]:
    return {
        "item_id": record.item_id,
        "group": record.group,
        "input_path": str(record.input_path),
        "output_path": str(output_path),
        "sha256": _file_sha256(output_path),
        "seed": str(SEED),
        "strength": str(STRENGTH),
        "model": MODEL_ID,
        "prompt": record.prompt,
        "elapsed_seconds": (
            "" if elapsed_seconds is None else f"{elapsed_seconds:.6f}"
        ),
    }


def _load_completed_rows(
    manifest_path: Path,
    *,
    records: Sequence[InputRecord],
    run_root: Path,
) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    if manifest_path.is_symlink():
        raise ValueError(f"generated manifest cannot be a symlink: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != GENERATED_FIELDS:
            raise ValueError(
                "generated.csv belongs to a different protocol; "
                f"expected {GENERATED_FIELDS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    by_id = {record.item_id: record for record in records}
    completed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        record = by_id.get(row["item_id"])
        if record is None or record.item_id in completed:
            raise ValueError(f"generated.csv:{row_number} has an unknown/duplicate item")
        output_path = (run_root / "images" / record.output_relative_path).absolute()
        if (
            row["group"] != record.group
            or Path(row["input_path"]) != record.input_path
            or Path(row["output_path"]) != output_path
            or row["seed"] != str(SEED)
            or float(row["strength"]) != STRENGTH
            or row["model"] != MODEL_ID
            or row["prompt"] != record.prompt
        ):
            raise ValueError(f"generated.csv:{row_number} differs from FLUX img2img protocol")
        elapsed_seconds = row["elapsed_seconds"].strip()
        if elapsed_seconds:
            elapsed_value = float(elapsed_seconds)
            if elapsed_value < 0:
                raise ValueError(
                    f"generated.csv:{row_number} has negative elapsed_seconds"
                )
            if elapsed_value == 0:
                row["elapsed_seconds"] = ""
        _validate_existing_png(output_path)
        if row["sha256"] != _file_sha256(output_path):
            raise ValueError(f"generated.csv:{row_number} output hash differs")
        completed[record.item_id] = dict(row)
    return completed


def _generate(
    *,
    records: Sequence[InputRecord],
    selected: Sequence[InputRecord],
    run_root: Path,
    pipeline_loader: Callable[[str], object],
    require_cuda: bool,
) -> tuple[list[dict[str, str]], int, int, int, float, float | None]:
    manifest_path = run_root / "generated.csv"
    completed = _load_completed_rows(
        manifest_path, records=records, run_root=run_root
    )
    recovered_without_manifest = 0
    for record in selected:
        if record.item_id in completed:
            continue
        output_path = (run_root / "images" / record.output_relative_path).absolute()
        if output_path.exists() or output_path.is_symlink():
            _validate_existing_png(output_path)
            completed[record.item_id] = _expected_row(
                record, output_path=output_path, elapsed_seconds=None
            )
            recovered_without_manifest += 1

    pending = [record for record in selected if record.item_id not in completed]
    resumed = len(selected) - len(pending)
    ordered_completed = list(completed.values())
    _atomic_write_csv(manifest_path, ordered_completed)
    print(
        f"FID_V2_GENERATE done={resumed}/{len(selected)} resumed={resumed}",
        flush=True,
    )

    started = time.perf_counter()
    if pending:
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("FLUX.1-dev img2img generation requires CUDA")
        with _exclusive_gpu_lock(run_root.parent):
            pipe = pipeline_loader(MODEL_ID)
            try:
                for index, record in enumerate(pending, start=1):
                    item_started = time.perf_counter()
                    with Image.open(record.input_path) as source_image:
                        source = source_image.convert("RGB").resize(
                            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
                        )
                    generator = torch.Generator(device="cpu").manual_seed(SEED)
                    result = pipe(
                        prompt=record.prompt,
                        image=source,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                        strength=STRENGTH,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=generator,
                    )
                    image = result.images[0].convert("RGB")
                    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError(
                            f"FLUX returned {image.size}, expected "
                            f"{IMAGE_SIZE}x{IMAGE_SIZE}"
                        )
                    output_path = (
                        run_root / "images" / record.output_relative_path
                    ).absolute()
                    _atomic_save_png(image, output_path)
                    completed[record.item_id] = _expected_row(
                        record,
                        output_path=output_path,
                        elapsed_seconds=time.perf_counter() - item_started,
                    )
                    ordered_completed = list(completed.values())
                    _atomic_write_csv(manifest_path, ordered_completed)
                    done = resumed + index
                    elapsed = time.perf_counter() - started
                    eta_minutes = (len(selected) - done) * elapsed / index / 60
                    print(
                        f"FID_V2_GENERATE done={done}/{len(selected)} "
                        f"item_id={record.item_id} eta_minutes={eta_minutes:.1f}",
                        flush=True,
                    )
            finally:
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    selected_rows = [completed[record.item_id] for record in selected]
    elapsed_seconds = time.perf_counter() - started
    recorded_elapsed = [
        float(row["elapsed_seconds"])
        for row in selected_rows
        if row["elapsed_seconds"]
    ]
    seconds_per_image = (
        sum(recorded_elapsed) / len(recorded_elapsed) if recorded_elapsed else None
    )
    return (
        selected_rows,
        len(pending),
        resumed,
        recovered_without_manifest,
        elapsed_seconds,
        seconds_per_image,
    )


@contextmanager
def _staged_flat(
    paths: Sequence[Path], *, run_root: Path, prefix: str
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir=run_root) as raw_directory:
        directory = Path(raw_directory)
        for index, path in enumerate(paths):
            (directory / f"{index:04d}{path.suffix.lower()}").symlink_to(path)
        yield directory


def run_fid_v2_img2img_evaluation(
    *,
    run_id: str,
    strength: float,
    limit: int | None,
    item_ids: Sequence[str] | None,
    runs_root: Path,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    pipeline_loader: Callable[[str], object] = load_flux_img2img_pipeline,
    fid_runner: Callable[[Path, Path], tuple[float, dict[str, str]]] = compute_fid,
    require_cuda: bool = True,
) -> dict[str, Any]:
    _validate_run_id(run_id)
    if float(strength) != STRENGTH:
        raise ValueError(
            f"fixed protocol strength is {STRENGTH}; received {strength}"
        )
    dataset_root = dataset_root.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    if runs_root.is_symlink():
        raise ValueError(f"RUNS_ROOT cannot be a symlink: {runs_root}")
    run_root = (runs_root / run_id).absolute()
    if run_root.is_symlink():
        raise ValueError(f"run root cannot be a symlink: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    records, selection = load_records(dataset_root)
    selected = _select_records(records, limit=limit, item_ids=item_ids)
    total_started = time.perf_counter()
    (
        generated_rows,
        generated_now,
        resumed,
        recovered_without_manifest,
        generation_elapsed,
        seconds_per_image,
    ) = _generate(
        records=records,
        selected=selected,
        run_root=run_root,
        pipeline_loader=pipeline_loader,
        require_cuda=require_cuda,
    )

    print(f"FID_V2_MEASURE count={len(selected)}", flush=True)
    with _staged_flat(
        [record.input_path for record in selected],
        run_root=run_root,
        prefix=".fidv2-real-",
    ) as real_directory, _staged_flat(
        [Path(row["output_path"]) for row in generated_rows],
        run_root=run_root,
        prefix=".fidv2-gen-",
    ) as generated_directory:
        fid_value, clean_fid = fid_runner(real_directory, generated_directory)

    measurement: dict[str, Any] = {
        "fid": float(fid_value),
        "count": len(selected),
        "smoke": len(selected) < len(records),
        "limit": limit,
        "target": {"operator": "<=", "value": TARGET_FID},
        "verdict": "PASS" if fid_value <= TARGET_FID else "FAIL",
        "generation_elapsed_seconds": generation_elapsed,
        "seconds_per_image": seconds_per_image,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    if item_ids is not None:
        measurement["selection_mode"] = "selected"
        measurement["item_ids"] = list(item_ids)

    source_dataset = selection.get("source_dataset", {})
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "root": str(dataset_root),
            "real_set": "deterministic FID_v2 rule-based thumbnails",
            "manifest": str(dataset_root / "manifest.csv"),
            "manifest_sha256": _file_sha256(dataset_root / "manifest.csv"),
            "source_dataset": source_dataset,
            "selection": selection,
            "generated_manifest": str(run_root / "generated.csv"),
            "generated_manifest_sha256": _file_sha256(run_root / "generated.csv"),
        },
        "protocol": {
            "model": MODEL_ID,
            "generation_mode": "image-to-image",
            "image_conditioning": True,
            "strength": STRENGTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            "generator_device": "cpu",
            "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "input_preprocess": f"RGB {IMAGE_SIZE}x{IMAGE_SIZE} LANCZOS",
            "output": f"{IMAGE_SIZE}x{IMAGE_SIZE} RGB PNG lossless",
            "offload": "sequential_cpu_offload",
            "prompt_protocol": "manifest.csv prompt",
            "clean_fid": {
                **clean_fid,
                "real_set": "selected FID_v2 thumbnail set",
                "gen_set": "FLUX img2img reconstructed thumbnail set",
            },
            "determinism": measurement_determinism_state(),
            "license_note": LICENSE_NOTE,
        },
        "measurement": measurement,
        "generation_validation": {
            "count": len(selected),
            "generated_now": generated_now,
            "resumed": resumed,
            "recovered_without_manifest": recovered_without_manifest,
            "pipeline_image_argument": True,
            "hashes_verified": True,
        },
    }
    _atomic_write_json(run_root / "fid_v2.json", result)
    print(
        f"FID_V2_IMG2IMG fid={float(fid_value):.12f} count={len(selected)} "
        f"verdict={measurement['verdict']}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--item-id", action="append", dest="item_ids")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    result = run_fid_v2_img2img_evaluation(
        run_id=args.run_id,
        strength=args.strength,
        limit=args.limit,
        item_ids=args.item_ids,
        runs_root=Path(runs_root_value),
        dataset_root=args.dataset_root,
    )
    print(
        f"result={Path(runs_root_value).expanduser().absolute() / args.run_id / 'fid_v2.json'}"
    )
    print(f"FID={result['measurement']['fid']}")


if __name__ == "__main__":
    main()
