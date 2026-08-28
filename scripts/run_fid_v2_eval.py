"""Generate FID_v2 thumbnails from metadata-only prompts and score their FID.

The deterministic thumbnails in ``dataset/fid_v2/input`` are the real set X.
FLUX.1-dev receives metadata-derived text prompts only and produces generated
set Y; the reference thumbnails and source product photos are never passed to
the diffusion pipeline. FID is then measured between X and Y as two sets.

FLUX.1-dev is used only for this non-commercial benchmark/evaluation. Its
license must be reviewed again before any commercial adoption; the existing
SDXL path remains the commercial fallback outside this FID_v2 experiment.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

# This allocator setting must exist before torch (directly or transitively) is
# imported. It reduces fragmentation during FLUX sequential CPU offload.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from diffusers import FluxPipeline
from PIL import Image

from metrics.fid import compute_fid, measurement_determinism_state
from scripts.build_fid_v2_thumbnails import dominant_hsv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "fid_v2"
DEFAULT_PRODUCT_DATASET = PROJECT_ROOT / "dataset" / "fid"
RUN_ID = "fid_v2_flux"
TARGET = {"operator": "<=", "value": 10.0}
SCHEMA_VERSION = 2

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_NUM_INFERENCE_STEPS = 30
FLUX_GUIDANCE_SCALE = 3.5
IMAGE_SIZE = 1024
SEED = 0
FLUX_LICENSE_NOTE = (
    "FLUX.1-dev is used for non-commercial benchmark/evaluation only; "
    "commercial adoption requires a license review, with SDXL retained as fallback."
)

GENERATED_FIELDS = [
    "item_id",
    "reference_path",
    "output_path",
    "sha256",
    "seed",
    "model",
    "prompt",
    "elapsed_seconds",
]


@dataclass(frozen=True)
class PromptRecord:
    item_id: str
    reference_path: Path
    output_relative_path: Path
    prompt: str


@dataclass(frozen=True)
class GenerationReport:
    selected_rows: tuple[dict[str, str], ...]
    generated_now: int
    resumed: int
    elapsed_seconds: float
    seconds_per_image: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: str) -> str:
    return " ".join(value.split())


def colour_name(hue: float, saturation: float) -> str:
    """Map the deterministic dominant HSV value to a stable prompt colour name."""

    if saturation < 0.14:
        return "neutral gray"
    hue = hue % 1.0
    bands = (
        (0.04, "red"),
        (0.09, "orange"),
        (0.16, "yellow"),
        (0.25, "lime"),
        (0.43, "green"),
        (0.52, "teal"),
        (0.58, "cyan"),
        (0.70, "blue"),
        (0.78, "indigo"),
        (0.88, "purple"),
        (0.96, "magenta"),
        (1.00, "red"),
    )
    return next(name for upper, name in bands if hue < upper)


def _safe_relative_path(raw: str, *, label: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} must be a safe relative path: {raw}")
    return path


def build_thumbnail_prompt(row: dict[str, str], *, product_root: Path) -> str:
    """Describe the rule-built thumbnail using metadata and product colour only."""

    required = ("item_no", "대분류", "중분류", "소분류", "상품명", "source_product")
    missing = [field for field in required if not _normalise(row.get(field, ""))]
    if missing:
        raise ValueError(f"FID_v2 manifest row lacks required fields: {missing}")

    product_relative = _safe_relative_path(
        row["source_product"], label="source_product"
    )
    if product_relative.parts[0] != "input":
        raise ValueError(f"source_product must be below input/: {product_relative}")
    product_path = (product_root / product_relative).absolute()
    if not product_path.is_file():
        raise FileNotFoundError(f"source product image is missing: {product_path}")
    with Image.open(product_path) as product:
        hue, saturation = dominant_hsv(product)

    large = _normalise(row["대분류"])
    middle = _normalise(row["중분류"])
    small = _normalise(row["소분류"])
    title = _normalise(row["상품명"])
    tone = colour_name(hue, saturation)
    return (
        f"Influencer-style e-commerce product thumbnail, {tone}-tone vertical "
        "gradient background, a white rounded card with a soft shadow holding "
        f"the featured {small} product, a \"{middle}\" category badge, a red "
        f"\"NEW\" corner ribbon, product title \"{title}\" rendered in Korean, "
        f"{large} / {middle} / {small} product context, a rounded frame, polished "
        "commercial graphic design, square 1024x1024 composition."
    )


def load_prompt_records(dataset_root: Path, product_root: Path) -> list[PromptRecord]:
    source = dataset_root / "manifest.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"FID_v2 manifest is empty: {source}")

    records: list[PromptRecord] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        item_id = _normalise(row.get("item_no", ""))
        if not item_id:
            raise ValueError(f"manifest.csv:{row_number} item_no is empty")
        if item_id in seen:
            raise ValueError(f"manifest.csv:{row_number} duplicates item_no {item_id}")
        seen.add(item_id)

        thumbnail = _safe_relative_path(
            row.get("thumbnail", ""), label=f"manifest.csv:{row_number} thumbnail"
        )
        if thumbnail.parts[0] != "input":
            raise ValueError(f"thumbnail must be below input/: {thumbnail}")
        reference_path = (dataset_root / thumbnail).absolute()
        if not reference_path.is_file():
            raise FileNotFoundError(f"thumbnail is missing: {reference_path}")
        records.append(
            PromptRecord(
                item_id=item_id,
                reference_path=reference_path,
                output_relative_path=Path(*thumbnail.parts[1:]).with_suffix(".png"),
                prompt=build_thumbnail_prompt(row, product_root=product_root),
            )
        )
    return records


def load_flux_pipeline(model_id: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # Model CPU offload still exceeds the RTX 3090's 24 GiB in this workload.
    pipe.enable_sequential_cpu_offload()
    return pipe


def _atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError(f"refusing to replace output symlink: {output_path}")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        image.save(temp_path, format="PNG", compress_level=9)
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
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
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


@contextmanager
def _exclusive_gpu_lock(runs_root: Path) -> Iterator[None]:
    # Share the existing generation lock so web SDXL jobs and this FLUX job
    # cannot concurrently overcommit the same GPU.
    lock_path = runs_root / ".sdxl-generation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another generation process holds {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} mode=flux-txt2img\n".encode())
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_completed_rows(
    manifest_path: Path,
    *,
    records: Sequence[PromptRecord],
    run_dir: Path,
) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    if manifest_path.is_symlink():
        raise ValueError(f"generated manifest cannot be a symlink: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != GENERATED_FIELDS:
            raise ValueError(
                "generated.csv belongs to a different generation protocol; "
                f"expected {GENERATED_FIELDS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    expected = {record.item_id: record for record in records}
    completed: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        item_id = row["item_id"]
        record = expected.get(item_id)
        if record is None or item_id in completed:
            raise ValueError(f"generated.csv:{row_number} has an unknown/duplicate item")
        expected_output = (run_dir / "images" / record.output_relative_path).absolute()
        output_path = Path(row["output_path"])
        if (
            Path(row["reference_path"]) != record.reference_path
            or output_path != expected_output
            or row["prompt"] != record.prompt
            or row["model"] != FLUX_MODEL_ID
            or int(row["seed"]) != SEED
        ):
            raise ValueError(f"generated.csv:{row_number} differs from FLUX protocol")
        if not output_path.is_file() or output_path.is_symlink():
            raise ValueError(f"generated.csv:{row_number} output is missing or unsafe")
        if file_sha256(output_path) != row["sha256"]:
            raise ValueError(f"generated.csv:{row_number} output hash differs")
        float(row["elapsed_seconds"])
        completed[item_id] = dict(row)
    return completed


def generate_flux_images(
    *,
    records: Sequence[PromptRecord],
    runs_root: Path,
    limit: int | None,
    pipeline_loader: Callable[[str], object] = load_flux_pipeline,
    require_cuda: bool = True,
) -> GenerationReport:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than 0")
    selected = list(records[:limit]) if limit is not None else list(records)
    if len(selected) < 2:
        raise ValueError("FID_v2 requires at least two selected images")

    runs_root = runs_root.expanduser().absolute()
    run_dir = runs_root / RUN_ID
    manifest_path = run_dir / "generated.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_rows(
        manifest_path, records=records, run_dir=run_dir
    )
    pending = [record for record in selected if record.item_id not in completed]
    started = time.perf_counter()

    if pending:
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("FLUX.1-dev generation requires CUDA")
        with _exclusive_gpu_lock(runs_root):
            pipe = pipeline_loader(FLUX_MODEL_ID)
            try:
                for index, record in enumerate(pending, start=1):
                    item_started = time.perf_counter()
                    generator = torch.Generator(device="cpu").manual_seed(SEED)
                    result = pipe(
                        prompt=record.prompt,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                        num_inference_steps=FLUX_NUM_INFERENCE_STEPS,
                        guidance_scale=FLUX_GUIDANCE_SCALE,
                        generator=generator,
                    )
                    image = result.images[0].convert("RGB")
                    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError(
                            f"FLUX returned {image.size}, expected {IMAGE_SIZE}x{IMAGE_SIZE}"
                        )
                    output_path = (
                        run_dir / "images" / record.output_relative_path
                    ).absolute()
                    _atomic_save_png(image, output_path)
                    row = {
                        "item_id": record.item_id,
                        "reference_path": str(record.reference_path),
                        "output_path": str(output_path),
                        "sha256": file_sha256(output_path),
                        "seed": str(SEED),
                        "model": FLUX_MODEL_ID,
                        "prompt": record.prompt,
                        "elapsed_seconds": f"{time.perf_counter() - item_started:.6f}",
                    }
                    completed[record.item_id] = row
                    ordered = [
                        completed[item.item_id]
                        for item in records
                        if item.item_id in completed
                    ]
                    _atomic_write_csv(manifest_path, ordered)
                    print(
                        f"FID_V2_GENERATE done={len(selected) - len(pending) + index}/"
                        f"{len(selected)} item_id={record.item_id}"
                    )
            finally:
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    selected_rows = tuple(completed[record.item_id] for record in selected)
    elapsed = time.perf_counter() - started
    recorded_elapsed = sum(float(row["elapsed_seconds"]) for row in selected_rows)
    return GenerationReport(
        selected_rows=selected_rows,
        generated_now=len(pending),
        resumed=len(selected) - len(pending),
        elapsed_seconds=elapsed,
        seconds_per_image=recorded_elapsed / len(selected_rows),
    )


@contextmanager
def _staged_flat(paths: Sequence[Path], prefix: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as raw_directory:
        directory = Path(raw_directory)
        for index, path in enumerate(paths):
            (directory / f"{index:04d}{path.suffix.lower()}").symlink_to(path)
        yield directory


def measure(
    dataset_root: Path,
    product_root: Path,
    runs_root: Path,
    limit: int | None,
    *,
    pipeline_loader: Callable[[str], object] = load_flux_pipeline,
    fid_runner: Callable[[Path, Path], tuple[float, dict[str, str]]] = compute_fid,
    require_cuda: bool = True,
) -> dict:
    dataset_root = dataset_root.expanduser().absolute()
    product_root = product_root.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    records = load_prompt_records(dataset_root, product_root)
    selected = records[:limit] if limit is not None else records

    total_started = time.perf_counter()
    generation = generate_flux_images(
        records=records,
        runs_root=runs_root,
        limit=limit,
        pipeline_loader=pipeline_loader,
        require_cuda=require_cuda,
    )
    generated_paths = [Path(row["output_path"]) for row in generation.selected_rows]
    reference_paths = [record.reference_path for record in selected]
    with _staged_flat(reference_paths, "fidv2-real-") as real_dir, _staged_flat(
        generated_paths, "fidv2-gen-"
    ) as gen_dir:
        fid_value, clean_fid = fid_runner(real_dir, gen_dir)

    verdict = "PASS" if fid_value <= TARGET["value"] else "FAIL"
    total_elapsed = time.perf_counter() - total_started
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "dataset": {
            "root": str(dataset_root),
            "real_set": "deterministic FID_v2 influencer-style thumbnails",
            "manifest": str(dataset_root / "manifest.csv"),
            "manifest_sha256": file_sha256(dataset_root / "manifest.csv"),
            "selection": json.loads(
                (dataset_root / "selection.json").read_text(encoding="utf-8")
            ),
        },
        "protocol": {
            "model": FLUX_MODEL_ID,
            "generation_mode": "text-to-image",
            "image_conditioning": False,
            "seed": SEED,
            "generator_device": "cpu",
            "num_inference_steps": FLUX_NUM_INFERENCE_STEPS,
            "guidance_scale": FLUX_GUIDANCE_SCALE,
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "offload": "sequential_cpu_offload",
            "allocator": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
            "prompt_protocol": (
                "deterministic manifest metadata + dominant product colour; "
                "legacy manifest prompt ignored"
            ),
            "sample_prompt": selected[0].prompt,
            "license_note": FLUX_LICENSE_NOTE,
            "clean_fid": {
                **clean_fid,
                "real_set": "FID_v2 thumbnail set",
                "gen_set": "FLUX prompt-only generated set",
            },
            "determinism": measurement_determinism_state(),
        },
        "measurement": {
            "fid": float(fid_value),
            "count": len(generated_paths),
            "smoke": limit is not None,
            "limit": limit,
            "target": TARGET,
            "verdict": verdict,
            "generation_elapsed_seconds": generation.elapsed_seconds,
            "seconds_per_image": generation.seconds_per_image,
            "total_elapsed_seconds": total_elapsed,
        },
        "generation_validation": {
            "count": len(generated_paths),
            "generated_now": generation.generated_now,
            "resumed": generation.resumed,
            "pipeline_image_argument": False,
            "hashes_verified": True,
        },
    }
    _atomic_write_json(runs_root / RUN_ID / "fid_v2.json", result)
    return result


def verify_only(runs_root: Path, dataset_root: Path, product_root: Path) -> dict:
    runs_root = runs_root.expanduser().absolute()
    dataset_root = dataset_root.expanduser().absolute()
    product_root = product_root.expanduser().absolute()
    run_dir = runs_root / RUN_ID
    stored = json.loads((run_dir / "fid_v2.json").read_text(encoding="utf-8"))
    count = int(stored["measurement"]["count"])
    records = load_prompt_records(dataset_root, product_root)
    completed = _load_completed_rows(
        run_dir / "generated.csv", records=records, run_dir=run_dir
    )
    selected = records[:count]
    rows = [completed[record.item_id] for record in selected]
    with _staged_flat(
        [record.reference_path for record in selected], "fidv2-vreal-"
    ) as real_dir, _staged_flat(
        [Path(row["output_path"]) for row in rows], "fidv2-vgen-"
    ) as gen_dir:
        fid_value, _ = compute_fid(real_dir, gen_dir)
    recorded = float(stored["measurement"]["fid"])
    if abs(fid_value - recorded) > 1e-9:
        raise ValueError(f"FID mismatch: recomputed {fid_value} != recorded {recorded}")
    print(f"FID_V2_VERIFIED run_id={RUN_ID} fid={fid_value:.12f} count={count}")
    return stored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCT_DATASET)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    runs_root = Path(runs_root_value) if runs_root_value else PROJECT_ROOT / "dataset" / "runs"

    if args.verify_only:
        verify_only(runs_root, args.dataset_root, args.product_root)
        return

    result = measure(
        dataset_root=args.dataset_root,
        product_root=args.product_root,
        runs_root=runs_root,
        limit=args.limit,
    )
    measurement = result["measurement"]
    print(json.dumps(result["protocol"], ensure_ascii=False))
    print(
        f"FID_V2 fid={measurement['fid']:.12f} count={measurement['count']} "
        f"elapsed_seconds={measurement['generation_elapsed_seconds']:.3f} "
        f"seconds_per_image={measurement['seconds_per_image']:.3f} "
        f"verdict={measurement['verdict']}"
    )


if __name__ == "__main__":
    main()
