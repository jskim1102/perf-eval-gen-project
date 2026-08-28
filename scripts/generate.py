#!/usr/bin/env python3
"""Generate deterministic SDXL img2img reconstructions from an EVAL500 manifest."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT = "a high quality studio product photograph on a clean white background"
IMAGE_SIZE = (1024, 1024)
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength"]
PROMPT_FIELD = "prompt"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class GenerationConfig:
    strength: float
    seed: int
    run_id: str
    limit: int | None = None
    model: str = MODEL_ID

    def __post_init__(self) -> None:
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("strength must be greater than 0 and at most 1")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be between 0 and 2**63 - 1")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be greater than 0")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        validate_run_id(self.run_id)


@dataclass(frozen=True)
class InputRecord:
    input_path: Path
    output_relative_path: Path


@dataclass(frozen=True)
class GenerationReport:
    run_root: Path
    generated_manifest: Path
    count: int
    elapsed_seconds: float
    peak_vram_gib: float
    hashes: tuple[str, ...]


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {raw_path}")
    return path


def load_manifest(
    manifest_path: Path,
    *,
    item_ids: Sequence[str] | None = None,
) -> list[InputRecord]:
    manifest_path = manifest_path.expanduser().absolute()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"input manifest not found: {manifest_path}")

    eval_root = manifest_path.parent.parent
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "selected_path" not in reader.fieldnames:
            raise ValueError("input manifest must contain selected_path")
        rows = list(reader)
    if not rows:
        raise ValueError(f"input manifest is empty: {manifest_path}")
    if item_ids is not None:
        requested = list(item_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("item_ids must be a non-empty unique list")
        if reader.fieldnames is None or "item_id" not in reader.fieldnames:
            raise ValueError("input manifest must contain item_id for selected generation")
        available = {row.get("item_id", "") for row in rows}
        missing = [item_id for item_id in requested if item_id not in available]
        if missing:
            raise ValueError(f"item_ids are not in the input manifest: {missing}")
        requested_set = set(requested)
        rows = [row for row in rows if row.get("item_id") in requested_set]

    records: list[InputRecord] = []
    seen_inputs: set[Path] = set()
    seen_outputs: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        selected = _safe_relative_path(
            row.get("selected_path", ""),
            label=f"input.csv:{row_number} selected_path",
        )
        if not selected.parts or selected.parts[0] != "input" or len(selected.parts) < 2:
            raise ValueError(
                f"input.csv:{row_number} selected_path must be below input/: {selected}"
            )
        input_path = (eval_root / selected).absolute()
        if not input_path.is_file():
            raise FileNotFoundError(f"manifest input is missing: {input_path}")

        output_relative = Path(*selected.parts[1:]).with_suffix(".png")
        if input_path in seen_inputs:
            raise ValueError(f"duplicate input path in manifest: {input_path}")
        if output_relative in seen_outputs:
            raise ValueError(f"duplicate output path in manifest: {output_relative}")
        seen_inputs.add(input_path)
        seen_outputs.add(output_relative)
        records.append(
            InputRecord(
                input_path=input_path,
                output_relative_path=output_relative,
            )
        )
    return records


def load_pipeline(
    model_id: str,
) -> StableDiffusionXLImg2ImgPipeline | StableDiffusionImg2ImgPipeline:
    pipeline_class = (
        StableDiffusionXLImg2ImgPipeline
        if model_id == MODEL_ID
        else StableDiffusionImg2ImgPipeline
    )
    pipeline = pipeline_class.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    return pipeline


@contextmanager
def _exclusive_gpu_lock(runs_root: Path) -> Iterator[None]:
    lock_path = runs_root / ".sdxl-generation.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another SDXL generation process holds the GPU lock: {lock_path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _prepare_input(input_path: Path) -> Image.Image:
    with Image.open(input_path) as source:
        return source.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def _atomic_save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_symlink():
        raise ValueError(f"refusing to replace an output symlink: {output_path}")
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


def _atomic_write_manifest(
    path: Path,
    rows: Sequence[dict[str, str]],
    *,
    fieldnames: Sequence[str] = GENERATED_FIELDS,
) -> None:
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _verified_completed_rows(
    *,
    manifest_path: Path,
    records: Sequence[InputRecord],
    output_paths: Sequence[Path],
    config: GenerationConfig,
    prompts: Mapping[Path, str] | None,
) -> dict[Path, dict[str, str]]:
    """Load only resumable rows whose paths, protocol, file, and hash still match."""

    if not manifest_path.exists():
        return {}
    if manifest_path.is_symlink():
        raise ValueError(f"generated manifest cannot be a symlink: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = (
            [*GENERATED_FIELDS, PROMPT_FIELD]
            if prompts is not None
            else GENERATED_FIELDS
        )
        if list(reader.fieldnames or []) != expected_fields:
            raise ValueError(f"generated manifest fields differ: {reader.fieldnames}")
        rows = list(reader)

    expected = {
        record.input_path: output_path
        for record, output_path in zip(records, output_paths, strict=True)
    }
    completed: dict[Path, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        input_path = Path(row["input_path"])
        output_path = Path(row["output_path"])
        if input_path in completed:
            raise ValueError(f"generated.csv:{row_number} duplicates input_path")
        if input_path not in expected or output_path != expected[input_path]:
            raise ValueError(
                f"generated.csv:{row_number} paths differ from the requested manifest"
            )
        if int(row["seed"]) != config.seed or float(row["strength"]) != config.strength:
            raise ValueError(
                f"generated.csv:{row_number} seed/strength differs from requested protocol"
            )
        if prompts is not None and row[PROMPT_FIELD] != prompts[input_path]:
            raise ValueError(f"generated.csv:{row_number} prompt differs from requested protocol")
        if not output_path.is_file() or output_path.is_symlink():
            raise ValueError(f"generated.csv:{row_number} output is missing or a symlink")
        actual_hash = file_sha256(output_path)
        if actual_hash != row["sha256"]:
            raise ValueError(f"generated.csv:{row_number} output sha256 differs")
        completed[input_path] = dict(row)
    return completed


def _ensure_output_scope(
    *, run_root: Path, records: Sequence[InputRecord]
) -> list[Path]:
    if run_root.is_symlink():
        raise ValueError(f"run directory cannot be a symlink: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    images_root = run_root / "images"
    if images_root.is_symlink():
        raise ValueError(f"images directory cannot be a symlink: {images_root}")
    images_root.mkdir(parents=True, exist_ok=True)

    output_paths = [(images_root / record.output_relative_path).absolute() for record in records]
    expected = set(output_paths)
    existing = {path.absolute() for path in images_root.rglob("*.png")}
    unexpected = sorted(existing - expected)
    if unexpected:
        raise ValueError(
            "run contains PNGs outside the requested manifest/limit; use a new run_id: "
            + ", ".join(str(path) for path in unexpected[:3])
        )

    resolved_run_root = run_root.resolve()
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_path.parent.resolve().relative_to(resolved_run_root)
        except ValueError as exc:
            raise ValueError(f"output path escapes run directory: {output_path}") from exc
    return output_paths


def _configure_determinism(*, require_cuda: bool) -> None:
    torch.manual_seed(0)
    if require_cuda:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def run_generation(
    *,
    manifest_path: Path,
    runs_root: Path,
    config: GenerationConfig,
    pipeline_loader: Callable[[str], Any] = load_pipeline,
    require_cuda: bool = True,
    item_ids: Sequence[str] | None = None,
    prompt_resolver: Callable[[InputRecord], str] | None = None,
) -> GenerationReport:
    records = load_manifest(manifest_path, item_ids=item_ids)
    if config.limit is not None:
        records = records[: config.limit]
    if not records:
        raise ValueError("no manifest inputs selected for generation")

    prompts: dict[Path, str] | None = None
    if prompt_resolver is not None:
        prompts = {}
        for record in records:
            prompt = prompt_resolver(record)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"prompt resolver returned an empty prompt: {record.input_path}")
            prompts[record.input_path] = prompt

    runs_root = runs_root.expanduser().absolute()
    if runs_root.is_symlink():
        raise ValueError(f"RUNS_ROOT cannot be a symlink: {runs_root}")
    runs_root.mkdir(parents=True, exist_ok=True)
    run_root = runs_root / config.run_id

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SDXL generation")
    device = "cuda" if require_cuda else "cpu"
    _configure_determinism(require_cuda=require_cuda)

    started_at = time.perf_counter()
    with _exclusive_gpu_lock(runs_root):
        output_paths = _ensure_output_scope(run_root=run_root, records=records)
        generated_manifest = run_root / "generated.csv"
        completed = _verified_completed_rows(
            manifest_path=generated_manifest,
            records=records,
            output_paths=output_paths,
            config=config,
            prompts=prompts,
        )
        pending_count = sum(record.input_path not in completed for record in records)
        if require_cuda and pending_count:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pipeline = None
        if pending_count:
            pipeline = pipeline_loader(config.model)
            pipeline.vae.enable_tiling()
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(device)

        rows: list[dict[str, str]] = []
        hashes: list[str] = []
        fields = (
            [*GENERATED_FIELDS, PROMPT_FIELD]
            if prompts is not None
            else GENERATED_FIELDS
        )
        for index, (record, output_path) in enumerate(
            zip(records, output_paths, strict=True), start=1
        ):
            resumed = completed.get(record.input_path)
            if resumed is not None:
                rows.append(resumed)
                hashes.append(resumed["sha256"])
                print(
                    f"resumed={index}/{len(records)} sha256={resumed['sha256']} "
                    f"output={output_path}",
                    flush=True,
                )
                if index % 10 == 0:
                    _atomic_write_manifest(generated_manifest, rows, fieldnames=fields)
                continue
            if pipeline is None:
                raise AssertionError("pending generation has no loaded pipeline")
            input_image = _prepare_input(record.input_path)
            prompt = prompts[record.input_path] if prompts is not None else PROMPT
            generator = torch.Generator(device=device).manual_seed(config.seed)
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    image=input_image,
                    strength=config.strength,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=generator,
                )
            if not result.images or not isinstance(result.images[0], Image.Image):
                raise RuntimeError("SDXL pipeline returned no PIL image")
            generated = result.images[0]
            if generated.size != IMAGE_SIZE:
                raise RuntimeError(
                    "SDXL pipeline returned an unexpected size: "
                    f"{generated.width}x{generated.height}"
                )
            generated = generated.convert("RGB")
            _atomic_save_png(generated, output_path)
            output_hash = file_sha256(output_path)
            hashes.append(output_hash)
            row = {
                "input_path": str(record.input_path),
                "output_path": str(output_path),
                "sha256": output_hash,
                "seed": str(config.seed),
                "strength": str(config.strength),
            }
            if prompts is not None:
                row[PROMPT_FIELD] = prompt
            rows.append(row)
            print(
                f"generated={index}/{len(records)} sha256={output_hash} "
                f"output={output_path}",
                flush=True,
            )
            if index % 10 == 0:
                _atomic_write_manifest(generated_manifest, rows, fieldnames=fields)

        _atomic_write_manifest(generated_manifest, rows, fieldnames=fields)
        if require_cuda and pending_count:
            torch.cuda.synchronize()
            peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            peak_vram_gib = 0.0

    elapsed_seconds = time.perf_counter() - started_at
    return GenerationReport(
        run_root=run_root,
        generated_manifest=generated_manifest,
        count=len(rows),
        elapsed_seconds=elapsed_seconds,
        peak_vram_gib=peak_vram_gib,
        hashes=tuple(hashes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    report = run_generation(
        manifest_path=args.manifest,
        runs_root=Path(runs_root_value),
        config=GenerationConfig(
            strength=args.strength,
            seed=args.seed,
            limit=args.limit,
            run_id=args.run_id,
            model=args.model,
        ),
    )
    print(f"run_root={report.run_root}")
    print(f"generated_manifest={report.generated_manifest}")
    print(f"count={report.count}")
    print(f"model={args.model}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"peak_vram_gib={report.peak_vram_gib:.3f}")


if __name__ == "__main__":
    main()
