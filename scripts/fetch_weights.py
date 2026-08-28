#!/usr/bin/env python3
"""Cache SDXL fp16 weights and optionally run a single-image GPU smoke test."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
EVAL500_ROOT = Path("/home/kim_3090/datasets/abo/curated/eval500")
INPUT_MANIFEST = EVAL500_ROOT / "manifests" / "input.csv"
IMAGE_SIZE = (1024, 1024)
STRENGTH = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate one 1024x1024 image after caching the model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/sdxl-smoke.png"),
        help="PNG output path used with --smoke.",
    )
    return parser.parse_args()


def first_eval500_input() -> Path:
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError(f"EVAL500 input manifest not found: {INPUT_MANIFEST}")

    with INPUT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise RuntimeError(f"EVAL500 input manifest is empty: {INPUT_MANIFEST}")

    for key in ("selected_path", "path", "input_path", "file"):
        value = row.get(key)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = EVAL500_ROOT / candidate
        if candidate.is_file():
            return candidate

    raise RuntimeError(
        "The first EVAL500 manifest row has no usable image path; "
        f"columns={list(row)}"
    )


def load_pipeline() -> StableDiffusionXLImg2ImgPipeline:
    return StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )


def run_smoke(pipe: StableDiffusionXLImg2ImgPipeline, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SDXL smoke test")

    input_path = first_eval500_input()
    with Image.open(input_path) as source:
        input_image = source.convert("RGB").resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=False)
    pipe.to("cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started_at = time.perf_counter()

    with torch.inference_mode():
        generated = pipe(
            prompt="a high quality studio product photograph on a clean white background",
            image=input_image,
            strength=STRENGTH,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images[0]

    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)

    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Smoke output must be a PNG path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated.save(output_path, format="PNG")

    print(f"model={MODEL_ID}")
    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"size={generated.width}x{generated.height}")
    print(f"strength={STRENGTH}")
    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    print(f"peak_vram_gib={peak_vram_gib:.3f}")


def main() -> None:
    args = parse_args()
    pipe = load_pipeline()
    if args.smoke:
        run_smoke(pipe, args.out)
    else:
        print(f"cached_model={MODEL_ID}")


if __name__ == "__main__":
    main()
