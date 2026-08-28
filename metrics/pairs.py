"""Load fixed original/generated image pairs without relying on file ordering."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength"]
PROMPT_FIELD = "prompt"


@dataclass(frozen=True)
class ImagePair:
    item_id: str
    group: str
    input_path: Path
    output_path: Path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _pair_input_path(pair_manifest: Path, row: dict[str, str], row_number: int) -> Path:
    selected_path = Path(row.get("selected_path", ""))
    if selected_path.is_absolute() or ".." in selected_path.parts:
        raise ValueError(
            f"pair manifest row {row_number} has unsafe selected_path: {selected_path}"
        )
    if not selected_path.parts or selected_path.parts[0] != "input":
        raise ValueError(
            f"pair manifest row {row_number} selected_path must be below input/"
        )
    input_path = (pair_manifest.parent.parent / selected_path).absolute()
    if not input_path.is_file():
        raise FileNotFoundError(f"pair input does not exist: {input_path}")
    return input_path


def load_image_pairs(
    pair_manifest: Path,
    generated_manifest: Path,
    *,
    require_all: bool = True,
) -> list[ImagePair]:
    """Join by absolute input path while preserving pair-manifest row order."""

    pair_manifest = pair_manifest.expanduser().absolute()
    generated_manifest = generated_manifest.expanduser().absolute()
    generated_fields, generated_rows = _read_csv(generated_manifest)
    if generated_fields not in (GENERATED_FIELDS, [*GENERATED_FIELDS, PROMPT_FIELD]):
        raise ValueError(f"generated.csv fields differ: {generated_fields}")

    generated_by_input: dict[Path, Path] = {}
    for row_number, row in enumerate(generated_rows, start=2):
        input_path = Path(row["input_path"])
        output_path = Path(row["output_path"])
        if not input_path.is_absolute() or not output_path.is_absolute():
            raise ValueError(f"generated.csv row {row_number} paths must be absolute")
        if input_path in generated_by_input:
            raise ValueError(f"duplicate generated input_path: {input_path}")
        if not output_path.is_file():
            raise FileNotFoundError(f"generated output does not exist: {output_path}")
        generated_by_input[input_path] = output_path

    pair_fields, pair_rows = _read_csv(pair_manifest)
    required_fields = {"item_id", "group", "selected_path"}
    if not required_fields.issubset(pair_fields):
        raise ValueError(
            f"pair manifest lacks required fields: {sorted(required_fields - set(pair_fields))}"
        )

    pairs: list[ImagePair] = []
    seen_item_ids: set[str] = set()
    for row_number, row in enumerate(pair_rows, start=2):
        input_path = _pair_input_path(pair_manifest, row, row_number)
        output_path = generated_by_input.get(input_path)
        if output_path is None:
            if require_all:
                raise ValueError(f"no generated output for fixed pair input: {input_path}")
            continue
        item_id = row["item_id"]
        if item_id in seen_item_ids:
            raise ValueError(f"duplicate pair item_id: {item_id}")
        seen_item_ids.add(item_id)
        pairs.append(
            ImagePair(
                item_id=item_id,
                group=row["group"],
                input_path=input_path,
                output_path=output_path,
            )
        )
    if not pairs:
        raise ValueError("no fixed image pairs matched generated.csv")
    return pairs


def load_rgb_arrays(pair: ImagePair) -> tuple[np.ndarray, np.ndarray]:
    """Resize original to generated dimensions and return RGB uint8 arrays."""

    with Image.open(pair.output_path) as output_image:
        generated_rgb = output_image.convert("RGB")
        output_size = generated_rgb.size
        generated = np.asarray(generated_rgb, dtype=np.uint8).copy()
    with Image.open(pair.input_path) as input_image:
        original_rgb = input_image.convert("RGB").resize(
            output_size, Image.Resampling.LANCZOS
        )
        original = np.asarray(original_rgb, dtype=np.uint8).copy()
    if original.shape != generated.shape:
        raise ValueError(
            f"pair array shapes differ after resize: {original.shape} != {generated.shape}"
        )
    return original, generated
