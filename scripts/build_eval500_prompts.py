#!/usr/bin/env python3
"""Build deterministic per-image prompts for the fixed EVAL500 dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_METADATA_ROOT = Path("/home/kim_3090/datasets/abo/listings/metadata")
DEFAULT_INPUT_MANIFEST = Path(
    "/home/kim_3090/datasets/abo/curated/eval500/manifests/input.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/kim_3090/datasets/abo/curated/eval500/manifests/prompts.csv"
)
PROMPT_TEMPLATE = (
    "a high quality studio product photograph of {item_name}, "
    "on a clean white background"
)
PROMPT_FIELDS = ["item_id", "prompt", "name_source"]
NAME_MAX_WORDS = 15


@dataclass(frozen=True)
class PromptBuildReport:
    output_path: Path
    count: int
    english_count: int
    fallback_count: int
    sha256: str


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"input manifest does not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"item_id", "product_type"}
        if not required.issubset(fields):
            raise ValueError(
                f"input manifest lacks required fields: {sorted(required - set(fields))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"input manifest is empty: {path}")
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        item_id = _normalize_whitespace(row.get("item_id", ""))
        if not item_id:
            raise ValueError(f"input manifest row {row_number} has an empty item_id")
        if item_id in seen:
            raise ValueError(f"duplicate input manifest item_id: {item_id}")
        seen.add(item_id)
        row["item_id"] = item_id
    return rows


def _target_metadata(metadata_root: Path, item_ids: set[str]) -> dict[str, str | None]:
    metadata_root = metadata_root.expanduser().absolute()
    files = sorted(metadata_root.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"no metadata JSON gzip files found below {metadata_root}")
    matched: dict[str, str | None] = {}
    for metadata_path in files:
        with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    listing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {metadata_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(listing, dict):
                    continue
                item_id = listing.get("item_id")
                if item_id not in item_ids:
                    continue
                if item_id not in matched:
                    matched[item_id] = None
                if matched[item_id] is None:
                    matched[item_id] = _english_name(listing)
    missing = sorted(item_ids - set(matched))
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            f"missing target metadata for {len(missing)} item_id(s): {preview}{suffix}"
        )
    return matched


def _english_name(listing: dict[str, Any]) -> str | None:
    names = listing.get("item_name")
    if not isinstance(names, list):
        return None
    for entry in names:
        if not isinstance(entry, dict):
            continue
        language_tag = entry.get("language_tag")
        value = entry.get("value")
        if not isinstance(language_tag, str) or not language_tag.lower().startswith("en"):
            continue
        if not isinstance(value, str):
            continue
        normalized = _normalize_whitespace(value)
        if normalized:
            return normalized
    return None


def _prompt_row(
    manifest_row: dict[str, str], english_name: str | None
) -> dict[str, str]:
    name = english_name
    if name is None:
        name = _normalize_whitespace(
            manifest_row.get("product_type", "").lower().replace("_", " ")
        )
        source = "fallback"
    else:
        source = "en"
    words = name.split()
    if not words:
        raise ValueError(
            f"item_id {manifest_row['item_id']} has no usable English name or product_type"
        )
    truncated_name = " ".join(words[:NAME_MAX_WORDS])
    return {
        "item_id": manifest_row["item_id"],
        "prompt": PROMPT_TEMPLATE.format(item_name=truncated_name),
        "name_source": source,
    }


def _atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=".prompts.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_prompt_manifest(
    *,
    metadata_root: Path = DEFAULT_METADATA_ROOT,
    input_manifest: Path = DEFAULT_INPUT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> PromptBuildReport:
    """Join EVAL500 item IDs to ABO names and atomically write prompts.csv."""

    manifest_rows = _manifest_rows(input_manifest)
    metadata = _target_metadata(
        metadata_root,
        {row["item_id"] for row in manifest_rows},
    )
    prompt_rows = [
        _prompt_row(row, metadata[row["item_id"]]) for row in manifest_rows
    ]
    output_path = output_path.expanduser().absolute()
    _atomic_write_csv(output_path, prompt_rows)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    english_count = sum(row["name_source"] == "en" for row in prompt_rows)
    return PromptBuildReport(
        output_path=output_path,
        count=len(prompt_rows),
        english_count=english_count,
        fallback_count=len(prompt_rows) - english_count,
        sha256=digest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_prompt_manifest(
        metadata_root=args.metadata_root,
        input_manifest=args.input_manifest,
        output_path=args.output,
    )
    payload = asdict(report)
    payload["output_path"] = str(report.output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
