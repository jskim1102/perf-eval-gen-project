#!/usr/bin/env python3
"""Materialize the frozen V2 web datasets inside this project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FID_SOURCE = Path("/home/kim_3090/datasets/aihub-product/fid500-v2")
DEFAULT_PSNR_SSIM_SOURCE = Path("/home/kim_3090/datasets/abo/curated/eval500")
DEFAULT_FID_DESTINATION = PROJECT_ROOT / "dataset" / "fid"
DEFAULT_PSNR_SSIM_DESTINATION = PROJECT_ROOT / "dataset" / "psnr_ssim"
FID_GENERATION_FIELDS = [
    "item_id",
    "group",
    "width",
    "height",
    "sha256",
    "source_path",
    "selected_path",
]
PROMPT_FIELDS = ["item_id", "prompt", "name_source"]


@dataclass(frozen=True)
class DatasetBuildReport:
    destination_root: Path
    count: int
    manifest_sha256: str
    prompt_manifest_sha256: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_destination(root: Path) -> Path:
    root = root.expanduser().absolute()
    if root.is_symlink():
        raise ValueError(f"destination root cannot be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError(f"destination root is not a directory: {root}")
    return root


def _safe_relative(raw: str, *, label: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"{label} is not a safe relative path: {raw}")
    return relative


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"source file does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.parent.is_symlink():
        raise ValueError(f"destination path cannot contain a symlink: {destination}")
    if destination.is_file() and _sha256(destination) == _sha256(source):
        return
    temporary: Path | None = None
    try:
        with source.open("rb") as source_handle, tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as destination_handle:
            temporary = Path(destination_handle.name)
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_csv(
    path: Path,
    *,
    fields: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"CSV destination cannot contain a symlink: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError(f"JSON destination cannot contain a symlink: {path}")
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
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


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            return fields, list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read CSV {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return value


def build_fid_web_dataset(
    *,
    source_root: Path = DEFAULT_FID_SOURCE,
    destination_root: Path = DEFAULT_FID_DESTINATION,
    expected_count: int = 500,
) -> DatasetBuildReport:
    """Copy only the final V2 FID reference set, excluding its selection pool."""

    source_root = source_root.expanduser().absolute()
    destination_root = _prepare_destination(destination_root)
    manifest_path = source_root / "manifest.csv"
    selection_path = source_root / "selection.json"
    fields, rows = _read_csv(manifest_path)
    required = {"item_no", "대분류", "zip_member", "width", "height", "sha256"}
    if not required.issubset(fields):
        raise ValueError(f"FID manifest lacks fields: {sorted(required - set(fields))}")
    selection = _read_json(selection_path)
    if len(rows) != expected_count or selection.get("counts", {}).get(
        "final_count"
    ) != expected_count:
        raise ValueError(
            f"FID final count differs from {expected_count}: "
            f"{len(rows)}/{selection.get('counts', {}).get('final_count')}"
        )
    manifest_hash = _sha256(manifest_path)
    if selection.get("manifest_sha256") != manifest_hash:
        raise ValueError("FID selection manifest SHA-256 differs from manifest.csv")
    category_directories = selection.get("rules", {}).get("category_directories")
    if not isinstance(category_directories, dict):
        raise ValueError("FID selection lacks category directory rules")

    generation_rows: list[dict[str, str]] = []
    seen_items: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        item_no = row["item_no"]
        if item_no in seen_items:
            raise ValueError(f"duplicate FID item_no: {item_no}")
        seen_items.add(item_no)
        try:
            category_directory = str(category_directories[row["대분류"]])
        except KeyError as exc:
            raise ValueError(
                f"FID manifest row {row_number} has no category directory"
            ) from exc
        relative = _safe_relative(
            f"input/{category_directory}/{item_no}.jpg",
            label=f"FID manifest row {row_number}",
        )
        source_image = source_root / relative
        if _sha256(source_image) != row["sha256"]:
            raise ValueError(f"FID source image SHA-256 differs: {source_image}")
        destination_image = destination_root / relative
        _atomic_copy(source_image, destination_image)
        if _sha256(destination_image) != row["sha256"]:
            raise ValueError(f"FID copied image SHA-256 differs: {destination_image}")
        generation_rows.append(
            {
                "item_id": item_no,
                "group": row["대분류"],
                "width": row["width"],
                "height": row["height"],
                "sha256": row["sha256"],
                "source_path": row["zip_member"],
                "selected_path": relative.as_posix(),
            }
        )

    _atomic_copy(manifest_path, destination_root / "manifest.csv")
    web_selection = json.loads(json.dumps(selection))
    if "pool_manifest" in web_selection:
        web_selection["pool_manifest"] = (
            "not materialized; frozen selection provenance only"
        )
    method_details = web_selection.get("method_details")
    if isinstance(method_details, dict) and "feature_cache" in method_details:
        method_details["feature_cache"] = (
            "not materialized; frozen selection provenance only"
        )
    _atomic_write_json(destination_root / "selection.json", web_selection)
    _atomic_write_csv(
        destination_root / "manifests" / "input.csv",
        fields=FID_GENERATION_FIELDS,
        rows=generation_rows,
    )
    return DatasetBuildReport(
        destination_root=destination_root,
        count=len(rows),
        manifest_sha256=manifest_hash,
    )


def build_psnr_ssim_web_dataset(
    *,
    source_root: Path = DEFAULT_PSNR_SSIM_SOURCE,
    destination_root: Path = DEFAULT_PSNR_SSIM_DESTINATION,
    expected_count: int = 100,
) -> DatasetBuildReport:
    """Materialize the fixed PSNR/SSIM pairs and their per-image V2 prompts."""

    source_root = source_root.expanduser().absolute()
    destination_root = _prepare_destination(destination_root)
    manifests = source_root / "manifests"
    pair_path = manifests / "psnr_ssim_100.csv"
    prompt_path = manifests / "prompts.csv"
    fields, rows = _read_csv(pair_path)
    required = {
        "split",
        "group",
        "product_type",
        "item_id",
        "image_id",
        "width",
        "height",
        "sha256",
        "source_path",
        "selected_path",
    }
    if not required.issubset(fields):
        raise ValueError(
            f"PSNR/SSIM manifest lacks fields: {sorted(required - set(fields))}"
        )
    if len(rows) != expected_count:
        raise ValueError(
            f"PSNR/SSIM pair count differs from {expected_count}: {len(rows)}"
        )
    prompt_fields, prompt_rows = _read_csv(prompt_path)
    if prompt_fields != PROMPT_FIELDS:
        raise ValueError(f"prompt manifest fields differ: {prompt_fields}")
    prompts_by_item: dict[str, dict[str, str]] = {}
    for row in prompt_rows:
        item_id = row["item_id"]
        if item_id in prompts_by_item:
            raise ValueError(f"duplicate prompt item_id: {item_id}")
        prompts_by_item[item_id] = row

    copied_rows: list[dict[str, str]] = []
    selected_prompts: list[dict[str, str]] = []
    seen_items: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        item_id = row["item_id"]
        if item_id in seen_items:
            raise ValueError(f"duplicate PSNR/SSIM item_id: {item_id}")
        seen_items.add(item_id)
        relative = _safe_relative(
            row["selected_path"], label=f"pair manifest row {row_number}"
        )
        if not relative.parts or relative.parts[0] != "input":
            raise ValueError(
                f"pair manifest row {row_number} selected_path must be below input/"
            )
        source_image = source_root / relative
        if _sha256(source_image) != row["sha256"]:
            raise ValueError(f"PSNR/SSIM source image SHA-256 differs: {source_image}")
        destination_image = destination_root / relative
        _atomic_copy(source_image, destination_image)
        if _sha256(destination_image) != row["sha256"]:
            raise ValueError(f"PSNR/SSIM copied image SHA-256 differs: {destination_image}")
        copied_rows.append({**row, "source_path": str(destination_image.absolute())})
        try:
            selected_prompts.append(prompts_by_item[item_id])
        except KeyError as exc:
            raise ValueError(f"prompt manifest is missing pair item_id: {item_id}") from exc

    output_manifests = destination_root / "manifests"
    for name in ("input.csv", "psnr_ssim_100.csv"):
        _atomic_write_csv(output_manifests / name, fields=fields, rows=copied_rows)
    _atomic_write_csv(
        output_manifests / "prompts.csv",
        fields=PROMPT_FIELDS,
        rows=selected_prompts,
    )
    prompt_hash = _sha256(output_manifests / "prompts.csv")
    source_pair_hash = _sha256(pair_path)
    protocol = {
        "schema_version": 2,
        "dataset": "psnr_ssim_100_v2",
        "source": {
            "origin": "ABO curated EVAL500 fixed PSNR/SSIM subset",
            "fixed_pair_manifest_sha256": source_pair_hash,
        },
        "selection": {
            "input_count": len(copied_rows),
            "psnr_ssim_pair_count": len(copied_rows),
            "group_quotas": dict(Counter(row["group"] for row in copied_rows)),
        },
        "prompt_protocol": {
            "mode": "per-image",
            "template": (
                "a high quality studio product photograph of {item_name}, "
                "on a clean white background"
            ),
            "name_source": (
                "ABO item_name language_tag starts with en; "
                "fallback to manifest product_type"
            ),
            "name_max_words": 15,
            "manifest": "manifests/prompts.csv",
            "manifest_sha256": prompt_hash,
        },
    }
    _atomic_write_json(output_manifests / "protocol.json", protocol)
    return DatasetBuildReport(
        destination_root=destination_root,
        count=len(copied_rows),
        manifest_sha256=_sha256(output_manifests / "input.csv"),
        prompt_manifest_sha256=prompt_hash,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fid-source", type=Path, default=DEFAULT_FID_SOURCE)
    parser.add_argument(
        "--psnr-ssim-source", type=Path, default=DEFAULT_PSNR_SSIM_SOURCE
    )
    parser.add_argument(
        "--fid-destination", type=Path, default=DEFAULT_FID_DESTINATION
    )
    parser.add_argument(
        "--psnr-ssim-destination",
        type=Path,
        default=DEFAULT_PSNR_SSIM_DESTINATION,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = {
        "fid": asdict(
            build_fid_web_dataset(
                source_root=args.fid_source,
                destination_root=args.fid_destination,
            )
        ),
        "psnr_ssim": asdict(
            build_psnr_ssim_web_dataset(
                source_root=args.psnr_ssim_source,
                destination_root=args.psnr_ssim_destination,
            )
        ),
    }
    for report in reports.values():
        report["destination_root"] = str(report["destination_root"])
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
