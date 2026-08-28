#!/usr/bin/env python3
"""Build the AI Hub real-world FID500 v2 pool and contribution-selected set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence, TextIO
from zipfile import ZipFile

import numpy as np
import torch
from cleanfid import fid as clean_fid
from diffusers.utils import logging as diffusers_logging
from scipy.linalg import svdvals

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.fid import configure_measurement_determinism
from scripts.generate import (
    GENERATED_FIELDS,
    MODEL_ID,
    PROMPT_FIELD,
    GenerationConfig,
    file_sha256,
    run_generation,
)
from scripts.run_eval import DEFAULT_PILOT_PATH, SEED, resolve_strength
from scripts.run_fid_eval import build_generation_manifest
from scripts.select_fid_dataset import (
    CATEGORY_DIRECTORIES,
    DEFAULT_ARCHIVES,
    DEFAULT_SOURCE_ROOT,
    MANIFEST_FIELDS,
    ArchiveSpec,
    _file_sha256,
    _load_selected_metadata,
    _parse_xml,
    _write_selected_images,
    _zip_metadata,
    choose_product_images,
    scan_archives,
)


DEFAULT_DATASET_ROOT = Path("/home/kim_3090/datasets/aihub-product/fid500-v2")
DEFAULT_V1_DATASET_ROOT = Path("/home/kim_3090/datasets/aihub-product/fid500")
POOL_RUN_ID = "fid500-v2-pool"
FINAL_RUN_ID = "fid500-v2"
V1_RUN_ID = "fid500"
POOL_RULE_VERSION = "aihub-product-fid500-v2-full-pool"
FINAL_RULE_VERSION = "aihub-product-fid500-v2-fid-contribution"
SELECTION_METHOD = (
    "FID-contribution-based selection (실사형 평가셋 구축, plan.md §4.6 2026-08-11)"
)
FEATURE_EXTRACTOR = "clean-fid Inception-v3"
PAIR_DISTANCE = "squared Euclidean distance"
PROMPT_TEMPLATE = (
    "a high quality studio product photograph of {img_prod_nm}, "
    "{div_m} {div_s}, on a clean white background"
)
PRODUCT_NAME_MAX_WORDS = 15
POOL_MANIFEST_FIELDS = (
    *MANIFEST_FIELDS,
    "소분류",
    PROMPT_FIELD,
    "prompt_name_source",
    "prompt_name_truncated",
)


@dataclass(frozen=True)
class SelectionResult:
    indices: tuple[int, ...]
    initial_fid: float
    final_fid: float
    swaps: tuple[dict[str, Any], ...]
    target_count: int
    pool_count: int
    target_fid: float


@dataclass(frozen=True)
class ProductPrompt:
    prompt: str
    product_name: str
    div_m: str
    div_s: str
    name_source: str
    name_truncated: bool


def _normalize_whitespace(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def prompt_protocol() -> dict[str, Any]:
    return {
        "template": PROMPT_TEMPLATE,
        "product_name_max_words": PRODUCT_NAME_MAX_WORDS,
        "product_name_truncation": "first 15 whitespace-delimited words",
        "fallback_order": ["img_prod_nm", "div_s", "div_m"],
        "whitespace_normalization": "strip and collapse consecutive whitespace",
        "label_text_policy": "preserve Korean and all other label text",
    }


def build_product_prompt(
    *, img_prod_nm: str | None, div_m: str | None, div_s: str | None
) -> ProductPrompt:
    """Build the recorded per-image prompt from normalized AI Hub label fields."""

    normalized_name = _normalize_whitespace(img_prod_nm)
    normalized_div_m = _normalize_whitespace(div_m)
    normalized_div_s = _normalize_whitespace(div_s)
    if normalized_name:
        product_name = normalized_name
        name_source = "img_prod_nm"
    elif normalized_div_s:
        product_name = normalized_div_s
        name_source = "div_s"
    elif normalized_div_m:
        product_name = normalized_div_m
        name_source = "div_m"
    else:
        raise ValueError("AI Hub metadata has no product name fallback")

    words = product_name.split()
    name_truncated = len(words) > PRODUCT_NAME_MAX_WORDS
    product_name = " ".join(words[:PRODUCT_NAME_MAX_WORDS])
    rendered = PROMPT_TEMPLATE.format(
        img_prod_nm=product_name,
        div_m=normalized_div_m,
        div_s=normalized_div_s,
    )
    return ProductPrompt(
        prompt=_normalize_whitespace(rendered),
        product_name=product_name,
        div_m=normalized_div_m,
        div_s=normalized_div_s,
        name_source=name_source,
        name_truncated=name_truncated,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValueError(f"cannot read CSV {path}: {exc}") from exc


def _atomic_write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
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


def _selection_source(
    source_root: Path, archive_specs: Sequence[ArchiveSpec]
) -> dict[str, Any]:
    return {
        "source_dataset": {
            "name": "AI Hub 상품 이미지",
            "url": "https://aihub.or.kr/aidata/34145",
            "builder": "NIA",
            "year": 2020,
            "attribution": "NIA AI 학습용 데이터 구축사업 결과",
        },
        "source_root": str(source_root),
        "source_zips": [
            _zip_metadata(source_root / spec.source_zip) for spec in archive_specs
        ],
        "label_zips": [
            _zip_metadata(source_root / spec.label_zip) for spec in archive_specs
        ],
    }


def _enrich_manifest_rows(
    *,
    source_root: Path,
    rows: Sequence[dict[str, str]],
    archive_specs: Sequence[ArchiveSpec],
) -> list[dict[str, str]]:
    specs_by_source = {spec.source_zip: spec for spec in archive_specs}
    if len(specs_by_source) != len(archive_specs):
        raise ValueError("archive_specs source ZIP names must be unique")
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(rows):
        try:
            spec = specs_by_source[row["zip_file"]]
        except KeyError as exc:
            raise ValueError(f"pool manifest has unknown source ZIP: {row['zip_file']}") from exc
        grouped.setdefault(spec.label_zip, []).append((index, row))

    enriched: list[dict[str, str] | None] = [None] * len(rows)
    for label_zip_name, group in sorted(grouped.items()):
        with ZipFile(source_root / label_zip_name) as label_zip:
            for index, row in group:
                source_member = PurePosixPath(row["zip_member"])
                metadata_member = str(
                    source_member.with_name(f"{source_member.stem}_meta.xml")
                )
                root = _parse_xml(
                    label_zip.read(metadata_member),
                    archive=label_zip_name,
                    member=metadata_member,
                )
                metadata_item = _normalize_whitespace(root.findtext(".//item_no"))
                if metadata_item != row["item_no"]:
                    raise ValueError(
                        f"metadata item_no mismatch in {label_zip_name}:{metadata_member}"
                    )
                raw_name = root.findtext(".//img_prod_nm")
                prompt = build_product_prompt(
                    img_prod_nm=raw_name,
                    div_m=root.findtext(".//div_m"),
                    div_s=root.findtext(".//div_s"),
                )
                value = dict(row)
                value["중분류"] = prompt.div_m
                value["상품명"] = _normalize_whitespace(raw_name)
                value["소분류"] = prompt.div_s
                value[PROMPT_FIELD] = prompt.prompt
                value["prompt_name_source"] = prompt.name_source
                value["prompt_name_truncated"] = str(prompt.name_truncated).lower()
                enriched[index] = value
    if any(row is None for row in enriched):
        raise AssertionError("not every pool manifest row received prompt metadata")
    return [row for row in enriched if row is not None]


def _ensure_pool_prompt_contract(
    *,
    source_root: Path,
    pool_root: Path,
    archive_specs: Sequence[ArchiveSpec],
) -> None:
    manifest_path = pool_root / "manifest.csv"
    selection_path = pool_root / "selection.json"
    rows = _read_csv(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_fields = set(POOL_MANIFEST_FIELDS)
    if rows and set(rows[0]) == expected_fields:
        if selection.get("prompt_protocol") != prompt_protocol():
            raise ValueError("existing pool prompt protocol differs from code")
        return
    if not rows or set(rows[0]) != set(MANIFEST_FIELDS):
        raise ValueError("existing pool manifest fields are neither v1 nor prompt-aware v2")

    backup_manifest = pool_root / "manifest.pre-per-image-prompt.csv"
    backup_selection = pool_root / "selection.pre-per-image-prompt.json"
    if not backup_manifest.exists():
        shutil.copy2(manifest_path, backup_manifest)
    if not backup_selection.exists():
        shutil.copy2(selection_path, backup_selection)
    enriched = _enrich_manifest_rows(
        source_root=source_root,
        rows=rows,
        archive_specs=archive_specs,
    )
    _atomic_write_csv(manifest_path, enriched, POOL_MANIFEST_FIELDS)
    selection["manifest_sha256"] = file_sha256(manifest_path)
    selection["prompt_protocol"] = prompt_protocol()
    selection.setdefault("rules", {})["prompt"] = (
        "per-image AI Hub metadata prompt; see prompt_protocol"
    )
    _atomic_write_json(selection_path, selection)


def _pool_summary(pool_root: Path) -> dict[str, Any]:
    manifest = pool_root / "manifest.csv"
    selection_path = pool_root / "selection.json"
    if not manifest.is_file() or not selection_path.is_file():
        raise ValueError(f"incomplete existing pool: {pool_root}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = _read_csv(manifest)
    if selection.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("existing pool manifest hash differs from selection.json")
    if int(selection["counts"]["final_count"]) != len(rows):
        raise ValueError("existing pool count differs from selection.json")
    return {
        "pool_count": len(rows),
        "pool_root": str(pool_root),
        "manifest_sha256": selection["manifest_sha256"],
        "existing": True,
    }


def build_pool(
    *,
    source_root: Path,
    pool_root: Path,
    seed: int | str = SEED,
    archive_specs: Sequence[ArchiveSpec] = DEFAULT_ARCHIVES,
    progress: TextIO | None = sys.stderr,
) -> dict[str, Any]:
    """Extract one preferred eligible image for every product, with no quota cut."""

    source_root = source_root.expanduser().resolve()
    pool_root = pool_root.expanduser().absolute()
    if pool_root.exists() or pool_root.is_symlink():
        _ensure_pool_prompt_contract(
            source_root=source_root,
            pool_root=pool_root,
            archive_specs=archive_specs,
        )
        return _pool_summary(pool_root)
    if pool_root == source_root or pool_root.is_relative_to(source_root):
        raise ValueError("pool cannot be created inside the read-only source root")

    scan = scan_archives(source_root, archive_specs, progress=progress)
    products = choose_product_images(scan.candidates, seed=seed)
    metadata = _load_selected_metadata(source_root, products)
    pool_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = pool_root.with_name(f".{pool_root.name}.building-{os.getpid()}")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"staging path already exists: {staging_root}")
    staging_root.mkdir()
    rows = _write_selected_images(
        source_root=source_root,
        staging_root=staging_root,
        selected=products,
        metadata=metadata,
        progress=progress,
    )
    rows = _enrich_manifest_rows(
        source_root=source_root,
        rows=rows,
        archive_specs=archive_specs,
    )
    manifest_path = staging_root / "manifest.csv"
    _atomic_write_csv(manifest_path, rows, POOL_MANIFEST_FIELDS)
    category_counts = dict(Counter(candidate.category for candidate in products))
    selection = {
        "schema_version": 1,
        "selection_rule_version": POOL_RULE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed) if str(seed).isdigit() else str(seed),
        "quota": category_counts,
        "counts": {
            **scan.counts,
            "unique_product_count": len(products),
            "final_count": len(products),
        },
        "manifest_sha256": _file_sha256(manifest_path),
        "prompt_protocol": prompt_protocol(),
        **_selection_source(source_root, archive_specs),
        "rules": {
            "eligibility": (
                "exactly one object; square image; width and height >= 1000; "
                "source JPG exists"
            ),
            "one_per_product": (
                "height 00 > 30 > 60; size m > s; "
                "sha256(seed:filename) ascending"
            ),
            "quota_selection": "none; full eligible product pool",
            "category_directories": dict(CATEGORY_DIRECTORIES),
            "prompt": "per-image AI Hub metadata prompt; see prompt_protocol",
        },
    }
    _atomic_write_json(staging_root / "selection.json", selection)
    os.replace(staging_root, pool_root)
    return {
        "pool_count": len(products),
        "pool_root": str(pool_root),
        "manifest_sha256": selection["manifest_sha256"],
        "category_counts": category_counts,
        "existing": False,
    }


def feature_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Compute clean-FID's Gaussian distance through an equivalent low-rank form."""

    real = np.asarray(real_features, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError("real and generated feature arrays must have the same 2-D shape")
    if real.shape[0] < 2:
        raise ValueError("FID requires at least two paired feature rows")
    real_centered = real - real.mean(axis=0, keepdims=True)
    generated_centered = generated - generated.mean(axis=0, keepdims=True)
    scale = float(real.shape[0] - 1)
    covariance_cross = real_centered @ generated_centered.T / scale
    covariance_term = (
        np.square(real_centered).sum() / scale
        + np.square(generated_centered).sum() / scale
        - 2.0 * svdvals(covariance_cross, check_finite=False).sum()
    )
    mean_term = np.square(real.mean(axis=0) - generated.mean(axis=0)).sum()
    value = float(mean_term + covariance_term)
    return max(value, 0.0) if value > -1e-9 else value


def _tie_rank(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def select_feature_subset(
    *,
    real_features: np.ndarray,
    generated_features: np.ndarray,
    item_ids: Sequence[str],
    target_count: int,
    target_fid: float,
    seed: int = SEED,
    max_swaps: int = 50,
    candidate_width: int = 4,
) -> SelectionResult:
    """Select low pair-distance items, then make deterministic improving swaps."""

    real = np.asarray(real_features)
    generated = np.asarray(generated_features)
    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError("feature arrays must have the same 2-D shape")
    if len(item_ids) != real.shape[0] or len(set(item_ids)) != len(item_ids):
        raise ValueError("item_ids must uniquely match feature rows")
    if not 2 <= target_count <= len(item_ids):
        raise ValueError("target_count must be between 2 and pool size")
    if target_fid < 0 or max_swaps < 0 or candidate_width <= 0:
        raise ValueError("target_fid/max_swaps/candidate_width are invalid")

    pair_distance = np.square(real - generated).sum(axis=1)
    ranking = sorted(
        range(len(item_ids)),
        key=lambda index: (
            float(pair_distance[index]),
            _tie_rank(seed, item_ids[index]),
            item_ids[index],
        ),
    )
    selected = set(ranking[:target_count])
    initial_indices = sorted(selected)
    initial_fid = feature_fid(real[initial_indices], generated[initial_indices])
    current_fid = initial_fid
    swaps: list[dict[str, Any]] = []

    while current_fid > target_fid and len(swaps) < max_swaps:
        selected_indices = sorted(selected)
        residual = (real[selected_indices] - generated[selected_indices]).mean(axis=0)
        removal_candidates = sorted(
            selected,
            key=lambda index: (
                -float(np.dot(real[index] - generated[index], residual)),
                -float(pair_distance[index]),
                _tie_rank(seed, item_ids[index]),
            ),
        )[:candidate_width]
        addition_candidates = sorted(
            set(range(len(item_ids))) - selected,
            key=lambda index: (
                float(np.dot(real[index] - generated[index], residual)),
                float(pair_distance[index]),
                _tie_rank(seed, item_ids[index]),
            ),
        )[:candidate_width]
        best: tuple[float, str, int, int] | None = None
        for removed in removal_candidates:
            for added in addition_candidates:
                trial = sorted((selected - {removed}) | {added})
                trial_fid = feature_fid(real[trial], generated[trial])
                key = (
                    trial_fid,
                    f"{_tie_rank(seed, item_ids[removed])}:{_tie_rank(seed, item_ids[added])}",
                    removed,
                    added,
                )
                if best is None or key < best:
                    best = key
        if best is None or best[0] >= current_fid - 1e-12:
            break
        trial_fid, _, removed, added = best
        before = current_fid
        selected.remove(removed)
        selected.add(added)
        current_fid = float(trial_fid)
        swaps.append(
            {
                "removed_item_id": item_ids[removed],
                "added_item_id": item_ids[added],
                "fid_before": before,
                "fid_after": current_fid,
            }
        )

    return SelectionResult(
        indices=tuple(sorted(selected)),
        initial_fid=float(initial_fid),
        final_fid=float(current_fid),
        swaps=tuple(swaps),
        target_count=target_count,
        pool_count=len(item_ids),
        target_fid=float(target_fid),
    )


def _validated_generated_rows(
    run_root: Path, *, seed: int, strength: float
) -> dict[str, dict[str, str]]:
    path = run_root / "generated.csv"
    if not path.exists():
        return {}
    rows = _read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        input_path = Path(row["input_path"])
        output_path = Path(row["output_path"])
        item_id = input_path.stem
        if item_id in result:
            raise ValueError(f"generated.csv:{row_number} duplicates item {item_id}")
        if int(row["seed"]) != seed or float(row["strength"]) != strength:
            raise ValueError(f"generated.csv:{row_number} protocol mismatch")
        if not input_path.is_file() or not output_path.is_file() or output_path.is_symlink():
            raise ValueError(f"generated.csv:{row_number} path is missing or unsafe")
        if file_sha256(output_path) != row["sha256"]:
            raise ValueError(f"generated.csv:{row_number} output hash mismatch")
        if PROMPT_FIELD in row and not row[PROMPT_FIELD].strip():
            raise ValueError(f"generated.csv:{row_number} prompt is empty")
        result[item_id] = dict(row)
    return result


def load_dataset_prompt_resolver(
    dataset_root: Path,
) -> Any:
    """Return an InputRecord resolver backed by the dataset's recorded prompt column."""

    dataset_root = dataset_root.expanduser().absolute()
    rows = _read_csv(dataset_root / "manifest.csv")
    prompts = {row["item_no"]: row.get(PROMPT_FIELD, "") for row in rows}
    if not prompts or any(not value.strip() for value in prompts.values()):
        raise ValueError(f"dataset prompt column is missing or empty: {dataset_root}")

    def resolve(record: Any) -> str:
        try:
            return prompts[record.input_path.stem]
        except KeyError as exc:
            raise ValueError(f"input has no recorded dataset prompt: {record.input_path}") from exc

    return resolve


def seed_reused_outputs(
    *,
    source_dataset_root: Path,
    source_run_root: Path,
    target_dataset_root: Path,
    target_run_root: Path,
    seed: int,
    strength: float,
) -> int:
    """Seed a target run from SHA-identical product inputs and verified outputs."""

    source_dataset_root = source_dataset_root.expanduser().absolute()
    target_dataset_root = target_dataset_root.expanduser().absolute()
    source_run_root = source_run_root.expanduser().absolute()
    target_run_root = target_run_root.expanduser().absolute()
    source_manifest = {row["item_no"]: row for row in _read_csv(source_dataset_root / "manifest.csv")}
    target_manifest = {row["item_no"]: row for row in _read_csv(target_dataset_root / "manifest.csv")}
    source_generated = _validated_generated_rows(
        source_run_root, seed=seed, strength=strength
    )
    existing = _validated_generated_rows(
        target_run_root, seed=seed, strength=strength
    )
    generation_rows = _read_csv(build_generation_manifest(dataset_root=target_dataset_root))
    targets = {row["item_id"]: row for row in generation_rows}
    reusable_count = 0

    for item_id, target in targets.items():
        source_row = source_manifest.get(item_id)
        target_row = target_manifest.get(item_id)
        source_output = source_generated.get(item_id)
        if source_row is None or target_row is None or source_output is None:
            continue
        source_input = source_dataset_root / target["selected_path"]
        target_input = target_dataset_root / target["selected_path"]
        if not source_input.is_file():
            source_input = Path(source_output["input_path"])
        if (
            source_row["sha256"] != target_row["sha256"]
            or file_sha256(source_input) != source_row["sha256"]
            or file_sha256(target_input) != target_row["sha256"]
        ):
            continue
        reusable_count += 1
        relative = Path(target["selected_path"]).relative_to("input").with_suffix(".png")
        destination = target_run_root / "images" / relative
        prior = existing.get(item_id)
        if prior is not None:
            if Path(prior["input_path"]) != target_input or Path(prior["output_path"]) != destination:
                raise ValueError(f"existing target row paths differ for item {item_id}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"untracked target output already exists: {destination}")
        shutil.copy2(Path(source_output["output_path"]), destination)
        output_hash = file_sha256(destination)
        if output_hash != source_output["sha256"]:
            raise ValueError(f"copied output hash differs for item {item_id}")
        existing[item_id] = {
            "input_path": str(target_input),
            "output_path": str(destination),
            "sha256": output_hash,
            "seed": str(seed),
            "strength": str(strength),
        }
        if PROMPT_FIELD in source_output:
            existing[item_id][PROMPT_FIELD] = source_output[PROMPT_FIELD]

    if existing:
        ordered = [existing[row["item_id"]] for row in generation_rows if row["item_id"] in existing]
        fields = (
            [*GENERATED_FIELDS, PROMPT_FIELD]
            if any(PROMPT_FIELD in row for row in ordered)
            else GENERATED_FIELDS
        )
        _atomic_write_csv(target_run_root / "generated.csv", ordered, fields)
    return reusable_count


def _feature_inputs(
    pool_root: Path, pool_run_root: Path
) -> tuple[list[str], list[Path], list[Path], list[str], list[str]]:
    manifest_rows = _read_csv(pool_root / "manifest.csv")
    generation_rows = _read_csv(build_generation_manifest(dataset_root=pool_root))
    generated_rows = _validated_generated_rows(pool_run_root, seed=SEED, strength=0.15)
    if len(manifest_rows) != len(generation_rows) or len(generated_rows) != len(generation_rows):
        raise ValueError("pool inputs and generated outputs must be complete before feature extraction")
    manifest_by_item = {row["item_no"]: row for row in manifest_rows}
    item_ids: list[str] = []
    real_paths: list[Path] = []
    generated_paths: list[Path] = []
    input_hashes: list[str] = []
    output_hashes: list[str] = []
    for row in generation_rows:
        item_id = row["item_id"]
        real_path = pool_root / row["selected_path"]
        generated = generated_rows[item_id]
        if file_sha256(real_path) != manifest_by_item[item_id]["sha256"]:
            raise ValueError(f"pool input hash differs for item {item_id}")
        item_ids.append(item_id)
        real_paths.append(real_path)
        generated_paths.append(Path(generated["output_path"]))
        input_hashes.append(manifest_by_item[item_id]["sha256"])
        output_hashes.append(generated["sha256"])
    return item_ids, real_paths, generated_paths, input_hashes, output_hashes


def extract_pool_features(
    *, pool_root: Path, pool_run_root: Path, cache_path: Path
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Extract or verify cached paired clean-fid Inception features."""

    pool_root = pool_root.expanduser().absolute()
    pool_run_root = pool_run_root.expanduser().absolute()
    cache_path = cache_path.expanduser().absolute()
    item_ids, real_paths, generated_paths, input_hashes, output_hashes = _feature_inputs(
        pool_root, pool_run_root
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                cached["item_ids"].tolist() != item_ids
                or cached["input_sha256"].tolist() != input_hashes
                or cached["output_sha256"].tolist() != output_hashes
            ):
                raise ValueError("feature cache identity/hashes differ from the current pool")
            return (
                np.asarray(cached["real_features"]),
                np.asarray(cached["generated_features"]),
                tuple(item_ids),
            )

    configure_measurement_determinism(seed=SEED)
    device = torch.device("cuda")
    extractor = clean_fid.build_feature_extractor(
        "clean", device=device, use_dataparallel=False
    )
    real_features = clean_fid.get_files_features(
        [str(path) for path in real_paths],
        model=extractor,
        num_workers=0,
        batch_size=32,
        device=device,
        mode="clean",
        description="FID v2 real pool",
        verbose=True,
    )
    generated_features = clean_fid.get_files_features(
        [str(path) for path in generated_paths],
        model=extractor,
        num_workers=0,
        batch_size=32,
        device=device,
        mode="clean",
        description="FID v2 generated pool",
        verbose=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{cache_path.name}.",
            suffix=".npz",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                real_features=real_features,
                generated_features=generated_features,
                item_ids=np.asarray(item_ids),
                input_sha256=np.asarray(input_hashes),
                output_sha256=np.asarray(output_hashes),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return real_features, generated_features, tuple(item_ids)


def build_final_dataset(
    *,
    pool_root: Path,
    pool_run_root: Path,
    output_root: Path,
    final_run_root: Path,
    selection: SelectionResult,
    seed: int,
    strength: float,
    feature_cache_path: Path,
) -> dict[str, Any]:
    """Materialize selected inputs and preseed their deterministic pool outputs."""

    pool_root = pool_root.expanduser().absolute()
    pool_run_root = pool_run_root.expanduser().absolute()
    output_root = output_root.expanduser().absolute()
    final_run_root = final_run_root.expanduser().absolute()
    pool_rows = _read_csv(pool_root / "manifest.csv")
    if selection.pool_count != len(pool_rows):
        raise ValueError("selection pool_count differs from pool manifest")
    if len(selection.indices) != selection.target_count:
        raise ValueError("selection index count differs from target_count")
    selected_rows = [pool_rows[index] for index in selection.indices]
    selected_item_ids = [row["item_no"] for row in selected_rows]

    final_manifest = output_root / "manifest.csv"
    final_selection = output_root / "selection.json"
    final_input = output_root / "input"
    existing_artifacts = [path.exists() or path.is_symlink() for path in (final_manifest, final_selection, final_input)]
    if any(existing_artifacts):
        if not all(existing_artifacts):
            raise ValueError(f"incomplete existing final dataset: {output_root}")
        existing_rows = _read_csv(final_manifest)
        existing_contract = json.loads(final_selection.read_text(encoding="utf-8"))
        if (
            [row["item_no"] for row in existing_rows] != selected_item_ids
            or existing_contract.get("selection_method") != SELECTION_METHOD
        ):
            raise ValueError("existing final dataset differs from requested selection")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        staging = output_root.parent / f".{output_root.name}.final-building-{os.getpid()}"
        if staging.exists() or staging.is_symlink():
            raise FileExistsError(f"final staging path already exists: {staging}")
        staging.mkdir()
        for row in selected_rows:
            category_directory = CATEGORY_DIRECTORIES[row["대분류"]]
            source = pool_root / "input" / category_directory / f"{row['item_no']}.jpg"
            destination = staging / "input" / category_directory / f"{row['item_no']}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if file_sha256(destination) != row["sha256"]:
                raise ValueError(f"final copied input hash differs for item {row['item_no']}")
        staging_manifest = staging / "manifest.csv"
        _atomic_write_csv(staging_manifest, selected_rows, POOL_MANIFEST_FIELDS)
        pool_contract = json.loads((pool_root / "selection.json").read_text(encoding="utf-8"))
        category_counts = dict(Counter(row["대분류"] for row in selected_rows))
        method_details: dict[str, Any] = {
            "feature_extractor": FEATURE_EXTRACTOR,
            "clean_fid_mode": "clean",
            "pair_distance": PAIR_DISTANCE,
            "initial_selection": "pair distance ascending",
            "greedy": "deterministic improving swaps evaluated with feature FID",
            "seed": seed,
            "target_fid": selection.target_fid,
            "initial_fid": selection.initial_fid,
            "final_fid": selection.final_fid,
            "swap_count": len(selection.swaps),
            "swaps": list(selection.swaps),
            "feature_cache": str(feature_cache_path.expanduser().absolute()),
        }
        if feature_cache_path.is_file():
            method_details["feature_cache_sha256"] = file_sha256(feature_cache_path)
        contract = {
            "schema_version": 1,
            "selection_rule_version": FINAL_RULE_VERSION,
            "selection_method": SELECTION_METHOD,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "quota": category_counts,
            "counts": {
                "pool_count": len(pool_rows),
                "final_count": len(selected_rows),
            },
            "manifest_sha256": file_sha256(staging_manifest),
            "source_dataset": pool_contract["source_dataset"],
            "prompt_protocol": pool_contract["prompt_protocol"],
            "source_root": pool_contract["source_root"],
            "source_zips": pool_contract["source_zips"],
            "label_zips": pool_contract["label_zips"],
            "pool_manifest": str((pool_root / "manifest.csv").absolute()),
            "pool_manifest_sha256": pool_contract["manifest_sha256"],
            "method_details": method_details,
            "rules": {
                **pool_contract["rules"],
                "quota_selection": "none; FID-contribution-based pool-to-500 selection",
                "category_directories": dict(CATEGORY_DIRECTORIES),
            },
        }
        _atomic_write_json(staging / "selection.json", contract)
        os.replace(staging / "input", final_input)
        os.replace(staging_manifest, final_manifest)
        os.replace(staging / "selection.json", final_selection)
        staging.rmdir()

    build_generation_manifest(dataset_root=output_root)
    reused = seed_reused_outputs(
        source_dataset_root=pool_root,
        source_run_root=pool_run_root,
        target_dataset_root=output_root,
        target_run_root=final_run_root,
        seed=seed,
        strength=strength,
    )
    if reused != len(selected_rows):
        raise ValueError(f"final run seeded {reused}/{len(selected_rows)} pool outputs")
    return {
        "pool_count": len(pool_rows),
        "final_count": len(selected_rows),
        "initial_fid": selection.initial_fid,
        "final_fid": selection.final_fid,
        "swap_count": len(selection.swaps),
        "manifest_sha256": file_sha256(final_manifest),
        "output_root": str(output_root),
        "final_run_root": str(final_run_root),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pool", "generate", "select", "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--v1-dataset-root", type=Path, default=DEFAULT_V1_DATASET_ROOT)
    parser.add_argument("--v1-run-id", default=V1_RUN_ID)
    parser.add_argument("--pool-run-id", default=POOL_RUN_ID)
    parser.add_argument("--final-run-id", default=FINAL_RUN_ID)
    parser.add_argument("--strength-from", type=Path, default=DEFAULT_PILOT_PATH)
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--target-fid", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().absolute()
    pool_root = dataset_root / "pool"
    reports: dict[str, Any] = {}
    if args.stage in {"pool", "all"}:
        reports["pool"] = build_pool(
            source_root=args.source_root,
            pool_root=pool_root,
            seed=args.seed,
        )
    if args.stage == "pool":
        print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    runs_root_value = args.runs_root or (
        Path(os.environ["RUNS_ROOT"]) if os.environ.get("RUNS_ROOT") else None
    )
    if runs_root_value is None:
        raise RuntimeError("RUNS_ROOT or --runs-root is required")
    runs_root = runs_root_value.expanduser().absolute()
    strength = resolve_strength(explicit=None, strength_from=args.strength_from)
    pool_run_root = runs_root / args.pool_run_id

    if args.stage in {"generate", "all"}:
        diffusers_logging.set_verbosity_error()
        generation_manifest = build_generation_manifest(dataset_root=pool_root)
        report = run_generation(
            manifest_path=generation_manifest,
            runs_root=runs_root,
            config=GenerationConfig(
                strength=strength,
                seed=args.seed,
                run_id=args.pool_run_id,
                model=MODEL_ID,
            ),
            prompt_resolver=load_dataset_prompt_resolver(pool_root),
        )
        reports["generation"] = {
            "count": report.count,
            "reused_v1_count": 0,
            "per_image_prompt": True,
            "elapsed_seconds": report.elapsed_seconds,
            "peak_vram_gib": report.peak_vram_gib,
            "run_root": str(report.run_root),
        }
    if args.stage == "generate":
        print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    cache_path = pool_root / "features.npz"
    real_features, generated_features, item_ids = extract_pool_features(
        pool_root=pool_root,
        pool_run_root=pool_run_root,
        cache_path=cache_path,
    )
    selection = select_feature_subset(
        real_features=real_features,
        generated_features=generated_features,
        item_ids=item_ids,
        target_count=args.target_count,
        target_fid=args.target_fid,
        seed=args.seed,
    )
    reports["selection"] = build_final_dataset(
        pool_root=pool_root,
        pool_run_root=pool_run_root,
        output_root=dataset_root,
        final_run_root=runs_root / args.final_run_id,
        selection=selection,
        seed=args.seed,
        strength=strength,
        feature_cache_path=cache_path,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
