#!/usr/bin/env python3
"""Create a deterministic, non-destructive 500+500 evaluation dataset view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SOURCE_ROOT = Path("/home/kim_3090/datasets/abo/curated")
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT / "eval500"
DEFAULT_SEED = "perf-eval-gen-eval500-v1"


@dataclass(frozen=True)
class ImageRecord:
    split: str
    group: str
    product_type: str
    item_id: str
    image_id: str
    source_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class SelectionResult:
    input: tuple[ImageRecord, ...]
    quotas: tuple[tuple[str, int], ...]


def _score(seed: str, record: ImageRecord, purpose: str) -> str:
    value = "\0".join(
        (seed, purpose, record.split, record.group, record.item_id, record.image_id)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _largest_remainder_quotas(capacities: dict[str, int], total: int) -> dict[str, int]:
    available = sum(capacities.values())
    if total <= 0:
        raise ValueError("selection total must be positive")
    if total > available:
        raise ValueError(f"requested {total} images, but common capacity is {available}")

    exact = {group: Fraction(total * capacity, available) for group, capacity in capacities.items()}
    quotas = {group: int(value) for group, value in exact.items()}
    remainder_count = total - sum(quotas.values())
    remainder_order = sorted(
        capacities,
        key=lambda group: (exact[group] - quotas[group], group),
        reverse=True,
    )
    for group in remainder_order[:remainder_count]:
        quotas[group] += 1
    return quotas


def _unique_products(records: Iterable[ImageRecord], seed: str) -> list[ImageRecord]:
    by_item: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_item[record.item_id].append(record)
    return [
        min(candidates, key=lambda record: _score(seed, record, "deduplicate-product"))
        for candidates in by_item.values()
    ]


def select_records(
    records: Sequence[ImageRecord], *, total_per_split: int, seed: str
) -> SelectionResult:
    if not records or {record.split for record in records} != {"input"}:
        raise ValueError("records must contain only the input split")

    eligible: dict[str, list[ImageRecord]] = defaultdict(list)
    candidates = (
        record
        for record in records
        if record.width == record.height and min(record.width, record.height) >= 1000
    )
    for record in _unique_products(candidates, seed):
        eligible[record.group].append(record)

    capacities = {group: len(candidates) for group, candidates in eligible.items()}
    quotas = _largest_remainder_quotas(capacities, total_per_split)

    chosen: list[ImageRecord] = []
    for group in sorted(quotas):
        ranked = sorted(
            eligible[group],
            key=lambda record: _score(seed, record, "select-evaluation-image"),
        )
        chosen.extend(ranked[: quotas[group]])
    selected = tuple(sorted(chosen, key=lambda record: (record.group, record.item_id)))

    return SelectionResult(
        input=selected,
        quotas=tuple(sorted(quotas.items())),
    )


def select_pair_subset(
    records: Sequence[ImageRecord], *, total: int, seed: str
) -> tuple[ImageRecord, ...]:
    by_group: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    quotas = _largest_remainder_quotas(
        {group: len(candidates) for group, candidates in by_group.items()}, total
    )
    selected: list[ImageRecord] = []
    for group in sorted(quotas):
        ranked = sorted(
            by_group[group], key=lambda record: _score(seed, record, "select-pair-image")
        )
        selected.extend(ranked[: quotas[group]])
    return tuple(sorted(selected, key=lambda record: (record.group, record.item_id)))


def load_records(
    source_root: Path, *, splits: Sequence[str] = ("input",)
) -> list[ImageRecord]:
    metadata_by_filename: dict[str, dict[str, str]] = {}
    with (source_root / "pool.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = f"{row['product_type']}__{row['image_id']}.jpg"
            if filename in metadata_by_filename:
                raise ValueError(f"duplicate image filename in pool.csv: {filename}")
            metadata_by_filename[filename] = row

    records: list[ImageRecord] = []
    for split in splits:
        split_root = source_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing source split: {split_root}")
        for source_path in sorted(split_root.glob("*/*")):
            if not source_path.is_file() or source_path.is_symlink():
                continue
            row = metadata_by_filename.get(source_path.name)
            if row is None:
                raise ValueError(f"missing pool.csv metadata for {source_path}")
            if row["group"] != source_path.parent.name:
                raise ValueError(f"group mismatch for {source_path}")
            records.append(
                ImageRecord(
                    split=split,
                    group=row["group"],
                    product_type=row["product_type"],
                    item_id=row["item_id"],
                    image_id=row["image_id"],
                    source_path=source_path.resolve(),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata_subsets(
    records: Sequence[ImageRecord], *, source_root: Path, output_root: Path
) -> dict[str, object]:
    """Write traceable subsets of the three source metadata CSV files."""
    if not output_root.is_dir():
        raise FileNotFoundError(f"metadata output directory does not exist: {output_root}")

    ordered_records = sorted(records, key=lambda record: (record.split, record.group, record.item_id))
    selected_by_key = {(record.item_id, record.image_id): record for record in ordered_records}
    if len(selected_by_key) != len(ordered_records):
        raise ValueError("selected records contain duplicate item_id/image_id keys")

    source_rows: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    source_hashes: dict[str, str] = {}
    for filename in ("pool.csv", "index-main.csv", "features.csv"):
        source_path = source_root / filename
        source_hashes[filename] = _file_sha256(source_path)
        with source_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"metadata CSV has no header: {source_path}")
            source_rows[filename] = (reader.fieldnames, list(reader))

    pool_fields, pool_source_rows = source_rows["pool.csv"]
    pool_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in pool_source_rows:
        key = (row["item_id"], row["image_id"])
        if key in selected_by_key:
            if key in pool_by_key:
                raise ValueError(f"duplicate selected row in pool.csv: {key}")
            pool_by_key[key] = row
    if set(pool_by_key) != set(selected_by_key):
        missing = sorted(set(selected_by_key) - set(pool_by_key))
        raise ValueError(f"pool.csv is missing selected records: {missing[:5]}")

    index_fields, index_source_rows = source_rows["index-main.csv"]
    index_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in index_source_rows:
        key = (row["item_id"], row["image_id"])
        record = selected_by_key.get(key)
        if record is None:
            continue
        pool_row = pool_by_key[key]
        if (
            row["group"] != record.group
            or row["product_type"] != record.product_type
            or row["path"] != pool_row["path"]
        ):
            continue
        existing = index_by_key.get(key)
        if existing is not None and existing != row:
            raise ValueError(f"conflicting selected rows in index-main.csv: {key}")
        index_by_key[key] = row
    if set(index_by_key) != set(selected_by_key):
        missing = sorted(set(selected_by_key) - set(index_by_key))
        raise ValueError(f"index-main.csv is missing selected records: {missing[:5]}")

    feature_fields, feature_source_rows = source_rows["features.csv"]
    selected_by_filename = {record.source_path.name: record for record in ordered_records}
    features_by_filename: dict[str, dict[str, str]] = {}
    for row in feature_source_rows:
        filename = Path(row["path"]).name
        if filename in selected_by_filename:
            if filename in features_by_filename:
                raise ValueError(f"duplicate selected row in features.csv: {filename}")
            features_by_filename[filename] = row
    if set(features_by_filename) != set(selected_by_filename):
        missing = sorted(set(selected_by_filename) - set(features_by_filename))
        raise ValueError(f"features.csv is missing selected records: {missing[:5]}")

    rows_by_filename: dict[str, list[dict[str, str]]] = {
        "pool.csv": [pool_by_key[(record.item_id, record.image_id)] for record in ordered_records],
        "index-main.csv": [
            index_by_key[(record.item_id, record.image_id)] for record in ordered_records
        ],
        "features.csv": [features_by_filename[record.source_path.name] for record in ordered_records],
    }
    fields_by_filename = {
        "pool.csv": pool_fields,
        "index-main.csv": index_fields,
        "features.csv": feature_fields,
    }

    for filename in rows_by_filename:
        destination = output_root / filename
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"metadata subset already exists: {destination}")

    for filename, rows in rows_by_filename.items():
        fields = [*fields_by_filename[filename], "split", "selected_path"]
        with (output_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record, row in zip(ordered_records, rows, strict=True):
                writer.writerow(
                    {
                        **row,
                        "split": record.split,
                        "selected_path": str(
                            Path(record.split) / record.group / record.source_path.name
                        ),
                    }
                )

    return {
        "source_csv_sha256": dict(sorted(source_hashes.items())),
        "subset_counts": {
            filename: len(rows) for filename, rows in sorted(rows_by_filename.items())
        },
    }


def _manifest_rows(records: Sequence[ImageRecord], output_root: Path) -> list[dict[str, object]]:
    rows = []
    for record in records:
        selected_path = Path(record.split) / record.group / record.source_path.name
        rows.append(
            {
                "split": record.split,
                "group": record.group,
                "product_type": record.product_type,
                "item_id": record.item_id,
                "image_id": record.image_id,
                "width": record.width,
                "height": record.height,
                "sha256": _file_sha256(record.source_path),
                "source_path": str(record.source_path),
                "selected_path": str(selected_path),
            }
        )
    return rows


def _validate_output_path(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve(strict=False)
    protected = (source_root, source_root / "input", source_root / "ref")
    if output_root in protected:
        raise ValueError(f"output path would overwrite source data: {output_root}")
    if output_root.is_relative_to(source_root / "input") or output_root.is_relative_to(
        source_root / "ref"
    ):
        raise ValueError(f"output path cannot be nested in a source split: {output_root}")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output path already exists: {output_root}")


def write_selection(
    selection: SelectionResult,
    pair_subset: Sequence[ImageRecord],
    *,
    source_root: Path,
    output_root: Path,
    seed: str,
) -> None:
    _validate_output_path(source_root, output_root)

    # Hash all selected sources before creating output so source/read failures leave no partial view.
    rows_by_split = {"input": _manifest_rows(selection.input, output_root)}
    pair_item_ids = {record.item_id for record in pair_subset}

    output_root.mkdir(parents=True, exist_ok=False)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir()
    metadata_summary = write_metadata_subsets(
        selection.input,
        source_root=source_root,
        output_root=output_root,
    )

    manifest_fields = [
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
    ]
    for split, records in (("input", selection.input),):
        for record in records:
            destination = output_root / split / record.group / record.source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            relative_source = os.path.relpath(record.source_path, destination.parent)
            destination.symlink_to(relative_source)

        with (manifest_root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerows(rows_by_split[split])

    pair_rows = [
        row for row in rows_by_split["input"] if str(row["item_id"]) in pair_item_ids
    ]
    with (manifest_root / "psnr_ssim_100.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(pair_rows)

    protocol = {
        "schema_version": 1,
        "seed": seed,
        "source_root": str(source_root.resolve()),
        "selection": {
            "input_count": len(selection.input),
            "psnr_ssim_pair_count": len(pair_subset),
            "group_quotas": dict(selection.quotas),
        },
        "fid": {
            "real_reference_set": "input",
            "generated_set": "model outputs corresponding to input",
            "comparison": "set-level Inception feature distributions",
        },
        "eligibility": {
            "source_membership": "pool.csv and curated input split",
            "minimum_width": 1000,
            "minimum_height": 1000,
            "aspect_ratio": "exactly 1:1",
            "unique_product_per_split": True,
            "selection_after_eligibility": "deterministic SHA-256 ordering with fixed seed",
        },
        "metadata": metadata_summary,
    }
    with (manifest_root / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _summary(selection: SelectionResult, pair_subset: Sequence[ImageRecord]) -> dict[str, object]:
    return {
        "input_count": len(selection.input),
        "psnr_ssim_pair_count": len(pair_subset),
        "group_quotas": dict(selection.quotas),
        "fid_real_reference_set": "input",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--pair-count", type=int, default=100)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    records = load_records(args.source_root)
    selection = select_records(records, total_per_split=args.total, seed=args.seed)
    pair_subset = select_pair_subset(
        selection.input, total=args.pair_count, seed=f"{args.seed}:psnr-ssim"
    )
    print(json.dumps(_summary(selection, pair_subset), ensure_ascii=False, indent=2))
    if not args.dry_run:
        write_selection(
            selection,
            pair_subset,
            source_root=args.source_root,
            output_root=args.output_root,
            seed=args.seed,
        )
        print(f"created non-destructive selection view: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
