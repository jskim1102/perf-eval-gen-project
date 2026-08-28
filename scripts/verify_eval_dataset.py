#!/usr/bin/env python3
"""Verify the existing EVAL500 view without writing anywhere under curated/."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECT_SCRIPT = PROJECT_ROOT / "scripts" / "select_eval_dataset.py"
DEFAULT_SOURCE_ROOT = Path("/home/kim_3090/datasets/abo/curated")
DEFAULT_EVAL_ROOT = DEFAULT_SOURCE_ROOT / "eval500"
DEFAULT_SEED = "perf-eval-gen-eval500-v1"
EXPECTED_GROUP_QUOTAS = {
    "가구": 58,
    "사무공구": 100,
    "생활일반": 138,
    "잡화": 82,
    "조명": 70,
    "주방": 30,
    "침구": 22,
}
MANIFEST_FIELDS = [
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
METADATA_FILES = ("pool.csv", "index-main.csv", "features.csv")
DETERMINISTIC_FILES = (
    "pool.csv",
    "index-main.csv",
    "features.csv",
    "manifests/input.csv",
    "manifests/psnr_ssim_100.csv",
    "manifests/protocol.json",
)


class VerificationError(ValueError):
    """The existing selection does not satisfy its fixed contract."""


@dataclass(frozen=True)
class DatasetContract:
    input_count: int = 500
    pair_count: int = 100
    group_quotas: Mapping[str, int] = field(
        default_factory=lambda: dict(EXPECTED_GROUP_QUOTAS)
    )
    seed: str = DEFAULT_SEED


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _check(reader.fieldnames is not None, f"CSV has no header: {path}")
            return list(reader.fieldnames or []), list(reader)
    except OSError as exc:
        raise VerificationError(f"cannot read CSV {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_relative_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    _check(not path.is_absolute(), f"{label} must be relative: {raw_path}")
    _check(".." not in path.parts, f"{label} cannot traverse parents: {raw_path}")
    return path


def _tree_signature(root: Path) -> tuple[tuple[Any, ...], ...]:
    if not root.exists() and not root.is_symlink():
        return (("<missing>",),)
    entries: list[tuple[Any, ...]] = []
    for path in sorted(root.rglob("*")):
        stat_result = path.lstat()
        kind = "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file"
        link_target = os.readlink(path) if path.is_symlink() else ""
        entries.append(
            (
                str(path.relative_to(root)),
                kind,
                stat_result.st_size,
                stat_result.st_mtime_ns,
                link_target,
            )
        )
    return tuple(entries)


def _protected_tree_signature(
    source_root: Path, eval_root: Path
) -> tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]:
    return tuple(
        (str(root), _tree_signature(root))
        for root in (source_root / "input", source_root / "ref", eval_root)
    )


def _verify_input_manifest(
    *,
    source_root: Path,
    eval_root: Path,
    contract: DatasetContract,
    strict: bool,
) -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    fields, rows = _read_csv(eval_root / "manifests" / "input.csv")
    _check(fields == MANIFEST_FIELDS, f"input manifest fields differ: {fields}")
    _check(
        len(rows) == contract.input_count,
        f"input manifest rows: expected {contract.input_count}, got {len(rows)}",
    )

    keys: list[tuple[str, str]] = []
    item_ids: list[str] = []
    selected_paths: list[str] = []
    observed_quotas: Counter[str] = Counter()
    source_input_root = (source_root / "input").resolve()

    for row_number, row in enumerate(rows, start=2):
        label = f"input.csv:{row_number}"
        _check(row["split"] == "input", f"{label} split must be input")
        key = (row["item_id"], row["image_id"])
        keys.append(key)
        item_ids.append(row["item_id"])
        observed_quotas[row["group"]] += 1

        try:
            width = int(row["width"])
            height = int(row["height"])
        except ValueError as exc:
            raise VerificationError(f"{label} has non-integer dimensions") from exc
        _check(width == height, f"{label} is not square: {width}x{height}")
        _check(width >= 1000 and height >= 1000, f"{label} is below 1000px")

        filename = f"{row['product_type']}__{row['image_id']}.jpg"
        expected_relative = Path("input") / row["group"] / filename
        selected_relative = _safe_relative_path(
            row["selected_path"], label=f"{label} selected_path"
        )
        _check(
            selected_relative == expected_relative,
            f"{label} selected_path does not match its metadata",
        )
        selected_paths.append(str(selected_relative))
        selected_file = eval_root / selected_relative
        _check(selected_file.is_symlink(), f"{label} is not a symlink: {selected_file}")
        _check(selected_file.is_file(), f"{label} symlink target is missing")

        source_file = Path(row["source_path"])
        _check(source_file.is_absolute(), f"{label} source_path must be absolute")
        _check(source_file.is_file(), f"{label} source file is missing: {source_file}")
        resolved_source = source_file.resolve()
        _check(
            _is_relative_to(resolved_source, source_input_root),
            f"{label} source_path is outside curated/input",
        )
        _check(
            selected_file.resolve() == resolved_source,
            f"{label} symlink does not target source_path",
        )
        if strict:
            try:
                with Image.open(selected_file) as image:
                    actual_size = image.size
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise VerificationError(f"{label} image is unreadable: {exc}") from exc
            _check(
                actual_size == (width, height),
                f"{label} pixel dimensions differ: "
                f"manifest={width}x{height}, actual={actual_size[0]}x{actual_size[1]}",
            )
            actual_hash = _file_sha256(selected_file)
            _check(
                actual_hash == row["sha256"],
                f"{label} sha256 mismatch: expected {row['sha256']}, got {actual_hash}",
            )

    _check(len(set(keys)) == len(keys), "input manifest has duplicate item/image keys")
    _check(len(set(item_ids)) == len(item_ids), "input manifest item_id is not unique")
    _check(
        len(set(selected_paths)) == len(selected_paths),
        "input manifest selected_path is not unique",
    )
    _check(
        dict(observed_quotas) == dict(contract.group_quotas),
        f"observed group quotas differ: {dict(observed_quotas)}",
    )

    input_root = eval_root / "input"
    leaves = [
        path
        for path in input_root.rglob("*")
        if path.is_symlink() or not path.is_dir()
    ]
    _check(
        len(leaves) == contract.input_count,
        f"input view leaves: expected {contract.input_count}, got {len(leaves)}",
    )
    _check(all(path.is_symlink() for path in leaves), "input view contains a non-symlink")
    actual_relative_paths = {str(path.relative_to(eval_root)) for path in leaves}
    _check(
        actual_relative_paths == set(selected_paths),
        "input symlink paths differ from manifests/input.csv",
    )
    return rows, {key: row for key, row in zip(keys, rows, strict=True)}


def _verify_pair_manifest(
    *,
    eval_root: Path,
    contract: DatasetContract,
    input_by_key: Mapping[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    fields, rows = _read_csv(eval_root / "manifests" / "psnr_ssim_100.csv")
    _check(fields == MANIFEST_FIELDS, f"pair manifest fields differ: {fields}")
    _check(
        len(rows) == contract.pair_count,
        f"pair manifest rows: expected {contract.pair_count}, got {len(rows)}",
    )
    pair_keys = [(row["item_id"], row["image_id"]) for row in rows]
    pair_key_set = set(pair_keys)
    input_key_set = set(input_by_key)
    _check(len(pair_key_set) == len(pair_keys), "pair manifest contains duplicate rows")
    _check(pair_key_set < input_key_set, "pair manifest is not a proper input subset")
    for key, row in zip(pair_keys, rows, strict=True):
        _check(
            row == input_by_key[key],
            f"pair manifest row differs from input manifest: {key}",
        )
    return rows


def _verify_metadata_subsets(
    *,
    source_root: Path,
    eval_root: Path,
    contract: DatasetContract,
    input_rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    expected_selected_paths = {row["selected_path"] for row in input_rows}
    expected_keys = {(row["item_id"], row["image_id"]) for row in input_rows}
    report: dict[str, Any] = {}

    for filename in METADATA_FILES:
        source_fields, source_rows = _read_csv(source_root / filename)
        subset_fields, subset_rows = _read_csv(eval_root / filename)
        _check(
            subset_fields == [*source_fields, "split", "selected_path"],
            f"{filename} subset fields do not extend its own source fields",
        )
        _check(
            len(subset_rows) == contract.input_count,
            f"{filename} rows: expected {contract.input_count}, got {len(subset_rows)}",
        )

        def source_signature(row: Mapping[str, str]) -> tuple[str, ...]:
            return tuple(row[field] for field in source_fields)

        source_counter = Counter(source_signature(row) for row in source_rows)
        multiplicities = [source_counter[source_signature(row)] for row in subset_rows]
        missing_count = sum(count == 0 for count in multiplicities)
        _check(
            missing_count == 0,
            f"{filename} subset row is not traceable to its own source",
        )
        _check(
            all(row["split"] == "input" for row in subset_rows),
            f"{filename} subset contains a non-input row",
        )
        subset_selected_paths = [row["selected_path"] for row in subset_rows]
        _check(
            len(set(subset_selected_paths)) == len(subset_selected_paths),
            f"{filename} selected_path is not unique",
        )
        _check(
            set(subset_selected_paths) == expected_selected_paths,
            f"{filename} selected paths differ from the input manifest",
        )
        if filename in {"pool.csv", "index-main.csv"}:
            subset_keys = {(row["item_id"], row["image_id"]) for row in subset_rows}
            _check(
                subset_keys == expected_keys,
                f"{filename} item/image keys differ from the input manifest",
            )

        report[filename] = {
            "rows": len(subset_rows),
            "missing_from_source": missing_count,
            "source_multiplicity": {
                str(value): count
                for value, count in sorted(Counter(multiplicities).items())
            },
        }

    pool_bytes = (eval_root / "pool.csv").read_bytes()
    index_bytes = (eval_root / "index-main.csv").read_bytes()
    report["pool_index_relation"] = (
        "identical" if pool_bytes == index_bytes else "distinct"
    )
    report["pool_index_assessment"] = (
        "identical_independently_traceable"
        if pool_bytes == index_bytes
        else "distinct_independently_traceable"
    )
    return report


def _verify_protocol(
    *,
    source_root: Path,
    eval_root: Path,
    contract: DatasetContract,
) -> None:
    protocol_path = eval_root / "manifests" / "protocol.json"
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        selection = protocol["selection"]
        metadata = protocol["metadata"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise VerificationError(f"protocol.json is invalid: {exc}") from exc

    _check(protocol.get("schema_version") == 1, "protocol schema_version must be 1")
    _check(protocol.get("seed") == contract.seed, "protocol seed differs")
    _check(
        Path(protocol.get("source_root", "")).resolve() == source_root.resolve(),
        "protocol source_root differs",
    )
    _check(
        selection.get("input_count") == contract.input_count,
        "protocol input_count differs",
    )
    _check(
        selection.get("psnr_ssim_pair_count") == contract.pair_count,
        "protocol pair count differs",
    )
    _check(
        selection.get("group_quotas") == dict(contract.group_quotas),
        "protocol group quotas differ",
    )
    expected_source_hashes = {
        filename: _file_sha256(source_root / filename) for filename in METADATA_FILES
    }
    _check(
        metadata.get("source_csv_sha256") == expected_source_hashes,
        "protocol source CSV hashes differ from the actual source files",
    )
    expected_counts = {filename: contract.input_count for filename in METADATA_FILES}
    _check(
        metadata.get("subset_counts") == expected_counts,
        "protocol metadata subset counts differ",
    )


def _symlink_target_map(eval_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(eval_root)): str(path.resolve())
        for path in sorted((eval_root / "input").rglob("*"))
        if path.is_symlink()
    }


def _verify_deterministic_rerun(
    *,
    source_root: Path,
    eval_root: Path,
    contract: DatasetContract,
) -> int:
    before = _protected_tree_signature(source_root, eval_root)
    with tempfile.TemporaryDirectory(
        prefix="perf-eval-gen-verify-", dir="/tmp"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory).resolve()
        _check(
            not _is_relative_to(temporary_root, source_root.resolve()),
            "temporary verification root is inside curated source",
        )
        generated_root = temporary_root / "eval500"
        command = [
            sys.executable,
            str(SELECT_SCRIPT),
            "--source-root",
            str(source_root),
            "--output-root",
            str(generated_root),
            "--total",
            str(contract.input_count),
            "--pair-count",
            str(contract.pair_count),
            "--seed",
            contract.seed,
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        _check(
            completed.returncode == 0,
            "deterministic rerun failed: "
            + (completed.stderr.strip() or completed.stdout.strip()),
        )

        for relative_path in DETERMINISTIC_FILES:
            existing = eval_root / relative_path
            regenerated = generated_root / relative_path
            _check(
                existing.read_bytes() == regenerated.read_bytes(),
                f"deterministic rerun differs: {relative_path}",
            )
        _check(
            _symlink_target_map(eval_root) == _symlink_target_map(generated_root),
            "deterministic rerun symlink view differs",
        )

    after = _protected_tree_signature(source_root, eval_root)
    _check(before == after, "strict verification changed a protected dataset tree")
    return len(DETERMINISTIC_FILES)


def verify_dataset(
    *,
    source_root: Path,
    eval_root: Path,
    contract: DatasetContract | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    contract = contract or DatasetContract()
    source_root = source_root.expanduser().resolve()
    eval_root = eval_root.expanduser().resolve()
    _check(source_root.is_dir(), f"source root is missing: {source_root}")
    _check(eval_root.is_dir(), f"EVAL500 root is missing: {eval_root}")
    _check(
        not _is_relative_to(eval_root, source_root / "input")
        and not _is_relative_to(eval_root, source_root / "ref"),
        "EVAL500 cannot be nested inside a protected source split",
    )

    input_rows, input_by_key = _verify_input_manifest(
        source_root=source_root,
        eval_root=eval_root,
        contract=contract,
        strict=strict,
    )
    pair_rows = _verify_pair_manifest(
        eval_root=eval_root,
        contract=contract,
        input_by_key=input_by_key,
    )
    metadata_report = _verify_metadata_subsets(
        source_root=source_root,
        eval_root=eval_root,
        contract=contract,
        input_rows=input_rows,
    )
    _verify_protocol(
        source_root=source_root,
        eval_root=eval_root,
        contract=contract,
    )

    deterministic_files = 0
    protected_trees_unchanged = True
    if strict:
        deterministic_files = _verify_deterministic_rerun(
            source_root=source_root,
            eval_root=eval_root,
            contract=contract,
        )

    return {
        "status": "PASS",
        "strict": strict,
        "input_symlinks": len(input_rows),
        "pair_count": len(pair_rows),
        "group_quotas": dict(Counter(row["group"] for row in input_rows)),
        "metadata": metadata_report,
        "deterministic_files": deterministic_files,
        "protected_trees_unchanged": protected_trees_unchanged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        report = verify_dataset(
            source_root=args.source_root,
            eval_root=args.eval_root,
            strict=args.strict,
        )
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("PASS: existing EVAL500 satisfies the fixed selection contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
