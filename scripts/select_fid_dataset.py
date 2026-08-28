#!/usr/bin/env python3
"""Build a deterministic FID reference set from AI Hub 「상품 이미지」.

Dataset: AI Hub 「상품 이미지」, aihub.or.kr/aidata/34145, NIA, 2020.
When publishing results derived from this dataset, identify it as an outcome of
the NIA AI training-data program (NIA 사업결과) as required by its source terms.

Only ZIP central directories, label XML entries, and the final selected image
entries are read. The read-only NAS archives are never extracted wholesale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence, TextIO
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile, ZipInfo


DEFAULT_SOURCE_ROOT = Path("/mnt/nas/2026/상품 이미지/Training")
DEFAULT_OUTPUT_ROOT = Path("/home/kim_3090/datasets/aihub-product/fid500")
DEFAULT_SEED = 0
SELECTION_RULE_VERSION = "aihub-product-fid500-v1"

# 500 * (1041 / 2228, 803 / 2228, 384 / 2228), with rounding adjusted to 500.
DEFAULT_QUOTAS = {"생활용품": 232, "이/미용": 179, "홈클린": 89}
CATEGORY_DIRECTORIES = {"생활용품": "생활용품", "이/미용": "이_미용", "홈클린": "홈클린"}
HEIGHT_PRIORITY = {"00": 0, "30": 1, "60": 2}
SIZE_PRIORITY = {"m": 0, "s": 1}
FILENAME_PATTERN = re.compile(
    r"^(?P<item_no>\d+)_(?P<height>00|30|60)_(?P<size>m|s)_(?P<angle>\d+)\.jpg$",
    re.IGNORECASE,
)
MANIFEST_FIELDS = (
    "item_no",
    "대분류",
    "중분류",
    "상품명",
    "zip_file",
    "zip_member",
    "source_filename",
    "width",
    "height",
    "sha256",
)


@dataclass(frozen=True)
class ArchiveSpec:
    label_zip: str
    source_zip: str
    category: str


DEFAULT_ARCHIVES = (
    ArchiveSpec("[라벨]생활용품1.zip", "[원천]생활용품1.zip", "생활용품"),
    ArchiveSpec("[라벨]생활용품2.zip", "[원천]생활용품2.zip", "생활용품"),
    ArchiveSpec("[라벨]생활용품3.zip", "[원천]생활용품3.zip", "생활용품"),
    ArchiveSpec("[라벨]이_미용1.zip", "[원천]이_미용1.zip", "이/미용"),
    ArchiveSpec("[라벨]이_미용3.zip", "[원천]이_미용3.zip", "이/미용"),
    ArchiveSpec("[라벨]홈클린.zip", "[원천]홈클린.zip", "홈클린"),
)


@dataclass(frozen=True)
class Candidate:
    item_no: str
    category: str
    label_zip: str
    source_zip: str
    annotation_member: str
    metadata_member: str
    source_member: str
    source_filename: str
    width: int
    height: int
    height_code: str
    size_code: str


@dataclass(frozen=True)
class ScanResult:
    candidates: tuple[Candidate, ...]
    counts: dict[str, int]


def _hash_rank(seed: int | str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_xml(data: bytes, *, archive: str, member: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ValueError(f"malformed XML in {archive}:{member}: {exc}") from exc


def _required_int(root: ElementTree.Element, path: str, *, context: str) -> int:
    value = root.findtext(path)
    try:
        return int(value or "")
    except ValueError as exc:
        raise ValueError(f"missing or invalid {path} in {context}") from exc


def _annotation_infos(infos: Iterable[ZipInfo]) -> list[ZipInfo]:
    return sorted(
        (
            info
            for info in infos
            if not info.is_dir()
            and info.filename.lower().endswith(".xml")
            and not info.filename.lower().endswith("_meta.xml")
        ),
        key=lambda info: info.filename,
    )


def scan_archives(
    source_root: Path,
    archive_specs: Sequence[ArchiveSpec] = DEFAULT_ARCHIVES,
    *,
    progress: TextIO | None = sys.stderr,
) -> ScanResult:
    """Scan label XML and source ZIP names without reading source image bytes."""
    source_root = source_root.expanduser().resolve()
    candidates: list[Candidate] = []
    candidate_count = 0
    single_object_count = 0

    for spec in archive_specs:
        label_path = source_root / spec.label_zip
        source_path = source_root / spec.source_zip
        if not label_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing archive pair: {label_path} / {source_path}")
        if progress is not None:
            print(f"[scan] {spec.label_zip}: reading source member names", file=progress, flush=True)
        try:
            with ZipFile(source_path) as source_zip:
                source_members = {
                    info.filename for info in source_zip.infolist() if not info.is_dir()
                }
            with ZipFile(label_path) as label_zip:
                label_infos = label_zip.infolist()
                label_members = {
                    info.filename for info in label_infos if not info.is_dir()
                }
                annotations = _annotation_infos(label_infos)
                if progress is not None:
                    print(
                        f"[scan] {spec.label_zip}: {len(annotations)} annotations",
                        file=progress,
                        flush=True,
                    )
                for index, info in enumerate(annotations, start=1):
                    candidate_count += 1
                    root = _parse_xml(
                        label_zip.read(info), archive=spec.label_zip, member=info.filename
                    )
                    objects = root.findall("object")
                    if len(objects) != 1:
                        continue
                    single_object_count += 1
                    context = f"{spec.label_zip}:{info.filename}"
                    width = _required_int(root, "size/width", context=context)
                    height = _required_int(root, "size/height", context=context)
                    if width != height or width < 1000:
                        continue
                    filename = (root.findtext("filename") or "").strip()
                    match = FILENAME_PATTERN.fullmatch(filename)
                    if match is None:
                        # The source also contains legacy turntable files such as
                        # ``18.jpg``. They carry no height/size code, so they cannot
                        # participate in the contracted 00/30/60, m/s ranking.
                        continue
                    annotation_path = PurePosixPath(info.filename)
                    item_no = annotation_path.parent.name.partition("_")[0]
                    if not item_no.isdigit():
                        raise ValueError(
                            f"missing item_no in annotation directory for {context}"
                        )
                    source_member = str(annotation_path.with_name(filename))
                    metadata_member = str(
                        annotation_path.with_name(f"{annotation_path.stem}_meta.xml")
                    )
                    if source_member not in source_members:
                        continue
                    if metadata_member not in label_members:
                        raise ValueError(f"missing metadata XML: {spec.label_zip}:{metadata_member}")
                    candidates.append(
                        Candidate(
                            # A small number of source filenames have a mistyped
                            # numeric prefix. The enclosing product directory and
                            # metadata item_no agree and are the product identity.
                            item_no=item_no,
                            category=spec.category,
                            label_zip=spec.label_zip,
                            source_zip=spec.source_zip,
                            annotation_member=info.filename,
                            metadata_member=metadata_member,
                            source_member=source_member,
                            source_filename=filename,
                            width=width,
                            height=height,
                            height_code=match.group("height"),
                            size_code=match.group("size").lower(),
                        )
                    )
                    if progress is not None and index % 10000 == 0:
                        print(
                            f"[scan] {spec.label_zip}: {index}/{len(annotations)}",
                            file=progress,
                            flush=True,
                        )
        except BadZipFile as exc:
            raise ValueError(f"unreadable ZIP archive in pair {spec}: {exc}") from exc
        if progress is not None:
            print(
                f"[scan] {spec.label_zip}: eligible so far {len(candidates)}",
                file=progress,
                flush=True,
            )

    ordered = tuple(
        sorted(candidates, key=lambda item: (item.category, item.item_no, item.source_filename))
    )
    return ScanResult(
        candidates=ordered,
        counts={
            "candidate_count": candidate_count,
            "single_object_count": single_object_count,
            "eligible_candidate_count": len(ordered),
        },
    )


def choose_product_images(
    candidates: Sequence[Candidate], *, seed: int | str
) -> tuple[Candidate, ...]:
    by_item: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_item[candidate.item_no].append(candidate)

    selected: list[Candidate] = []
    for item_no, options in by_item.items():
        categories = {option.category for option in options}
        if len(categories) != 1:
            raise ValueError(f"item_no {item_no} appears in multiple categories: {categories}")
        selected.append(
            min(
                options,
                key=lambda option: (
                    HEIGHT_PRIORITY[option.height_code],
                    SIZE_PRIORITY[option.size_code],
                    _hash_rank(seed, option.source_filename),
                    option.label_zip,
                    option.source_member,
                ),
            )
        )
    return tuple(sorted(selected, key=lambda item: (item.category, item.item_no)))


def select_by_quota(
    products: Sequence[Candidate],
    *,
    seed: int | str,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
) -> tuple[Candidate, ...]:
    if not quotas or any(value <= 0 for value in quotas.values()):
        raise ValueError("quotas must contain positive counts")
    by_category: dict[str, list[Candidate]] = defaultdict(list)
    for product in products:
        by_category[product.category].append(product)

    selected: list[Candidate] = []
    for category in sorted(quotas):
        quota = quotas[category]
        available = by_category.get(category, [])
        if len(available) < quota:
            raise ValueError(
                f"category {category} has {len(available)} products, below quota {quota}"
            )
        ranked = sorted(
            available,
            key=lambda product: (_hash_rank(seed, product.item_no), product.item_no),
        )
        selected.extend(ranked[:quota])
    return tuple(selected)


def _zip_metadata(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"filename": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _category_directory(category: str) -> str:
    directory = CATEGORY_DIRECTORIES.get(category, category.replace("/", "_"))
    if not directory or directory in {".", ".."} or "/" in directory or "\\" in directory:
        raise ValueError(f"unsafe category directory: {category!r}")
    return directory


def _load_selected_metadata(
    source_root: Path, selected: Sequence[Candidate]
) -> dict[Candidate, dict[str, str]]:
    by_label_zip: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        by_label_zip[candidate.label_zip].append(candidate)

    metadata: dict[Candidate, dict[str, str]] = {}
    for label_name, candidates in sorted(by_label_zip.items()):
        with ZipFile(source_root / label_name) as label_zip:
            for candidate in sorted(candidates, key=lambda item: item.metadata_member):
                root = _parse_xml(
                    label_zip.read(candidate.metadata_member),
                    archive=label_name,
                    member=candidate.metadata_member,
                )
                item_no = (root.findtext(".//item_no") or "").strip()
                if item_no != candidate.item_no:
                    raise ValueError(
                        f"metadata item_no mismatch in {label_name}:{candidate.metadata_member}"
                    )
                fallback_name = PurePosixPath(candidate.source_member).parent.name.split("_", 1)[-1]
                metadata[candidate] = {
                    "중분류": (root.findtext(".//div_m") or "").strip(),
                    "상품명": (root.findtext(".//img_prod_nm") or fallback_name).strip(),
                }
    return metadata


def _write_selected_images(
    *,
    source_root: Path,
    staging_root: Path,
    selected: Sequence[Candidate],
    metadata: Mapping[Candidate, Mapping[str, str]],
    progress: TextIO | None,
) -> list[dict[str, str | int]]:
    by_source_zip: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        by_source_zip[candidate.source_zip].append(candidate)

    rows: list[dict[str, str | int]] = []
    for source_name, candidates in sorted(by_source_zip.items()):
        if progress is not None:
            print(
                f"[extract] {source_name}: {len(candidates)} selected entries",
                file=progress,
                flush=True,
            )
        with ZipFile(source_root / source_name) as source_zip:
            for candidate in sorted(candidates, key=lambda item: (item.category, item.item_no)):
                destination = (
                    staging_root
                    / "input"
                    / _category_directory(candidate.category)
                    / f"{candidate.item_no}.jpg"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with source_zip.open(candidate.source_member) as source, destination.open("xb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        target.write(chunk)
                rows.append(
                    {
                        "item_no": candidate.item_no,
                        "대분류": candidate.category,
                        "중분류": metadata[candidate]["중분류"],
                        "상품명": metadata[candidate]["상품명"],
                        "zip_file": candidate.source_zip,
                        "zip_member": candidate.source_member,
                        "source_filename": candidate.source_filename,
                        "width": candidate.width,
                        "height": candidate.height,
                        "sha256": digest.hexdigest(),
                    }
                )
    return sorted(rows, key=lambda row: (str(row["대분류"]), str(row["item_no"])))


def build_dataset(
    *,
    source_root: Path,
    output_root: Path,
    seed: int | str = DEFAULT_SEED,
    archive_specs: Sequence[ArchiveSpec] = DEFAULT_ARCHIVES,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    progress: TextIO | None = sys.stderr,
) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve(strict=False)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output path already exists: {output_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("output path cannot be inside the read-only source root")

    scan = scan_archives(source_root, archive_specs, progress=progress)
    products = choose_product_images(scan.candidates, seed=seed)
    selected = select_by_quota(products, seed=seed, quotas=quotas)
    actual_quotas = dict(Counter(candidate.category for candidate in selected))
    if actual_quotas != dict(quotas):
        raise AssertionError(f"quota mismatch after selection: {actual_quotas}")
    metadata = _load_selected_metadata(source_root, selected)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.with_name(f".{output_root.name}.building-{os.getpid()}")
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(f"staging path already exists: {staging_root}")
    staging_root.mkdir()
    rows = _write_selected_images(
        source_root=source_root,
        staging_root=staging_root,
        selected=selected,
        metadata=metadata,
        progress=progress,
    )
    manifest_path = staging_root / "manifest.csv"
    with manifest_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        **scan.counts,
        "unique_product_count": len(products),
        "final_count": len(selected),
    }
    selection = {
        "schema_version": 1,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed) if isinstance(seed, int) or str(seed).isdigit() else str(seed),
        "quota": dict(quotas),
        "counts": counts,
        "manifest_sha256": _file_sha256(manifest_path),
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
        "rules": {
            "eligibility": "exactly one object; square image; width and height >= 1000; source JPG exists",
            "one_per_product": "height 00 > 30 > 60; size m > s; sha256(seed:filename) ascending",
            "quota_selection": "sha256(seed:item_no) ascending within each category",
            "category_directories": dict(CATEGORY_DIRECTORIES),
        },
    }
    with (staging_root / "selection.json").open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(staging_root, output_root)
    return {
        **counts,
        "quota_counts": actual_quotas,
        "output_root": str(output_root),
    }


def verify_dataset(output_root: Path) -> dict[str, object]:
    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.csv"
    selection_path = output_root / "selection.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"missing manifest.csv or selection.json under {output_root}")
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed selection.json: {exc}") from exc
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"manifest fields mismatch: {reader.fieldnames}")
        rows = list(reader)

    quotas = {str(key): int(value) for key, value in selection.get("quota", {}).items()}
    expected_count = sum(quotas.values())
    if len(rows) != expected_count:
        raise ValueError(f"manifest count mismatch: {len(rows)} != {expected_count}")
    if len({row["item_no"] for row in rows}) != len(rows):
        raise ValueError("manifest item_no values are not unique")
    actual_quotas = dict(Counter(row["대분류"] for row in rows))
    if actual_quotas != quotas:
        raise ValueError(f"manifest quota mismatch: {actual_quotas} != {quotas}")
    if selection.get("selection_rule_version") != SELECTION_RULE_VERSION:
        raise ValueError("selection rule version mismatch")
    counts = selection.get("counts", {})
    if int(counts.get("final_count", -1)) != expected_count:
        raise ValueError("selection final_count mismatch")
    if selection.get("manifest_sha256") != _file_sha256(manifest_path):
        raise ValueError("manifest sha256 mismatch")

    expected_paths: set[Path] = set()
    for row in rows:
        width = int(row["width"])
        height = int(row["height"])
        if width != height or width < 1000:
            raise ValueError(f"manifest dimensions are ineligible for item {row['item_no']}")
        image_path = (
            output_root
            / "input"
            / _category_directory(row["대분류"])
            / f"{row['item_no']}.jpg"
        )
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"selected image is missing or not a regular file: {image_path}")
        digest = _file_sha256(image_path)
        if digest != row["sha256"]:
            raise ValueError(
                f"sha256 mismatch for item {row['item_no']}: {digest} != {row['sha256']}"
            )
        expected_paths.add(image_path.resolve())
    actual_paths = {
        path.resolve() for path in (output_root / "input").rglob("*.jpg") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("selected image file set does not match manifest")
    return {"final_count": len(rows), "quota_counts": actual_quotas}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify:
        summary = verify_dataset(args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        print("FID500_VERIFY_OK")
        return 0
    summary = build_dataset(
        source_root=args.source_root,
        output_root=args.out,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
