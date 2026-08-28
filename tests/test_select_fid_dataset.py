from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts.select_fid_dataset import (
    ArchiveSpec,
    build_dataset,
    choose_product_images,
    scan_archives,
    select_by_quota,
    verify_dataset,
)


def _annotation(filename: str, *, objects: int = 1, width: int = 1200) -> str:
    object_xml = "".join(
        "<object><name>상품</name><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>10</ymax>"
        "</bndbox></object>"
        for _ in range(objects)
    )
    return (
        "<annotation>"
        f"<filename>{filename}</filename>"
        f"<size><width>{width}</width><height>{width}</height><depth>3</depth></size>"
        f"{object_xml}</annotation>"
    )


def _meta(item_no: str, category: str, filename: str) -> str:
    return (
        "<comp_cd><div_cd>"
        f"<item_no>{item_no}</item_no>"
        f"<div_l>{category}</div_l>"
        "<div_m>중분류</div_m>"
        f"<img_prod_nm>상품 {item_no}</img_prod_nm>"
        "</div_cd>"
        f"<annotation><filename>{filename}</filename></annotation>"
        "</comp_cd>"
    )


def _write_pair(
    root: Path,
    spec: ArchiveSpec,
    records: list[dict[str, object]],
) -> None:
    with (
        ZipFile(root / spec.label_zip, "w", ZIP_DEFLATED) as labels,
        ZipFile(root / spec.source_zip, "w", ZIP_DEFLATED) as sources,
    ):
        for record in records:
            item_no = str(record["item_no"])
            filename = str(record.get("filename", f"{item_no}_00_m_1.jpg"))
            directory = f"{item_no}_상품{item_no}"
            stem = Path(filename).stem
            labels.writestr(
                f"{directory}/{stem}.xml",
                _annotation(
                    filename,
                    objects=int(record.get("objects", 1)),
                    width=int(record.get("width", 1200)),
                ),
            )
            labels.writestr(
                f"{directory}/{stem}_meta.xml",
                _meta(
                    item_no,
                    str(record.get("metadata_category", spec.category)),
                    filename,
                ),
            )
            if record.get("has_source", True):
                sources.writestr(
                    f"{directory}/{filename}",
                    f"jpeg-bytes-{spec.category}-{item_no}-{filename}".encode(),
                )


def _spec(category: str = "생활용품") -> ArchiveSpec:
    safe = category.replace("/", "_")
    return ArchiveSpec(
        label_zip=f"[라벨]{safe}.zip",
        source_zip=f"[원천]{safe}.zip",
        category=category,
    )


def test_scan_filters_multi_object_small_and_missing_source(tmp_path: Path) -> None:
    spec = _spec()
    _write_pair(
        tmp_path,
        spec,
        [
            {"item_no": "100"},
            {"item_no": "200", "objects": 2},
            {"item_no": "300", "width": 800},
            {"item_no": "400", "has_source": False},
            {"item_no": "500", "filename": "18.jpg"},
        ],
    )

    result = scan_archives(tmp_path, (spec,))

    assert [candidate.item_no for candidate in result.candidates] == ["100"]
    assert result.counts == {
        "candidate_count": 5,
        "single_object_count": 4,
        "eligible_candidate_count": 1,
    }


def test_product_choice_prefers_height_then_size_then_seed_hash(tmp_path: Path) -> None:
    spec = _spec()
    filenames = [
        "100_60_m_1.jpg",
        "100_30_m_1.jpg",
        "100_00_s_1.jpg",
        "100_00_m_1.jpg",
        "100_00_m_2.jpg",
    ]
    _write_pair(
        tmp_path,
        spec,
        [{"item_no": "100", "filename": filename} for filename in filenames],
    )
    candidates = scan_archives(tmp_path, (spec,)).candidates

    selected = choose_product_images(candidates, seed=0)

    expected = min(
        ("100_00_m_1.jpg", "100_00_m_2.jpg"),
        key=lambda name: hashlib.sha256(f"0:{name}".encode()).hexdigest(),
    )
    assert len(selected) == 1
    assert selected[0].source_filename == expected


def test_scan_uses_product_directory_item_number_when_filename_prefix_is_wrong(
    tmp_path: Path,
) -> None:
    spec = _spec()
    _write_pair(
        tmp_path,
        spec,
        [{"item_no": "500", "filename": "600_00_m_1.jpg"}],
    )

    candidates = scan_archives(tmp_path, (spec,)).candidates

    assert len(candidates) == 1
    assert candidates[0].item_no == "500"


def test_quota_cut_is_deterministic_and_seeded(tmp_path: Path) -> None:
    spec = _spec()
    records = [{"item_no": str(100 + index)} for index in range(12)]
    _write_pair(tmp_path, spec, records)
    products = choose_product_images(scan_archives(tmp_path, (spec,)).candidates, seed=0)

    first = select_by_quota(products, seed=0, quotas={"생활용품": 4})
    repeated = select_by_quota(products, seed=0, quotas={"생활용품": 4})
    changed = select_by_quota(products, seed=1, quotas={"생활용품": 4})
    expected = sorted(
        (candidate.item_no for candidate in products),
        key=lambda item_no: hashlib.sha256(f"0:{item_no}".encode()).hexdigest(),
    )[:4]

    assert [candidate.item_no for candidate in first] == expected
    assert first == repeated
    assert {candidate.item_no for candidate in changed} != {
        candidate.item_no for candidate in first
    }


def test_build_and_verify_manifest_hashes_and_detects_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    specs = (_spec("생활용품"), _spec("이/미용"), _spec("홈클린"))
    for spec, count in zip(specs, (4, 4, 3), strict=True):
        offset = {"생활용품": 100, "이/미용": 200, "홈클린": 300}[spec.category]
        _write_pair(
            source_root,
            spec,
            [
                {
                    "item_no": str(offset + index),
                    "metadata_category": (
                        "의약외품" if spec.category == "생활용품" else spec.category
                    ),
                }
                for index in range(count)
            ],
        )
    output = tmp_path / "fid500"
    quotas = {"생활용품": 2, "이/미용": 2, "홈클린": 1}

    summary = build_dataset(
        source_root=source_root,
        output_root=output,
        seed=0,
        archive_specs=specs,
        quotas=quotas,
    )
    verified = verify_dataset(output)
    with (output / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))

    assert summary["final_count"] == verified["final_count"] == 5
    assert summary["quota_counts"] == quotas
    assert len(rows) == 5
    assert set(rows[0]) == {
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
    }
    assert selection["seed"] == 0
    assert selection["quota"] == quotas
    assert len(selection["source_zips"]) == 3
    assert len(selection["label_zips"]) == 3
    assert (output / "input" / "이_미용").is_dir()

    changed = output / "input" / "생활용품" / f"{rows[0]['item_no']}.jpg"
    if not changed.is_file():
        changed = next((output / "input").rglob("*.jpg"))
    changed.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_dataset(output)
