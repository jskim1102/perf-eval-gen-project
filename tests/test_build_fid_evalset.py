from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from PIL import Image
from cleanfid.fid import fid_from_feats

from scripts.build_fid_evalset import (
    PROMPT_TEMPLATE,
    SELECTION_METHOD,
    build_product_prompt,
    build_final_dataset,
    build_pool,
    feature_fid,
    seed_reused_outputs,
    select_feature_subset,
)
from scripts.run_fid_eval import build_generation_manifest
from scripts.select_fid_dataset import ArchiveSpec


MANIFEST_FIELDS = [
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
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotation(filename: str, *, width: int = 1200) -> str:
    return (
        "<annotation>"
        f"<filename>{filename}</filename>"
        f"<size><width>{width}</width><height>{width}</height><depth>3</depth></size>"
        "<object><name>상품</name></object>"
        "</annotation>"
    )


def _metadata(item_no: str) -> str:
    return (
        "<comp_cd><div_cd>"
        f"<item_no>{item_no}</item_no>"
        "<div_m>fixture</div_m>"
        "<div_s>fixture detail</div_s>"
        f"<img_prod_nm>상품 {item_no}</img_prod_nm>"
        "</div_cd></comp_cd>"
    )


def _write_pair(
    root: Path,
    spec: ArchiveSpec,
    records: list[tuple[str, str]],
) -> None:
    with (
        ZipFile(root / spec.label_zip, "w", ZIP_DEFLATED) as labels,
        ZipFile(root / spec.source_zip, "w", ZIP_DEFLATED) as sources,
    ):
        for item_no, filename in records:
            directory = f"{item_no}_상품"
            stem = Path(filename).stem
            labels.writestr(f"{directory}/{stem}.xml", _annotation(filename))
            labels.writestr(
                f"{directory}/{stem}_meta.xml",
                _metadata(item_no),
            )
            sources.writestr(
                f"{directory}/{filename}",
                f"jpeg-{item_no}-{filename}".encode(),
            )


def _source_spec(category: str, suffix: str) -> ArchiveSpec:
    return ArchiveSpec(
        label_zip=f"[라벨]{suffix}.zip",
        source_zip=f"[원천]{suffix}.zip",
        category=category,
    )


def _write_dataset(root: Path, rows: list[dict[str, str]]) -> None:
    prompt_aware = bool(rows and "prompt" in rows[0])
    fields = (
        [
            *MANIFEST_FIELDS,
            "소분류",
            "prompt",
            "prompt_name_source",
            "prompt_name_truncated",
        ]
        if prompt_aware
        else MANIFEST_FIELDS
    )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["대분류"]] = counts.get(row["대분류"], 0) + 1
    selection = {
        "schema_version": 1,
        "selection_rule_version": "fixture",
        "seed": 0,
        "quota": counts,
        "counts": {"final_count": len(rows)},
        "manifest_sha256": _sha256(root / "manifest.csv"),
        "source_dataset": {"name": "fixture", "builder": "NIA"},
        "source_root": "/fixture/source",
        "source_zips": [{"filename": "source.zip", "size": 1, "mtime_ns": 1}],
        "label_zips": [{"filename": "label.zip", "size": 1, "mtime_ns": 1}],
        "rules": {
            "category_directories": {
                "생활용품": "생활용품",
                "이/미용": "이_미용",
            }
        },
    }
    if prompt_aware:
        selection["prompt_protocol"] = {
            "template": PROMPT_TEMPLATE,
            "product_name_max_words": 15,
            "product_name_truncation": "first 15 whitespace-delimited words",
            "fallback_order": ["img_prod_nm", "div_s", "div_m"],
            "whitespace_normalization": "strip and collapse consecutive whitespace",
            "label_text_policy": "preserve Korean and all other label text",
        }
    (root / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8"
    )


def test_pool_uses_full_eligible_product_set_without_quota_cut(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    household = _source_spec("생활용품", "생활")
    beauty = _source_spec("이/미용", "미용")
    _write_pair(
        source_root,
        household,
        [
            ("100", "100_30_m_1.jpg"),
            ("100", "100_00_s_1.jpg"),
            ("200", "200_00_m_1.jpg"),
        ],
    )
    _write_pair(source_root, beauty, [("300", "300_00_m_1.jpg")])
    pool_root = tmp_path / "fid500-v2" / "pool"

    summary = build_pool(
        source_root=source_root,
        pool_root=pool_root,
        seed=0,
        archive_specs=(household, beauty),
        progress=None,
    )

    with (pool_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selection = json.loads((pool_root / "selection.json").read_text(encoding="utf-8"))
    assert summary["pool_count"] == 3
    assert [row["item_no"] for row in rows] == ["100", "200", "300"]
    assert rows[0]["source_filename"] == "100_00_s_1.jpg"
    assert selection["counts"]["unique_product_count"] == 3
    assert selection["counts"]["final_count"] == 3
    assert selection["manifest_sha256"] == _sha256(pool_root / "manifest.csv")
    assert selection["rules"]["quota_selection"] == "none; full eligible product pool"
    assert len(selection["source_zips"]) == 2
    assert rows[0]["소분류"] == "fixture detail"
    assert rows[0]["prompt"] == (
        "a high quality studio product photograph of 상품 100, fixture fixture detail, "
        "on a clean white background"
    )
    assert rows[0]["prompt_name_source"] == "img_prod_nm"
    assert rows[0]["prompt_name_truncated"] == "false"
    assert selection["prompt_protocol"]["template"] == PROMPT_TEMPLATE


def test_product_prompt_normalizes_truncates_and_records_fallback() -> None:
    long_name = "  " + "   ".join(f"word-{index}" for index in range(18)) + "  "

    prompt = build_product_prompt(
        img_prod_nm=long_name,
        div_m="  중   분류 ",
        div_s=" 소   분류 ",
    )
    fallback = build_product_prompt(
        img_prod_nm="   ",
        div_m=" 중분류 ",
        div_s="  소   분류 ",
    )

    assert prompt.product_name == " ".join(f"word-{index}" for index in range(15))
    assert prompt.name_source == "img_prod_nm"
    assert prompt.name_truncated is True
    assert "중 분류 소 분류" in prompt.prompt
    assert fallback.product_name == "소 분류"
    assert fallback.name_source == "div_s"
    assert fallback.name_truncated is False
    with pytest.raises(ValueError, match="no product name fallback"):
        build_product_prompt(img_prod_nm="", div_m=" ", div_s="\t")


def test_feature_fid_matches_cleanfid_and_selection_is_deterministic() -> None:
    real = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    generated = real + np.asarray(
        [
            [0.01, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.00, 0.00, 1.00],
            [2.00, 2.00, 2.00],
        ]
    )

    assert feature_fid(real, generated) == pytest.approx(
        float(fid_from_feats(real, generated)), abs=1e-10
    )
    first = select_feature_subset(
        real_features=real,
        generated_features=generated,
        item_ids=("a", "b", "c", "d", "e"),
        target_count=3,
        target_fid=100.0,
        seed=0,
    )
    repeated = select_feature_subset(
        real_features=real,
        generated_features=generated,
        item_ids=("a", "b", "c", "d", "e"),
        target_count=3,
        target_fid=100.0,
        seed=0,
    )
    assert first == repeated
    assert first.indices == (0, 1, 2)
    assert first.initial_fid == first.final_fid
    assert first.swaps == ()


def test_reuse_requires_matching_input_sha_and_rewrites_run_paths(tmp_path: Path) -> None:
    source_dataset = tmp_path / "v1"
    target_dataset = tmp_path / "v2" / "pool"
    source_run = tmp_path / "runs" / "fid500"
    target_run = tmp_path / "runs" / "fid500-v2-pool"
    for root in (source_dataset, target_dataset):
        (root / "input" / "생활용품").mkdir(parents=True)

    source_rows: list[dict[str, str]] = []
    target_rows: list[dict[str, str]] = []
    for index, item_no in enumerate(("100", "200")):
        source_image = source_dataset / "input" / "생활용품" / f"{item_no}.jpg"
        target_image = target_dataset / "input" / "생활용품" / f"{item_no}.jpg"
        source_image.write_bytes(f"source-{item_no}".encode())
        target_image.write_bytes(
            source_image.read_bytes() if index == 0 else b"different-input"
        )
        for rows, image in ((source_rows, source_image), (target_rows, target_image)):
            rows.append(
                {
                    "item_no": item_no,
                    "대분류": "생활용품",
                    "중분류": "fixture",
                    "상품명": item_no,
                    "zip_file": "source.zip",
                    "zip_member": f"folder/{item_no}.jpg",
                    "source_filename": f"{item_no}_00_m_1.jpg",
                    "width": "1200",
                    "height": "1200",
                    "sha256": _sha256(image),
                }
            )
    _write_dataset(source_dataset, source_rows)
    _write_dataset(target_dataset, target_rows)
    build_generation_manifest(dataset_root=target_dataset)

    (source_run / "images" / "생활용품").mkdir(parents=True)
    generated_rows = []
    for item_no in ("100", "200"):
        output = source_run / "images" / "생활용품" / f"{item_no}.png"
        Image.new("RGB", (16, 16), (10, 20, 30)).save(output)
        generated_rows.append(
            {
                "input_path": str(source_dataset / "input" / "생활용품" / f"{item_no}.jpg"),
                "output_path": str(output),
                "sha256": _sha256(output),
                "seed": "0",
                "strength": "0.15",
            }
        )
    with (source_run / "generated.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_path", "output_path", "sha256", "seed", "strength"],
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    reused = seed_reused_outputs(
        source_dataset_root=source_dataset,
        source_run_root=source_run,
        target_dataset_root=target_dataset,
        target_run_root=target_run,
        seed=0,
        strength=0.15,
    )

    assert reused == 1
    with (target_run / "generated.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["input_path"] == str(
        target_dataset / "input" / "생활용품" / "100.jpg"
    )
    assert rows[0]["output_path"] == str(
        target_run / "images" / "생활용품" / "100.png"
    )
    assert Path(rows[0]["output_path"]).read_bytes() == Path(
        generated_rows[0]["output_path"]
    ).read_bytes()
    assert not (target_run / "images" / "생활용품" / "200.png").exists()


def test_final_dataset_records_truthful_method_and_seeds_selected_run(
    tmp_path: Path,
) -> None:
    pool_root = tmp_path / "fid500-v2" / "pool"
    pool_run = tmp_path / "runs" / "fid500-v2-pool"
    final_root = pool_root.parent
    final_run = tmp_path / "runs" / "fid500-v2"
    (pool_root / "input" / "생활용품").mkdir(parents=True)
    (pool_run / "images" / "생활용품").mkdir(parents=True)
    rows: list[dict[str, str]] = []
    generated_rows: list[dict[str, str]] = []
    for index, item_no in enumerate(("100", "200", "300")):
        input_path = pool_root / "input" / "생활용품" / f"{item_no}.jpg"
        input_path.write_bytes(f"input-{item_no}".encode())
        output_path = pool_run / "images" / "생활용품" / f"{item_no}.png"
        Image.new("RGB", (8, 8), (index + 1, 2, 3)).save(output_path)
        rows.append(
            {
                "item_no": item_no,
                "대분류": "생활용품",
                "중분류": "fixture",
                "상품명": item_no,
                "zip_file": "source.zip",
                "zip_member": f"folder/{item_no}.jpg",
                "source_filename": f"{item_no}_00_m_1.jpg",
                "width": "1200",
                "height": "1200",
                "sha256": _sha256(input_path),
                "소분류": "fixture detail",
                "prompt": f"fixture product photograph of {item_no}",
                "prompt_name_source": "img_prod_nm",
                "prompt_name_truncated": "false",
            }
        )
        generated_rows.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "seed": "0",
                "strength": "0.15",
                "prompt": f"fixture product photograph of {item_no}",
            }
        )
    _write_dataset(pool_root, rows)
    with (pool_run / "generated.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_path",
                "output_path",
                "sha256",
                "seed",
                "strength",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    result = select_feature_subset(
        real_features=np.asarray([[0.0], [0.1], [5.0]]),
        generated_features=np.asarray([[0.0], [0.2], [9.0]]),
        item_ids=("100", "200", "300"),
        target_count=2,
        target_fid=10.0,
        seed=0,
    )
    summary = build_final_dataset(
        pool_root=pool_root,
        pool_run_root=pool_run,
        output_root=final_root,
        final_run_root=final_run,
        selection=result,
        seed=0,
        strength=0.15,
        feature_cache_path=pool_root / "features.npz",
    )

    selection = json.loads((final_root / "selection.json").read_text(encoding="utf-8"))
    assert summary["final_count"] == 2
    assert selection["selection_method"] == SELECTION_METHOD
    assert selection["counts"] == {"pool_count": 3, "final_count": 2}
    assert selection["method_details"]["feature_extractor"] == "clean-fid Inception-v3"
    assert selection["method_details"]["pair_distance"] == "squared Euclidean distance"
    assert selection["method_details"]["seed"] == 0
    assert selection["manifest_sha256"] == _sha256(final_root / "manifest.csv")
    with (final_run / "generated.csv").open(encoding="utf-8", newline="") as handle:
        final_generated = list(csv.DictReader(handle))
    assert len(final_generated) == 2
    assert [row["prompt"] for row in final_generated] == [
        "fixture product photograph of 100",
        "fixture product photograph of 200",
    ]
    assert all(Path(row["input_path"]).is_relative_to(final_root) for row in final_generated)
    assert all(Path(row["output_path"]).is_relative_to(final_run) for row in final_generated)
