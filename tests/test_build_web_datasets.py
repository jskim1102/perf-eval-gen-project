from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.build_web_datasets import (
    build_fid_web_dataset,
    build_psnr_ssim_web_dataset,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_fid_web_dataset_copies_only_the_frozen_final_set(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-fid"
    destination = tmp_path / "web/fid"
    images = []
    for item_no, category, directory in (
        ("10", "생활용품", "생활용품"),
        ("20", "이/미용", "이_미용"),
    ):
        path = source / "input" / directory / f"{item_no}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{item_no}".encode())
        images.append((item_no, category, path))
    fields = [
        "item_no",
        "대분류",
        "zip_member",
        "width",
        "height",
        "sha256",
        "prompt",
    ]
    source.mkdir(exist_ok=True)
    with (source / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_no, category, image in images:
            writer.writerow(
                {
                    "item_no": item_no,
                    "대분류": category,
                    "zip_member": f"archive/{image.name}",
                    "width": "1000",
                    "height": "1000",
                    "sha256": _sha256(image),
                    "prompt": f"prompt {item_no}",
                }
            )
    selection = {
        "counts": {"final_count": 2, "pool_count": 9},
        "manifest_sha256": _sha256(source / "manifest.csv"),
        "selection_rule_version": "v2",
        "seed": 0,
        "source_dataset": {"name": "fixture"},
        "pool_manifest": str(source / "pool/manifest.csv"),
        "method_details": {"feature_cache": str(source / "pool/features.npz")},
        "rules": {
            "category_directories": {"생활용품": "생활용품", "이/미용": "이_미용"}
        },
        "prompt_protocol": {"template": "fixture"},
    }
    (source / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8"
    )
    (source / "pool").mkdir()
    (source / "pool/unused.jpg").write_bytes(b"do-not-copy")

    report = build_fid_web_dataset(
        source_root=source,
        destination_root=destination,
        expected_count=2,
    )

    assert report.count == 2
    assert (destination / "manifest.csv").read_bytes() == (
        source / "manifest.csv"
    ).read_bytes()
    web_selection = json.loads((destination / "selection.json").read_text())
    assert web_selection["pool_manifest"] == "not materialized; frozen selection provenance only"
    assert web_selection["method_details"]["feature_cache"] == (
        "not materialized; frozen selection provenance only"
    )
    assert str(source) not in json.dumps(web_selection)
    assert (destination / "input/생활용품/10.jpg").read_bytes() == b"image-10"
    assert (destination / "input/이_미용/20.jpg").read_bytes() == b"image-20"
    assert not (destination / "pool").exists()


def test_build_psnr_ssim_web_dataset_materializes_fixed_pairs_and_prompts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "eval500"
    manifests = source / "manifests"
    curated_input = tmp_path / "curated/input"
    manifests.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    fields = [
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
    for index in range(2):
        image = curated_input / "가구" / f"TYPE__IMAGE{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"input-{index}".encode())
        selected = source / "input/가구" / image.name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.symlink_to(image)
        rows.append(
            {
                "split": "input",
                "group": "가구",
                "product_type": "TYPE",
                "item_id": f"ITEM{index}",
                "image_id": f"IMAGE{index}",
                "width": "1000",
                "height": "1000",
                "sha256": _sha256(image),
                "source_path": str(image),
                "selected_path": f"input/가구/{image.name}",
            }
        )
    with (manifests / "psnr_ssim_100.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (manifests / "prompts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "prompt", "name_source"]
        )
        writer.writeheader()
        writer.writerow({"item_id": "ITEM1", "prompt": "prompt 1", "name_source": "en"})
        writer.writerow(
            {"item_id": "ITEM0", "prompt": "prompt 0", "name_source": "fallback"}
        )
    destination = tmp_path / "web/psnr_ssim"

    report = build_psnr_ssim_web_dataset(
        source_root=source,
        destination_root=destination,
        expected_count=2,
    )

    assert report.count == 2
    with (destination / "manifests/input.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        copied_rows = list(csv.DictReader(handle))
    assert copied_rows == [
        {**row, "source_path": str((destination / row["selected_path"]).absolute())}
        for row in rows
    ]
    assert (destination / "manifests/input.csv").read_bytes() == (
        destination / "manifests/psnr_ssim_100.csv"
    ).read_bytes()
    with (destination / "manifests/prompts.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        prompts = list(csv.DictReader(handle))
    assert [row["item_id"] for row in prompts] == ["ITEM0", "ITEM1"]
    assert (destination / "input/가구/TYPE__IMAGE0.jpg").read_bytes() == b"input-0"
    protocol = json.loads((destination / "manifests/protocol.json").read_text())
    assert protocol["schema_version"] == 2
    assert protocol["selection"]["input_count"] == 2
    assert protocol["selection"]["psnr_ssim_pair_count"] == 2
    assert protocol["selection"]["group_quotas"] == {"가구": 2}
