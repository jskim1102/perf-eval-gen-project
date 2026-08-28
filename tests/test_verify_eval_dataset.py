from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.select_eval_dataset import (
    load_records,
    select_pair_subset,
    select_records,
    write_selection,
)
from scripts.verify_eval_dataset import (
    DatasetContract,
    VerificationError,
    verify_dataset,
)


BASE_FIELDS = [
    "group",
    "product_type",
    "item_id",
    "image_id",
    "path",
    "width",
    "height",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_selection_view(tmp_path: Path) -> tuple[Path, Path, DatasetContract]:
    source_root = tmp_path / "curated"
    output_root = tmp_path / "generated-eval500"
    source_root.mkdir()

    rows: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    for group_index, group in enumerate(("가구", "조명")):
        group_root = source_root / "input" / group
        group_root.mkdir(parents=True)
        for item_index in range(2):
            image_id = f"image-{group_index}-{item_index}"
            item_id = f"item-{group_index}-{item_index}"
            filename = f"PRODUCT__{image_id}.jpg"
            image_path = group_root / filename
            Image.new(
                "RGB",
                (1000, 1000),
                (40 + group_index * 80, 50 + item_index * 70, 60),
            ).save(image_path)
            rows.append(
                {
                    "group": group,
                    "product_type": "PRODUCT",
                    "item_id": item_id,
                    "image_id": image_id,
                    "path": f"input/{group}/{filename}",
                    "width": "1000",
                    "height": "1000",
                }
            )
            feature_rows.append(
                {
                    "path": f"input/{group}/{filename}",
                    "group": group,
                    "ring_med": "255.0",
                    "ring_std": "0.0",
                    "ink": "0.0",
                    "sat": "0.0",
                }
            )

    write_csv(source_root / "pool.csv", BASE_FIELDS, rows)
    # One identical duplicate proves that provenance accepts source multiplicity
    # without confusing it with a conflicting row.
    write_csv(source_root / "index-main.csv", BASE_FIELDS, [*rows, rows[0]])
    write_csv(
        source_root / "features.csv",
        ["path", "group", "ring_med", "ring_std", "ink", "sat"],
        feature_rows,
    )

    seed = "verification-test-seed"
    records = load_records(source_root)
    selection = select_records(records, total_per_split=4, seed=seed)
    pairs = select_pair_subset(selection.input, total=2, seed=f"{seed}:psnr-ssim")
    write_selection(
        selection,
        pairs,
        source_root=source_root,
        output_root=output_root,
        seed=seed,
    )
    contract = DatasetContract(
        input_count=4,
        pair_count=2,
        group_quotas=dict(selection.quotas),
        seed=seed,
    )
    return source_root, output_root, contract


def test_strict_verification_accepts_independently_traceable_equal_subsets(
    tmp_path: Path,
) -> None:
    source_root, output_root, contract = build_selection_view(tmp_path)

    report = verify_dataset(
        source_root=source_root,
        eval_root=output_root,
        contract=contract,
        strict=True,
    )

    assert report["input_symlinks"] == 4
    assert report["pair_count"] == 2
    assert report["metadata"]["pool_index_relation"] == "identical"
    assert report["metadata"]["pool.csv"]["missing_from_source"] == 0
    assert report["metadata"]["index-main.csv"]["missing_from_source"] == 0
    assert report["metadata"]["index-main.csv"]["source_multiplicity"] == {
        "1": 3,
        "2": 1,
    }
    assert report["deterministic_files"] == 6
    assert report["protected_trees_unchanged"] is True


def test_protocol_fid_metadata_is_optional_for_eval500_verification(
    tmp_path: Path,
) -> None:
    source_root, output_root, contract = build_selection_view(tmp_path)
    protocol_path = output_root / "manifests" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    del protocol["fid"]
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    report = verify_dataset(
        source_root=source_root,
        eval_root=output_root,
        contract=contract,
        strict=False,
    )

    assert report["status"] == "PASS"


def test_metadata_subset_row_must_come_from_its_own_source(tmp_path: Path) -> None:
    source_root, output_root, contract = build_selection_view(tmp_path)
    index_subset = output_root / "index-main.csv"
    with index_subset.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["path"] = "not/from/index-main.jpg"
    write_csv(index_subset, fieldnames, rows)

    with pytest.raises(VerificationError, match="not traceable to its own source"):
        verify_dataset(
            source_root=source_root,
            eval_root=output_root,
            contract=contract,
            strict=False,
        )


def test_strict_verification_detects_changed_image_bytes(tmp_path: Path) -> None:
    source_root, output_root, contract = build_selection_view(tmp_path)
    with (source_root / "input" / "가구" / "PRODUCT__image-0-0.jpg").open(
        "ab"
    ) as handle:
        handle.write(b"changed-after-selection")

    with pytest.raises(VerificationError, match="sha256 mismatch"):
        verify_dataset(
            source_root=source_root,
            eval_root=output_root,
            contract=contract,
            strict=True,
        )


def test_strict_verification_checks_real_pixel_dimensions(tmp_path: Path) -> None:
    source_root, output_root, contract = build_selection_view(tmp_path)
    Image.new("RGB", (900, 1000), (1, 2, 3)).save(
        source_root / "input" / "가구" / "PRODUCT__image-0-0.jpg"
    )

    with pytest.raises(VerificationError, match="pixel dimensions"):
        verify_dataset(
            source_root=source_root,
            eval_root=output_root,
            contract=contract,
            strict=True,
        )
