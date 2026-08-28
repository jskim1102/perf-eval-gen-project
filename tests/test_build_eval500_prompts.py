from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_eval500_prompts import (
    PROMPT_FIELDS,
    PROMPT_TEMPLATE,
    build_prompt_manifest,
)


INPUT_FIELDS = ["item_id", "product_type", "selected_path"]


def _write_input_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_metadata(path: Path, listings: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for listing in listings:
            handle.write(json.dumps(listing, ensure_ascii=False) + "\n")
    return path


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_builder_prefers_english_normalizes_whitespace_and_truncates_15_words(
    tmp_path: Path,
) -> None:
    input_manifest = _write_input_manifest(
        tmp_path / "eval500/manifests/input.csv",
        [
            {
                "item_id": "item-en",
                "product_type": "HOME_BED_AND_BATH",
                "selected_path": "input/group/image.jpg",
            }
        ],
    )
    words = [f"word-{index}" for index in range(1, 18)]
    _write_metadata(
        tmp_path / "metadata/listings_a.json.gz",
        [
            {
                "item_id": "item-en",
                "item_name": [
                    {"language_tag": "ko_KR", "value": "한국어 이름"},
                    {
                        "language_tag": "en_US",
                        "value": "  " + "  \n ".join(words) + "  ",
                    },
                ],
            }
        ],
    )

    output = tmp_path / "eval500/manifests/prompts.csv"
    report = build_prompt_manifest(
        metadata_root=tmp_path / "metadata",
        input_manifest=input_manifest,
        output_path=output,
    )

    fields, rows = _read_rows(output)
    expected_name = " ".join(words[:15])
    assert fields == PROMPT_FIELDS
    assert rows == [
        {
            "item_id": "item-en",
            "prompt": PROMPT_TEMPLATE.format(item_name=expected_name),
            "name_source": "en",
        }
    ]
    assert report.count == 1
    assert report.english_count == 1
    assert report.fallback_count == 0
    assert report.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()


def test_builder_uses_normalized_product_type_fallback_and_manifest_order(
    tmp_path: Path,
) -> None:
    input_manifest = _write_input_manifest(
        tmp_path / "eval500/manifests/input.csv",
        [
            {
                "item_id": "item-fallback",
                "product_type": "HARDWARE_HANDLE",
                "selected_path": "input/group/a.jpg",
            },
            {
                "item_id": "item-en",
                "product_type": "CHAIR",
                "selected_path": "input/group/b.jpg",
            },
        ],
    )
    _write_metadata(
        tmp_path / "metadata/listings_f.json.gz",
        [
            {
                "item_id": "item-en",
                "item_name": [{"language_tag": "en", "value": "Desk Chair"}],
            },
            {
                "item_id": "item-fallback",
                "item_name": [{"language_tag": "de_DE", "value": "Griff"}],
            },
        ],
    )

    output = tmp_path / "eval500/manifests/prompts.csv"
    build_prompt_manifest(
        metadata_root=tmp_path / "metadata",
        input_manifest=input_manifest,
        output_path=output,
    )
    first = output.read_bytes()
    report = build_prompt_manifest(
        metadata_root=tmp_path / "metadata",
        input_manifest=input_manifest,
        output_path=output,
    )

    _, rows = _read_rows(output)
    assert [row["item_id"] for row in rows] == ["item-fallback", "item-en"]
    assert rows[0] == {
        "item_id": "item-fallback",
        "prompt": PROMPT_TEMPLATE.format(item_name="hardware handle"),
        "name_source": "fallback",
    }
    assert rows[1]["name_source"] == "en"
    assert output.read_bytes() == first
    assert report.sha256 == hashlib.sha256(first).hexdigest()


def test_builder_rejects_missing_target_metadata_without_output(tmp_path: Path) -> None:
    input_manifest = _write_input_manifest(
        tmp_path / "eval500/manifests/input.csv",
        [
            {
                "item_id": "target",
                "product_type": "CHAIR",
                "selected_path": "input/group/image.jpg",
            }
        ],
    )
    _write_metadata(tmp_path / "metadata/listings_0.json.gz", [])
    output = tmp_path / "eval500/manifests/prompts.csv"

    with pytest.raises(ValueError, match="missing"):
        build_prompt_manifest(
            metadata_root=tmp_path / "metadata",
            input_manifest=input_manifest,
            output_path=output,
        )

    assert not output.exists()


def test_builder_uses_first_english_name_across_duplicate_marketplace_listings(
    tmp_path: Path,
) -> None:
    input_manifest = _write_input_manifest(
        tmp_path / "eval500/manifests/input.csv",
        [
            {
                "item_id": "target",
                "product_type": "CHAIR",
                "selected_path": "input/group/image.jpg",
            }
        ],
    )
    _write_metadata(
        tmp_path / "metadata/listings_0.json.gz",
        [
            {
                "item_id": "target",
                "item_name": [{"language_tag": "de_DE", "value": "Stuhl"}],
            }
        ],
    )
    _write_metadata(
        tmp_path / "metadata/listings_1.json.gz",
        [
            {
                "item_id": "target",
                "item_name": [{"language_tag": "en_GB", "value": "First Chair"}],
            },
            {
                "item_id": "target",
                "item_name": [{"language_tag": "en_US", "value": "Second Chair"}],
            },
        ],
    )
    output = tmp_path / "eval500/manifests/prompts.csv"

    build_prompt_manifest(
        metadata_root=tmp_path / "metadata",
        input_manifest=input_manifest,
        output_path=output,
    )

    _, rows = _read_rows(output)
    assert rows == [
        {
            "item_id": "target",
            "prompt": PROMPT_TEMPLATE.format(item_name="First Chair"),
            "name_source": "en",
        }
    ]
