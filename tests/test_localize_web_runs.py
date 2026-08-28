from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.localize_web_runs import localize_fid_run, localize_psnr_ssim_run


GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength", "prompt"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_localizers_rebase_every_web_input_and_output_below_project_dataset(
    tmp_path: Path,
) -> None:
    external_runs = tmp_path / "external-runs"
    local_runs = tmp_path / "project/dataset/runs"

    fid_dataset = tmp_path / "project/dataset/fid"
    fid_input = fid_dataset / "input/생활용품/10.jpg"
    fid_input.parent.mkdir(parents=True)
    fid_input.write_bytes(b"fid-input")
    _write_csv(
        fid_dataset / "manifests/input.csv",
        ["item_id", "group", "width", "height", "sha256", "source_path", "selected_path"],
        [{
            "item_id": "10",
            "group": "생활용품",
            "width": "1000",
            "height": "1000",
            "sha256": _sha256(fid_input),
            "source_path": "archive/10.jpg",
            "selected_path": "input/생활용품/10.jpg",
        }],
    )
    (fid_dataset / "manifest.csv").write_text("fixture\n")
    fid_source_output = external_runs / "fid500-v2/images/생활용품/10.png"
    fid_source_output.parent.mkdir(parents=True)
    fid_source_output.write_bytes(b"fid-output")
    _write_csv(
        external_runs / "fid500-v2/generated.csv",
        GENERATED_FIELDS,
        [{
            "input_path": "/external/fid/input/생활용품/10.jpg",
            "output_path": str(fid_source_output),
            "sha256": _sha256(fid_source_output),
            "seed": "0",
            "strength": "0.15",
            "prompt": "fid prompt",
        }],
    )
    (external_runs / "fid500-v2/fid500.json").write_text(
        json.dumps({
            "run_id": "fid500-v2",
            "dataset": {
                "root": "/external/fid",
                "input_directory": "/external/fid/input",
                "selection_manifest": "/external/fid/manifest.csv",
                "generation_manifest": "/external/fid/manifests/input.csv",
                "generation_manifest_sha256": "old",
                "generated_manifest_sha256": "old",
            },
            "measurement": {"count": 1},
        }),
        encoding="utf-8",
    )

    localize_fid_run(
        source_runs_root=external_runs,
        destination_runs_root=local_runs,
        dataset_root=fid_dataset,
        expected_count=1,
    )

    with (local_runs / "fid500-v2/generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        fid_row = next(csv.DictReader(handle))
    assert fid_row["input_path"] == str(fid_input.absolute())
    assert fid_row["output_path"].startswith(str(local_runs.absolute()))
    localized_result = json.loads(
        (local_runs / "fid500-v2/fid500.json").read_text()
    )
    assert localized_result["dataset"]["root"] == str(fid_dataset.absolute())
    assert "/external/" not in json.dumps(localized_result)

    pair_dataset = tmp_path / "project/dataset/psnr_ssim"
    pair_input = pair_dataset / "input/가구/CHAIR__IMAGE.jpg"
    pair_input.parent.mkdir(parents=True)
    pair_input.write_bytes(b"pair-input")
    pair_fields = [
        "split", "group", "product_type", "item_id", "image_id", "width",
        "height", "sha256", "source_path", "selected_path",
    ]
    pair_row = {
        "split": "input", "group": "가구", "product_type": "CHAIR",
        "item_id": "ITEM", "image_id": "IMAGE", "width": "1000",
        "height": "1000", "sha256": _sha256(pair_input),
        "source_path": str(pair_input),
        "selected_path": "input/가구/CHAIR__IMAGE.jpg",
    }
    _write_csv(pair_dataset / "manifests/input.csv", pair_fields, [pair_row])
    _write_csv(
        pair_dataset / "manifests/prompts.csv",
        ["item_id", "prompt", "name_source"],
        [{"item_id": "ITEM", "prompt": "pair prompt", "name_source": "en"}],
    )
    pair_source_output = external_runs / "main-v2/images/가구/CHAIR__IMAGE.png"
    pair_source_output.parent.mkdir(parents=True)
    pair_source_output.write_bytes(b"pair-output")
    _write_csv(
        external_runs / "main-v2/generated.csv",
        GENERATED_FIELDS,
        [{
            "input_path": "/external/eval/input/가구/CHAIR__IMAGE.jpg",
            "output_path": str(pair_source_output),
            "sha256": _sha256(pair_source_output),
            "seed": "0",
            "strength": "0.15",
            "prompt": "pair prompt",
        }],
    )

    localize_psnr_ssim_run(
        source_runs_root=external_runs,
        destination_runs_root=local_runs,
        dataset_root=pair_dataset,
        expected_count=1,
    )

    with (local_runs / "main-v2-100/generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        pair_generated = next(csv.DictReader(handle))
    assert pair_generated["input_path"] == str(pair_input.absolute())
    assert pair_generated["output_path"].startswith(str(local_runs.absolute()))
    assert "/external/" not in json.dumps(pair_generated)
