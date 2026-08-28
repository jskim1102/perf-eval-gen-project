from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from metrics.runner import measure_run
from metrics.schema import ResultsValidationError, validate_results


INPUT_FIELDS = [
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
GENERATED_FIELDS = ["input_path", "output_path", "sha256", "seed", "strength"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    eval_root = tmp_path / "eval500"
    runs_root = tmp_path / "runs"
    run_root = runs_root / "schema-test"
    input_manifest = eval_root / "manifests" / "input.csv"
    pair_manifest = eval_root / "manifests" / "psnr_ssim_100.csv"
    input_manifest.parent.mkdir(parents=True)
    (eval_root / "input" / "가구").mkdir(parents=True)
    (run_root / "images" / "가구").mkdir(parents=True)

    input_rows: list[dict[str, str]] = []
    generated_rows: list[dict[str, str]] = []
    for index, delta in enumerate((5, 14)):
        filename = f"PRODUCT__image-{index}.jpg"
        input_path = eval_root / "input" / "가구" / filename
        output_path = run_root / "images" / "가구" / filename.replace(".jpg", ".png")
        image = Image.new("RGB", (48, 48), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 7, 40, 42), fill=(60 + index * 50, 90, 130))
        image.save(input_path, quality=100)
        with Image.open(input_path) as source:
            pixels = source.convert("RGB")
        pixels = pixels.point(lambda value: min(255, value + delta))
        pixels.save(output_path, format="PNG")
        input_rows.append(
            {
                "split": "input",
                "group": "가구",
                "product_type": "PRODUCT",
                "item_id": f"item-{index}",
                "image_id": f"image-{index}",
                "width": "48",
                "height": "48",
                "sha256": _sha256(input_path),
                "source_path": str(input_path),
                "selected_path": f"input/가구/{filename}",
            }
        )
        generated_rows.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": _sha256(output_path),
                "seed": "0",
                "strength": "0.25",
            }
        )

    for path in (input_manifest, pair_manifest):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INPUT_FIELDS)
            writer.writeheader()
            writer.writerows(input_rows)
    with (run_root / "generated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
        writer.writeheader()
        writer.writerows(reversed(generated_rows))
    return eval_root, runs_root, input_manifest, pair_manifest


def test_measure_run_writes_and_validates_canonical_results(tmp_path: Path) -> None:
    eval_root, runs_root, input_manifest, pair_manifest = _build_run(tmp_path)

    results = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        created_at="2026-08-05T12:00:00+09:00",
    )

    written = json.loads((runs_root / "schema-test" / "results.json").read_text())
    assert written == results
    assert list(results) == [
        "run_id",
        "created_at",
        "protocol",
        "env",
        "inputs",
        "metrics",
        "pairs",
        "targets",
        "baseline_ref",
    ]
    assert results["protocol"]["n_input"] == 2
    assert results["protocol"]["n_pairs"] == 2
    assert results["inputs"]["input_manifest_sha256"] == _sha256(input_manifest)
    assert results["inputs"]["generated_manifest_sha256"] == _sha256(
        runs_root / "schema-test" / "generated.csv"
    )
    assert [pair["item_id"] for pair in results["pairs"]] == ["item-0", "item-1"]
    assert len(results["metrics"]["psnr"]["per_image"]) == 2
    assert len(results["metrics"]["ssim"]["per_image"]) == 2
    assert "fid" not in results["protocol"]
    assert set(results["metrics"]) == {"psnr", "ssim"}
    assert results["targets"] == {"psnr": 25.0, "ssim": 0.9}
    assert all(results["env"][key] for key in results["env"] if key != "determinism")
    assert results["env"]["determinism"] == {
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "torch_seed": 0,
    }
    validate_results(
        results,
        eval_root=eval_root,
        runs_root=runs_root,
        require_determinism=True,
    )


def test_schema_reads_legacy_env_but_can_require_determinism(tmp_path: Path) -> None:
    eval_root, runs_root, input_manifest, pair_manifest = _build_run(tmp_path)
    results = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    historical = deepcopy(results)
    historical["protocol"]["fid"] = {
        "impl": "clean-fid",
        "mode": "clean",
        "feature": "inception_v3",
        "input": "299x299 RGB",
        "resize": "bicubic+antialias",
        "real_set": "input x2",
        "gen_set": "reconstructed y2",
    }
    historical["metrics"]["fid"] = 0.125
    historical["targets"]["fid"] = 10.0
    validate_results(historical, eval_root=eval_root, runs_root=runs_root)

    legacy = deepcopy(results)
    del legacy["env"]["determinism"]

    validate_results(legacy, eval_root=eval_root, runs_root=runs_root)
    with pytest.raises(ResultsValidationError, match="env.determinism is required"):
        validate_results(
            legacy,
            eval_root=eval_root,
            runs_root=runs_root,
            require_determinism=True,
        )


def test_same_generated_images_produce_identical_metrics(tmp_path: Path) -> None:
    _, runs_root, input_manifest, pair_manifest = _build_run(tmp_path)

    first = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        created_at="2026-08-05T12:00:00+09:00",
    )
    second = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        created_at="2026-08-05T12:01:00+09:00",
    )

    assert first["metrics"] == second["metrics"]
    first_without_time = {key: value for key, value in first.items() if key != "created_at"}
    second_without_time = {
        key: value for key, value in second.items() if key != "created_at"
    }
    assert first_without_time == second_without_time


def test_measure_run_accepts_prompt_generated_csv_and_records_optional_protocol(
    tmp_path: Path,
) -> None:
    eval_root, runs_root, input_manifest, pair_manifest = _build_run(tmp_path)
    generated_manifest = runs_root / "schema-test/generated.csv"
    with generated_manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with generated_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*GENERATED_FIELDS, "prompt"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prompt": "per-image prompt"})
    prompt_protocol = {
        "mode": "per-image",
        "template": "a high quality studio product photograph of {item_name}, on a clean white background",
        "name_source": "ABO item_name language_tag starts with en; fallback to manifest product_type",
        "name_max_words": 15,
        "truncation": "first 15 whitespace-delimited words",
        "whitespace_normalization": "strip and collapse consecutive whitespace",
        "manifest_path": str((eval_root / "manifests/prompts.csv").absolute()),
        "manifest_sha256": "1" * 64,
    }

    results = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt_protocol=prompt_protocol,
    )

    assert results["protocol"]["prompt_protocol"] == prompt_protocol
    validate_results(results, eval_root=eval_root, runs_root=runs_root)

    invalid = deepcopy(results)
    invalid["protocol"]["prompt_protocol"]["manifest_sha256"] = "not-a-hash"
    with pytest.raises(ResultsValidationError, match="prompt protocol manifest SHA-256"):
        validate_results(invalid, eval_root=eval_root, runs_root=runs_root)


def test_schema_rejects_pair_metric_length_and_path_drift(tmp_path: Path) -> None:
    eval_root, runs_root, input_manifest, pair_manifest = _build_run(tmp_path)
    results = measure_run(
        run_id="schema-test",
        runs_root=runs_root,
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        strength=0.25,
        seed=0,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )

    wrong_length = deepcopy(results)
    wrong_length["metrics"]["ssim"]["per_image"].pop()
    with pytest.raises(ResultsValidationError, match="pair metric lengths"):
        validate_results(wrong_length, eval_root=eval_root, runs_root=runs_root)

    wrong_path = deepcopy(results)
    wrong_path["pairs"][0]["output_path"] = "/tmp/outside.png"
    with pytest.raises(ResultsValidationError, match="outside RUNS_ROOT"):
        validate_results(wrong_path, eval_root=eval_root, runs_root=runs_root)
