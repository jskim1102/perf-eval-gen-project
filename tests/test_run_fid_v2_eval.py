from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from scripts.run_fid_v2_eval import (
    FLUX_GUIDANCE_SCALE,
    FLUX_MODEL_ID,
    FLUX_NUM_INFERENCE_STEPS,
    RUN_ID,
    build_thumbnail_prompt,
    measure,
)


class _PipelineResult:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _FakeFluxPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _PipelineResult:
        self.calls.append(kwargs)
        assert "image" not in kwargs
        assert kwargs["height"] == 1024
        assert kwargs["width"] == 1024
        assert kwargs["num_inference_steps"] == FLUX_NUM_INFERENCE_STEPS
        assert kwargs["guidance_scale"] == FLUX_GUIDANCE_SCALE
        generator = kwargs["generator"]
        assert generator.device.type == "cpu"
        assert generator.initial_seed() == 0
        return _PipelineResult(Image.new("RGB", (1024, 1024), (42, 80, 120)))


def _build_dataset(tmp_path: Path, *, count: int = 2) -> tuple[Path, Path]:
    dataset_root = tmp_path / "fid_v2"
    product_root = tmp_path / "fid"
    rows: list[dict[str, str]] = []
    for index in range(count):
        item_no = str(15000 + index)
        category = "생활용품"
        product_path = product_root / "input" / category / f"{item_no}.jpg"
        thumbnail_path = dataset_root / "input" / category / f"{item_no}.png"
        product_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), (205, 48 + index, 35)).save(product_path)
        Image.new("RGB", (128, 128), (235, 220, 210)).save(thumbnail_path)
        rows.append(
            {
                "item_no": item_no,
                "대분류": category,
                "중분류": "위생용품",
                "소분류": "탈취제",
                "상품명": f"테스트 상품 {index}",
                "source_product": f"input/{category}/{item_no}.jpg",
                "thumbnail": f"input/{category}/{item_no}.png",
                "prompt": "OLD WHITE BACKGROUND PROMPT MUST NOT BE USED",
                "sha256": f"fixture-{index}",
            }
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dataset_root / "selection.json").write_text(
        json.dumps({"count": count}, ensure_ascii=False), encoding="utf-8"
    )
    return dataset_root, product_root


def test_thumbnail_prompt_uses_metadata_and_product_colour_not_legacy_prompt(
    tmp_path: Path,
) -> None:
    dataset_root, product_root = _build_dataset(tmp_path, count=1)
    with (dataset_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))

    first = build_thumbnail_prompt(row, product_root=product_root)
    second = build_thumbnail_prompt(row, product_root=product_root)

    assert first == second
    assert "테스트 상품 0" in first
    assert '"위생용품" category badge' in first
    assert "red \"NEW\" corner ribbon" in first
    assert "red-tone vertical gradient background" in first
    assert "white rounded card with a soft shadow" in first
    assert "rounded frame" in first
    assert "OLD WHITE BACKGROUND PROMPT" not in first


def test_measure_is_prompt_only_records_flux_protocol_and_resumes(
    tmp_path: Path,
) -> None:
    dataset_root, product_root = _build_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    pipeline = _FakeFluxPipeline()
    observed: dict[str, int] = {}

    def fake_fid(real_directory: Path, generated_directory: Path):
        observed["real"] = len(list(real_directory.glob("*.png")))
        observed["generated"] = len(list(generated_directory.glob("*.png")))
        return 18.25, {
            "impl": "clean-fid",
            "mode": "clean",
            "feature": "inception_v3",
            "input": "299x299 RGB",
            "resize": "bicubic+antialias",
        }

    result = measure(
        dataset_root=dataset_root,
        product_root=product_root,
        runs_root=runs_root,
        limit=2,
        pipeline_loader=lambda _model: pipeline,
        fid_runner=fake_fid,
        require_cuda=False,
    )

    assert len(pipeline.calls) == 2
    assert all("OLD WHITE BACKGROUND PROMPT" not in call["prompt"] for call in pipeline.calls)
    assert observed == {"real": 2, "generated": 2}
    assert result["run_id"] == RUN_ID == "fid_v2_flux"
    assert result["protocol"]["model"] == FLUX_MODEL_ID
    assert result["protocol"]["generation_mode"] == "text-to-image"
    assert result["protocol"]["image_conditioning"] is False
    assert "non-commercial" in result["protocol"]["license_note"].lower()
    assert result["measurement"]["fid"] == 18.25
    assert result["measurement"]["count"] == 2
    assert result["generation_validation"]["generated_now"] == 2
    assert result["generation_validation"]["resumed"] == 0

    with (runs_root / RUN_ID / "generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "reference_path" in rows[0]
    assert "input_path" not in rows[0]
    assert rows[0]["model"] == FLUX_MODEL_ID

    resumed_pipeline = _FakeFluxPipeline()
    resumed = measure(
        dataset_root=dataset_root,
        product_root=product_root,
        runs_root=runs_root,
        limit=2,
        pipeline_loader=lambda _model: resumed_pipeline,
        fid_runner=fake_fid,
        require_cuda=False,
    )
    assert resumed_pipeline.calls == []
    assert resumed["generation_validation"]["generated_now"] == 0
    assert resumed["generation_validation"]["resumed"] == 2
