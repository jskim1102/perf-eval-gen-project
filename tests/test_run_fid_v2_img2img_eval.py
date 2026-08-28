from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

import scripts.run_fid_v2_img2img_eval as fidv2_img2img


class _PipelineResult:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _FakeFluxImg2ImgPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> _PipelineResult:
        self.calls.append(kwargs)
        source = kwargs["image"]
        assert isinstance(source, Image.Image)
        assert source.mode == "RGB"
        assert source.size == (1024, 1024)
        assert kwargs["strength"] == fidv2_img2img.STRENGTH
        assert kwargs["num_inference_steps"] == fidv2_img2img.NUM_INFERENCE_STEPS
        assert kwargs["guidance_scale"] == fidv2_img2img.GUIDANCE_SCALE
        assert kwargs["height"] == 1024
        assert kwargs["width"] == 1024
        generator = kwargs["generator"]
        assert generator.device.type == "cpu"
        assert generator.initial_seed() == 0
        return _PipelineResult(Image.new("RGB", (1024, 1024), (42, 80, 120)))


def _build_dataset(tmp_path: Path, *, count: int = 3) -> tuple[Path, list[dict[str, str]]]:
    dataset_root = tmp_path / "fid_v2"
    rows: list[dict[str, str]] = []
    for index in range(count):
        item_id = str(15000 + index)
        group = "생활용품" if index < 2 else "홈클린"
        thumbnail = dataset_root / "input" / group / f"{item_id}.png"
        thumbnail.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 96), (220 - index, 210, 200)).save(thumbnail)
        rows.append(
            {
                "item_no": item_id,
                "대분류": group,
                "중분류": "위생용품",
                "소분류": "탈취제",
                "상품명": f"테스트 상품 {index}",
                "source_product": f"input/{group}/{item_id}.jpg",
                "thumbnail": f"input/{group}/{item_id}.png",
                "prompt": f"manifest prompt {index}",
                "sha256": f"fixture-{index}",
            }
        )
    with (dataset_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (dataset_root / "selection.json").write_text(
        json.dumps(
            {
                "count": count,
                "source_dataset": {
                    "name": "AI Hub 상품 이미지",
                    "url": "https://aihub.or.kr/aidata/34145",
                    "attribution": "NIA AI 학습용 데이터 구축사업 결과",
                },
                "not_ai_generated": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return dataset_root, rows


def test_img2img_measurement_uses_manifest_thumbnail_prompt_and_selected_order(
    tmp_path: Path,
) -> None:
    dataset_root, rows = _build_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    pipeline = _FakeFluxImg2ImgPipeline()
    observed: dict[str, int] = {}

    def fake_fid(real_directory: Path, generated_directory: Path):
        observed["real"] = len(list(real_directory.glob("*.png")))
        observed["generated"] = len(list(generated_directory.glob("*.png")))
        assert real_directory.parent == runs_root / "fidv2try-fixture"
        assert generated_directory.parent == runs_root / "fidv2try-fixture"
        return 8.25, {
            "impl": "clean-fid",
            "mode": "clean",
            "feature": "inception_v3",
            "input": "299x299 RGB",
            "resize": "bicubic+antialias",
        }

    result = fidv2_img2img.run_fid_v2_img2img_evaluation(
        run_id="fidv2try-fixture",
        strength=fidv2_img2img.STRENGTH,
        limit=None,
        item_ids=[rows[2]["item_no"], rows[0]["item_no"]],
        runs_root=runs_root,
        dataset_root=dataset_root,
        pipeline_loader=lambda _model: pipeline,
        fid_runner=fake_fid,
        require_cuda=False,
    )

    assert [call["prompt"] for call in pipeline.calls] == [
        rows[2]["prompt"],
        rows[0]["prompt"],
    ]
    assert observed == {"real": 2, "generated": 2}
    assert result["run_id"] == "fidv2try-fixture"
    assert result["schema_version"] == 1
    assert result["protocol"]["model"] == "black-forest-labs/FLUX.1-dev"
    assert result["protocol"]["generation_mode"] == "image-to-image"
    assert result["protocol"]["strength"] == 0.15
    assert result["protocol"]["num_inference_steps"] == 30
    assert result["protocol"]["guidance_scale"] == 3.5
    assert result["protocol"]["seed"] == 0
    assert result["measurement"]["fid"] == 8.25
    assert result["measurement"]["verdict"] == "PASS"
    assert result["measurement"]["smoke"] is True
    assert result["measurement"]["selection_mode"] == "selected"
    assert result["measurement"]["item_ids"] == ["15002", "15000"]
    assert result["generation_validation"]["pipeline_image_argument"] is True

    with (runs_root / "fidv2try-fixture/generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        generated_rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in generated_rows] == ["15002", "15000"]
    assert [Path(row["input_path"]) for row in generated_rows] == [
        dataset_root / "input/홈클린/15002.png",
        dataset_root / "input/생활용품/15000.png",
    ]
    assert all(Path(row["output_path"]).is_file() for row in generated_rows)


@pytest.mark.parametrize(
    ("case_name", "limit", "item_indexes", "expected_smoke"),
    [
        ("full-default", None, None, False),
        ("partial-limit", 2, None, True),
        ("partial-selected", None, [0, 1], True),
        ("full-selected", None, [0, 1, 2], False),
    ],
)
def test_img2img_smoke_reflects_measured_set_size(
    tmp_path: Path,
    case_name: str,
    limit: int | None,
    item_indexes: list[int] | None,
    expected_smoke: bool,
) -> None:
    dataset_root, rows = _build_dataset(tmp_path)
    item_ids = (
        [rows[index]["item_no"] for index in item_indexes]
        if item_indexes is not None
        else None
    )

    result = fidv2_img2img.run_fid_v2_img2img_evaluation(
        run_id=f"fidv2try-smoke-{case_name}",
        strength=fidv2_img2img.STRENGTH,
        limit=limit,
        item_ids=item_ids,
        runs_root=tmp_path / "runs",
        dataset_root=dataset_root,
        pipeline_loader=lambda _model: _FakeFluxImg2ImgPipeline(),
        fid_runner=lambda _real, _generated: (8.0, {"impl": "clean-fid"}),
        require_cuda=False,
    )

    assert result["measurement"]["smoke"] is expected_smoke


def test_img2img_resume_preserves_rows_from_previous_selection(tmp_path: Path) -> None:
    dataset_root, rows = _build_dataset(tmp_path, count=4)
    runs_root = tmp_path / "runs"

    def fake_fid(_real: Path, _generated: Path):
        return 8.0, {"impl": "clean-fid"}

    common = {
        "run_id": "fidv2try-preserve-ledger",
        "strength": fidv2_img2img.STRENGTH,
        "limit": None,
        "runs_root": runs_root,
        "dataset_root": dataset_root,
        "fid_runner": fake_fid,
        "require_cuda": False,
    }
    first_ids = [row["item_no"] for row in rows[:2]]
    second_ids = [row["item_no"] for row in rows[2:]]

    fidv2_img2img.run_fid_v2_img2img_evaluation(
        **common,
        item_ids=first_ids,
        pipeline_loader=lambda _model: _FakeFluxImg2ImgPipeline(),
    )
    second = fidv2_img2img.run_fid_v2_img2img_evaluation(
        **common,
        item_ids=second_ids,
        pipeline_loader=lambda _model: _FakeFluxImg2ImgPipeline(),
    )

    with (runs_root / "fidv2try-preserve-ledger/generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        generated_rows = list(csv.DictReader(handle))
    assert [row["item_id"] for row in generated_rows] == first_ids + second_ids
    assert all(Path(row["output_path"]).is_file() for row in generated_rows)
    assert second["measurement"]["item_ids"] == second_ids
    assert second["generation_validation"]["generated_now"] == 2


def test_img2img_resume_skips_existing_png_even_after_manifest_loss(tmp_path: Path) -> None:
    dataset_root, rows = _build_dataset(tmp_path, count=2)
    runs_root = tmp_path / "runs"
    first_pipeline = _FakeFluxImg2ImgPipeline()

    def fake_fid(_real: Path, _generated: Path):
        return 12.0, {"impl": "clean-fid"}

    kwargs = {
        "run_id": "fidv2try-resume",
        "strength": fidv2_img2img.STRENGTH,
        "limit": None,
        "item_ids": [row["item_no"] for row in rows],
        "runs_root": runs_root,
        "dataset_root": dataset_root,
        "fid_runner": fake_fid,
        "require_cuda": False,
    }
    first = fidv2_img2img.run_fid_v2_img2img_evaluation(
        **kwargs,
        pipeline_loader=lambda _model: first_pipeline,
    )
    assert first["generation_validation"]["generated_now"] == 2
    (runs_root / "fidv2try-resume/generated.csv").unlink()
    missing_output = runs_root / "fidv2try-resume/images/생활용품/15001.png"
    missing_output.unlink()

    resumed_pipeline = _FakeFluxImg2ImgPipeline()
    resumed = fidv2_img2img.run_fid_v2_img2img_evaluation(
        **kwargs,
        pipeline_loader=lambda _model: resumed_pipeline,
    )

    assert len(resumed_pipeline.calls) == 1
    assert resumed["generation_validation"]["generated_now"] == 1
    assert resumed["generation_validation"]["resumed"] == 1
    assert resumed["generation_validation"]["recovered_without_manifest"] == 1
    manifest_path = runs_root / "fidv2try-resume/generated.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        recovered_rows = list(csv.DictReader(handle))
    assert recovered_rows[0]["elapsed_seconds"] == ""
    measured_elapsed = float(recovered_rows[1]["elapsed_seconds"])
    assert measured_elapsed > 0
    assert resumed["measurement"]["seconds_per_image"] == pytest.approx(
        measured_elapsed
    )

    final_pipeline = _FakeFluxImg2ImgPipeline()
    final = fidv2_img2img.run_fid_v2_img2img_evaluation(
        **kwargs,
        pipeline_loader=lambda _model: final_pipeline,
    )
    assert final_pipeline.calls == []
    assert final["measurement"]["seconds_per_image"] == pytest.approx(
        measured_elapsed
    )


def test_img2img_rejects_protocol_changes_and_conflicting_selection(tmp_path: Path) -> None:
    dataset_root, _ = _build_dataset(tmp_path)
    common = {
        "run_id": "fidv2try-invalid",
        "runs_root": tmp_path / "runs",
        "dataset_root": dataset_root,
        "pipeline_loader": lambda _model: _FakeFluxImg2ImgPipeline(),
        "fid_runner": lambda _real, _generated: (1.0, {}),
        "require_cuda": False,
    }

    with pytest.raises(ValueError, match="fixed protocol strength"):
        fidv2_img2img.run_fid_v2_img2img_evaluation(
            **common, strength=0.2, limit=2, item_ids=None
        )
    with pytest.raises(ValueError, match="cannot be used together"):
        fidv2_img2img.run_fid_v2_img2img_evaluation(
            **common, strength=0.15, limit=2, item_ids=["15000", "15001"]
        )
    with pytest.raises(ValueError, match="not in FID_v2 manifest"):
        fidv2_img2img.run_fid_v2_img2img_evaluation(
            **common, strength=0.15, limit=None, item_ids=["15000", "missing"]
        )


def test_flux_loader_enables_sequential_cpu_offload(monkeypatch) -> None:
    class FakePipeline:
        offload_enabled = False

        def enable_sequential_cpu_offload(self) -> None:
            self.offload_enabled = True

    pipeline = FakePipeline()
    monkeypatch.setattr(
        fidv2_img2img.FluxImg2ImgPipeline,
        "from_pretrained",
        lambda model_id, torch_dtype: pipeline,
    )

    loaded = fidv2_img2img.load_flux_img2img_pipeline(fidv2_img2img.MODEL_ID)

    assert loaded is pipeline
    assert pipeline.offload_enabled is True
