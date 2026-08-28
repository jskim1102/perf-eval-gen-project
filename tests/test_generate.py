from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from scripts.generate import GenerationConfig, load_manifest, run_generation
from scripts.verify_generated import (
    GeneratedVerificationError,
    verify_generation_source_has_no_copy_or_link_path,
    verify_run,
)


MANIFEST_FIELDS = [
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


def _write_input(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (1100, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((220, 180, 880, 920), fill=color)
    draw.ellipse((350, 330, 750, 730), fill=(40, 50, 60))
    image.save(path, format="JPEG", quality=95)


def _build_manifest(tmp_path: Path, *, count: int = 2) -> Path:
    eval_root = tmp_path / "eval500"
    input_root = eval_root / "input" / "가구"
    manifest_path = eval_root / "manifests" / "input.csv"
    input_root.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)

    rows = []
    for index in range(count):
        filename = f"PRODUCT__image-{index}.jpg"
        image_path = input_root / filename
        _write_input(image_path, color=(70 + index * 40, 90, 150))
        rows.append(
            {
                "split": "input",
                "group": "가구",
                "product_type": "PRODUCT",
                "item_id": f"item-{index}",
                "image_id": f"image-{index}",
                "width": "1100",
                "height": "1100",
                "sha256": "unused-in-this-test",
                "source_path": str(image_path),
                "selected_path": f"input/가구/{filename}",
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


@dataclass
class _PipelineResult:
    images: list[Image.Image]


class _FakeVae:
    def enable_tiling(self) -> None:
        return None


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.calls: list[dict[str, object]] = []

    def set_progress_bar_config(self, *, disable: bool) -> None:
        assert disable is True

    def to(self, device: str) -> "_FakePipeline":
        assert device == "cpu"
        return self

    def __call__(self, **kwargs: object) -> _PipelineResult:
        self.calls.append(kwargs)
        source = np.asarray(kwargs["image"], dtype=np.uint16)
        seed = int(kwargs["generator"].initial_seed())
        strength = float(kwargs["strength"])
        # Deterministic, visibly model-like stand-in: all output pixels are newly
        # computed from the input, seed, and strength.
        delta = 12 + seed % 5 + round(strength * 20)
        generated = np.clip(source + delta, 0, 255).astype(np.uint8)
        return _PipelineResult(images=[Image.fromarray(generated, mode="RGB")])


def _fake_pipeline_loader(model_id: str) -> _FakePipeline:
    assert model_id
    return _FakePipeline()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_generation_writes_limited_lossless_manifest_and_images(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    runs_root = tmp_path / "runs"
    config = GenerationConfig(strength=0.25, seed=7, limit=1, run_id="unit-smoke")

    report = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=config,
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )

    fields, rows = _read_csv(runs_root / "unit-smoke" / "generated.csv")
    assert fields == GENERATED_FIELDS
    assert len(rows) == 1
    row = rows[0]
    assert Path(row["input_path"]).is_absolute()
    assert Path(row["output_path"]).is_absolute()
    assert row["seed"] == "7"
    assert row["strength"] == "0.25"
    assert row["sha256"] == report.hashes[0]
    with Image.open(row["output_path"]) as generated:
        assert generated.format == "PNG"
        assert generated.size == (1024, 1024)
        assert generated.mode == "RGB"


def test_same_seed_and_strength_produce_identical_hashes(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path)
    runs_root = tmp_path / "runs"

    first = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=GenerationConfig(strength=0.25, seed=0, limit=2, run_id="first"),
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )
    second = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=GenerationConfig(strength=0.25, seed=0, limit=2, run_id="second"),
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )

    assert second.hashes == first.hashes


def test_same_run_resumes_verified_outputs_without_loading_pipeline(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path, count=2)
    runs_root = tmp_path / "runs"
    config = GenerationConfig(strength=0.25, seed=0, run_id="resume")
    first = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=config,
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )

    def fail_loader(_model_id: str) -> _FakePipeline:
        raise AssertionError("a fully completed run must not reload the pipeline")

    second = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=config,
        pipeline_loader=fail_loader,
        require_cuda=False,
    )

    assert second.hashes == first.hashes
    assert second.count == first.count == 2


def test_per_image_prompts_are_recorded_and_resume_detects_changes(
    tmp_path: Path,
) -> None:
    manifest_path = _build_manifest(tmp_path, count=2)
    runs_root = tmp_path / "runs"
    pipeline = _FakePipeline()
    config = GenerationConfig(strength=0.25, seed=0, run_id="per-image-prompt")

    run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=config,
        pipeline_loader=lambda _model: pipeline,
        require_cuda=False,
        prompt_resolver=lambda record: f"photograph of {record.input_path.stem}",
    )

    fields, rows = _read_csv(runs_root / "per-image-prompt" / "generated.csv")
    assert fields == [*GENERATED_FIELDS, "prompt"]
    assert [row["prompt"] for row in rows] == [
        "photograph of PRODUCT__image-0",
        "photograph of PRODUCT__image-1",
    ]
    assert [call["prompt"] for call in pipeline.calls] == [row["prompt"] for row in rows]

    with pytest.raises(ValueError, match="prompt differs"):
        run_generation(
            manifest_path=manifest_path,
            runs_root=runs_root,
            config=config,
            pipeline_loader=_fake_pipeline_loader,
            require_cuda=False,
            prompt_resolver=lambda record: f"changed {record.input_path.stem}",
        )


def test_generation_checkpoints_every_ten_images_for_crash_resume(
    tmp_path: Path,
) -> None:
    manifest_path = _build_manifest(tmp_path, count=11)
    runs_root = tmp_path / "runs"

    class FailsAfterTen(_FakePipeline):
        def __call__(self, **kwargs: object) -> _PipelineResult:
            if len(self.calls) == 10:
                raise RuntimeError("fixture interruption")
            return super().__call__(**kwargs)

    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_generation(
            manifest_path=manifest_path,
            runs_root=runs_root,
            config=GenerationConfig(strength=0.25, seed=0, run_id="checkpoint"),
            pipeline_loader=lambda _model: FailsAfterTen(),
            require_cuda=False,
        )

    _, checkpointed = _read_csv(runs_root / "checkpoint" / "generated.csv")
    assert len(checkpointed) == 10
    completed = run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=GenerationConfig(strength=0.25, seed=0, run_id="checkpoint"),
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )
    assert completed.count == 11


def test_manifest_and_run_id_reject_path_escape(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path, count=1)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["selected_path"] = "../outside.jpg"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="selected_path"):
        load_manifest(manifest_path)
    with pytest.raises(ValueError, match="run_id"):
        GenerationConfig(strength=0.25, seed=0, limit=1, run_id="../escape")


def test_strict_verifier_accepts_material_foreground_changes(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path, count=1)
    runs_root = tmp_path / "runs"
    run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=GenerationConfig(strength=0.25, seed=0, limit=1, run_id="changed"),
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )

    report = verify_run(runs_root=runs_root, run_id="changed", strict=True)

    assert report["count"] == 1
    assert report["all_model_output_checks_passed"] is True
    assert report["images"][0]["foreground_mean_abs_diff"] >= 3.0


def test_verify_script_runs_as_a_direct_file(tmp_path: Path) -> None:
    manifest_path = _build_manifest(tmp_path, count=1)
    runs_root = tmp_path / "runs"
    run_generation(
        manifest_path=manifest_path,
        runs_root=runs_root,
        config=GenerationConfig(strength=0.25, seed=0, limit=1, run_id="direct"),
        pipeline_loader=_fake_pipeline_loader,
        require_cuda=False,
    )
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, RUNS_ROOT=str(runs_root))

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_generated.py"),
            "--run-id",
            "direct",
            "--strict",
        ],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"all_model_output_checks_passed": true' in completed.stdout


@pytest.mark.parametrize("reencode", [False, True])
def test_strict_verifier_only_rejects_zero_foreground_change(
    tmp_path: Path, reencode: bool
) -> None:
    manifest_path = _build_manifest(tmp_path, count=1)
    input_record = load_manifest(manifest_path)[0]
    runs_root = tmp_path / "runs"
    run_root = runs_root / "pass-through"
    output_path = run_root / "images" / "가구" / "PRODUCT__image-0.png"
    output_path.parent.mkdir(parents=True)

    with Image.open(input_record.input_path) as source:
        resized = source.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    if reencode:
        jpeg_path = tmp_path / "reencoded.jpg"
        resized.save(jpeg_path, format="JPEG", quality=95)
        with Image.open(jpeg_path) as encoded:
            encoded.convert("RGB").save(output_path, format="PNG")
    else:
        resized.save(output_path, format="PNG")

    from scripts.generate import file_sha256

    with (run_root / "generated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "input_path": str(input_record.input_path),
                "output_path": str(output_path),
                "sha256": file_sha256(output_path),
                "seed": "0",
                "strength": "0.25",
            }
        )

    if not reencode:
        with pytest.raises(GeneratedVerificationError, match="pass-through"):
            verify_run(runs_root=runs_root, run_id="pass-through", strict=True)
        return

    report = verify_run(runs_root=runs_root, run_id="pass-through", strict=True)
    assert report["generation_validation"]["all_foreground_mae_positive"] is True
    assert report["generation_validation"]["foreground_mae"][
        "below_threshold_count"
    ] == 1
    assert report["generation_validation"]["foreground_mae"][
        "observation_threshold"
    ] == 3.0


def test_generation_source_rejects_copy_or_link_calls(tmp_path: Path) -> None:
    safe = tmp_path / "safe.py"
    unsafe = tmp_path / "unsafe.py"
    safe.write_text("output.save(target)\n", encoding="utf-8")
    unsafe.write_text("import shutil\nshutil.copy(source, target)\n", encoding="utf-8")

    assert verify_generation_source_has_no_copy_or_link_path(safe) is True
    with pytest.raises(GeneratedVerificationError, match="copy/link"):
        verify_generation_source_has_no_copy_or_link_path(unsafe)
