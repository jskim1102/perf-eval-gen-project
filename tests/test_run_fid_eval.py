from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from scripts.generate import GenerationConfig, run_generation
from scripts.run_eval import NoSelectedStrengthError
from scripts.run_fid_eval import (
    GENERATION_MANIFEST_FIELDS,
    build_generation_manifest,
    parse_args,
    resolve_fid_strength,
    run_fid_evaluation,
    verify_fid_evaluation,
)


SOURCE_FIELDS = [
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


class _PipelineResult:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _FakeVae:
    def enable_tiling(self) -> None:
        return None


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVae()

    def set_progress_bar_config(self, *, disable: bool) -> None:
        assert disable is True

    def to(self, device: str) -> "_FakePipeline":
        assert device == "cpu"
        return self

    def __call__(self, **kwargs: object) -> _PipelineResult:
        source = np.asarray(kwargs["image"], dtype=np.uint16)
        generated = np.clip(source + 17, 0, 255).astype(np.uint8)
        return _PipelineResult(Image.fromarray(generated, mode="RGB"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_dataset(
    tmp_path: Path, *, count: int = 3, per_image_prompts: bool = False
) -> Path:
    dataset_root = tmp_path / "fid500"
    categories = ["생활용품", "이/미용", "홈클린"]
    category_directories = {
        "생활용품": "생활용품",
        "이/미용": "이_미용",
        "홈클린": "홈클린",
    }
    rows: list[dict[str, str]] = []
    for index in range(count):
        category = categories[index % len(categories)]
        item_no = str(15000 + index)
        image_path = (
            dataset_root
            / "input"
            / category_directories[category]
            / f"{item_no}.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (1100, 1100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((180, 150, 900, 940), fill=(50 + index * 30, 90, 140))
        draw.ellipse((330, 300, 760, 760), fill=(30, 45, 65))
        image.save(image_path, format="JPEG", quality=95)
        row = {
                "item_no": item_no,
                "대분류": category,
                "중분류": "fixture",
                "상품명": f"fixture-{index}",
                "zip_file": f"source-{index}.zip",
                "zip_member": f"folder/{item_no}_00_m_1.jpg",
                "source_filename": f"{item_no}_00_m_1.jpg",
                "width": "2988",
                "height": "2988",
                # Deliberately not the image digest: derivation must copy this
                # frozen value, not inspect/recompute image metadata.
                "sha256": f"frozen-sha-{index}",
            }
        if per_image_prompts:
            row["prompt"] = f"a fixture product photograph of {item_no}"
        rows.append(row)

    manifest_path = dataset_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [*SOURCE_FIELDS, "prompt"] if per_image_prompts else SOURCE_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "schema_version": 1,
        "selection_rule_version": "aihub-product-fid500-v1",
        "seed": 0,
        "quota": {category: categories.count(category) for category in categories},
        "counts": {"final_count": count},
        "manifest_sha256": _sha256(manifest_path),
        "source_dataset": {
            "name": "AI Hub 상품 이미지",
            "url": "https://aihub.or.kr/aidata/34145",
            "builder": "NIA",
            "year": 2020,
            "attribution": "NIA AI 학습용 데이터 구축사업 결과",
        },
        "rules": {"category_directories": category_directories},
    }
    if per_image_prompts:
        selection["prompt_protocol"] = {
            "template": "fixture template {img_prod_nm}",
            "product_name_max_words": 15,
            "fallback_order": ["img_prod_nm", "div_s", "div_m"],
            "whitespace_normalization": "strip and collapse consecutive whitespace",
        }
    (dataset_root / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False), encoding="utf-8"
    )
    return dataset_root


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _cpu_generation_runner(**kwargs):
    return run_generation(
        manifest_path=kwargs["manifest_path"],
        runs_root=kwargs["runs_root"],
        config=kwargs["config"],
        pipeline_loader=lambda _model: _FakePipeline(),
        require_cuda=False,
        item_ids=kwargs.get("item_ids"),
        prompt_resolver=kwargs.get("prompt_resolver"),
    )


def test_generation_manifest_is_a_metadata_only_derivative(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path, count=3)
    protected = [
        dataset_root / "manifest.csv",
        dataset_root / "selection.json",
        *sorted((dataset_root / "input").rglob("*.jpg")),
    ]
    before = {path: (_sha256(path), path.stat().st_mtime_ns) for path in protected}

    derived = build_generation_manifest(dataset_root=dataset_root)

    fields, rows = _read_csv(derived)
    assert fields == GENERATION_MANIFEST_FIELDS
    assert rows[0] == {
        "item_id": "15000",
        "group": "생활용품",
        "width": "2988",
        "height": "2988",
        "sha256": "frozen-sha-0",
        "source_path": "folder/15000_00_m_1.jpg",
        "selected_path": "input/생활용품/15000.jpg",
    }
    assert rows[1]["selected_path"] == "input/이_미용/15001.jpg"
    assert len(rows) == 3
    assert before == {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in protected
    }


def test_strength_has_no_fallback_and_explicit_value_must_match_pilot(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": None}), encoding="utf-8")
    with pytest.raises(NoSelectedStrengthError, match="selected is null"):
        resolve_fid_strength(
            explicit=None,
            strength_from=pilot,
            default_pilot_path=pilot,
        )

    pilot.write_text(json.dumps({"selected": 0.23}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match pilot selected"):
        resolve_fid_strength(
            explicit=0.24,
            strength_from=None,
            default_pilot_path=pilot,
        )
    assert resolve_fid_strength(
        explicit=0.23,
        strength_from=None,
        default_pilot_path=pilot,
    ) == (0.23, pilot.absolute())


def test_cli_accepts_a_defaulted_dataset_root_override(tmp_path: Path) -> None:
    args = parse_args(
        ["--run-id", "fid500-v2", "--dataset-root", str(tmp_path / "fid500-v2")]
    )
    assert args.dataset_root == tmp_path / "fid500-v2"


def test_full_run_reuses_original_real_directory_and_records_protocol(
    tmp_path: Path,
) -> None:
    dataset_root = _build_dataset(tmp_path, count=3)
    runs_root = tmp_path / "runs"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.23}), encoding="utf-8")
    observed: dict[str, Path] = {}

    def fake_fid(real_directory: Path, generated_directory: Path):
        observed["real"] = real_directory
        observed["generated"] = generated_directory
        return 7.25, {
            "impl": "clean-fid",
            "mode": "clean",
            "feature": "inception_v3",
            "input": "299x299 RGB",
            "resize": "bicubic+antialias",
            "real_set": "input x3",
            "gen_set": "reconstructed y3",
        }

    result = run_fid_evaluation(
        run_id="fid-unit-full",
        strength=0.23,
        strength_source=pilot,
        limit=None,
        runs_root=runs_root,
        dataset_root=dataset_root,
        generation_runner=_cpu_generation_runner,
        fid_runner=fake_fid,
    )

    assert observed["real"] == (dataset_root / "input").absolute()
    assert observed["generated"] == (
        runs_root / "fid-unit-full" / "images"
    ).absolute()
    assert result["dataset"]["source_dataset"]["builder"] == "NIA"
    assert result["dataset"]["manifest_sha256"] == _sha256(
        dataset_root / "manifest.csv"
    )
    assert result["protocol"]["strength"] == 0.23
    assert result["protocol"]["strength_source"] == str(pilot.absolute())
    assert result["protocol"]["model"] == GenerationConfig(
        strength=0.23, seed=0, run_id="contract"
    ).model
    assert result["protocol"]["clean_fid"]["mode"] == "clean"
    assert result["protocol"]["determinism"]["torch_seed"] == 0
    assert result["measurement"] == {
        "count": 3,
        "smoke": False,
        "limit": None,
        "fid": 7.25,
        "target": {"operator": "<=", "value": 10.0},
        "verdict": "PASS",
    }
    assert json.loads(
        (runs_root / "fid-unit-full" / "fid500.json").read_text(encoding="utf-8")
    ) == result


def test_full_run_uses_and_records_dataset_per_image_prompt_protocol(
    tmp_path: Path,
) -> None:
    dataset_root = _build_dataset(tmp_path, count=3, per_image_prompts=True)
    runs_root = tmp_path / "runs"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.23}), encoding="utf-8")

    result = run_fid_evaluation(
        run_id="fid-unit-prompts",
        strength=0.23,
        strength_source=pilot,
        limit=None,
        runs_root=runs_root,
        dataset_root=dataset_root,
        generation_runner=_cpu_generation_runner,
        fid_runner=lambda *_args: (
            7.0,
            {
                "impl": "clean-fid",
                "mode": "clean",
                "feature": "inception_v3",
                "input": "299x299 RGB",
                "resize": "bicubic+antialias",
                "real_set": "input x3",
                "gen_set": "reconstructed y3",
            },
        ),
    )

    with (runs_root / "fid-unit-prompts" / "generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        generated_rows = list(csv.DictReader(handle))
    assert [row["prompt"] for row in generated_rows] == [
        f"a fixture product photograph of {15000 + index}" for index in range(3)
    ]
    assert result["protocol"]["prompt"] == "per-image dataset prompt"
    assert result["protocol"]["prompt_protocol"] == json.loads(
        (dataset_root / "selection.json").read_text(encoding="utf-8")
    )["prompt_protocol"]


def test_smoke_subset_is_outside_dataset_and_verify_only_is_exact(
    tmp_path: Path,
) -> None:
    dataset_root = _build_dataset(tmp_path, count=3)
    runs_root = tmp_path / "runs"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.23}), encoding="utf-8")
    observed_real_directories: list[Path] = []

    def matching_fid(real_directory: Path, generated_directory: Path):
        real_directory = real_directory.absolute()
        observed_real_directories.append(real_directory)
        assert not real_directory.is_relative_to(dataset_root.absolute())
        assert len(list(real_directory.rglob("*.jpg"))) == 2
        assert len(list(generated_directory.rglob("*.png"))) == 2
        return 8.75, {
            "impl": "clean-fid",
            "mode": "clean",
            "feature": "inception_v3",
            "input": "299x299 RGB",
            "resize": "bicubic+antialias",
            "real_set": "input x2",
            "gen_set": "reconstructed y2",
        }

    result = run_fid_evaluation(
        run_id="fid-unit-smoke",
        strength=0.23,
        strength_source=pilot,
        limit=2,
        runs_root=runs_root,
        dataset_root=dataset_root,
        generation_runner=_cpu_generation_runner,
        fid_runner=matching_fid,
    )
    assert result["measurement"]["smoke"] is True
    assert result["measurement"]["count"] == 2

    verified = verify_fid_evaluation(
        run_id="fid-unit-smoke",
        runs_root=runs_root,
        dataset_root=dataset_root,
        fid_runner=matching_fid,
    )
    assert verified == result
    assert len(observed_real_directories) == 2

    with pytest.raises(ValueError, match="recorded FID differs"):
        verify_fid_evaluation(
            run_id="fid-unit-smoke",
            runs_root=runs_root,
            dataset_root=dataset_root,
            fid_runner=lambda *_args: (8.750000000000002, {}),
        )


def test_selected_fid_run_measures_exact_arbitrary_selected_set(tmp_path: Path) -> None:
    dataset_root = _build_dataset(tmp_path, count=3)
    runs_root = tmp_path / "runs"
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.23}), encoding="utf-8")
    observed_inputs: list[str] = []

    def selected_fid(real_directory: Path, generated_directory: Path):
        observed_inputs.extend(sorted(path.stem for path in real_directory.glob("*.jpg")))
        assert len(list(generated_directory.rglob("*.png"))) == 2
        return 6.5, {
            "impl": "clean-fid",
            "mode": "clean",
            "feature": "inception_v3",
            "input": "299x299 RGB",
            "resize": "bicubic+antialias",
            "real_set": "input x2",
            "gen_set": "reconstructed y2",
        }

    result = run_fid_evaluation(
        run_id="fidtry-unit-selected",
        strength=0.23,
        strength_source=pilot,
        limit=None,
        item_ids=["15002", "15000"],
        runs_root=runs_root,
        dataset_root=dataset_root,
        generation_runner=_cpu_generation_runner,
        fid_runner=selected_fid,
    )

    with (runs_root / "fidtry-unit-selected/generated.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        generated = list(csv.DictReader(handle))
    assert {Path(row["input_path"]).stem for row in generated} == {"15000", "15002"}
    assert len(observed_inputs) == 2
    assert result["measurement"]["count"] == 2
    assert result["measurement"]["selection_mode"] == "selected"
    assert result["measurement"]["item_ids"] == ["15002", "15000"]
    assert result["measurement"]["fid"] == 6.5
