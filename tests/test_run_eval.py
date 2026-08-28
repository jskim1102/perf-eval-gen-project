from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.generate import InputRecord, MODEL_ID, SD15_MODEL_ID
from scripts.run_eval import (
    NoSelectedStrengthError,
    PROMPT_PROTOCOL_TEMPLATE,
    verification_defaults,
    verify_evaluation,
    resolve_strength,
    run_evaluation,
)


@dataclass(frozen=True)
class _GenerationReport:
    count: int
    elapsed_seconds: float = 2.5
    peak_vram_gib: float = 8.0


def _manifest(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["selected_path"])
        writer.writeheader()
        for index in range(count):
            writer.writerow({"selected_path": f"input/group/image-{index}.jpg"})
    return path


def _measured(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "metrics": {
            "psnr": {"mean": 30.0},
            "ssim": {"mean": 0.93},
        },
    }


def _generation_verification(count: int) -> dict:
    return {
        "count": count,
        "generation_validation": {
            "image_count": count,
            "all_output_sha256_differ_from_input": True,
            "all_foreground_mae_positive": True,
            "copy_or_link_path_absent": True,
            "foreground_definition": "any input RGB channel < 245",
            "changed_pixel_delta": 3.0,
            "foreground_mae": {
                "observation_threshold": 3.0,
                "minimum": 1.0,
                "p5": 1.1,
                "median": 2.0,
                "p95": 4.0,
                "maximum": 5.0,
                "below_threshold_count": 1,
                "below_threshold_ratio": 0.5,
            },
            "foreground_changed_fraction": {
                "observation_threshold": 0.25,
                "minimum": 0.1,
                "p5": 0.11,
                "median": 0.2,
                "p95": 0.4,
                "maximum": 0.5,
                "below_threshold_count": 1,
                "below_threshold_ratio": 0.5,
            },
        },
    }


def test_resolve_strength_rejects_a_pilot_without_a_selected_value(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": None, "best": {"strength": 0.15}}))

    with pytest.raises(NoSelectedStrengthError, match="selected is null"):
        resolve_strength(explicit=None, strength_from=pilot)


def test_resolve_strength_reads_the_fixed_pilot_selection(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.15}))

    assert resolve_strength(explicit=None, strength_from=pilot) == 0.15


def test_verification_defaults_enforce_full_main_and_baseline_contracts(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({"selected": 0.15}))

    assert verification_defaults(
        run_id="main",
        limit=None,
        explicit_strength=None,
        strength_from=None,
        requested_model=MODEL_ID,
        pilot_path=pilot,
    ) == (0.15, MODEL_ID, 500, 100)
    assert verification_defaults(
        run_id="baseline",
        limit=None,
        explicit_strength=None,
        strength_from=None,
        requested_model=MODEL_ID,
        pilot_path=pilot,
    ) == (0.15, SD15_MODEL_ID, 500, 100)
    assert verification_defaults(
        run_id="main-v2",
        limit=None,
        explicit_strength=None,
        strength_from=None,
        requested_model=MODEL_ID,
        pilot_path=pilot,
    ) == (0.15, MODEL_ID, 500, 100)
    assert verification_defaults(
        run_id="baseline-v2",
        limit=None,
        explicit_strength=None,
        strength_from=None,
        requested_model=MODEL_ID,
        pilot_path=pilot,
    ) == (0.15, SD15_MODEL_ID, 500, 100)


def test_run_evaluation_uses_fixed_full_manifests_and_protocol(
    tmp_path: Path, capsys
) -> None:
    input_manifest = _manifest(tmp_path / "eval500/manifests/input.csv", 500)
    pair_manifest = _manifest(tmp_path / "eval500/manifests/psnr_ssim_100.csv", 100)
    calls: dict[str, object] = {}

    def fake_generate(**kwargs):
        calls["generation"] = kwargs
        return _GenerationReport(count=500)

    def fake_measure(**kwargs):
        calls["measurement"] = kwargs
        return _measured(kwargs["run_id"])

    outcome = run_evaluation(
        run_id="main",
        strength=0.25,
        model=MODEL_ID,
        limit=None,
        runs_root=tmp_path / "runs",
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
        generated_verifier=lambda **kwargs: _generation_verification(500),
    )

    generation = calls["generation"]
    measurement = calls["measurement"]
    assert generation["manifest_path"] == input_manifest
    assert generation["config"].limit is None
    assert generation["config"].strength == 0.25
    assert generation["config"].seed == 0
    assert measurement["input_manifest"] == input_manifest
    assert measurement["pair_manifest"] == pair_manifest
    assert measurement["require_all_pairs"] is True
    assert outcome.generation.count == 500
    assert outcome.results["run_id"] == "main"
    output = capsys.readouterr().out.lower()
    assert "fid=" not in output
    assert "psnr=30.000000" in output
    assert "ssim=0.930000" in output


def test_limited_run_uses_the_fixed_pair_prefix_for_generation(tmp_path: Path) -> None:
    input_manifest = _manifest(tmp_path / "eval500/manifests/input.csv", 500)
    pair_manifest = _manifest(tmp_path / "eval500/manifests/psnr_ssim_100.csv", 100)
    calls: dict[str, object] = {}

    def fake_generate(**kwargs):
        calls.update(kwargs)
        return _GenerationReport(count=4)

    def fake_measure(**kwargs):
        calls["measurement"] = kwargs
        return _measured(kwargs["run_id"])

    run_evaluation(
        run_id="web-smoke",
        strength=0.25,
        model=MODEL_ID,
        limit=4,
        runs_root=tmp_path / "runs",
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
        generated_verifier=lambda **kwargs: _generation_verification(4),
    )

    assert calls["manifest_path"] == pair_manifest
    assert calls["config"].limit == 4
    assert calls["measurement"]["input_manifest"] == input_manifest
    assert calls["measurement"]["require_all_pairs"] is False


def test_run_evaluation_uses_per_image_prompts_and_records_the_manifest_contract(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval500"
    input_path = eval_root / "input/group/image.jpg"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"input")
    input_manifest = eval_root / "manifests/input.csv"
    input_manifest.parent.mkdir(parents=True)
    with input_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "product_type", "selected_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "item-1",
                "product_type": "CHAIR",
                "selected_path": "input/group/image.jpg",
            }
        )
    pair_manifest = input_manifest.parent / "psnr_ssim_100.csv"
    pair_manifest.write_bytes(input_manifest.read_bytes())
    prompt_manifest = input_manifest.parent / "prompts.csv"
    with prompt_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id", "prompt", "name_source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "item-1",
                "prompt": "a high quality studio product photograph of Desk Chair, on a clean white background",
                "name_source": "en",
            }
        )

    observed: dict[str, object] = {}

    def fake_generate(**kwargs):
        observed["prompt"] = kwargs["prompt_resolver"](
            InputRecord(input_path=input_path, output_relative_path=Path("group/image.png"))
        )
        return _GenerationReport(count=1)

    def fake_measure(**kwargs):
        observed["prompt_protocol"] = kwargs["prompt_protocol"]
        return _measured(kwargs["run_id"])

    run_evaluation(
        run_id="main-v2",
        strength=0.15,
        model=MODEL_ID,
        limit=None,
        runs_root=tmp_path / "runs",
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        prompt_manifest=prompt_manifest,
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
        generated_verifier=lambda **kwargs: _generation_verification(1),
    )

    assert observed["prompt"] == (
        "a high quality studio product photograph of Desk Chair, on a clean white background"
    )
    protocol = observed["prompt_protocol"]
    assert protocol["template"] == PROMPT_PROTOCOL_TEMPLATE
    assert protocol["name_max_words"] == 15
    assert protocol["manifest_path"] == str(prompt_manifest.absolute())
    assert protocol["manifest_sha256"] == hashlib.sha256(
        prompt_manifest.read_bytes()
    ).hexdigest()


def test_inaccessible_baseline_model_falls_back_and_records_the_protocol_note(
    tmp_path: Path,
) -> None:
    input_manifest = _manifest(tmp_path / "eval500/manifests/input.csv", 500)
    pair_manifest = _manifest(tmp_path / "eval500/manifests/psnr_ssim_100.csv", 100)
    attempted_models: list[str] = []
    measured: dict[str, object] = {}

    def fake_generate(**kwargs):
        attempted_models.append(kwargs["config"].model)
        if len(attempted_models) == 1:
            raise OSError("model access denied")
        return _GenerationReport(count=500)

    def fake_measure(**kwargs):
        measured.update(kwargs)
        return _measured(kwargs["run_id"])

    outcome = run_evaluation(
        run_id="baseline",
        strength=0.15,
        model="stable-diffusion-v1-5/stable-diffusion-v1-5",
        limit=None,
        runs_root=tmp_path / "runs",
        input_manifest=input_manifest,
        pair_manifest=pair_manifest,
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
        generated_verifier=lambda **kwargs: _generation_verification(500),
        allow_baseline_fallback=True,
    )

    assert attempted_models == [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        MODEL_ID,
    ]
    assert outcome.model == MODEL_ID
    assert outcome.strength == 0.25
    assert measured["model"] == MODEL_ID
    assert measured["strength"] == 0.25
    assert "requested baseline model" in measured["note"]


def test_verify_evaluation_checks_counts_protocol_and_actual_manifest_hashes(
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval500"
    input_manifest = eval_root / "manifests" / "input.csv"
    input_path = eval_root / "input" / "group" / "input.png"
    second_input_path = eval_root / "input" / "group" / "input-2.png"
    runs_root = tmp_path / "runs"
    run_root = runs_root / "main"
    output_path = run_root / "images" / "group" / "output.png"
    second_output_path = run_root / "images" / "group" / "output-2.png"
    input_manifest.parent.mkdir(parents=True)
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_manifest.write_text("fixed manifest\n", encoding="utf-8")
    input_path.write_bytes(b"input")
    second_input_path.write_bytes(b"input-2")
    output_path.write_bytes(b"output")
    second_output_path.write_bytes(b"output-2")

    generated_manifest = run_root / "generated.csv"
    with generated_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_path", "output_path", "sha256", "seed", "strength"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "seed": "0",
                "strength": "0.15",
            }
        )
        writer.writerow(
            {
                "input_path": str(second_input_path),
                "output_path": str(second_output_path),
                "sha256": hashlib.sha256(second_output_path.read_bytes()).hexdigest(),
                "seed": "0",
                "strength": "0.15",
            }
        )
    result = {
        "run_id": "main",
        "created_at": "2026-08-05T10:00:00+09:00",
        "protocol": {
            "seed": 0,
            "n_input": 2,
            "n_pairs": 1,
            "strength": 0.15,
            "model": MODEL_ID,
            "steps": 30,
            "guidance": 5.0,
            "scheduler": "EulerDiscreteScheduler",
            "output": "1024x1024 PNG",
            "fid": {
                "impl": "clean-fid",
                "mode": "clean",
                "feature": "inception_v3",
                "input": "299x299 RGB",
                "resize": "bicubic+antialias",
                "real_set": "input x2",
                "gen_set": "reconstructed y2",
            },
            "psnr": {"resize": "lanczos", "dtype": "uint8", "data_range": 255},
            "ssim": {
                "gaussian_weights": True,
                "sigma": 1.5,
                "use_sample_covariance": False,
                "data_range": 255,
                "channel": "per-channel mean",
            },
        },
        "env": {
            "python": "3.12",
            "torch": "x",
            "diffusers": "x",
            "cleanfid": "x",
            "skimage": "x",
            "gpu": "x",
            "driver": "x",
            "cuda": "x",
            "determinism": {
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "torch_seed": 0,
            },
        },
        "inputs": {
            "input_manifest_sha256": hashlib.sha256(
                input_manifest.read_bytes()
            ).hexdigest(),
            "generated_manifest_sha256": hashlib.sha256(
                generated_manifest.read_bytes()
            ).hexdigest(),
        },
        "metrics": {
            "fid": 1.0,
            "psnr": {"mean": 30.0, "std": 0.0, "per_image": [30.0]},
            "ssim": {"mean": 0.93, "std": 0.0, "per_image": [0.93]},
        },
        "pairs": [
            {
                "item_id": "item-1",
                "group": "group",
                "input_path": str(input_path),
                "output_path": str(output_path),
            }
        ],
        "targets": {"fid": 10.0, "psnr": 25.0, "ssim": 0.9},
        "baseline_ref": None,
    }
    result["generation_validation"] = _generation_verification(2)[
        "generation_validation"
    ]
    result_path = run_root / "results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    bytes_before = result_path.read_bytes()

    verifier_calls: list[dict] = []

    def fake_verifier(**kwargs):
        verifier_calls.append(kwargs)
        return _generation_verification(2)

    verified = verify_evaluation(
        run_id="main",
        runs_root=runs_root,
        input_manifest=input_manifest,
        expected_strength=0.15,
        expected_model=MODEL_ID,
        expected_count=2,
        expected_pair_count=1,
        generated_verifier=fake_verifier,
    )
    assert verified["metrics"]["fid"] == 1.0
    assert verifier_calls == [
        {"runs_root": runs_root.absolute(), "run_id": "main", "strict": True}
    ]
    assert verified["env"]["determinism"] == {
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "torch_seed": 0,
    }
    assert result_path.read_bytes() == bytes_before

    result["inputs"]["generated_manifest_sha256"] = "0" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="generated manifest hash"):
        verify_evaluation(
            run_id="main",
            runs_root=runs_root,
            input_manifest=input_manifest,
            expected_count=2,
            expected_pair_count=1,
            generated_verifier=lambda **kwargs: _generation_verification(2),
        )
