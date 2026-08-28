from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reproduce_check import ReproductionMismatch, compare_runs, reproduce


def _result(
    run_id: str,
    *,
    fid: float = 7.0,
    prompt_manifest: Path | None = None,
) -> dict:
    result = {
        "run_id": run_id,
        "protocol": {
            "strength": 0.25,
            "model": "stabilityai/stable-diffusion-xl-base-1.0",
        },
        "metrics": {
            "fid": fid,
            "psnr": {"mean": 31.0, "std": 1.0, "per_image": [31.0]},
            "ssim": {"mean": 0.93, "std": 0.01, "per_image": [0.93]},
        },
    }
    if prompt_manifest is not None:
        result["protocol"]["prompt_protocol"] = {
            "manifest_path": str(prompt_manifest.absolute()),
        }
    return result


def _write_result(runs_root: Path, result: dict) -> None:
    path = runs_root / result["run_id"] / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")


def test_reproduce_invokes_a_fresh_run_with_the_original_protocol(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    prompt_manifest = tmp_path / "prompts.csv"
    original = _result("main", prompt_manifest=prompt_manifest)
    _write_result(runs_root, original)
    observed: dict[str, object] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        reproduced = _result(kwargs["run_id"])
        _write_result(runs_root, reproduced)
        return object()

    report = reproduce(
        against="main",
        run_id="main-repro",
        runs_root=runs_root,
        run_runner=fake_run,
    )

    assert observed["strength"] == 0.25
    assert observed["model"] == original["protocol"]["model"]
    assert observed["limit"] is None
    assert observed["prompt_manifest"] == prompt_manifest.absolute()
    assert report["identical"] is True


def test_compare_runs_ignores_historical_fid_drift(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _write_result(runs_root, _result("main"))
    _write_result(runs_root, _result("main-repro", fid=7.1))

    report = compare_runs(against="main", run_id="main-repro", runs_root=runs_root)

    assert report["identical"] is True
    assert report["differences"] == {}


def test_compare_runs_writes_an_actionable_diff_on_psnr_drift(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    _write_result(runs_root, _result("main"))
    reproduced = _result("main-repro")
    reproduced["metrics"]["psnr"]["mean"] = 30.5
    _write_result(runs_root, reproduced)

    with pytest.raises(ReproductionMismatch, match="determinism controls"):
        compare_runs(against="main", run_id="main-repro", runs_root=runs_root)

    diff = json.loads(
        (runs_root / "main-repro" / "reproduction-diff.json").read_text()
    )
    assert diff["identical"] is False
    assert diff["differences"]["psnr_mean"] == {
        "expected": 31.0,
        "actual": 30.5,
    }
    assert "seed" in diff["possible_causes"]


def test_compare_runs_reports_per_image_drift_even_when_means_match(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    main = _result("main")
    reproduced = _result("main-repro")
    reproduced["metrics"]["psnr"]["per_image"] = [30.5, 31.5]
    _write_result(runs_root, main)
    _write_result(runs_root, reproduced)

    with pytest.raises(ReproductionMismatch):
        compare_runs(against="main", run_id="main-repro", runs_root=runs_root)

    diff = json.loads(
        (runs_root / "main-repro" / "reproduction-diff.json").read_text()
    )
    assert diff["differences"]["metrics_payload"] == {
        "expected_sha256": diff["expected_metrics_sha256"],
        "actual_sha256": diff["actual_metrics_sha256"],
    }
