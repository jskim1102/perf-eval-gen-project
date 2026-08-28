from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.pilot import (
    DEFAULT_STRENGTHS,
    FID_NOTE,
    SELECTION_RULE,
    reselect_pilot,
    run_pilot,
    verify_pilot,
)


@dataclass(frozen=True)
class _GenerationReport:
    elapsed_seconds: float = 1.5
    peak_vram_gib: float = 9.75


def _measurement(fid: float, psnr: float, ssim: float) -> dict:
    return {
        "metrics": {
            "fid": fid,
            "psnr": {"mean": psnr, "std": 0.1, "per_image": [psnr]},
            "ssim": {"mean": ssim, "std": 0.01, "per_image": [ssim]},
        }
    }


def test_pilot_selects_highest_psnr_and_ignores_fid_order(tmp_path: Path) -> None:
    pair_manifest = tmp_path / "psnr_ssim_100.csv"
    pair_manifest.write_text("fixed\n", encoding="utf-8")
    measurements = {
        0.15: _measurement(6.0, 31.0, 0.96),
        0.20: _measurement(4.0, 29.0, 0.94),
        0.25: _measurement(7.0, 27.0, 0.92),
        0.30: _measurement(9.5, 25.1, 0.901),
        0.35: _measurement(11.0, 24.5, 0.89),
        0.40: _measurement(13.0, 23.0, 0.86),
    }

    def fake_generate(**kwargs):
        assert kwargs["config"].limit == 50
        assert kwargs["manifest_path"] == pair_manifest
        return _GenerationReport()

    def fake_measure(**kwargs):
        return measurements[kwargs["strength"]]

    summary = run_pilot(
        n=50,
        strengths=DEFAULT_STRENGTHS,
        runs_root=tmp_path / "external-runs",
        pair_manifest=pair_manifest,
        summary_root=tmp_path / "runs" / "pilot",
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
    )

    assert summary["selected"] == 0.15
    assert summary["best"]["strength"] == 0.15
    assert summary["selection_rule"] == SELECTION_RULE
    assert summary["fid_note"] == FID_NOTE
    assert summary["results"][0]["meets_psnr_ssim_gate"] is True
    assert "meets_all_targets" not in summary["results"][0]
    assert all("fid" not in entry for entry in summary["results"])
    assert len(summary["results"]) == 6
    assert [entry["strength"] for entry in summary["results"]] == DEFAULT_STRENGTHS
    assert (tmp_path / "runs" / "pilot" / "pilot.json").is_file()
    markdown = (tmp_path / "runs" / "pilot" / "pilot.md").read_text(encoding="utf-8")
    assert "| 0.20 | 29.000000 | 0.940000 | PASS |" in markdown
    assert FID_NOTE in markdown
    verify_pilot(tmp_path / "runs" / "pilot" / "pilot.json")


def test_pilot_records_null_only_when_no_strength_meets_psnr_ssim_gate(
    tmp_path: Path,
) -> None:
    pair_manifest = tmp_path / "psnr_ssim_100.csv"
    pair_manifest.write_text("fixed\n", encoding="utf-8")

    def fake_generate(**kwargs):
        return _GenerationReport()

    def fake_measure(**kwargs):
        strength = kwargs["strength"]
        return _measurement(10.5 + strength, 24.9 - strength, 0.899 - strength / 100)

    summary = run_pilot(
        n=50,
        strengths=DEFAULT_STRENGTHS,
        runs_root=tmp_path / "external-runs",
        pair_manifest=pair_manifest,
        summary_root=tmp_path / "runs" / "pilot",
        generation_runner=fake_generate,
        measurement_runner=fake_measure,
    )

    assert summary["selected"] is None
    assert summary["best"]["strength"] in DEFAULT_STRENGTHS
    written = json.loads(
        (tmp_path / "runs" / "pilot" / "pilot.json").read_text(encoding="utf-8")
    )
    assert written["selected"] is None
    assert "No strength met the PSNR and SSIM gate" in (
        tmp_path / "runs" / "pilot" / "pilot.md"
    ).read_text(encoding="utf-8")


def test_reselect_reuses_existing_measurements_without_generation(tmp_path: Path) -> None:
    summary_root = tmp_path / "runs" / "pilot"
    summary_root.mkdir(parents=True)
    summary_path = summary_root / "pilot.json"
    original = {
        "created_at": "2026-08-05T00:00:00+09:00",
        "n": 50,
        "seed": 0,
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "strengths": [0.15, 0.20],
        "targets": {"fid": 10.0, "psnr": 25.0, "ssim": 0.9},
        "selection_rule": "obsolete rule",
        "pair_manifest": "/fixed/pairs.csv",
        "pair_manifest_sha256": "a" * 64,
        "runs_root": str(tmp_path / "external-runs"),
        "results": [
            {
                "run_id": "pilot-s015",
                "strength": 0.15,
                "fid": 16.0,
                "psnr": 32.0,
                "ssim": 0.93,
                "meets_all_targets": False,
                "generation_count": 50,
                "elapsed_seconds": 1.0,
                "peak_vram_gib": 9.0,
            },
            {
                "run_id": "pilot-s020",
                "strength": 0.20,
                "fid": 19.0,
                "psnr": 31.0,
                "ssim": 0.92,
                "meets_all_targets": False,
                "generation_count": 50,
                "elapsed_seconds": 1.0,
                "peak_vram_gib": 9.0,
            },
        ],
        "selected": None,
        "best": {"strength": 0.15},
    }
    summary_path.write_text(json.dumps(original), encoding="utf-8")

    reselected = reselect_pilot(summary_path)

    assert reselected["selected"] == 0.15
    assert reselected["selection_rule"] == SELECTION_RULE
    assert reselected["fid_note"] == FID_NOTE
    assert [entry["fid"] for entry in reselected["results"]] == [16.0, 19.0]
    assert all(entry["meets_psnr_ssim_gate"] for entry in reselected["results"])
    assert not any("meets_all_targets" in entry for entry in reselected["results"])
    assert (summary_root / "pilot.md").is_file()
