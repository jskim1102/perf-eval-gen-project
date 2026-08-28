from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from metrics.fid import FID_PARAMETERS, compute_fid


def _write_fid_images(root: Path, *, count: int = 8) -> None:
    root.mkdir(parents=True)
    for index in range(count):
        image = Image.new("RGB", (360, 340), (245, 246, 247))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (30 + index, 40, 250, 280 - index),
            fill=(30 + index * 20, 80 + index * 7, 160 - index * 9),
        )
        image.save(root / f"sample-{index:02d}.png")


def test_fid_parameters_and_cleanfid_call_are_hard_fixed(
    tmp_path: Path, monkeypatch
) -> None:
    real_dir = tmp_path / "real"
    generated_dir = tmp_path / "generated"
    _write_fid_images(real_dir, count=2)
    _write_fid_images(generated_dir, count=2)
    observed: dict[str, object] = {}

    def fake_compute_fid(*args, **kwargs) -> float:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return 3.25

    monkeypatch.setattr("metrics.fid.clean_fid.compute_fid", fake_compute_fid)

    value, parameters = compute_fid(real_dir, generated_dir)

    assert value == 3.25
    assert parameters == {
        **FID_PARAMETERS,
        "real_set": "input x2",
        "gen_set": "reconstructed y2",
    }
    assert observed["args"] == (str(real_dir), str(generated_dir))
    assert observed["kwargs"] == {
        "mode": "clean",
        "model_name": "inception_v3",
        "num_workers": 0,
        "batch_size": 32,
        "device": "cuda",
        "verbose": False,
        "use_dataparallel": False,
    }


def test_fid_call_fixes_measurement_determinism_before_cleanfid(
    tmp_path: Path, monkeypatch
) -> None:
    real_dir = tmp_path / "real"
    generated_dir = tmp_path / "generated"
    _write_fid_images(real_dir, count=2)
    _write_fid_images(generated_dir, count=2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    def fake_compute_fid(*args, **kwargs) -> float:
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.initial_seed() == 0
        return 3.25

    monkeypatch.setattr("metrics.fid.clean_fid.compute_fid", fake_compute_fid)

    compute_fid(real_dir, generated_dir)


def test_identical_image_sets_have_fid_below_one(tmp_path: Path) -> None:
    image_dir = tmp_path / "same"
    _write_fid_images(image_dir)

    value, _ = compute_fid(image_dir, image_dir)

    assert isinstance(value, float)
    assert value < 1.0
