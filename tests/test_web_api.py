from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from web.server import (
    DEFAULT_EVAL500_ROOT,
    DEFAULT_FID500_ROOT,
    DEFAULT_FID_V2_ROOT,
    DEFAULT_RUNS_ROOT,
    create_app,
)
from web.frontend import handler_factory


def make_dataset(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    curated_root = tmp_path / "curated"
    eval500_root = curated_root / "eval500"
    trusted_input_root = curated_root / "input"
    (eval500_root / "manifests").mkdir(parents=True)
    (eval500_root / "input").mkdir()
    trusted_input_root.mkdir()

    protocol = {
        "seed": "test-seed",
        "selection": {
            "input_count": 500,
            "psnr_ssim_pair_count": 100,
            "group_quotas": {"가구": 500},
        },
        "source_root": str(curated_root),
    }
    (eval500_root / "manifests" / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False),
        encoding="utf-8",
    )

    source_image = trusted_input_root / "sample.png"
    Image.new("RGB", (8, 8), (22, 44, 66)).save(source_image)
    eval_image = eval500_root / "input" / "sample.png"
    eval_image.symlink_to(source_image)
    return eval500_root, eval_image, protocol


def write_input_manifest(eval500_root: Path, count: int) -> None:
    rows = ["item_id", *(f"ITEM{index}" for index in range(count))]
    (eval500_root / "manifests" / "input.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def write_result(
    runs_root: Path,
    run_id: str,
    *,
    n_input: int,
    psnr: float,
    mtime_ns: int,
) -> Path:
    result_path = runs_root / run_id / "results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "protocol": {"n_input": n_input},
                "metrics": {"psnr": {"mean": psnr}},
            }
        ),
        encoding="utf-8",
    )
    os.utime(result_path, ns=(mtime_ns, mtime_ns))
    return result_path


def wait_for_state(client: TestClient, expected: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["state"] == expected:
            return status
        time.sleep(0.02)
    raise AssertionError(f"state did not become {expected!r}: {status}")


def test_dataset_returns_fixed_root_and_unmodified_protocol(tmp_path: Path) -> None:
    eval500_root, _, protocol = make_dataset(tmp_path)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd="",
    )

    with TestClient(app) as client:
        response = client.get("/api/dataset")

    assert response.status_code == 200
    assert response.json() == {
        "root": str(eval500_root.resolve()),
        "protocol": protocol,
    }


def test_production_web_commands_force_only_the_project_dataset_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    external = tmp_path / "external-dataset"
    monkeypatch.setenv("EVAL500_ROOT", str(external / "eval"))
    monkeypatch.setenv("FID500_ROOT", str(external / "fid"))
    monkeypatch.setenv("FID_V2_ROOT", str(external / "fid_v2"))
    monkeypatch.setenv("RUNS_ROOT", str(external / "runs"))
    monkeypatch.setenv("EVAL_CMD", "python scripts/run_eval.py")
    monkeypatch.setenv("TRY_CMD", "python scripts/try_generate.py")
    monkeypatch.setenv("FID_CMD", "python scripts/run_fid_eval.py")
    monkeypatch.setenv(
        "FID_V2_CMD", "python scripts/run_fid_v2_img2img_eval.py"
    )

    app = create_app()
    commands = "\n".join(
        (
            app.state.run_manager._eval_cmd,
            app.state.run_manager._try_cmd,
            app.state.run_manager._fid_cmd,
            app.state.run_manager._fidv2_cmd,
        )
    )
    with TestClient(app) as client:
        dataset = client.get("/api/dataset")

    assert dataset.status_code == 200
    assert dataset.json()["root"] == str(DEFAULT_EVAL500_ROOT.resolve())
    assert str(DEFAULT_EVAL500_ROOT.resolve()) in commands
    assert str(DEFAULT_FID500_ROOT.resolve()) in commands
    assert str(DEFAULT_FID_V2_ROOT.resolve()) in commands
    assert app.state.run_manager._runs_root == DEFAULT_RUNS_ROOT.absolute()
    assert str(external) not in commands


def test_results_returns_file_bytes_without_transforming_json(tmp_path: Path) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    result_dir = runs_root / "fixture"
    result_dir.mkdir(parents=True)
    raw_result = '{\n  "run_id": "fixture",\n  "metrics": {"fid": 8.75}\n}\n'
    (result_dir / "results.json").write_text(raw_result, encoding="utf-8")
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        response = client.get("/api/results", params={"run_id": "fixture"})
        missing = client.get("/api/results", params={"run_id": "missing"})
        traversal = client.get("/api/results", params={"run_id": "../fixture"})

    assert response.status_code == 200
    assert response.content == raw_result.encode()
    assert missing.status_code == 404
    assert traversal.status_code == 422


def test_results_default_excludes_fixture_even_when_it_is_newest(tmp_path: Path) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    write_input_manifest(eval500_root, 500)
    runs_root = tmp_path / "runs"
    expected = write_result(
        runs_root,
        "measured",
        n_input=500,
        psnr=32.87,
        mtime_ns=100,
    )
    fixture = runs_root / "fixture" / "results.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(
        (Path(__file__).resolve().parents[1] / "web/fixtures/results.fixture.json")
        .read_bytes()
    )
    os.utime(fixture, ns=(400, 400))
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        default = client.get("/api/results")
        explicit_fixture = client.get("/api/results", params={"run_id": "fixture"})

    assert default.status_code == 200
    assert default.content == expected.read_bytes()
    assert default.json()["run_id"] == "measured"
    assert default.json()["metrics"]["psnr"]["mean"] == 32.87
    assert explicit_fixture.status_code == 200


def test_results_default_requires_the_project_prompt_manifest_contract(
    tmp_path: Path,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    write_input_manifest(eval500_root, 100)
    prompt_manifest = eval500_root / "manifests/prompts.csv"
    prompt_manifest.write_text("item_id,prompt,name_source\nITEM0,prompt,en\n")
    prompt_hash = hashlib.sha256(prompt_manifest.read_bytes()).hexdigest()
    runs_root = tmp_path / "runs"
    valid = write_result(
        runs_root,
        "main-v2-100",
        n_input=100,
        psnr=32.8,
        mtime_ns=100,
    )
    payload = json.loads(valid.read_text())
    payload["protocol"]["prompt_protocol"] = {"manifest_sha256": prompt_hash}
    valid.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(valid, ns=(100, 100))
    write_result(
        runs_root,
        "legacy-newer",
        n_input=100,
        psnr=99.0,
        mtime_ns=200,
    )
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        response = client.get("/api/results")

    assert response.status_code == 200
    assert response.json()["run_id"] == "main-v2-100"


def test_results_default_excludes_trial_and_partial_runs_or_returns_404(
    tmp_path: Path,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    write_input_manifest(eval500_root, 500)
    runs_root = tmp_path / "runs"
    write_result(
        runs_root,
        "try-newest",
        n_input=500,
        psnr=99.0,
        mtime_ns=300,
    )
    write_result(
        runs_root,
        "partial-newer",
        n_input=4,
        psnr=98.0,
        mtime_ns=200,
    )
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        default = client.get("/api/results")
        explicit_trial = client.get(
            "/api/results", params={"run_id": "try-newest"}
        )

    assert default.status_code == 404
    assert explicit_trial.status_code == 404


def test_results_default_uses_run_id_as_deterministic_mtime_tiebreak(
    tmp_path: Path,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    write_input_manifest(eval500_root, 500)
    runs_root = tmp_path / "runs"
    write_result(
        runs_root,
        "alpha",
        n_input=500,
        psnr=31.0,
        mtime_ns=100,
    )
    expected = write_result(
        runs_root,
        "zulu",
        n_input=500,
        psnr=32.0,
        mtime_ns=100,
    )
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        responses = [client.get("/api/results") for _ in range(3)]

    assert all(response.status_code == 200 for response in responses)
    assert all(response.content == expected.read_bytes() for response in responses)


def test_image_only_serves_allowed_roots_and_trusted_eval_symlinks(
    tmp_path: Path,
) -> None:
    eval500_root, eval_image, _ = make_dataset(tmp_path)
    runs_root = tmp_path / "runs"
    run_image = runs_root / "main" / "images" / "generated.png"
    run_image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (88, 99, 111)).save(run_image)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(outside)
    escaping_link = run_image.parent / "escape.png"
    escaping_link.symlink_to(outside)
    app = create_app(runs_root=runs_root, eval500_root=eval500_root)

    with TestClient(app) as client:
        eval_response = client.get("/api/image", params={"path": str(eval_image)})
        run_response = client.get("/api/image", params={"path": str(run_image)})
        outside_response = client.get("/api/image", params={"path": str(outside)})
        escape_response = client.get(
            "/api/image", params={"path": str(escaping_link)}
        )
        missing_response = client.get(
            "/api/image", params={"path": str(runs_root / "missing.png")}
        )

    assert eval_response.status_code == 200
    assert run_response.status_code == 200
    assert outside_response.status_code == 403
    assert escape_response.status_code == 403
    assert missing_response.status_code == 404


def test_run_rejects_concurrency_and_keeps_only_last_50_log_lines(
    tmp_path: Path,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    marker = tmp_path / "starts.log"
    stub = tmp_path / "stub_eval.py"
    stub.write_text(
        """
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("marker")
parser.add_argument("--strength", required=True)
parser.add_argument("--run-id", required=True)
args = parser.parse_args()
with open(args.marker, "a", encoding="utf-8") as handle:
    handle.write(f"{args.run_id} {args.strength}\\n")
time.sleep(0.35)
for index in range(1, 61):
    print(f"generate {index}/60", flush=True)
print("measure 1/2", flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    eval_cmd = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(str(stub)),
            shlex.quote(str(marker)),
        ]
    )
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd=eval_cmd,
    )

    with TestClient(app) as client:
        first = client.post("/api/run", json={"strength": 0.25})
        assert first.status_code == 200
        run_id = first.json()["run_id"]

        second = client.post("/api/run", json={"strength": 0.3})
        assert second.status_code == 409
        assert second.json() == {"run_id": run_id}

        status = wait_for_state(client, "done")

    assert marker.read_text(encoding="utf-8").splitlines() == [f"{run_id} 0.25"]
    assert status["done"] == 60
    assert status["total"] == 60
    assert len(status["log"]) == 50
    assert status["log"][0] == "generate 12/60"
    assert status["log"][-1] == "measure 1/2"


def test_run_validates_strength_and_requires_command(tmp_path: Path) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd="",
    )

    with TestClient(app) as client:
        unavailable = client.post("/api/run", json={"strength": 0.25})
        too_low = client.post("/api/run", json={"strength": -0.01})
        too_high = client.post("/api/run", json={"strength": 1.01})

    assert unavailable.status_code == 503
    assert too_low.status_code == 422
    assert too_high.status_code == 422


def test_bad_eval_command_transitions_to_error_and_allows_retry(
    tmp_path: Path,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd="'unterminated",
    )

    with TestClient(app) as client:
        first = client.post("/api/run", json={"strength": 0.25})
        assert first.status_code == 200
        status = wait_for_state(client, "error")
        second = client.post("/api/run", json={"strength": 0.25})

    assert status["run_id"] == first.json()["run_id"]
    assert any("failed to start evaluation" in line for line in status["log"])
    assert second.status_code == 200


def test_unexpected_execution_exception_converges_to_error_and_allows_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd="configured",
    )

    def fail_split(_command: str) -> list[str]:
        raise subprocess.SubprocessError("unexpected command failure")

    monkeypatch.setattr("web.server.shlex.split", fail_split)
    with TestClient(app) as client:
        first = client.post("/api/run", json={"strength": 0.25})
        status = wait_for_state(client, "error", timeout=0.5)
        second = client.post("/api/run", json={"strength": 0.25})
        retry_status = wait_for_state(client, "error", timeout=0.5)

    assert first.status_code == 200
    assert any("unexpected command failure" in line for line in status["log"])
    assert second.status_code == 200
    assert retry_status["run_id"] == second.json()["run_id"]


def test_execution_exception_terminates_kills_and_reaps_child(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eval500_root, _, _ = make_dataset(tmp_path)

    class ExplodingOutput:
        def __iter__(self):
            raise subprocess.SubprocessError("output stream failed")

    class HangingProcess:
        stdout = ExplodingOutput()

        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_calls: list[float | None] = []

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            if not self.killed:
                raise subprocess.TimeoutExpired("stub", timeout)
            return -9

    process = HangingProcess()
    monkeypatch.setattr("web.server.subprocess.Popen", lambda *args, **kwargs: process)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval500_root,
        eval_cmd="configured",
    )

    with TestClient(app) as client:
        first = client.post("/api/run", json={"strength": 0.25})
        status = wait_for_state(client, "error", timeout=0.5)
        second = client.post("/api/run", json={"strength": 0.25})
        retry_status = wait_for_state(client, "error", timeout=0.5)

    assert first.status_code == 200
    assert status["state"] == "error"
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls[:2] == [5.0, 5.0]
    assert app.state.run_manager._process is None
    assert second.status_code == 200
    assert retry_status["run_id"] == second.json()["run_id"]


def test_frontend_config_has_no_implicit_fixture_run() -> None:
    handler = handler_factory(8013)
    config_body = handler.config_body.decode("utf-8")

    assert '"backendPort":8013' in config_body
    assert "initialRunId" not in config_body


def test_frontend_is_wired_without_mock_metric_values() -> None:
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    javascript = (project_root / "web" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    stylesheet = (project_root / "web" / "static" / "style.css").read_text(
        encoding="utf-8"
    )

    assert 'data-mock="states"' not in html
    assert "6.42" not in html
    assert "27.83" not in html
    assert "0.874" not in html
    assert "item.input_path" in javascript
    assert "item.output_path" in javascript
    assert "status.run_id" in javascript
    assert '|| "fixture"' not in javascript
    assert 'new Set(["idle", "done", "error"])' in javascript
    assert "toFixed" in javascript
    assert 'id="view-pair"' in html
    assert 'id="view-fid"' in html
    assert "fid" not in javascript.lower()
    assert "grid-template-columns: repeat(2, 1fr)" in stylesheet
    assert "renderSelectedResults(result)" in javascript
    assert 'fetch(apiUrl("/api/results"' not in javascript
    assert "loadResults(" not in javascript
    assert 'clearSelectedResults("선택 실행 대기")' in javascript


def test_canonical_comparison_is_removed_without_breaking_trial_images() -> None:
    project_root = Path(__file__).resolve().parents[1]
    html = (project_root / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    javascript = (project_root / "web" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    stylesheet = (project_root / "web" / "static" / "style.css").read_text(
        encoding="utf-8"
    )
    canonical_results = html.split('<section id="try-result"', 1)[1].split(
        "</section>", 1
    )[0]

    for symbol in (
        'id="compare-input"',
        'id="compare-output"',
        'id="compare-foot"',
        '<figure class="compare">',
        "compare-caption",
        "compare-pair",
        "원본 ↔ 생성물 대조",
    ):
        assert symbol not in canonical_results
    assert "function setComparison(" not in javascript
    assert "setComparison(" not in javascript
    for selector in (
        ".compare {",
        ".compare-caption {",
        ".compare-pair {",
        ".compare-foot {",
    ):
        assert selector not in stylesheet

    for element_id in (
        "metric-psnr",
        "psnr-badge",
        "psnr-tolerance",
        "psnr-histogram",
        "metric-ssim",
        "ssim-badge",
        "ssim-tolerance",
        "ssim-histogram",
    ):
        assert f'id="{element_id}"' in canonical_results
    assert "function imageCell(" in javascript
    assert '.className = "compare-cell"' in javascript
    assert '.className = "compare-tag"' in javascript
    assert ".compare-cell {" in stylesheet
    assert ".compare-tag {" in stylesheet
    assert ".compare-cell img {" in stylesheet
