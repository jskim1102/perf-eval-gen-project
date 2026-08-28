from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from web.server import FID_V2_RUN_ID, FID_V2_STRENGTH, create_app


def _write_fidv2_dataset(root: Path) -> tuple[list[dict[str, str]], bytes]:
    rows = [
        {
            "item_no": "15021",
            "대분류": "생활용품",
            "중분류": "위생용품",
            "소분류": "탈취제",
            "상품명": "fixture one",
            "source_product": "input/생활용품/15021.jpg",
            "thumbnail": "input/생활용품/15021.png",
            "prompt": "prompt one",
            "sha256": "hash-one",
        },
        {
            "item_no": "15025",
            "대분류": "생활용품",
            "중분류": "위생용품",
            "소분류": "탈취제",
            "상품명": "fixture two",
            "source_product": "input/생활용품/15025.jpg",
            "thumbnail": "input/생활용품/15025.png",
            "prompt": "prompt two",
            "sha256": "hash-two",
        },
        {
            "item_no": "25001",
            "대분류": "홈클린",
            "중분류": "청소용품",
            "소분류": "세정제",
            "상품명": "fixture three",
            "source_product": "input/홈클린/25001.jpg",
            "thumbnail": "input/홈클린/25001.png",
            "prompt": "prompt three",
            "sha256": "hash-three",
        },
    ]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        image = root / row["thumbnail"]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (20, 40, 60)).save(image)
    selection = {
        "count": len(rows),
        "category_counts": {"생활용품": 2, "홈클린": 1},
        "not_ai_generated": True,
        "source_dataset": {
            "name": "AI Hub 상품 이미지",
            "url": "https://aihub.or.kr/aidata/34145",
            "attribution": "NIA AI 학습용 데이터 구축사업 결과",
        },
    }
    selection_bytes = (
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    (root / "selection.json").write_bytes(selection_bytes)
    return rows, selection_bytes


def _write_pilot(path: Path) -> Path:
    path.write_text(json.dumps({"selected": 0.15}), encoding="utf-8")
    return path


def _wait_terminal(client: TestClient, timeout: float = 3.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    status: dict[str, object] = {}
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["state"] in {"done", "error"}:
            return status
        time.sleep(0.01)
    raise AssertionError(f"run did not finish: {status}")


def _app(tmp_path: Path, **overrides: object):
    fidv2_root = tmp_path / "fid_v2"
    if not fidv2_root.exists():
        _write_fidv2_dataset(fidv2_root)
    options = {
        "runs_root": tmp_path / "runs",
        "fidv2_root": fidv2_root,
        "fidv2_cmd": "configured",
        "pilot_path": _write_pilot(tmp_path / "pilot.json"),
    }
    options.update(overrides)
    return create_app(**options)


def test_fidv2_config_dataset_and_items_expose_fixed_img2img_protocol(
    tmp_path: Path,
) -> None:
    fidv2_root = tmp_path / "fid_v2"
    rows, selection_bytes = _write_fidv2_dataset(fidv2_root)
    app = _app(tmp_path, fidv2_root=fidv2_root)

    with TestClient(app) as client:
        config = client.get("/api/fidv2/config")
        dataset = client.get("/api/fidv2/dataset")
        items = client.get("/api/fidv2/dataset/items")

    assert config.status_code == 200
    assert config.json() == {
        "strength": FID_V2_STRENGTH,
        "source": "fixed FLUX.1-dev img2img protocol",
        "model": "black-forest-labs/FLUX.1-dev",
        "generation_mode": "image-to-image",
        "num_inference_steps": 30,
        "guidance_scale": 3.5,
        "seed": 0,
        "seconds_per_image": 22,
        "detail": None,
        "available": True,
        "completed": False,
        "selected_available": True,
        "selected_detail": None,
    }
    assert dataset.status_code == 200
    assert dataset.content == selection_bytes
    assert items.status_code == 200
    assert items.json()["groups"] == [
        {"group": "생활용품", "count": 2},
        {"group": "홈클린", "count": 1},
    ]
    assert [item["item_id"] for item in items.json()["items"]] == [
        row["item_no"] for row in rows
    ]
    assert items.json()["items"][0] == {
        "item_id": "15021",
        "group": "생활용품",
        "product_type": "fixture one",
        "image_id": "15021",
        "input_path": str((fidv2_root / rows[0]["thumbnail"]).absolute()),
    }


def test_selected_fidv2_command_uses_fixed_protocol_and_selected_ids(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "argv.json"
    stub = tmp_path / "capture.py"
    stub.write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))\n",
        encoding="utf-8",
    )
    command = " ".join(
        [shlex.quote(sys.executable), shlex.quote(str(stub)), shlex.quote(str(marker))]
    )
    app = _app(tmp_path, fidv2_cmd=command)

    with TestClient(app) as client:
        response = client.post(
            "/api/fidv2/try", json={"item_ids": ["15025", "25001"]}
        )
        status = _wait_terminal(client)

    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert run_id.startswith("fidv2try-")
    assert status["run_id"] == run_id
    assert json.loads(marker.read_text(encoding="utf-8")) == [
        "--strength",
        "0.15",
        "--run-id",
        run_id,
        "--item-id",
        "15025",
        "--item-id",
        "25001",
    ]


def test_fidv2_selected_result_images_and_originals_are_scoped(tmp_path: Path) -> None:
    fidv2_root = tmp_path / "fid_v2"
    _write_fidv2_dataset(fidv2_root)
    runs_root = tmp_path / "runs"
    run_id = "fidv2try-selected"
    run_root = runs_root / run_id
    run_root.mkdir(parents=True)
    result_path = run_root / "fid_v2.json"
    result_path.write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    input_path = fidv2_root / "input/생활용품/15021.png"
    output_path = run_root / "images/생활용품/15021.png"
    output_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), (70, 80, 90)).save(output_path)
    with (run_root / "generated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "input_path",
                "output_path",
                "sha256",
                "seed",
                "strength",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item_id": "15021",
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": "hash",
                "seed": "0",
                "strength": "0.15",
            }
        )
    app = _app(
        tmp_path,
        fidv2_root=fidv2_root,
        runs_root=runs_root,
    )

    with TestClient(app) as client:
        result = client.get("/api/fidv2/try/results", params={"run_id": run_id})
        images = client.get("/api/fidv2/try/images", params={"run_id": run_id})
        original = client.get("/api/image", params={"path": str(input_path)})
        generated = client.get("/api/image", params={"path": str(output_path)})
        wrong_prefix = client.get(
            "/api/fidv2/try/results", params={"run_id": "fidtry-selected"}
        )

    assert result.status_code == 200
    assert result.content == result_path.read_bytes()
    assert images.status_code == 200
    assert images.json()["run_id"] == run_id
    assert images.json()["items"][0]["input_path"] == str(input_path)
    assert original.status_code == 200
    assert generated.status_code == 200
    assert wrong_prefix.status_code == 404


def test_fidv1_and_fidv2_share_the_same_run_manager_lock(tmp_path: Path) -> None:
    stub = tmp_path / "slow.py"
    stub.write_text(
        "import time\nprint('FID_V2_GENERATE done=1/2', flush=True)\ntime.sleep(0.2)\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(stub))}"
    app = _app(tmp_path, fid_cmd=command, fidv2_cmd=command)

    with TestClient(app) as client:
        fidv1 = client.post("/api/fid/run")
        blocked_fidv2 = client.post(
            "/api/fidv2/try", json={"item_ids": ["15021", "15025"]}
        )
        _wait_terminal(client)
        fidv2 = client.post(
            "/api/fidv2/try", json={"item_ids": ["15021", "15025"]}
        )
        blocked_fidv1 = client.post("/api/fid/run")
        _wait_terminal(client)

    assert fidv1.status_code == 200
    assert blocked_fidv2.status_code == 409
    assert blocked_fidv2.json() == {"run_id": fidv1.json()["run_id"]}
    assert fidv2.status_code == 200
    assert blocked_fidv1.status_code == 409
    assert blocked_fidv1.json() == {"run_id": fidv2.json()["run_id"]}


def test_fidv2_selection_validation_and_canonical_endpoints(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    app = _app(tmp_path, runs_root=runs_root)

    with TestClient(app) as client:
        too_small = client.post("/api/fidv2/try", json={"item_ids": ["15021"]})
        duplicate = client.post(
            "/api/fidv2/try", json={"item_ids": ["15021", "15021"]}
        )
        unknown = client.post(
            "/api/fidv2/try", json={"item_ids": ["15021", "missing"]}
        )
        missing_result = client.get("/api/fidv2/results")
        missing_images = client.get("/api/fidv2/images")

    assert too_small.status_code == 400
    assert duplicate.status_code == 400
    assert unknown.status_code == 400
    assert missing_result.status_code == 404
    assert missing_images.status_code == 404

    canonical = runs_root / FID_V2_RUN_ID
    canonical.mkdir(parents=True)
    (canonical / "fid_v2.json").write_text(
        json.dumps({"run_id": FID_V2_RUN_ID}) + "\n", encoding="utf-8"
    )
    with TestClient(_app(tmp_path, runs_root=runs_root)) as client:
        config = client.get("/api/fidv2/config")
        blocked = client.post("/api/fidv2/run")
    assert config.json()["completed"] is True
    assert config.json()["available"] is False
    assert blocked.status_code == 409


def test_fidv2_static_panel_is_complete_and_uses_its_own_api_namespace() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/static/index.html").read_text(encoding="utf-8")
    javascript = (root / "web/static/fidv2.js").read_text(encoding="utf-8")
    fidv2_view = html.split('id="view-fidv2"', 1)[1]

    assert 'src="fidv2.js"' in html
    assert 'id="fidv2-tree-groups"' in fidv2_view
    assert 'id="fidv2-run-button"' in fidv2_view
    assert 'id="fidv2-progress-track"' in fidv2_view
    assert 'id="fidv2-result-body"' in fidv2_view
    assert 'id="fidv2-result-generation-mode"' in fidv2_view
    assert "규칙 기반 합성 썸네일" in fidv2_view
    assert "AI 생성물이 아니다" in fidv2_view
    assert "500장 선택 시 약 3시간" in fidv2_view
    assert 'fetch(apiUrl("/api/fidv2/dataset/items")' in javascript
    assert 'fetch(apiUrl("/api/fidv2/config")' in javascript
    assert 'fetch(apiUrl("/api/fidv2/try")' in javascript
    assert 'fetch(apiUrl("/api/fidv2/try/results"' in javascript
    assert 'fetch(apiUrl("/api/fidv2/try/images"' in javascript
    assert 'fetch(apiUrl("/api/fidv2/results")' not in javascript
    assert 'fetch(apiUrl("/api/fidv2/run")' not in javascript
    assert "8013" not in javascript
    assert "5186" not in javascript
    assert "const FIDV2_PAGE_SIZE = 10" in javascript
    assert 'statusRunId.startsWith("fidv2try-")' in javascript
    assert "PERF_RUN_PROGRESS.snapshot" in javascript

    completed = subprocess.run(
        ["node", "--check", "web/static/fidv2.js"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
