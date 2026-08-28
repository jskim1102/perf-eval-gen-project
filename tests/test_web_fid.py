from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from web.server import FID_RUN_ID, create_app


def _write_selection(fid500_root: Path) -> tuple[dict[str, object], bytes]:
    selection = {
        "schema_version": 1,
        "selection_rule_version": "aihub-product-fid500-v1",
        "seed": 0,
        "quota": {"생활용품": 232, "이/미용": 179, "홈클린": 89},
        "counts": {"final_count": 500},
        "manifest_sha256": "frozen-manifest-hash",
        "source_dataset": {
            "name": "AI Hub 상품 이미지",
            "url": "https://aihub.or.kr/aidata/34145",
            "builder": "NIA",
            "year": 2020,
            "attribution": "NIA AI 학습용 데이터 구축사업 결과",
        },
        "rules": {
            "eligibility": "single object; square; at least 1000px",
            "one_per_product": "frozen preference order",
            "quota_selection": "seeded sha256 order",
        },
        "source_zips": [
            {"filename": f"source-{index}.zip", "size": index + 10, "mtime_ns": index + 20}
            for index in range(6)
        ],
        "label_zips": [
            {"filename": f"label-{index}.zip", "size": index + 30, "mtime_ns": index + 40}
            for index in range(6)
        ],
    }
    fid500_root.mkdir(parents=True)
    payload = (
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (fid500_root / "selection.json").write_bytes(payload)
    return selection, payload


def _write_pilot(path: Path, selected: object) -> Path:
    path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    return path


def _write_fid_items(fid500_root: Path) -> list[dict[str, str]]:
    rows = [
        {
            "item_no": "15021",
            "대분류": "생활용품",
            "중분류": "위생용품",
            "소분류": "탈취제",
            "상품명": "fixture one",
        },
        {
            "item_no": "15025",
            "대분류": "생활용품",
            "중분류": "위생용품",
            "소분류": "탈취제",
            "상품명": "fixture two",
        },
        {
            "item_no": "25001",
            "대분류": "홈클린",
            "중분류": "청소용품",
            "소분류": "세정제",
            "상품명": "fixture three",
        },
    ]
    fid500_root.mkdir(parents=True, exist_ok=True)
    with (fid500_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        image = fid500_root / "input" / row["대분류"] / f'{row["item_no"]}.jpg'
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f'image-{row["item_no"]}'.encode())
    return rows


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
    fid500_root = tmp_path / "fid500"
    if not fid500_root.exists():
        _write_selection(fid500_root)
    options = {
        "runs_root": tmp_path / "runs",
        "fid500_root": fid500_root,
        "pilot_path": _write_pilot(tmp_path / "pilot.json", 0.23),
        "fid_cmd": "configured",
    }
    options.update(overrides)
    return create_app(**options)


def test_fid_dataset_and_result_endpoints_return_files_unmodified(tmp_path: Path) -> None:
    fid500_root = tmp_path / "fid500"
    _, selection_bytes = _write_selection(fid500_root)
    runs_root = tmp_path / "runs"
    result_path = runs_root / FID_RUN_ID / "fid500.json"
    result_path.parent.mkdir(parents=True)
    result_bytes = json.dumps(
        {"run_id": FID_RUN_ID, "measurement": {"fid": 7.25}}
    ).encode() + b"\n"
    result_path.write_bytes(result_bytes)
    app = _app(tmp_path, fid500_root=fid500_root, runs_root=runs_root)

    with TestClient(app) as client:
        dataset = client.get("/api/fid/dataset")
        result = client.get("/api/fid/results")

    assert dataset.status_code == 200
    assert dataset.content == selection_bytes
    assert result.status_code == 200
    assert result.content == result_bytes


def test_fid_dataset_items_expose_all_selectable_images_by_group(tmp_path: Path) -> None:
    fid500_root = tmp_path / "fid500"
    _write_selection(fid500_root)
    _write_fid_items(fid500_root)
    app = _app(tmp_path, fid500_root=fid500_root)

    with TestClient(app) as client:
        response = client.get("/api/fid/dataset/items")

    assert response.status_code == 200
    payload = response.json()
    assert payload["groups"] == [
        {"group": "생활용품", "count": 2},
        {"group": "홈클린", "count": 1},
    ]
    assert [item["item_id"] for item in payload["items"]] == [
        "15021", "15025", "25001"
    ]
    assert payload["items"][0] == {
        "item_id": "15021",
        "group": "생활용품",
        "product_type": "fixture one",
        "image_id": "15021",
        "input_path": str(
            (fid500_root / "input/생활용품/15021.jpg").absolute()
        ),
    }


def test_fid_images_returns_all_generated_rows_and_serves_both_sides(
    tmp_path: Path,
) -> None:
    fid500_root = tmp_path / "fid500"
    _write_selection(fid500_root)
    runs_root = tmp_path / "runs"
    run_root = runs_root / FID_RUN_ID
    input_path = fid500_root / "input" / "생활용품" / "15019.jpg"
    output_path = run_root / "images" / "생활용품" / "15019.png"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"original-image")
    output_path.write_bytes(b"generated-image")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    escape = fid500_root / "input" / "escape.jpg"
    escape.symlink_to(outside)
    rows = [
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sha256": f"hash-{index}",
            "seed": "0",
            "strength": "0.23",
        }
        for index in range(23)
    ]
    with (run_root / "generated.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    app = _app(tmp_path, fid500_root=fid500_root, runs_root=runs_root)

    with TestClient(app) as client:
        response = client.get("/api/fid/images")
        original = client.get("/api/image", params={"path": str(input_path)})
        generated = client.get("/api/image", params={"path": str(output_path)})
        escaped = client.get("/api/image", params={"path": str(escape)})

    assert response.status_code == 200
    assert response.json() == {
        "run_id": FID_RUN_ID,
        "items": [dict(index=index + 1, **row) for index, row in enumerate(rows)],
    }
    assert original.status_code == 200
    assert original.content == b"original-image"
    assert generated.status_code == 200
    assert generated.content == b"generated-image"
    assert escaped.status_code == 403


def test_fid_images_requires_generated_manifest(tmp_path: Path) -> None:
    app = _app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/fid/images")
    assert response.status_code == 404


def test_fid_images_rebases_recorded_inputs_to_the_project_dataset(
    tmp_path: Path,
) -> None:
    fid500_root = tmp_path / "project-dataset/fid"
    _write_selection(fid500_root)
    local_input = fid500_root / "input/생활용품/15019.jpg"
    local_input.parent.mkdir(parents=True)
    local_input.write_bytes(b"local-copy")
    recorded_input = tmp_path / "external/fid500-v2/input/생활용품/15019.jpg"
    recorded_input.parent.mkdir(parents=True)
    recorded_input.write_bytes(b"external-copy")
    runs_root = tmp_path / "runs"
    run_root = runs_root / FID_RUN_ID
    output = run_root / "images/생활용품/15019.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")
    with (run_root / "generated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_path", "output_path", "sha256", "seed", "strength"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_path": str(recorded_input),
                "output_path": str(output),
                "sha256": "hash",
                "seed": "0",
                "strength": "0.15",
            }
        )
    app = _app(tmp_path, fid500_root=fid500_root, runs_root=runs_root)

    with TestClient(app) as client:
        response = client.get("/api/fid/images")

    assert response.status_code == 200
    assert response.json()["items"][0]["input_path"] == str(local_input.absolute())


def test_fid_run_rejects_completed_frozen_result_without_starting(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result_path = runs_root / FID_RUN_ID / "fid500.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({"run_id": FID_RUN_ID}) + "\n", encoding="utf-8")
    app = _app(tmp_path, runs_root=runs_root)
    started = False

    def capture(_strength: float) -> tuple[bool, str]:
        nonlocal started
        started = True
        return True, FID_RUN_ID

    app.state.run_manager.start_fid = capture
    with TestClient(app) as client:
        config = client.get("/api/fid/config")
        response = client.post("/api/fid/run")

    assert config.status_code == 200
    assert config.json()["completed"] is True
    assert config.json()["available"] is False
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]
    assert started is False


def test_fid_pilot_null_missing_and_malformed_are_reported_without_fallback(
    tmp_path: Path,
) -> None:
    cases = {
        "null": _write_pilot(tmp_path / "null.json", None),
        "missing": tmp_path / "missing.json",
        "malformed": tmp_path / "malformed.json",
    }
    cases["malformed"].write_text("{not-json", encoding="utf-8")

    for name, pilot_path in cases.items():
        case_root = tmp_path / name
        app = _app(case_root, pilot_path=pilot_path)
        started = False

        def capture(_strength: float) -> tuple[bool, str]:
            nonlocal started
            started = True
            return True, FID_RUN_ID

        app.state.run_manager.start_fid = capture
        with TestClient(app) as client:
            config = client.get("/api/fid/config")
            response = client.post("/api/fid/run")

        payload = config.json()
        assert config.status_code == 200
        assert payload["available"] is False
        assert payload["strength"] is None
        assert payload["source"] == str(pilot_path.absolute())
        assert payload["detail"]
        assert response.status_code == 503
        assert response.json()["detail"] == payload["detail"]
        assert started is False


def test_fid_command_uses_fixed_run_id_and_request_time_pilot_strength(
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
        [
            shlex.quote(sys.executable),
            shlex.quote(str(stub)),
            shlex.quote(str(marker)),
        ]
    )
    pilot_path = _write_pilot(tmp_path / "pilot.json", 0.19)
    app = _app(tmp_path, fid_cmd=command, pilot_path=pilot_path)
    _write_pilot(pilot_path, 0.31)

    with TestClient(app) as client:
        response = client.post("/api/fid/run")
        status = _wait_terminal(client)

    assert response.status_code == 200
    assert response.json() == {"run_id": FID_RUN_ID}
    assert status["run_id"] == FID_RUN_ID
    assert json.loads(marker.read_text(encoding="utf-8")) == [
        "--strength",
        "0.31",
        "--run-id",
        FID_RUN_ID,
    ]


def test_selected_fid_run_uses_selected_ids_even_when_canonical_result_exists(
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
    fid500_root = tmp_path / "fid500"
    _write_selection(fid500_root)
    _write_fid_items(fid500_root)
    runs_root = tmp_path / "runs"
    canonical = runs_root / FID_RUN_ID / "fid500.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("{}\n", encoding="utf-8")
    app = _app(
        tmp_path,
        fid500_root=fid500_root,
        runs_root=runs_root,
        fid_cmd=command,
    )

    with TestClient(app) as client:
        config = client.get("/api/fid/config")
        response = client.post(
            "/api/fid/try", json={"item_ids": ["15025", "25001"]}
        )
        status = _wait_terminal(client)

    assert config.status_code == 200
    assert config.json()["selected_available"] is True
    assert config.json()["selected_detail"] is None
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    assert run_id.startswith("fidtry-")
    assert status["run_id"] == run_id
    assert json.loads(marker.read_text(encoding="utf-8")) == [
        "--strength",
        "0.23",
        "--run-id",
        run_id,
        "--item-id",
        "15025",
        "--item-id",
        "25001",
    ]


def test_selected_fid_run_validates_selection(tmp_path: Path) -> None:
    fid500_root = tmp_path / "fid500"
    _write_selection(fid500_root)
    _write_fid_items(fid500_root)
    app = _app(tmp_path, fid500_root=fid500_root)

    with TestClient(app) as client:
        too_small = client.post("/api/fid/try", json={"item_ids": ["15021"]})
        duplicate = client.post(
            "/api/fid/try", json={"item_ids": ["15021", "15021"]}
        )
        unknown = client.post(
            "/api/fid/try", json={"item_ids": ["15021", "missing"]}
        )

    assert too_small.status_code == 400
    assert duplicate.status_code == 400
    assert unknown.status_code == 400


def test_selected_fid_result_and_images_are_scoped_to_selected_run(
    tmp_path: Path,
) -> None:
    fid500_root = tmp_path / "fid500"
    _write_selection(fid500_root)
    runs_root = tmp_path / "runs"
    run_id = "fidtry-selected"
    run_root = runs_root / run_id
    result = run_root / "fid500.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    input_path = fid500_root / "input/생활용품/15021.jpg"
    output_path = run_root / "images/생활용품/15021.png"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")
    with (run_root / "generated.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_path", "output_path", "sha256", "seed", "strength"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sha256": "hash",
                "seed": "0",
                "strength": "0.15",
            }
        )
    app = _app(tmp_path, fid500_root=fid500_root, runs_root=runs_root)

    with TestClient(app) as client:
        selected_result = client.get(
            "/api/fid/try/results", params={"run_id": run_id}
        )
        selected_images = client.get(
            "/api/fid/try/images", params={"run_id": run_id}
        )
        canonical = client.get(
            "/api/fid/try/results", params={"run_id": FID_RUN_ID}
        )

    assert selected_result.status_code == 200
    assert selected_result.json() == {"run_id": run_id}
    assert selected_images.status_code == 200
    assert selected_images.json()["run_id"] == run_id
    assert canonical.status_code == 404


def test_fid_eval_and_try_share_one_run_manager_lock(tmp_path: Path) -> None:
    stub = tmp_path / "slow.py"
    stub.write_text(
        "import time\nprint('generated=1/1', flush=True)\ntime.sleep(0.2)\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(stub))}"
    app = _app(tmp_path, eval_cmd=command, try_cmd=command, fid_cmd=command)

    with TestClient(app) as client:
        evaluation = client.post("/api/run", json={"strength": 0.23})
        blocked_fid = client.post("/api/fid/run")
        _wait_terminal(client)
        fid = client.post("/api/fid/run")
        blocked_try = client.post("/api/try", json={"item_ids": ["ITEM0"]})
        _wait_terminal(client)

    assert evaluation.status_code == 200
    assert blocked_fid.status_code == 409
    assert blocked_fid.json() == {"run_id": evaluation.json()["run_id"]}
    assert fid.status_code == 200
    assert fid.json() == {"run_id": FID_RUN_ID}
    assert blocked_try.status_code == 409
    assert blocked_try.json() == {"run_id": FID_RUN_ID}


def test_fid_static_page_uses_dynamic_image_pairs_and_numeric_pages() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/static/index.html").read_text(encoding="utf-8")
    javascript = (root / "web/static/fid.js").read_text(encoding="utf-8")
    css = (root / "web/static/fid.css").read_text(encoding="utf-8")
    fid_view = html.split('id="view-fid"', 1)[1].split('id="view-fidv2"', 1)[0]

    assert "<img" not in fid_view.lower()
    assert fid_view.lower().count("<button") == 3
    assert "<input" not in fid_view.lower()
    assert "<select" not in fid_view.lower()
    assert "/api/image" not in fid_view
    assert 'apiUrl("/api/image", {path})' in javascript
    assert "8013" not in html + javascript
    assert "5186" not in html + javascript
    assert html.count('src="/config.js"') == 1
    assert javascript.count('fetch(apiUrl("/api/status")') == 1
    assert 'fetch(apiUrl("/api/fid/dataset/items")' in javascript
    assert 'fetch(apiUrl("/api/fid/config")' in javascript
    assert 'fetch(apiUrl("/api/fid/results")' not in javascript
    assert 'fetch(apiUrl("/api/fid/images")' not in javascript
    assert 'fetch(apiUrl("/api/fid/run")' not in javascript
    assert 'fetch(apiUrl("/api/fid/dataset/items")' in javascript
    assert 'fetch(apiUrl("/api/fid/try")' in javascript
    assert 'fetch(apiUrl("/api/fid/try/results"' in javascript
    assert 'fetch(apiUrl("/api/fid/try/images"' in javascript
    for element_id in (
        "fid-selection-count",
        "fid-selection-fill",
        "fid-selection-reset",
        "fid-tree-groups",
    ):
        assert f'id="{element_id}"' in html
    assert "장별 FID 평균이 아닌 선택 집합의 단일 FID" in html
    for element_id in (
        "fid-progress",
        "fid-progress-stage",
        "fid-progress-count",
        "fid-progress-percent",
        "fid-progress-track",
        "fid-progress-fill",
    ):
        assert f'id="{element_id}"' in html
    assert "PERF_RUN_PROGRESS.snapshot" in javascript
    assert 'setAttribute("aria-valuenow", String(view.percent))' in javascript
    assert '`${view.percent}%`' in javascript
    assert "13.402" not in html + javascript
    assert 'tolerance.style.setProperty("--tick", `${tick}%`)' in javascript
    assert 'scale.style.setProperty("--tick", `${tick}%`)' in javascript
    assert "left: var(--tick);" in css
    assert "const FID_PAGE_SIZE = 10" in javascript
    assert "renderImagePage" in javascript
    assert "renderPageButtons" in javascript
    assert 'button.textContent = String(index + 1)' in javascript
    assert 'button.setAttribute("aria-current", "page")' in javascript


def test_metric_pages_are_integrated_behind_one_top_switch() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/static/index.html").read_text(encoding="utf-8")
    legacy = (root / "web/static/fid.html").read_text(encoding="utf-8")
    switch = (root / "web/static/view-switch.js").read_text(encoding="utf-8")

    assert 'id="metric-view-switch"' in html
    assert 'data-metric-view="pair"' in html
    assert 'data-metric-view="fid"' in html
    assert 'data-metric-view="fidv2"' in html
    assert 'id="view-pair"' in html
    assert 'id="view-fid"' in html
    assert 'id="view-fidv2"' in html
    assert 'id="view-fid" class="metric-view" hidden' in html
    assert 'href="fid.css"' in html
    assert 'src="app.js"' in html
    assert 'src="fid.js"' in html
    assert 'src="fidv2.js"' in html
    assert 'src="view-switch.js"' in html
    assert 'location.replace("/?view=fid")' in legacy
    assert 'pathname.endsWith("/fid.html")' in switch
    assert 'searchParams.get("view")' in switch
    assert 'button.setAttribute("aria-selected", String(selected))' in switch
    assert 'panel.hidden = panel.dataset.metricPanel !== view' in switch

    ids = [part.split('"', 1)[0] for part in html.split('id="')[1:]]
    assert len(ids) == len(set(ids))


def test_metric_view_switch_resolves_default_query_and_legacy_route() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
const helpers = require("./web/static/view-switch.js");
console.log(JSON.stringify([
  helpers.resolveView("/", ""),
  helpers.resolveView("/", "?view=fid"),
  helpers.resolveView("/", "?view=fidv2"),
  helpers.resolveView("/fid.html", ""),
  helpers.resolveView("/", "?view=unknown"),
]));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["pair", "fid", "fidv2", "fid", "pair"]


def test_fid_image_pagination_shows_ten_and_clamps_to_last_page() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
const helpers = require("./web/static/fid.js");
const items = Array.from({length: 23}, (_, index) => ({index: index + 1}));
console.log(JSON.stringify({
  first: helpers.paginateItems(items, 0, 10),
  last: helpers.paginateItems(items, 99, 10),
}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert [item["index"] for item in payload["first"]["items"]] == list(range(1, 11))
    assert payload["first"]["pageCount"] == 3
    assert [item["index"] for item in payload["last"]["items"]] == [21, 22, 23]
    assert payload["last"] == {
        "items": [{"index": 21}, {"index": 22}, {"index": 23}],
        "page": 2,
        "pageCount": 3,
        "start": 21,
        "end": 23,
        "total": 23,
    }


def test_fid_result_is_invalidated_when_selection_changes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
const helpers = require("./web/static/fid.js");
console.log(JSON.stringify([
  helpers.selectionMatches(["15021", "15025"], ["15025", "15021"]),
  helpers.selectionMatches(["15021"], ["15021", "15025"]),
]));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [True, False]

    javascript = (root / "web/static/fid.js").read_text(encoding="utf-8")
    assert "PERF_FID_VIEW.selectionMatches(" in javascript
    assert "현재 선택으로 측정 실행 필요" in javascript
