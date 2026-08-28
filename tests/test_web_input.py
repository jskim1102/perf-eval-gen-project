from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from web.server import create_app


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    curated_root = tmp_path / "curated"
    eval_root = curated_root / "eval500"
    source_root = curated_root / "input"
    rows: list[dict[str, str]] = []
    for index, group in enumerate(("가구", "가구", "조명")):
        source_path = source_root / group / f"source-{index}.jpg"
        selected_path = eval_root / "input" / group / f"TYPE{index}__IMAGE{index}.jpg"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (50 + index * 30, 80, 110)).save(source_path)
        selected_path.symlink_to(source_path)
        rows.append(
            {
                "split": "input",
                "group": group,
                "product_type": f"TYPE{index}",
                "item_id": f"ITEM{index}",
                "image_id": f"IMAGE{index}",
                "width": "32",
                "height": "32",
                "sha256": _sha256(source_path),
                "source_path": str(source_path),
                "selected_path": f"input/{group}/{selected_path.name}",
            }
        )
    manifests = eval_root / "manifests"
    manifests.mkdir(parents=True)
    for name, selected_rows in (
        ("input.csv", rows),
        ("psnr_ssim_100.csv", [rows[0], rows[2]]),
    ):
        with (manifests / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(selected_rows)
    (manifests / "protocol.json").write_text(
        json.dumps({"selection": {"input_count": 3}}, ensure_ascii=False),
        encoding="utf-8",
    )
    return eval_root, rows


def _wait_terminal(client: TestClient, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        status = client.get("/api/status").json()
        if status["state"] in {"done", "error"}:
            return status
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {status}")


def _pilot(tmp_path: Path, selected: float | None) -> Path:
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    return pilot_path


def _selection_helper_case(
    checked_indices: list[int],
    action: str,
    *,
    box_count: int = 12,
    limit: int = 500,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    script = f"""
const helpers = require("./web/static/app.js");
const initiallyChecked = new Set({json.dumps(checked_indices)});
const groups = [{{open: false}}, {{open: false}}];
const groupSize = Math.ceil({box_count} / groups.length);
const boxes = Array.from({{length: {box_count}}}, (_, index) => ({{
  value: `ITEM${{index}}`,
  checked: initiallyChecked.has(index),
  disabled: !initiallyChecked.has(index),
  closest(selector) {{ return selector === "details" ? groups[Math.floor(index / groupSize)] : null; }},
}}));
if ({json.dumps(action)} === "reset") helpers.resetSelection(boxes);
if ({json.dumps(action)} === "fill") helpers.fillSelection(boxes, {limit});
const view = helpers.selectionSnapshot(boxes, {limit}, false, true);
const lockedView = helpers.selectionSnapshot(boxes, {limit}, true, true);
const unreadyView = helpers.selectionSnapshot(boxes, {limit}, false, false);
console.log(JSON.stringify({{
  checked: boxes.map((box, index) => box.checked ? index : null).filter((value) => value !== null),
  groups: groups.map((group) => group.open),
  view,
  lockedView,
  unreadyView,
}}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _pagination_case(total: int, page: int) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    script = f"""
const helpers = require("./web/static/app.js");
const items = Array.from({{length: {total}}}, (_, index) => ({{item_id: `ITEM${{index}}`}}));
const view = helpers.paginateItems(items, {page}, 10);
console.log(JSON.stringify({{
  itemIds: view.items.map((item) => item.item_id),
  page: view.page,
  pageCount: view.pageCount,
  start: view.start,
  end: view.end,
  total: view.total,
}}));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _status_failure_case(
    status: dict[str, object],
    previous_state: str,
    previous_run_id: str | None,
) -> dict[str, str] | None:
    root = Path(__file__).resolve().parents[1]
    script = f"""
const helpers = require("./web/static/app.js");
console.log(JSON.stringify(helpers.statusFailure(
  {json.dumps(status)}, {json.dumps(previous_state)}, {json.dumps(previous_run_id)},
)));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_dataset_items_uses_manifest_fields_and_actual_group_counts(tmp_path: Path) -> None:
    eval_root, rows = _dataset(tmp_path)
    app = create_app(runs_root=tmp_path / "runs", eval500_root=eval_root)

    with TestClient(app) as client:
        response = client.get("/api/dataset/items")
        image = client.get("/api/image", params={"path": rows[0]["source_path"]})

    assert response.status_code == 200
    payload = response.json()
    assert payload["groups"] == [
        {"group": "가구", "count": 2, "pair_count": 1},
        {"group": "조명", "count": 1, "pair_count": 1},
    ]
    assert payload["items"] == [
        {
            "item_id": row["item_id"],
            "group": row["group"],
            "product_type": row["product_type"],
            "image_id": row["image_id"],
            "input_path": row["source_path"],
            "in_pair_set": row["item_id"] in {"ITEM0", "ITEM2"},
        }
        for row in rows
    ]
    assert all(list(item) == [
        "item_id", "group", "product_type", "image_id", "input_path", "in_pair_set"
    ] for item in payload["items"])
    assert image.status_code == 200


def test_try_api_validates_limit_shares_lock_and_separates_results(tmp_path: Path) -> None:
    eval_root, _ = _dataset(tmp_path)
    runs_root = tmp_path / "runs"
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import time\ntime.sleep(0.25)\nprint('generated=1/1', flush=True)\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(stub))}"
    pilot_path = _pilot(tmp_path, 0.15)
    app = create_app(
        runs_root=runs_root,
        eval500_root=eval_root,
        eval_cmd=command,
        try_cmd=command,
        pilot_path=pilot_path,
    )

    with TestClient(app) as client:
        empty = client.post("/api/try", json={"strength": 0.15, "item_ids": []})
        too_many = client.post(
            "/api/try",
            json={"strength": 0.15, "item_ids": [f"ITEM{i}" for i in range(501)]},
        )
        main = client.post("/api/run", json={"strength": 0.15})
        blocked_try = client.post(
            "/api/try", json={"strength": 0.15, "item_ids": ["ITEM0"]}
        )
        _wait_terminal(client)
        trial = client.post(
            "/api/try", json={"strength": 0.15, "item_ids": ["ITEM0"]}
        )
        blocked_main = client.post("/api/run", json={"strength": 0.15})
        _wait_terminal(client)

        trial_id = trial.json()["run_id"]
        trial_root = runs_root / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        (trial_root / "try.json").write_text('{"run_id":"try-test"}\n')
        (trial_root / "results.json").write_text('{"run_id":"wrong"}\n')
        try_result = client.get("/api/try/results", params={"run_id": trial_id})
        canonical_result = client.get("/api/results", params={"run_id": trial_id})

    assert empty.status_code == 400
    assert too_many.status_code == 400
    assert main.status_code == 200
    assert blocked_try.status_code == 409
    assert blocked_try.json()["run_id"] == main.json()["run_id"]
    assert trial.status_code == 200
    assert trial_id.startswith("try-")
    assert blocked_main.status_code == 409
    assert blocked_main.json()["run_id"] == trial_id
    assert try_result.status_code == 200
    assert canonical_result.status_code == 404


def test_try_api_accepts_100_items_and_rejects_101(tmp_path: Path) -> None:
    from scripts.try_generate import MAX_TRY_ITEMS as CLI_MAX_TRY_ITEMS
    from web.server import MAX_TRY_ITEMS as API_MAX_TRY_ITEMS

    eval_root, _ = _dataset(tmp_path)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval_root,
        try_cmd="configured",
        pilot_path=_pilot(tmp_path, 0.15),
    )
    observed: dict[str, object] = {}

    def capture(strength: float, item_ids: list[str]) -> tuple[bool, str]:
        observed.update(strength=strength, item_ids=item_ids)
        return True, "try-100"

    app.state.run_manager.start_try = capture
    selected = [f"ITEM{index}" for index in range(100)]

    with TestClient(app) as client:
        accepted = client.post("/api/try", json={"item_ids": selected})
        rejected = client.post("/api/try", json={"item_ids": [*selected, "ITEM100"]})

    assert API_MAX_TRY_ITEMS == CLI_MAX_TRY_ITEMS == 100
    assert accepted.status_code == 200
    assert observed == {"strength": 0.15, "item_ids": selected}
    assert rejected.status_code == 400


def test_try_request_strength_is_ignored_in_favor_of_pilot(tmp_path: Path) -> None:
    eval_root, _ = _dataset(tmp_path)
    pilot_path = _pilot(tmp_path, 0.23)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval_root,
        try_cmd="configured",
        pilot_path=pilot_path,
    )
    observed: dict[str, object] = {}

    def capture(strength: float, item_ids: list[str]) -> tuple[bool, str]:
        observed.update(strength=strength, item_ids=item_ids)
        return True, "try-captured"

    app.state.run_manager.start_try = capture

    with TestClient(app) as client:
        response = client.post(
            "/api/try",
            json={"strength": 0.91, "item_ids": ["ITEM0"]},
        )

    assert response.status_code == 200
    assert observed == {"strength": 0.23, "item_ids": ["ITEM0"]}


def test_try_default_pilot_path_matches_run_eval_contract() -> None:
    from scripts.run_eval import DEFAULT_PILOT_PATH as EVALUATION_PILOT_PATH
    from web.server import DEFAULT_PILOT_PATH as WEB_PILOT_PATH

    assert WEB_PILOT_PATH == EVALUATION_PILOT_PATH


def test_try_config_and_execution_share_injected_pilot_selected(
    tmp_path: Path,
) -> None:
    eval_root, _ = _dataset(tmp_path)
    pilot_path = _pilot(tmp_path, 0.31)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval_root,
        try_cmd="configured",
        pilot_path=pilot_path,
    )
    observed: dict[str, object] = {}

    def capture(strength: float, item_ids: list[str]) -> tuple[bool, str]:
        observed.update(strength=strength, item_ids=item_ids)
        return True, "try-captured"

    app.state.run_manager.start_try = capture
    _pilot(tmp_path, 0.37)

    with TestClient(app) as client:
        config = client.get("/api/try/config")
        response = client.post("/api/try", json={"item_ids": ["ITEM1"]})

    assert config.status_code == 200
    assert config.json() == {
        "strength": 0.37,
        "source": str(pilot_path.absolute()),
        "detail": None,
    }
    assert response.status_code == 200
    assert observed == {"strength": 0.37, "item_ids": ["ITEM1"]}


def test_try_rejects_null_pilot_selected_without_starting(tmp_path: Path) -> None:
    eval_root, _ = _dataset(tmp_path)
    pilot_path = _pilot(tmp_path, None)
    app = create_app(
        runs_root=tmp_path / "runs",
        eval500_root=eval_root,
        try_cmd="configured",
        pilot_path=pilot_path,
    )
    started = False

    def capture(strength: float, item_ids: list[str]) -> tuple[bool, str]:
        nonlocal started
        started = True
        return True, "try-should-not-start"

    app.state.run_manager.start_try = capture

    with TestClient(app) as client:
        config = client.get("/api/try/config")
        response = client.post("/api/try", json={"item_ids": ["ITEM0"]})

    assert config.status_code == 200
    assert config.json()["strength"] is None
    assert "selected is null" in config.json()["detail"]
    assert response.status_code == 503
    assert "selected is null" in response.json()["detail"]
    assert started is False


def test_try_reports_missing_and_malformed_pilot_without_fallback(
    tmp_path: Path,
) -> None:
    eval_root, _ = _dataset(tmp_path)
    missing = tmp_path / "missing-pilot.json"
    malformed = tmp_path / "malformed-pilot.json"
    malformed.write_text("{not-json", encoding="utf-8")

    for pilot_path in (missing, malformed):
        app = create_app(
            runs_root=tmp_path / "runs",
            eval500_root=eval_root,
            try_cmd="configured",
            pilot_path=pilot_path,
        )
        with TestClient(app) as client:
            config = client.get("/api/try/config")
            response = client.post("/api/try", json={"item_ids": ["ITEM0"]})

        assert config.status_code == 200
        assert config.json()["strength"] is None
        assert config.json()["detail"]
        assert response.status_code == 503
        assert response.json()["detail"] == config.json()["detail"]


def test_selection_reset_restores_empty_state_and_controls() -> None:
    result = _selection_helper_case([0, 4, 11], "reset")

    assert result["checked"] == []
    assert result["view"]["selectedCount"] == 0
    assert result["view"]["countText"] == "선택 0 / 최대 500장"
    assert result["view"]["etaText"] == "≈ 0초"
    assert result["view"]["executeDisabled"] is True
    assert result["view"]["checkboxDisabled"] == [False] * 12
    assert result["lockedView"]["helperDisabled"] is True


def test_selection_fill_uses_dom_order_and_stops_at_limit() -> None:
    result = _selection_helper_case([], "fill", box_count=501)

    assert result["checked"] == list(range(500))
    assert result["view"]["selectedCount"] == 500
    assert result["view"]["checkboxDisabled"] == [False] * 500 + [True]
    assert result["groups"] == [False, False]
    assert result["unreadyView"]["helperDisabled"] is False
    assert result["unreadyView"]["executeDisabled"] is True


def test_selection_fill_preserves_existing_items_before_adding() -> None:
    result = _selection_helper_case([500], "fill", box_count=501)

    assert result["checked"] == [*range(499), 500]
    assert 500 in result["checked"]
    assert result["view"]["selectedCount"] == 500


def test_selection_fill_at_limit_does_not_change_selection() -> None:
    initial = list(range(500))
    result = _selection_helper_case(initial, "fill", box_count=501)

    assert result["checked"] == initial
    assert result["view"]["selectedCount"] == 500


def test_try_result_pagination_uses_ten_items_and_clamps_page() -> None:
    first = _pagination_case(23, 0)
    second = _pagination_case(23, 1)
    last = _pagination_case(23, 99)

    assert first == {
        "itemIds": [f"ITEM{index}" for index in range(10)],
        "page": 0,
        "pageCount": 3,
        "start": 1,
        "end": 10,
        "total": 23,
    }
    assert second["itemIds"] == [f"ITEM{index}" for index in range(10, 20)]
    assert (second["start"], second["end"]) == (11, 20)
    assert last["itemIds"] == ["ITEM20", "ITEM21", "ITEM22"]
    assert (last["page"], last["start"], last["end"]) == (2, 21, 23)


def test_shared_run_progress_reaches_100_only_after_done() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
const helpers = require("./web/static/progress.js");
console.log(JSON.stringify([
  helpers.snapshot({state: "running", done: 0, total: 0}, 100),
  helpers.snapshot({state: "generating", done: 50, total: 100}, 100),
  helpers.snapshot({state: "measuring", done: 100, total: 100}, 100),
  helpers.snapshot({state: "done", done: 100, total: 100}, 100),
  helpers.snapshot({state: "error", done: 25, total: 100}, 100),
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
    payload = json.loads(completed.stdout)
    assert [entry["percent"] for entry in payload] == [0, 50, 99, 100, 25]
    assert payload[0]["total"] == 100
    assert payload[3]["done"] == 100


def test_error_status_message_includes_cause_and_is_deduplicated() -> None:
    javascript = (
        Path(__file__).resolve().parents[1] / "web/static/app.js"
    ).read_text(encoding="utf-8")
    status = {
        "state": "error",
        "run_id": "try-failed",
        "log": ["generated 2/3", "process exited with code 1"],
    }

    first = _status_failure_case(status, "running", "try-failed")
    repeated = _status_failure_case(status, "error", "try-failed")

    assert first == {
        "runId": "try-failed",
        "message": "실행 실패 · run_id try-failed · process exited with code 1",
    }
    assert repeated is None
    assert "PERF_EVAL_SELECTION.statusFailure(" in javascript
    assert "if (failure) showAlert(failure.message);" in javascript


def test_selection_result_is_invalidated_when_current_selection_changes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
const helpers = require("./web/static/app.js");
console.log(JSON.stringify([
  helpers.selectionMatches(["A", "B"], ["B", "A"]),
  helpers.selectionMatches(["A"], ["A", "B"]),
  helpers.resultMatchesSelection(
    {run_id: "try-current", items: [{item_id: "A"}, {item_id: "B"}]},
    ["B", "A"],
    "try-current",
  ),
  helpers.resultMatchesSelection(
    {run_id: "try-old", items: [{item_id: "A"}, {item_id: "B"}]},
    ["A", "B"],
    "try-current",
  ),
]));
"""
    completed = subprocess.run(
        ["node", "-e", script], cwd=root, text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [True, False, True, False]

    javascript = (root / "web/static/app.js").read_text(encoding="utf-8")
    assert "PERF_EVAL_SELECTION.selectionMatches(" in javascript
    assert "PERF_EVAL_SELECTION.resultMatchesSelection(" in javascript
    assert "현재 선택으로 측정 실행 필요" in javascript


def test_ssim_scale_label_uses_same_two_thirds_position_as_target_tick() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/static/index.html").read_text(encoding="utf-8")
    css = (root / "web/static/style.css").read_text(encoding="utf-8")
    ssim_metric = html.split('id="metric-ssim"', 1)[1].split("</article>", 1)[0]
    psnr_metric = html.split('id="metric-psnr"', 1)[1].split("</article>", 1)[0]

    assert 'class="tolerance-scale tolerance-scale-ssim"' in ssim_metric
    assert 'class="tolerance-scale tolerance-scale-ssim"' not in psnr_metric
    assert ".tolerance-scale-ssim span:nth-child(2)" in css
    assert "left: 66.7%;" in css


@dataclass(frozen=True)
class _GenerationReport:
    generated_manifest: Path


def test_try_cli_uses_generator_selection_and_writes_only_psnr_ssim(tmp_path: Path) -> None:
    from scripts.generate import InputRecord
    from scripts.try_generate import run_try

    eval_root, rows = _dataset(tmp_path)
    prompt_manifest = eval_root / "manifests" / "prompts.csv"
    with prompt_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["item_id", "prompt", "name_source"]
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "item_id": row["item_id"],
                    "prompt": f"per-image prompt {index}",
                    "name_source": "en",
                }
            )
    runs_root = tmp_path / "runs"
    observed: dict[str, object] = {}

    def fake_generation_runner(**kwargs):
        observed.update(kwargs)
        run_root = runs_root / kwargs["config"].run_id
        images_root = run_root / "images"
        images_root.mkdir(parents=True)
        generated_rows = []
        for row in (rows[0], rows[2]):
            input_path = eval_root / row["selected_path"]
            output_path = images_root / f"{row['item_id']}.png"
            with Image.open(input_path) as source:
                source.convert("RGB").point(lambda value: min(255, value + 5)).save(output_path)
            generated_rows.append({
                "input_path": str(input_path.absolute()),
                "output_path": str(output_path.absolute()),
                "sha256": _sha256(output_path),
                "seed": "0",
                "strength": "0.15",
            })
        manifest = run_root / "generated.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GENERATED_FIELDS)
            writer.writeheader()
            writer.writerows(generated_rows)
        return _GenerationReport(generated_manifest=manifest)

    payload = run_try(
        run_id="try-unit",
        item_ids=["ITEM0", "ITEM2"],
        strength=0.15,
        runs_root=runs_root,
        input_manifest=eval_root / "manifests" / "input.csv",
        prompt_manifest=prompt_manifest,
        generation_runner=fake_generation_runner,
    )

    assert observed["item_ids"] == ["ITEM0", "ITEM2"]
    assert observed["prompt_resolver"](
        InputRecord(
            input_path=(eval_root / rows[2]["selected_path"]).absolute(),
            output_relative_path=Path("unused.png"),
        )
    ) == "per-image prompt 2"
    assert payload == json.loads((runs_root / "try-unit" / "try.json").read_text())
    assert payload["run_id"] == "try-unit"
    assert payload["strength"] == 0.15
    assert payload["targets"] == {"psnr": 25.0, "ssim": 0.9}
    assert [item["item_id"] for item in payload["items"]] == ["ITEM0", "ITEM2"]
    assert all(set(item) == {
        "item_id", "group", "product_type", "image_id", "input_path",
        "output_path", "psnr", "ssim"
    } for item in payload["items"])
    assert set(payload["metrics"]) == {"psnr", "ssim"}
    assert math.isclose(
        payload["metrics"]["psnr"]["mean"],
        sum(item["psnr"] for item in payload["items"]) / len(payload["items"]),
    )
    assert math.isclose(
        payload["metrics"]["ssim"]["mean"],
        sum(item["ssim"] for item in payload["items"]) / len(payload["items"]),
    )
    assert "fid" not in json.dumps(payload).lower()


def test_input_frontend_contains_approved_paths_without_mock_only_copy() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "web/static/index.html").read_text(encoding="utf-8")
    css = (root / "web/static/style.css").read_text(encoding="utf-8")
    js = (root / "web/static/app.js").read_text(encoding="utf-8")
    pair_view = html.split('id="view-pair"', 1)[1].split('id="view-fid"', 1)[0]

    for removed_fixed_ui in (
        "읽기 전용 · 고정",
        "생성 전 고정 · 사후 재선별 금지",
        "체크는 시험해 볼 장을 고르는 것이며 측정 대상을 바꾸지 않는다.",
        "본 측정 PSNR·SSIM 고정 부분집합",
    ):
        assert removed_fixed_ui not in pair_view + js
    assert 'class="group-pair"' not in js
    assert 'class="item-star"' not in js
    assert ".group-pair" not in css
    assert ".item-star" not in css

    for element_id in ("dataset-tree", "try-result"):
        assert f'id="{element_id}"' in html
    for element_id in ("main-run", "progress", "strength", "run"):
        assert f'id="{element_id}"' not in html
    assert "500장 본 측정" not in html + js
    assert ">진행<" not in html
    assert "try-strength" not in html + js
    assert "tryStrength" not in js
    assert "0.15" not in html + js
    for element_id in (
        "try-progress",
        "try-progress-stage",
        "try-progress-count",
        "try-progress-percent",
        "try-progress-track",
        "try-progress-fill",
    ):
        assert f'id="{element_id}"' in html
    assert 'src="progress.js"' in html
    assert "PERF_RUN_PROGRESS.snapshot" in js
    assert 'setAttribute("aria-valuenow", String(view.percent))' in js
    assert '`${view.percent}%`' in js
    assert 'id="selection-count" class="meta-inline" aria-live="polite"' in html
    assert 'id="selection-fill"' in html
    assert 'id="selection-reset"' in html
    assert "선택 초기화" in html
    assert "최대 10장" not in html + js
    assert 'selectionFillButton.textContent = "모두 선택"' in js
    assert "const MAX_TRY_ITEMS = 100" in js
    assert js.count("const MAX_TRY_ITEMS =") == 1
    assert "fillSelection(checkboxes, MAX_TRY_ITEMS)" in js
    assert "resetSelection(checkboxes)" in js
    for element_id in (
        "try-pagination",
        "try-page-prev",
        "try-page-status",
        "try-page-next",
    ):
        assert f'id="{element_id}"' in html
    assert "const TRY_PAGE_SIZE = 10" in js
    assert "paginateItems" in js
    assert "result.metrics?.psnr?.mean" in js
    assert "result.metrics?.ssim?.mean" in js
    assert 'fetch(apiUrl("/api/try/config")' in js
    assert "측정값이 아님" not in html + js
    assert ".state-specimen" not in css
    assert ".specimen-label" not in css
    assert "columns: 2" in css
    assert 'startsWith("try-")' in js
    assert "renderSelectedResults(result)" in js
    assert "items.map((item) => item.psnr)" in js
    assert "items.map((item) => item.ssim)" in js
    assert 'fetch(apiUrl("/api/results"' not in js
    assert "loadResults(" not in js
    assert 'clearSelectedResults("선택 데이터 측정 중")' in js
    assert 'clearSelectedResults("선택 실행 대기")' in js
    assert "시험 결과" in html
    assert "product_type" in js and "image_id" in js
    assert 'fetch(apiUrl("/api/try")' in js
    assert "JSON.stringify({item_ids: itemIds})" in js
    assert 'fetch(apiUrl("/api/run")' not in js
    assert 'fetch(apiUrl("/api/status")' in js
    for removed_symbol in (
        "mainStrength",
        "runButton",
        "startRun",
        "statePill",
        "progressPanel",
        "progressStage",
        "progressTrack",
        "progressFill",
        "progressCount",
        "logTail",
    ):
        assert removed_symbol not in js
    for removed_selector in (
        ".panel-heavy",
        ".heavy-row",
        ".btn-run",
        ".state-pill",
        ".progress-row",
        ".progress-track",
        ".progress-fill",
        ".log-tail",
    ):
        assert removed_selector not in css
    assert ".btn-try" in css
    assert "fid" not in html.split('id="try-result"', 1)[1].split("</section>", 1)[0].lower()
