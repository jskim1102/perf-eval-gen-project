#!/usr/bin/env python3
"""Verify that the web wrapper executes the same real CLI as direct use."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.frontend import handler_factory
from web.server import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL500_ROOT = Path("/home/kim_3090/datasets/abo/curated/eval500")


class WiringVerificationError(RuntimeError):
    """The web path diverged from the canonical CLI path."""


def compare_results(web_result: dict[str, Any], cli_result: dict[str, Any]) -> None:
    if web_result.get("protocol") != cli_result.get("protocol"):
        raise WiringVerificationError("web and CLI protocol values differ")
    if web_result.get("metrics") != cli_result.get("metrics"):
        raise WiringVerificationError("web and CLI metrics differ")


def _wait_terminal(client: TestClient, *, timeout: float = 600.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get("/api/status")
        response.raise_for_status()
        last = response.json()
        if last["state"] in {"done", "error"}:
            return last
        time.sleep(0.2)
    raise WiringVerificationError(f"web evaluation did not finish: {last}")


_JS_IDENTIFIER = r"[A-Za-z_$][\w$]*"


def _function_body(javascript: str, name: str) -> tuple[str, str]:
    match = re.search(
        rf"(?:async\s+)?function\s+{re.escape(name)}\s*"
        rf"\(\s*(?P<parameter>{_JS_IDENTIFIER})?\s*\)\s*\{{",
        javascript,
    )
    if match is None:
        raise WiringVerificationError(f"frontend function is missing: {name}")
    opening_brace = match.end() - 1
    depth = 0
    for index in range(opening_brace, len(javascript)):
        if javascript[index] == "{":
            depth += 1
        elif javascript[index] == "}":
            depth -= 1
            if depth == 0:
                return match.group("parameter") or "", javascript[opening_brace + 1 : index]
    raise WiringVerificationError(f"frontend function is unterminated: {name}")


def _assert_frontend_contract(javascript: str | None = None) -> None:
    if javascript is None:
        javascript = (PROJECT_ROOT / "web/static/app.js").read_text(encoding="utf-8")
    config = handler_factory(8013).config_body.decode("utf-8")

    status_parameter, status_body = _function_body(javascript, "applyStatus")
    if not re.search(r"\.startsWith\([\"']try-[\"']\)", status_body):
        raise WiringVerificationError("try run_id detection is missing")
    if "loadTryResults(statusRunId)" not in status_body:
        raise WiringVerificationError("completed trial result adoption is missing")

    _, trial_body = _function_body(javascript, "loadTryResults")
    if 'apiUrl("/api/try/results", {run_id: runId})' not in trial_body:
        raise WiringVerificationError("trial results API request is missing")
    if "renderSelectedResults(result)" not in trial_body:
        raise WiringVerificationError("selected result summary rendering is missing")
    if 'fetch(apiUrl("/api/results"' in javascript or "function loadResults(" in javascript:
        raise WiringVerificationError("canonical result still drives the frontend summary")
    if 'clearSelectedResults("선택 실행 대기")' not in javascript:
        raise WiringVerificationError("initial selected-result empty state is missing")
    if '|| "fixture"' in javascript or "initialRunId" in config:
        raise WiringVerificationError("fixture remains an implicit initial result")


def _verify_bad_command(runs_root: Path, strength: float) -> dict[str, Any]:
    app = create_app(
        runs_root=runs_root,
        eval500_root=EVAL500_ROOT,
        eval_cmd="'unterminated",
    )
    with TestClient(app) as client:
        first = client.post("/api/run", json={"strength": strength})
        if first.status_code != 200:
            raise WiringVerificationError("bad-command request was not accepted for execution")
        status = _wait_terminal(client, timeout=10.0)
        if status["state"] != "error":
            raise WiringVerificationError("bad EVAL_CMD did not transition to error")
        if not any("failed to start evaluation" in line for line in status["log"]):
            raise WiringVerificationError("bad EVAL_CMD error cause is absent from the log")
        retry = client.post("/api/run", json={"strength": strength})
        if retry.status_code != 200:
            raise WiringVerificationError("run lock stayed active after command construction error")
        _wait_terminal(client, timeout=10.0)
    return status


def _run_web(*, runs_root: Path, strength: float, limit: int) -> dict[str, Any]:
    eval_cmd = shlex.join(
        [sys.executable, str(PROJECT_ROOT / "scripts/run_eval.py"), "--limit", str(limit)]
    )
    app = create_app(
        runs_root=runs_root,
        eval500_root=EVAL500_ROOT,
        eval_cmd=eval_cmd,
    )
    with TestClient(app) as client:
        response = client.post("/api/run", json={"strength": strength})
        if response.status_code != 200:
            raise WiringVerificationError(f"web run request failed: {response.text}")
        run_id = response.json()["run_id"]

        # A browser reload asks status again; the server-owned run_id must remain
        # sufficient for the newly loaded frontend to adopt the active result.
        reloaded_status = client.get("/api/status").json()
        if reloaded_status.get("run_id") != run_id:
            raise WiringVerificationError("status did not preserve run_id across reload")

        status = _wait_terminal(client)
        if status["state"] != "done":
            raise WiringVerificationError(
                "web evaluation failed: " + " | ".join(status.get("log", [])[-5:])
            )
        if status["done"] != limit or status["total"] != limit:
            raise WiringVerificationError(
                f"generation progress differs: {status['done']}/{status['total']}"
            )
        result_response = client.get("/api/results", params={"run_id": run_id})
        if result_response.status_code != 200:
            raise WiringVerificationError("completed web run has no results.json")
        result = result_response.json()
    if result["run_id"] != run_id:
        raise WiringVerificationError("web result run_id differs from status run_id")
    return result


def _run_cli(*, runs_root: Path, strength: float, limit: int) -> dict[str, Any]:
    run_id = f"cli-wire-{uuid.uuid4().hex[:12]}"
    environment = dict(os.environ, RUNS_ROOT=str(runs_root))
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_eval.py"),
            "--strength",
            str(strength),
            "--limit",
            str(limit),
            "--run-id",
            run_id,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WiringVerificationError(
            f"direct CLI exited with code {completed.returncode}"
        )
    return json.loads(
        (runs_root / run_id / "results.json").read_text(encoding="utf-8")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--compare-cli", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.strength <= 1.0:
        raise ValueError("--strength must be in (0, 1]")
    if not 2 <= args.limit <= 100:
        raise ValueError("--limit must be between 2 and 100")
    runs_root_value = os.environ.get("RUNS_ROOT")
    if not runs_root_value:
        raise RuntimeError("RUNS_ROOT environment variable is required")
    runs_root = Path(runs_root_value).expanduser().absolute()

    _assert_frontend_contract()
    bad_status = _verify_bad_command(runs_root, args.strength)
    web_result = _run_web(
        runs_root=runs_root,
        strength=args.strength,
        limit=args.limit,
    )
    cli_result = None
    if args.compare_cli:
        cli_result = _run_cli(
            runs_root=runs_root,
            strength=args.strength,
            limit=args.limit,
        )
        compare_results(web_result, cli_result)

    print(
        json.dumps(
            {
                "verified": True,
                "web_run_id": web_result["run_id"],
                "bad_command_state": bad_status["state"],
                "progress": f"{args.limit}/{args.limit}",
                "compare_cli": args.compare_cli,
                "metrics": web_result["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
