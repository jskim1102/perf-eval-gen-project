from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL500_ROOT = PROJECT_ROOT / "dataset" / "psnr_ssim"
DEFAULT_FID500_ROOT = PROJECT_ROOT / "dataset" / "fid"
DEFAULT_FID_V2_ROOT = PROJECT_ROOT / "dataset" / "fid_v2"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "dataset" / "runs"
DEFAULT_FIXTURE_RESULT = PROJECT_ROOT / "web" / "fixtures" / "results.fixture.json"
DEFAULT_PILOT_PATH = PROJECT_ROOT / "runs" / "pilot" / "pilot.json"
FID_RUN_ID = "fid500-v2"
FID_V2_RUN_ID = "fid_v2_img2img"
FID_V2_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FID_V2_GENERATION_MODE = "image-to-image"
FID_V2_STRENGTH = 0.15
FID_V2_NUM_INFERENCE_STEPS = 30
FID_V2_GUIDANCE_SCALE = 3.5
FID_V2_SEED = 0
FID_V2_SECONDS_PER_IMAGE = 22
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROGRESS_PATTERN = re.compile(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
MAX_TRY_ITEMS = 100
MIN_FID_TRY_ITEMS = 2
MAX_FID_TRY_ITEMS = 500
FID_GENERATED_FIELDS = {"input_path", "output_path", "sha256", "seed", "strength"}


class RunRequest(BaseModel):
    strength: float = Field(ge=0.0, le=1.0)


class TryRequest(BaseModel):
    item_ids: list[str]


class PilotStrengthError(RuntimeError):
    """The fixed pilot strength is unavailable or invalid."""


class RunManager:
    """Own one subprocess and expose a small, thread-safe status snapshot."""

    def __init__(
        self,
        eval_cmd: str,
        try_cmd: str = "",
        cwd: Path = PROJECT_ROOT,
        fid_cmd: str = "",
        fidv2_cmd: str = "",
        runs_root: Path = DEFAULT_RUNS_ROOT,
    ) -> None:
        self._eval_cmd = eval_cmd.strip()
        self._try_cmd = try_cmd.strip()
        self._fid_cmd = fid_cmd.strip()
        self._fidv2_cmd = fidv2_cmd.strip()
        self._cwd = cwd
        self._runs_root = runs_root.expanduser().absolute()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._run_id: str | None = None
        self._state = "idle"
        self._done = 0
        self._total = 0
        self._log: deque[str] = deque(maxlen=50)

    @property
    def available(self) -> bool:
        return bool(self._eval_cmd)

    @property
    def try_available(self) -> bool:
        return bool(self._try_cmd)

    @property
    def fid_available(self) -> bool:
        return bool(self._fid_cmd)

    @property
    def fidv2_available(self) -> bool:
        return bool(self._fidv2_cmd)

    def start(self, strength: float) -> tuple[bool, str]:
        return self._start(command=self._eval_cmd, strength=strength, item_ids=None)

    def start_try(self, strength: float, item_ids: list[str]) -> tuple[bool, str]:
        return self._start(
            command=self._try_cmd,
            strength=strength,
            item_ids=item_ids,
        )

    def start_fid(self, strength: float) -> tuple[bool, str]:
        return self._start(
            command=self._fid_cmd,
            strength=strength,
            item_ids=None,
            run_id=FID_RUN_ID,
        )

    def start_fid_try(
        self, strength: float, item_ids: list[str]
    ) -> tuple[bool, str]:
        return self._start(
            command=self._fid_cmd,
            strength=strength,
            item_ids=item_ids,
            run_id=self._new_run_id(prefix="fidtry"),
        )

    def start_fidv2(self, strength: float) -> tuple[bool, str]:
        return self._start(
            command=self._fidv2_cmd,
            strength=strength,
            item_ids=None,
            run_id=FID_V2_RUN_ID,
        )

    def start_fidv2_try(
        self, strength: float, item_ids: list[str]
    ) -> tuple[bool, str]:
        return self._start(
            command=self._fidv2_cmd,
            strength=strength,
            item_ids=item_ids,
            run_id=self._new_run_id(prefix="fidv2try"),
        )

    def _start(
        self,
        *,
        command: str,
        strength: float,
        item_ids: list[str] | None,
        run_id: str | None = None,
    ) -> tuple[bool, str]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False, self._run_id or "unknown"

            run_id = run_id or self._new_run_id(
                prefix="try" if item_ids is not None else "web"
            )
            self._run_id = run_id
            self._state = "running"
            self._done = 0
            self._total = 0
            self._log.clear()
            self._thread = threading.Thread(
                target=self._execute,
                args=(command, run_id, strength, item_ids),
                name=f"eval-{run_id}",
                daemon=True,
            )
            self._thread.start()
            return True, run_id

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "done": self._done,
                "total": self._total,
                "log": list(self._log),
                "run_id": self._run_id,
            }

    @staticmethod
    def _new_run_id(*, prefix: str = "web") -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"

    def _execute(
        self,
        command_text: str,
        run_id: str,
        strength: float,
        item_ids: list[str] | None,
    ) -> None:
        process: subprocess.Popen[str] | None = None
        failure: BaseException | None = None
        try:
            command = [
                *shlex.split(command_text),
                "--strength",
                format(strength, ".15g"),
                "--run-id",
                run_id,
            ]
            for item_id in item_ids or []:
                command.extend(("--item-id", item_id))
            process_env = os.environ.copy()
            process_env["RUNS_ROOT"] = str(self._runs_root)
            process = subprocess.Popen(
                command,
                cwd=self._cwd,
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._process = process

            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if line:
                    self._consume_log(line)

            return_code = process.wait()
            with self._lock:
                self._state = "done" if return_code == 0 else "error"
                if return_code != 0:
                    self._log.append(f"process exited with code {return_code}")
        except BaseException as exc:
            failure = exc
        finally:
            cleanup_failure: BaseException | None = None
            if failure is not None and process is not None:
                try:
                    self._terminate_and_reap(process)
                except BaseException as exc:
                    cleanup_failure = exc
            with self._lock:
                if failure is not None:
                    prefix = (
                        "failed to start evaluation"
                        if process is None
                        else "evaluation failed"
                    )
                    self._state = "error"
                    self._log.append(
                        f"{prefix}: {type(failure).__name__}: {failure}"
                    )
                if cleanup_failure is not None:
                    self._log.append(
                        "failed to reap evaluation process: "
                        f"{type(cleanup_failure).__name__}: {cleanup_failure}"
                    )
                if self._state not in {"done", "error"}:
                    self._state = "error"
                    self._log.append("evaluation ended without a terminal state")
                self._process = None
                self._thread = None

    @staticmethod
    def _terminate_and_reap(process: subprocess.Popen[str]) -> None:
        try:
            running = process.poll() is None
        except BaseException:
            running = True
        if not running:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            process.wait(timeout=0)
            return
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)

    def _consume_log(self, line: str) -> None:
        lowered = line.lower()
        progress = PROGRESS_PATTERN.search(line)
        with self._lock:
            self._log.append(line)
            if "measur" in lowered:
                self._state = "measuring"
            elif "generat" in lowered:
                self._state = "generating"
            if progress is not None and "generat" in lowered:
                self._done = int(progress.group("done"))
                self._total = int(progress.group("total"))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_run_id(run_id: str) -> str:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=422, detail="invalid run_id")
    return run_id


def _read_pilot_strength(pilot_path: Path) -> float:
    source = pilot_path.expanduser().absolute()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotStrengthError(
            f"cannot read pilot strength from {source}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PilotStrengthError(f"pilot payload is not an object in {source}")
    selected = payload.get("selected")
    if selected is None:
        raise PilotStrengthError(
            f"pilot selected is null in {source}; no strength met the selection gate"
        )
    try:
        strength = float(selected)
    except (TypeError, ValueError) as exc:
        raise PilotStrengthError(
            f"pilot selected is not numeric in {source}: {selected!r}"
        ) from exc
    if not 0.0 < strength <= 1.0:
        raise PilotStrengthError(
            f"pilot selected must be in (0, 1] in {source}, got {strength}"
        )
    return strength


def _input_manifest_count(eval500_root: Path) -> int | None:
    manifest_path = eval500_root / "manifests" / "input.csv"
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return None
            count = sum(1 for _ in reader)
    except OSError:
        return None
    return count if count > 0 else None


def _prompt_manifest_sha256(eval500_root: Path) -> str | None:
    prompt_manifest = eval500_root / "manifests" / "prompts.csv"
    try:
        return hashlib.sha256(prompt_manifest.read_bytes()).hexdigest()
    except OSError:
        return None


def _default_result_path(runs_root: Path, eval500_root: Path) -> Path | None:
    """Choose newest full-dataset result; break mtime ties by run ID."""
    expected_input_count = _input_manifest_count(eval500_root)
    if expected_input_count is None:
        return None
    expected_prompt_hash = _prompt_manifest_sha256(eval500_root)
    try:
        fixture_bytes = DEFAULT_FIXTURE_RESULT.read_bytes()
    except OSError:
        fixture_bytes = None
    try:
        run_roots = list(runs_root.iterdir())
    except OSError:
        return None

    candidates: list[tuple[int, str, Path]] = []
    for run_root in run_roots:
        run_id = run_root.name
        if (
            not run_root.is_dir()
            or run_id.startswith("try-")
            or RUN_ID_PATTERN.fullmatch(run_id) is None
        ):
            continue
        result_path = run_root / "results.json"
        try:
            if not result_path.is_file():
                continue
            resolved_result = result_path.resolve()
            if not _is_relative_to(resolved_result, runs_root):
                continue
            result_bytes = resolved_result.read_bytes()
            if fixture_bytes is not None and result_bytes == fixture_bytes:
                continue
            payload = json.loads(result_bytes)
            protocol = payload.get("protocol")
            if (
                payload.get("run_id") != run_id
                or not isinstance(protocol, dict)
                or protocol.get("n_input") != expected_input_count
            ):
                continue
            if expected_prompt_hash is not None:
                prompt_protocol = protocol.get("prompt_protocol")
                if (
                    not isinstance(prompt_protocol, dict)
                    or prompt_protocol.get("manifest_sha256") != expected_prompt_hash
                ):
                    continue
            modified_ns = resolved_result.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        candidates.append((modified_ns, run_id, resolved_result))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _read_dataset_items(eval500_root: Path) -> dict[str, list[dict[str, Any]]]:
    manifests = eval500_root / "manifests"
    input_path = manifests / "input.csv"
    pair_path = manifests / "psnr_ssim_100.csv"
    required = {
        "item_id",
        "group",
        "product_type",
        "image_id",
        "source_path",
    }

    def read(path: Path) -> list[dict[str, str]]:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
                if not required.issubset(fields):
                    raise ValueError(f"{path.name} lacks fields: {sorted(required - fields)}")
                return list(reader)
        except OSError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"dataset manifest not found: {path.name}",
            ) from exc

    input_rows = read(input_path)
    pair_rows = read(pair_path)
    pair_ids = {row["item_id"] for row in pair_rows}
    group_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    for row in input_rows:
        group_counts[row["group"]] = group_counts.get(row["group"], 0) + 1
    for row in pair_rows:
        pair_counts[row["group"]] = pair_counts.get(row["group"], 0) + 1
    groups = [
        {
            "group": group,
            "count": count,
            "pair_count": pair_counts.get(group, 0),
        }
        for group, count in group_counts.items()
    ]
    items = [
        {
            "item_id": row["item_id"],
            "group": row["group"],
            "product_type": row["product_type"],
            "image_id": row["image_id"],
            "input_path": row["source_path"],
            "in_pair_set": row["item_id"] in pair_ids,
        }
        for row in input_rows
    ]
    return {"groups": groups, "items": items}


def _read_fid_images(
    generated_path: Path,
    *,
    fid500_root: Path | None = None,
    run_id: str = FID_RUN_ID,
    dataset_name: str = "FID500",
) -> dict[str, Any]:
    try:
        with generated_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = sorted(FID_GENERATED_FIELDS - fields)
            if missing:
                raise ValueError(f"generated.csv lacks fields: {missing}")
            rows = list(reader)
    except OSError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"{dataset_name} generated manifest not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if fid500_root is not None:
        input_root = fid500_root.expanduser().absolute() / "input"
        for row in rows:
            recorded = Path(row["input_path"])
            candidate = input_root / recorded.parent.name / recorded.name
            if candidate.is_file():
                row["input_path"] = str(candidate.absolute())
    return {
        "run_id": run_id,
        "items": [dict(row, index=index) for index, row in enumerate(rows, start=1)],
    }


def _read_fid_dataset_items(fid500_root: Path) -> dict[str, Any]:
    manifest_path = fid500_root / "manifest.csv"
    required = {"item_no", "대분류", "상품명"}
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                raise ValueError(
                    f"manifest.csv lacks fields: {sorted(required - fields)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail="FID500 manifest not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_root = (fid500_root / "input").absolute()
    images_by_id: dict[str, Path] = {}
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        }:
            if path.stem in images_by_id:
                raise HTTPException(
                    status_code=500,
                    detail=f"FID500 input has duplicate item id: {path.stem}",
                )
            images_by_id[path.stem] = path.absolute()

    items: list[dict[str, str]] = []
    group_counts: dict[str, int] = {}
    for row in rows:
        item_id = row["item_no"]
        image_path = images_by_id.get(item_id)
        if image_path is None:
            raise HTTPException(
                status_code=500,
                detail=f"FID500 input image not found for item: {item_id}",
            )
        group = row["대분류"]
        group_counts[group] = group_counts.get(group, 0) + 1
        items.append(
            {
                "item_id": item_id,
                "group": group,
                "product_type": row["상품명"],
                "image_id": item_id,
                "input_path": str(image_path),
            }
        )
    return {
        "groups": [
            {"group": group, "count": count}
            for group, count in group_counts.items()
        ],
        "items": items,
    }


def _read_fidv2_dataset_items(fidv2_root: Path) -> dict[str, Any]:
    manifest_path = fidv2_root / "manifest.csv"
    required = {"item_no", "대분류", "상품명", "thumbnail"}
    try:
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not required.issubset(fields):
                raise ValueError(
                    f"manifest.csv lacks fields: {sorted(required - fields)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail="FID_v2 manifest not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    items: list[dict[str, str]] = []
    group_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        item_id = row["item_no"].strip()
        group = row["대분류"].strip()
        product_type = row["상품명"].strip()
        thumbnail = Path(row["thumbnail"])
        if (
            not item_id
            or not group
            or not product_type
            or thumbnail.is_absolute()
            or ".." in thumbnail.parts
            or not thumbnail.parts
            or thumbnail.parts[0] != "input"
        ):
            raise HTTPException(
                status_code=500,
                detail=f"FID_v2 manifest row {row_number} is invalid",
            )
        if item_id in seen_ids:
            raise HTTPException(
                status_code=500,
                detail=f"FID_v2 manifest duplicates item id: {item_id}",
            )
        seen_ids.add(item_id)
        image_path = (fidv2_root / thumbnail).absolute()
        if not image_path.is_file() or image_path.is_symlink():
            raise HTTPException(
                status_code=500,
                detail=f"FID_v2 thumbnail not found for item: {item_id}",
            )
        group_counts[group] = group_counts.get(group, 0) + 1
        items.append(
            {
                "item_id": item_id,
                "group": group,
                "product_type": product_type,
                "image_id": item_id,
                "input_path": str(image_path),
            }
        )
    return {
        "groups": [
            {"group": group, "count": count}
            for group, count in group_counts.items()
        ],
        "items": items,
    }


def _append_dataset_arguments(command: str, arguments: list[str]) -> str:
    command = command.strip()
    if not command:
        return ""
    return " ".join([command, *(shlex.quote(argument) for argument in arguments)])


def create_app(
    *,
    runs_root: Path | str | None = None,
    eval500_root: Path | str | None = None,
    fid500_root: Path | str | None = None,
    fidv2_root: Path | str | None = None,
    eval_cmd: str | None = None,
    try_cmd: str | None = None,
    fid_cmd: str | None = None,
    fidv2_cmd: str | None = None,
    pilot_path: Path | str | None = None,
) -> FastAPI:
    resolved_runs_root = Path(
        runs_root if runs_root is not None else DEFAULT_RUNS_ROOT
    ).expanduser().resolve()
    resolved_eval500_root = Path(
        eval500_root if eval500_root is not None else DEFAULT_EVAL500_ROOT
    ).expanduser().resolve()
    resolved_fid500_root = Path(
        fid500_root if fid500_root is not None else DEFAULT_FID500_ROOT
    ).expanduser().resolve()
    resolved_fidv2_root = Path(
        fidv2_root if fidv2_root is not None else DEFAULT_FID_V2_ROOT
    ).expanduser().resolve()
    resolved_curated_input = (resolved_eval500_root.parent / "input").resolve()
    if eval_cmd is None:
        resolved_eval_cmd = _append_dataset_arguments(
            os.environ.get("EVAL_CMD", ""),
            [
                "--input-manifest",
                str(resolved_eval500_root / "manifests" / "input.csv"),
                "--pair-manifest",
                str(resolved_eval500_root / "manifests" / "psnr_ssim_100.csv"),
                "--prompt-manifest",
                str(resolved_eval500_root / "manifests" / "prompts.csv"),
            ],
        )
    else:
        resolved_eval_cmd = eval_cmd
    if try_cmd is None:
        resolved_try_cmd = _append_dataset_arguments(
            os.environ.get("TRY_CMD", ""),
            [
                "--manifest",
                str(resolved_eval500_root / "manifests" / "input.csv"),
                "--prompt-manifest",
                str(resolved_eval500_root / "manifests" / "prompts.csv"),
            ],
        )
    else:
        resolved_try_cmd = try_cmd
    if fid_cmd is None:
        resolved_fid_cmd = _append_dataset_arguments(
            os.environ.get("FID_CMD", ""),
            ["--dataset-root", str(resolved_fid500_root)],
        )
    else:
        resolved_fid_cmd = fid_cmd
    if fidv2_cmd is None:
        resolved_fidv2_cmd = _append_dataset_arguments(
            os.environ.get("FID_V2_CMD", ""),
            ["--dataset-root", str(resolved_fidv2_root)],
        )
    else:
        resolved_fidv2_cmd = fidv2_cmd
    resolved_pilot_path = Path(pilot_path or DEFAULT_PILOT_PATH).expanduser().absolute()
    manager = RunManager(
        resolved_eval_cmd,
        resolved_try_cmd,
        fid_cmd=resolved_fid_cmd,
        fidv2_cmd=resolved_fidv2_cmd,
        runs_root=resolved_runs_root,
    )
    fid_result_path = resolved_runs_root / FID_RUN_ID / "fid500.json"
    fidv2_result_path = resolved_runs_root / FID_V2_RUN_ID / "fid_v2.json"

    api = FastAPI(title="perf-eval-gen web wrapper")
    frontend_port = os.environ.get("FRONTEND_PORT")
    if frontend_port:
        api.add_middleware(
            CORSMiddleware,
            allow_origin_regex=rf"^https?://[^/:]+:{re.escape(frontend_port)}$",
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )
    api.state.run_manager = manager

    @api.post("/api/run")
    def start_run(request: RunRequest) -> JSONResponse:
        if not manager.available:
            raise HTTPException(status_code=503, detail="EVAL_CMD is not configured")
        started, run_id = manager.start(request.strength)
        status_code = 200 if started else 409
        return JSONResponse(status_code=status_code, content={"run_id": run_id})

    @api.post("/api/try")
    def start_try(request: TryRequest) -> JSONResponse:
        if not 1 <= len(request.item_ids) <= MAX_TRY_ITEMS:
            raise HTTPException(
                status_code=400,
                detail=f"item_ids must contain 1 to {MAX_TRY_ITEMS} entries",
            )
        if len(request.item_ids) != len(set(request.item_ids)) or any(
            not item_id for item_id in request.item_ids
        ):
            raise HTTPException(status_code=400, detail="item_ids must be non-empty and unique")
        try:
            strength = _read_pilot_strength(resolved_pilot_path)
        except PilotStrengthError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not manager.try_available:
            raise HTTPException(status_code=503, detail="TRY_CMD is not configured")
        started, run_id = manager.start_try(strength, request.item_ids)
        return JSONResponse(
            status_code=200 if started else 409,
            content={"run_id": run_id},
        )

    @api.get("/api/try/config")
    def get_try_config() -> dict[str, Any]:
        try:
            strength = _read_pilot_strength(resolved_pilot_path)
            detail = None
        except PilotStrengthError as exc:
            strength = None
            detail = str(exc)
        return {
            "strength": strength,
            "source": str(resolved_pilot_path),
            "detail": detail,
        }

    @api.get("/api/fid/config")
    def get_fid_config() -> dict[str, Any]:
        try:
            strength = _read_pilot_strength(resolved_pilot_path)
            detail = None
        except PilotStrengthError as exc:
            strength = None
            detail = str(exc)
        completed = fid_result_path.is_file()
        if detail is None and not manager.fid_available:
            detail = "FID_CMD is not configured"
        selected_detail = detail
        canonical_detail = detail
        if canonical_detail is None and completed:
            canonical_detail = f"completed result already exists: {fid_result_path}"
        return {
            "strength": strength,
            "source": str(resolved_pilot_path),
            "detail": canonical_detail,
            "available": canonical_detail is None,
            "completed": completed,
            "selected_available": selected_detail is None,
            "selected_detail": selected_detail,
        }

    @api.post("/api/fid/run")
    def start_fid() -> JSONResponse:
        if fid_result_path.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"completed result already exists: {fid_result_path}",
            )
        try:
            strength = _read_pilot_strength(resolved_pilot_path)
        except PilotStrengthError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not manager.fid_available:
            raise HTTPException(status_code=503, detail="FID_CMD is not configured")
        started, run_id = manager.start_fid(strength)
        return JSONResponse(
            status_code=200 if started else 409,
            content={"run_id": run_id},
        )

    @api.post("/api/fid/try")
    def start_fid_try(request: TryRequest) -> JSONResponse:
        item_ids = request.item_ids
        if not MIN_FID_TRY_ITEMS <= len(item_ids) <= MAX_FID_TRY_ITEMS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"item_ids must contain {MIN_FID_TRY_ITEMS} to "
                    f"{MAX_FID_TRY_ITEMS} entries"
                ),
            )
        if len(item_ids) != len(set(item_ids)) or any(not item_id for item_id in item_ids):
            raise HTTPException(
                status_code=400, detail="item_ids must be non-empty and unique"
            )
        available_ids = {
            item["item_id"]
            for item in _read_fid_dataset_items(resolved_fid500_root)["items"]
        }
        missing = [item_id for item_id in item_ids if item_id not in available_ids]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"item_ids are not in FID500: {missing}",
            )
        try:
            strength = _read_pilot_strength(resolved_pilot_path)
        except PilotStrengthError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not manager.fid_available:
            raise HTTPException(status_code=503, detail="FID_CMD is not configured")
        started, run_id = manager.start_fid_try(strength, item_ids)
        return JSONResponse(
            status_code=200 if started else 409,
            content={"run_id": run_id},
        )

    @api.get("/api/fid/dataset")
    def get_fid_dataset() -> FileResponse:
        selection_path = resolved_fid500_root / "selection.json"
        if not selection_path.is_file():
            raise HTTPException(status_code=404, detail="FID500 selection not found")
        return FileResponse(selection_path, media_type="application/json")

    @api.get("/api/fid/dataset/items")
    def get_fid_dataset_items() -> dict[str, Any]:
        return _read_fid_dataset_items(resolved_fid500_root)

    @api.get("/api/fid/results")
    def get_fid_results() -> FileResponse:
        if not fid_result_path.is_file():
            raise HTTPException(status_code=404, detail="FID500 results not found")
        resolved_result = fid_result_path.resolve()
        if not _is_relative_to(resolved_result, resolved_runs_root):
            raise HTTPException(status_code=403, detail="FID500 results path is not allowed")
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/fid/images")
    def get_fid_images() -> dict[str, Any]:
        return _read_fid_images(
            resolved_runs_root / FID_RUN_ID / "generated.csv",
            fid500_root=resolved_fid500_root,
        )

    @api.get("/api/fid/try/results")
    def get_fid_try_results(run_id: str = Query(...)) -> FileResponse:
        safe_run_id = _validated_run_id(run_id)
        if not safe_run_id.startswith("fidtry-"):
            raise HTTPException(status_code=404, detail="selected FID result not found")
        result_path = resolved_runs_root / safe_run_id / "fid500.json"
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="selected FID result not found")
        resolved_result = result_path.resolve()
        if not _is_relative_to(resolved_result, resolved_runs_root):
            raise HTTPException(status_code=403, detail="selected FID result path is not allowed")
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/fid/try/images")
    def get_fid_try_images(run_id: str = Query(...)) -> dict[str, Any]:
        safe_run_id = _validated_run_id(run_id)
        if not safe_run_id.startswith("fidtry-"):
            raise HTTPException(status_code=404, detail="selected FID images not found")
        return _read_fid_images(
            resolved_runs_root / safe_run_id / "generated.csv",
            fid500_root=resolved_fid500_root,
            run_id=safe_run_id,
        )

    @api.get("/api/fidv2/config")
    def get_fidv2_config() -> dict[str, Any]:
        completed = fidv2_result_path.is_file()
        selected_detail = None
        if not manager.fidv2_available:
            selected_detail = "FID_V2_CMD is not configured"
        canonical_detail = selected_detail
        if canonical_detail is None and completed:
            canonical_detail = f"completed result already exists: {fidv2_result_path}"
        return {
            "strength": FID_V2_STRENGTH,
            "source": "fixed FLUX.1-dev img2img protocol",
            "model": FID_V2_MODEL_ID,
            "generation_mode": FID_V2_GENERATION_MODE,
            "num_inference_steps": FID_V2_NUM_INFERENCE_STEPS,
            "guidance_scale": FID_V2_GUIDANCE_SCALE,
            "seed": FID_V2_SEED,
            "seconds_per_image": FID_V2_SECONDS_PER_IMAGE,
            "detail": canonical_detail,
            "available": canonical_detail is None,
            "completed": completed,
            "selected_available": selected_detail is None,
            "selected_detail": selected_detail,
        }

    @api.post("/api/fidv2/run")
    def start_fidv2() -> JSONResponse:
        if fidv2_result_path.is_file():
            raise HTTPException(
                status_code=409,
                detail=f"completed result already exists: {fidv2_result_path}",
            )
        if not manager.fidv2_available:
            raise HTTPException(status_code=503, detail="FID_V2_CMD is not configured")
        started, run_id = manager.start_fidv2(FID_V2_STRENGTH)
        return JSONResponse(
            status_code=200 if started else 409,
            content={"run_id": run_id},
        )

    @api.post("/api/fidv2/try")
    def start_fidv2_try(request: TryRequest) -> JSONResponse:
        item_ids = request.item_ids
        if not MIN_FID_TRY_ITEMS <= len(item_ids) <= MAX_FID_TRY_ITEMS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"item_ids must contain {MIN_FID_TRY_ITEMS} to "
                    f"{MAX_FID_TRY_ITEMS} entries"
                ),
            )
        if len(item_ids) != len(set(item_ids)) or any(not item_id for item_id in item_ids):
            raise HTTPException(
                status_code=400, detail="item_ids must be non-empty and unique"
            )
        available_ids = {
            item["item_id"]
            for item in _read_fidv2_dataset_items(resolved_fidv2_root)["items"]
        }
        missing = [item_id for item_id in item_ids if item_id not in available_ids]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"item_ids are not in FID_v2: {missing}",
            )
        if not manager.fidv2_available:
            raise HTTPException(status_code=503, detail="FID_V2_CMD is not configured")
        started, run_id = manager.start_fidv2_try(FID_V2_STRENGTH, item_ids)
        return JSONResponse(
            status_code=200 if started else 409,
            content={"run_id": run_id},
        )

    @api.get("/api/fidv2/dataset")
    def get_fidv2_dataset() -> FileResponse:
        selection_path = resolved_fidv2_root / "selection.json"
        if not selection_path.is_file():
            raise HTTPException(status_code=404, detail="FID_v2 selection not found")
        return FileResponse(selection_path, media_type="application/json")

    @api.get("/api/fidv2/dataset/items")
    def get_fidv2_dataset_items() -> dict[str, Any]:
        return _read_fidv2_dataset_items(resolved_fidv2_root)

    @api.get("/api/fidv2/results")
    def get_fidv2_results() -> FileResponse:
        if not fidv2_result_path.is_file():
            raise HTTPException(status_code=404, detail="FID_v2 results not found")
        resolved_result = fidv2_result_path.resolve()
        if not _is_relative_to(resolved_result, resolved_runs_root):
            raise HTTPException(status_code=403, detail="FID_v2 results path is not allowed")
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/fidv2/images")
    def get_fidv2_images() -> dict[str, Any]:
        return _read_fid_images(
            resolved_runs_root / FID_V2_RUN_ID / "generated.csv",
            fid500_root=resolved_fidv2_root,
            run_id=FID_V2_RUN_ID,
            dataset_name="FID_v2",
        )

    @api.get("/api/fidv2/try/results")
    def get_fidv2_try_results(run_id: str = Query(...)) -> FileResponse:
        safe_run_id = _validated_run_id(run_id)
        if not safe_run_id.startswith("fidv2try-"):
            raise HTTPException(status_code=404, detail="selected FID_v2 result not found")
        result_path = resolved_runs_root / safe_run_id / "fid_v2.json"
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="selected FID_v2 result not found")
        resolved_result = result_path.resolve()
        if not _is_relative_to(resolved_result, resolved_runs_root):
            raise HTTPException(
                status_code=403, detail="selected FID_v2 result path is not allowed"
            )
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/fidv2/try/images")
    def get_fidv2_try_images(run_id: str = Query(...)) -> dict[str, Any]:
        safe_run_id = _validated_run_id(run_id)
        if not safe_run_id.startswith("fidv2try-"):
            raise HTTPException(status_code=404, detail="selected FID_v2 images not found")
        return _read_fid_images(
            resolved_runs_root / safe_run_id / "generated.csv",
            fid500_root=resolved_fidv2_root,
            run_id=safe_run_id,
            dataset_name="FID_v2",
        )

    @api.get("/api/status")
    def get_status() -> dict[str, Any]:
        return manager.snapshot()

    @api.get("/api/results")
    def get_results(run_id: str | None = Query(default=None)) -> FileResponse:
        if run_id is None:
            resolved_result = _default_result_path(
                resolved_runs_root,
                resolved_eval500_root,
            )
            if resolved_result is None:
                raise HTTPException(status_code=404, detail="canonical results not found")
        else:
            safe_run_id = _validated_run_id(run_id)
            if safe_run_id.startswith("try-"):
                raise HTTPException(status_code=404, detail="canonical results not found")
            result_path = resolved_runs_root / safe_run_id / "results.json"
            if not result_path.is_file():
                raise HTTPException(status_code=404, detail="results not found")
            resolved_result = result_path.resolve()
            if not _is_relative_to(resolved_result, resolved_runs_root):
                raise HTTPException(status_code=403, detail="results path is not allowed")
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/try/results")
    def get_try_results(run_id: str = Query(...)) -> FileResponse:
        safe_run_id = _validated_run_id(run_id)
        if not safe_run_id.startswith("try-"):
            raise HTTPException(status_code=404, detail="trial results not found")
        result_path = resolved_runs_root / safe_run_id / "try.json"
        if not result_path.is_file():
            raise HTTPException(status_code=404, detail="trial results not found")
        resolved_result = result_path.resolve()
        if not _is_relative_to(resolved_result, resolved_runs_root):
            raise HTTPException(status_code=403, detail="trial results path is not allowed")
        return FileResponse(resolved_result, media_type="application/json")

    @api.get("/api/dataset")
    def get_dataset() -> dict[str, Any]:
        protocol_path = resolved_eval500_root / "manifests" / "protocol.json"
        if not protocol_path.is_file():
            raise HTTPException(status_code=404, detail="dataset protocol not found")
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"dataset protocol is unreadable: {exc}",
            ) from exc
        return {"root": str(resolved_eval500_root), "protocol": protocol}

    @api.get("/api/dataset/items")
    def get_dataset_items() -> dict[str, list[dict[str, Any]]]:
        return _read_dataset_items(resolved_eval500_root)

    @api.get("/api/image")
    def get_image(path: str = Query(...)) -> FileResponse:
        requested = Path(path).expanduser()
        if not requested.is_absolute():
            raise HTTPException(status_code=403, detail="image path is not allowed")
        lexical_path = Path(os.path.abspath(requested))
        under_runs = _is_relative_to(lexical_path, resolved_runs_root)
        under_eval500 = _is_relative_to(lexical_path, resolved_eval500_root)
        under_fid500 = _is_relative_to(lexical_path, resolved_fid500_root)
        under_fidv2 = _is_relative_to(lexical_path, resolved_fidv2_root)
        under_curated_input = _is_relative_to(lexical_path, resolved_curated_input)
        if (
            not under_runs
            and not under_eval500
            and not under_fid500
            and not under_fidv2
            and not under_curated_input
        ):
            raise HTTPException(status_code=403, detail="image path is not allowed")
        if not lexical_path.is_file():
            raise HTTPException(status_code=404, detail="image not found")

        resolved_path = lexical_path.resolve()
        if under_runs and not _is_relative_to(resolved_path, resolved_runs_root):
            raise HTTPException(status_code=403, detail="image symlink is not allowed")
        if under_eval500 and not (
            _is_relative_to(resolved_path, resolved_eval500_root)
            or _is_relative_to(resolved_path, resolved_curated_input)
        ):
            raise HTTPException(status_code=403, detail="image symlink is not allowed")
        if under_fid500 and not _is_relative_to(resolved_path, resolved_fid500_root):
            raise HTTPException(status_code=403, detail="image symlink is not allowed")
        if under_fidv2 and not _is_relative_to(resolved_path, resolved_fidv2_root):
            raise HTTPException(status_code=403, detail="image symlink is not allowed")
        if under_curated_input and not _is_relative_to(
            resolved_path, resolved_curated_input
        ):
            raise HTTPException(status_code=403, detail="image symlink is not allowed")
        return FileResponse(resolved_path)

    return api


app = create_app()
