"""Validation for the canonical results.json contract."""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TOP_LEVEL_KEYS = [
    "run_id",
    "created_at",
    "protocol",
    "env",
    "inputs",
    "metrics",
    "pairs",
    "targets",
    "baseline_ref",
]
GENERATION_VALIDATION_KEYS = {
    "image_count",
    "all_output_sha256_differ_from_input",
    "all_foreground_mae_positive",
    "copy_or_link_path_absent",
    "foreground_definition",
    "changed_pixel_delta",
    "foreground_mae",
    "foreground_changed_fraction",
}
DISTRIBUTION_KEYS = {
    "observation_threshold",
    "minimum",
    "p5",
    "median",
    "p95",
    "maximum",
    "below_threshold_count",
    "below_threshold_ratio",
}
PROTOCOL_REQUIRED_KEYS = {
    "seed",
    "n_input",
    "n_pairs",
    "strength",
    "model",
    "steps",
    "guidance",
    "scheduler",
    "output",
    "psnr",
    "ssim",
}
PROTOCOL_OPTIONAL_KEYS = {"fid", "note", "prompt_protocol"}
PROMPT_PROTOCOL_KEYS = {
    "mode",
    "template",
    "name_source",
    "name_max_words",
    "truncation",
    "whitespace_normalization",
    "manifest_path",
    "manifest_sha256",
}
ENV_KEYS = {"python", "torch", "diffusers", "cleanfid", "skimage", "gpu", "driver", "cuda"}
DETERMINISM_KEYS = {
    "matmul_tf32",
    "cudnn_tf32",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "torch_seed",
}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ResultsValidationError(ValueError):
    """results.json does not satisfy the fixed cross-consumer contract."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ResultsValidationError(message)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _finite_number(value: Any, *, label: str) -> None:
    _check(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )


def validate_results(
    results: dict[str, Any],
    *,
    eval_root: Path,
    runs_root: Path,
    require_determinism: bool = False,
    require_generation_validation: bool = False,
) -> None:
    accepted_top_level = [TOP_LEVEL_KEYS, TOP_LEVEL_KEYS + ["generation_validation"]]
    _check(list(results) in accepted_top_level, "results.json top-level keys differ")
    if require_generation_validation:
        _check(
            "generation_validation" in results,
            "generation_validation is required",
        )
    _check(isinstance(results["run_id"], str) and results["run_id"], "run_id is invalid")
    try:
        datetime.fromisoformat(results["created_at"])
    except (TypeError, ValueError) as exc:
        raise ResultsValidationError("created_at must be ISO-8601") from exc

    protocol = results["protocol"]
    _check(isinstance(protocol, dict), "protocol must be an object")
    protocol_keys = set(protocol)
    _check(
        PROTOCOL_REQUIRED_KEYS <= protocol_keys
        <= PROTOCOL_REQUIRED_KEYS | PROTOCOL_OPTIONAL_KEYS,
        f"protocol keys differ: {sorted(protocol_keys)}",
    )
    _check(isinstance(protocol["seed"], int) and protocol["seed"] >= 0, "seed is invalid")
    _check(
        isinstance(protocol["n_input"], int) and protocol["n_input"] >= 2,
        "n_input must be at least two",
    )
    _check(
        isinstance(protocol["n_pairs"], int) and protocol["n_pairs"] >= 1,
        "n_pairs must be positive",
    )
    _finite_number(protocol["strength"], label="protocol.strength")
    _check(0.0 < float(protocol["strength"]) <= 1.0, "strength is out of range")
    _check(protocol["steps"] == 30, "steps must be 30")
    _check(float(protocol["guidance"]) == 5.0, "guidance must be 5.0")
    _check(protocol["scheduler"] == "EulerDiscreteScheduler", "scheduler differs")
    _check(protocol["output"] == "1024x1024 PNG", "output protocol differs")
    if "prompt_protocol" in protocol:
        prompt_protocol = protocol["prompt_protocol"]
        _check(
            isinstance(prompt_protocol, dict)
            and set(prompt_protocol) == PROMPT_PROTOCOL_KEYS,
            "prompt protocol keys differ",
        )
        _check(prompt_protocol["mode"] == "per-image", "prompt protocol mode differs")
        for key in (
            "template",
            "name_source",
            "truncation",
            "whitespace_normalization",
        ):
            _check(
                isinstance(prompt_protocol[key], str) and prompt_protocol[key],
                f"prompt protocol {key} must be a non-empty string",
            )
        _check(
            isinstance(prompt_protocol["name_max_words"], int)
            and not isinstance(prompt_protocol["name_max_words"], bool)
            and prompt_protocol["name_max_words"] == 15,
            "prompt protocol name_max_words must be 15",
        )
        _check(
            isinstance(prompt_protocol["manifest_path"], str)
            and prompt_protocol["manifest_path"],
            "prompt protocol manifest_path must be a non-empty string",
        )
        manifest_path = Path(prompt_protocol["manifest_path"])
        _check(
            manifest_path.is_absolute(),
            "prompt protocol manifest_path must be absolute",
        )
        _check(
            isinstance(prompt_protocol["manifest_sha256"], str)
            and HASH_PATTERN.fullmatch(prompt_protocol["manifest_sha256"]) is not None,
            "prompt protocol manifest SHA-256 is invalid",
        )

    environment = results["env"]
    _check(isinstance(environment, dict), "env must be an object")
    environment_keys = set(environment)
    _check(
        environment_keys in (ENV_KEYS, ENV_KEYS | {"determinism"}),
        "env keys differ",
    )
    if require_determinism:
        _check("determinism" in environment, "env.determinism is required")
    _check(
        all(isinstance(environment[key], str) and environment[key] for key in ENV_KEYS),
        "env values must be non-empty strings",
    )
    if "determinism" in environment:
        determinism = environment["determinism"]
        _check(
            isinstance(determinism, dict) and set(determinism) == DETERMINISM_KEYS,
            "env.determinism keys differ",
        )
        for key in DETERMINISM_KEYS - {"torch_seed"}:
            _check(
                isinstance(determinism[key], bool),
                f"env.determinism.{key} must be boolean",
            )
        _check(
            isinstance(determinism["torch_seed"], int)
            and not isinstance(determinism["torch_seed"], bool)
            and determinism["torch_seed"] >= 0,
            "env.determinism.torch_seed must be a non-negative integer",
        )

    inputs = results["inputs"]
    _check(
        isinstance(inputs, dict)
        and set(inputs) == {"input_manifest_sha256", "generated_manifest_sha256"},
        "inputs keys differ",
    )
    for key, value in inputs.items():
        _check(
            isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None,
            f"inputs.{key} must be a SHA-256",
        )

    metrics = results["metrics"]
    _check(isinstance(metrics, dict), "metrics must be an object")
    _check(
        {"psnr", "ssim"} <= set(metrics) <= {"fid", "psnr", "ssim"},
        "metrics keys differ",
    )
    if "fid" in metrics:
        _finite_number(metrics["fid"], label="metrics.fid")
    for metric_name in ("psnr", "ssim"):
        metric = metrics[metric_name]
        _check(
            isinstance(metric, dict) and list(metric) == ["mean", "std", "per_image"],
            f"metrics.{metric_name} keys differ",
        )
        _finite_number(metric["mean"], label=f"metrics.{metric_name}.mean")
        _finite_number(metric["std"], label=f"metrics.{metric_name}.std")
        _check(isinstance(metric["per_image"], list), f"metrics.{metric_name}.per_image must be a list")
        for index, value in enumerate(metric["per_image"]):
            _finite_number(value, label=f"metrics.{metric_name}.per_image[{index}]")

    pairs = results["pairs"]
    _check(isinstance(pairs, list), "pairs must be a list")
    n_pairs = protocol["n_pairs"]
    _check(
        len(pairs)
        == len(metrics["psnr"]["per_image"])
        == len(metrics["ssim"]["per_image"])
        == n_pairs,
        "pair metric lengths differ",
    )
    eval_root = eval_root.expanduser().absolute()
    runs_root = runs_root.expanduser().absolute()
    seen_item_ids: set[str] = set()
    for index, pair in enumerate(pairs):
        _check(
            isinstance(pair, dict)
            and list(pair) == ["item_id", "group", "input_path", "output_path"],
            f"pairs[{index}] keys differ",
        )
        item_id = pair["item_id"]
        _check(isinstance(item_id, str) and item_id, f"pairs[{index}].item_id is invalid")
        _check(item_id not in seen_item_ids, f"pairs[{index}] duplicates item_id")
        seen_item_ids.add(item_id)
        input_path = Path(pair["input_path"])
        output_path = Path(pair["output_path"])
        _check(input_path.is_absolute(), f"pairs[{index}].input_path must be absolute")
        _check(output_path.is_absolute(), f"pairs[{index}].output_path must be absolute")
        _check(_within(input_path, eval_root), f"pairs[{index}].input_path is outside EVAL500")
        _check(_within(output_path, runs_root), f"pairs[{index}].output_path is outside RUNS_ROOT")
        _check(input_path.is_file(), f"pairs[{index}].input_path does not exist")
        _check(output_path.is_file(), f"pairs[{index}].output_path does not exist")

    targets = results["targets"]
    _check(isinstance(targets, dict), "targets must be an object")
    _check(
        {"psnr", "ssim"} <= set(targets) <= {"fid", "psnr", "ssim"},
        "targets keys differ",
    )
    _check(targets["psnr"] == 25.0 and targets["ssim"] == 0.9, "targets differ")
    if "fid" in targets:
        _check(targets["fid"] == 10.0, "targets.fid differs")
    _check(
        results["baseline_ref"] is None or isinstance(results["baseline_ref"], str),
        "baseline_ref must be null or a string",
    )

    if "generation_validation" in results:
        validation = results["generation_validation"]
        _check(
            isinstance(validation, dict)
            and set(validation) == GENERATION_VALIDATION_KEYS,
            "generation_validation keys differ",
        )
        _check(
            validation["image_count"] == protocol["n_input"],
            "generation_validation.image_count differs from protocol.n_input",
        )
        for key in (
            "all_output_sha256_differ_from_input",
            "all_foreground_mae_positive",
            "copy_or_link_path_absent",
        ):
            _check(validation[key] is True, f"generation_validation.{key} must be true")
        _check(
            isinstance(validation["foreground_definition"], str)
            and validation["foreground_definition"],
            "generation_validation.foreground_definition is invalid",
        )
        _finite_number(
            validation["changed_pixel_delta"],
            label="generation_validation.changed_pixel_delta",
        )
        for name in ("foreground_mae", "foreground_changed_fraction"):
            distribution = validation[name]
            _check(
                isinstance(distribution, dict)
                and set(distribution) == DISTRIBUTION_KEYS,
                f"generation_validation.{name} keys differ",
            )
            for key in (
                "observation_threshold",
                "minimum",
                "p5",
                "median",
                "p95",
                "maximum",
                "below_threshold_ratio",
            ):
                _finite_number(
                    distribution[key],
                    label=f"generation_validation.{name}.{key}",
                )
            _check(
                isinstance(distribution["below_threshold_count"], int)
                and 0 <= distribution["below_threshold_count"] <= protocol["n_input"],
                f"generation_validation.{name}.below_threshold_count is invalid",
            )
            _check(
                0.0 <= float(distribution["below_threshold_ratio"]) <= 1.0,
                f"generation_validation.{name}.below_threshold_ratio is invalid",
            )
