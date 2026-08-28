from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.verify_web_wiring import WiringVerificationError, compare_results


def _result(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "created_at": "ignored",
        "protocol": {"strength": 0.25, "n_input": 4},
        "metrics": {
            "fid": 4.0,
            "psnr": {"mean": 30.0},
            "ssim": {"mean": 0.93},
        },
    }


def test_compare_results_accepts_only_matching_protocol_and_metrics() -> None:
    web = _result("web-id")
    cli = _result("cli-id")

    compare_results(web, cli)

    changed = deepcopy(cli)
    changed["metrics"]["fid"] = 4.1
    with pytest.raises(WiringVerificationError, match="metrics"):
        compare_results(web, changed)

    changed = deepcopy(cli)
    changed["protocol"]["strength"] = 0.3
    with pytest.raises(WiringVerificationError, match="protocol"):
        compare_results(web, changed)
