from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_replication_record_is_self_consistent() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_replication_record.py"),
            str(ROOT / "experiments" / "records" / "tum_fr2_desk_replication_v1.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout)
    assert audit["status"] == "ok"
    assert audit["decision"]["robust_support"] is True
    assert audit["interleaved_summary"]["psnr"]["favorable_pairs"] == 3
    assert audit["interleaved_summary"]["raw_depth_abs_rel"]["favorable_pairs"] == 2
