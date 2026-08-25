import hashlib
import json

from scripts.compact_uncertainty_report import compact_report


def test_compact_report_keeps_scalars_and_hashes_source(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(
        json.dumps(
            {
                "frames": [{"large": [1, 2, 3]}],
                "metadata": {"candidate_count": 3},
                "signals": {
                    "depth-gradient": {
                        "spearman": 0.25,
                        "auroc": 0.75,
                        "uncertainty_bins": [{"mean": 1.0}],
                        "sparsification": {"ause": 0.1, "curve": [1, 2]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    compact = compact_report(source)

    assert compact["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "frames" not in compact
    assert compact["signals"]["depth-gradient"]["spearman"] == 0.25
    assert compact["signals"]["depth-gradient"]["sparsification"] == {"ause": 0.1}
