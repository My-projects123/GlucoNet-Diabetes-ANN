from __future__ import annotations

from pathlib import Path

from diabetes_prediction.run import main


def test_profile_only_writes_reports(tiny_csv: Path, tmp_path: Path) -> None:
    code = main(
        [
            "--data",
            str(tiny_csv),
            "--profile-only",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "0",
        ]
    )
    assert code == 0
    assert (tmp_path / "data_profile.md").exists()
    assert (tmp_path / "data_profile.json").exists()
    assert (tmp_path / "figures" / "outcome_balance.png").exists()
    assert (tmp_path / "figures" / "feature_correlations.png").exists()
    assert not (tmp_path / "metrics.csv").exists()
