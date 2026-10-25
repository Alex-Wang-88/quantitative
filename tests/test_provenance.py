from __future__ import annotations

import os
import time
from pathlib import Path

from quantpaper.provenance import (
    DEFAULT_WALKFORWARD_PRODUCERS,
    FRESH,
    STALE,
    UNKNOWN,
    report_provenance,
)


def _write(path: Path, text: str = "x = 1\n", *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_stale_when_a_producer_is_newer_than_the_artifact(tmp_path) -> None:
    """The whole point: a report older than the code that produced it is not
    evidence for the current code."""

    now = time.time()
    root = tmp_path / "pkg"
    _write(root / "model.py", mtime=now - 7200)
    _write(root / "ml_backtest.py", mtime=now - 1800)
    artifact = _write(tmp_path / "walkforward.json", "{}", mtime=now - 3600)

    result = report_provenance(artifact, package_root=root)

    assert result["verdict"] == STALE
    assert result["newer_producers"] == ["ml_backtest.py"]
    assert 1700 < result["lag_seconds"] < 1900
    assert result["artifact_written_at"] < result["newest_producer_at"]
    assert "早于产出代码" in result["detail"]
    # The module that is older than the report must not be blamed.
    assert "model.py" not in result["newer_producers"]
    assert set(result["compared_producers"]) == {"model.py", "ml_backtest.py"}


def test_fresh_when_the_artifact_postdates_every_producer(tmp_path) -> None:
    now = time.time()
    root = tmp_path / "pkg"
    _write(root / "model.py", mtime=now - 7200)
    _write(root / "ml_backtest.py", mtime=now - 3600)
    artifact = _write(tmp_path / "walkforward.json", "{}", mtime=now - 60)

    result = report_provenance(artifact, package_root=root)

    assert result["verdict"] == FRESH
    assert result["newer_producers"] == []
    assert result["lag_seconds"] == 0.0
    assert "版本比对通过" in result["detail"]


def test_same_second_write_is_not_treated_as_stale(tmp_path) -> None:
    """A module rewritten in the same second as the report is not a version
    change; the tolerance keeps the check from crying wolf on parallel writes."""

    now = time.time()
    root = tmp_path / "pkg"
    _write(root / "ml_backtest.py", mtime=now)
    artifact = _write(tmp_path / "walkforward.json", "{}", mtime=now - 0.5)

    assert report_provenance(artifact, package_root=root)["verdict"] == FRESH
    strict = report_provenance(artifact, package_root=root, tolerance_seconds=0.0)
    assert strict["verdict"] == STALE


def test_missing_artifact_is_unknown_never_fresh(tmp_path) -> None:
    root = tmp_path / "pkg"
    _write(root / "ml_backtest.py")

    result = report_provenance(tmp_path / "nope.json", package_root=root)

    assert result["verdict"] == UNKNOWN
    assert result["artifact_written_at"] is None
    assert "未找到报告文件" in result["detail"]


def test_missing_producers_are_unknown_never_fresh(tmp_path) -> None:
    """If the check cannot run, it must not report success.  An unverifiable
    artifact is exactly the case this guard exists for."""

    artifact = _write(tmp_path / "walkforward.json", "{}")

    result = report_provenance(artifact, package_root=tmp_path / "empty")

    assert result["verdict"] == UNKNOWN
    assert result["compared_producers"] == []
    assert "未找到报告产出模块" in result["detail"]


def test_producers_accept_explicit_absolute_paths(tmp_path) -> None:
    now = time.time()
    _write(tmp_path / "elsewhere" / "custom_runner.py", mtime=now)
    artifact = _write(tmp_path / "walkforward.json", "{}", mtime=now - 600)

    result = report_provenance(
        artifact,
        producers=[tmp_path / "elsewhere" / "custom_runner.py"],
    )

    assert result["verdict"] == STALE
    assert result["newer_producers"] == ["custom_runner.py"]


def test_shipped_report_is_compared_against_the_shipped_modules() -> None:
    """Runs the guard against the real repository layout so the wiring is
    exercised outside a fixture.  Today this artifact IS stale (written
    2026-09-15, producer edited 2026-09-18); the assertion deliberately checks
    only that the comparison ran, so re-running the walk-forward does not break
    the test."""

    repo_root = Path(__file__).resolve().parents[1]
    result = report_provenance(repo_root / "data" / "reports" / "lightgbm-walkforward-3y.json")

    if result["artifact_written_at"] is None:
        return  # report not present in this checkout
    assert result["verdict"] in {FRESH, STALE}
    assert "ml_backtest.py" in result["compared_producers"]
    assert set(DEFAULT_WALKFORWARD_PRODUCERS).issuperset({"ml_backtest.py", "model.py"})
