"""Report provenance checks.

A published research artifact is only evidence for the code that produced it.
When a producing module is edited *after* the artifact was written, the numbers
in that artifact may come from a version of the code that no longer exists --
yet every consumer keeps citing it as current.

That is not hypothetical here.  ``data/reports/lightgbm-walkforward-3y.json``
was written on 2026-09-15 17:24 while ``ml_backtest.py`` was last edited on
2026-09-18 11:55, so the close report kept quoting a walk-forward summary whose
empty-fold handling had already been changed underneath it.

This module makes that comparison mechanical: given an artifact path and the
modules that produce it, report whether any producer is newer than the artifact.
It deliberately compares **file modification times**, which is a coarse but
honest signal -- it over-reports rather than under-reports, and it never claims
freshness it cannot prove.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")

#: Modules whose behaviour determines the numbers inside a walk-forward report.
#: Editing any of them invalidates an older report as evidence.
DEFAULT_WALKFORWARD_PRODUCERS: tuple[str, ...] = (
    "ml_backtest.py",
    "model.py",
    "features.py",
    "backtest.py",
    "domain.py",
)

FRESH = "FRESH"
STALE = "STALE"
UNKNOWN = "UNKNOWN"

#: Writes inside the same second are not treated as a version change.
DEFAULT_TOLERANCE_SECONDS = 1.0


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _format_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, SHANGHAI).isoformat(timespec="seconds")


def _resolve_producers(
    producers: Iterable[str | Path], package_root: Path
) -> list[tuple[str, Path]]:
    resolved: list[tuple[str, Path]] = []
    for entry in producers:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = package_root / candidate
        if candidate.is_file():
            resolved.append((candidate.name, candidate))
    return resolved


def report_provenance(
    artifact: str | Path,
    *,
    producers: Iterable[str | Path] | None = None,
    package_root: str | Path | None = None,
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Compare a report artifact against the modules that produce it.

    Returns a disclosure dict with a ``verdict`` of ``FRESH``, ``STALE`` or
    ``UNKNOWN``.  ``UNKNOWN`` means the check could not run (missing artifact or
    no resolvable producer modules) -- it is never reported as fresh, because an
    unverifiable artifact is exactly the case this guard exists for.
    """

    root = Path(package_root) if package_root is not None else _package_root()
    artifact_path = Path(artifact)
    selected = producers if producers is not None else DEFAULT_WALKFORWARD_PRODUCERS
    resolved = _resolve_producers(selected, root)

    base: dict[str, Any] = {
        "verdict": UNKNOWN,
        "artifact": artifact_path.name,
        "artifact_written_at": None,
        "newest_producer_at": None,
        "newer_producers": [],
        "compared_producers": [name for name, _ in resolved],
        "lag_seconds": None,
        "tolerance_seconds": float(tolerance_seconds),
        "detail": "",
    }

    if not artifact_path.is_file():
        base["detail"] = "未找到报告文件，无法比对代码版本；不视为有效证据"
        return base

    artifact_mtime = artifact_path.stat().st_mtime
    base["artifact_written_at"] = _format_mtime(artifact_mtime)

    if not resolved:
        base["detail"] = "未找到报告产出模块，无法比对代码版本；不视为有效证据"
        return base

    newest_name, newest_path = max(resolved, key=lambda item: item[1].stat().st_mtime)
    newest_mtime = newest_path.stat().st_mtime
    base["newest_producer_at"] = _format_mtime(newest_mtime)

    newer = sorted(
        name
        for name, path in resolved
        if path.stat().st_mtime > artifact_mtime + tolerance_seconds
    )
    if newer:
        base["verdict"] = STALE
        base["newer_producers"] = newer
        base["lag_seconds"] = round(newest_mtime - artifact_mtime, 3)
        base["detail"] = (
            f"报告早于产出代码：{', '.join(newer)} 在报告生成后又被修改，"
            f"最新修改 {base['newest_producer_at']}，报告写于 "
            f"{base['artifact_written_at']}（滞后 {base['lag_seconds']:.0f}s）；"
            "报告数字可能来自已经不存在的代码版本，需重跑后再引用"
        )
        return base

    base["verdict"] = FRESH
    base["newest_producer_at"] = _format_mtime(newest_mtime)
    base["lag_seconds"] = 0.0
    base["detail"] = (
        f"报告晚于全部 {len(resolved)} 个产出模块（最新 {newest_name} "
        f"{base['newest_producer_at']}），版本比对通过"
    )
    return base
