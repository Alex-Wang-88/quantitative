from __future__ import annotations

import json
import os
import time
from pathlib import Path

from quantpaper.config import AppConfig, ModelConfig, NotificationConfig
from quantpaper.data import make_demo_provider
from quantpaper.ledger import Ledger
import quantpaper.provenance
from quantpaper.report import ReportGenerator
from quantpaper.runtime import RuntimeService


def test_walkforward_metrics_are_attached_to_close_report(tmp_path) -> None:
    walkforward_path = tmp_path / "walkforward.json"
    walkforward_path.write_text(
        json.dumps(
            {
                "stats": {"fold_count": 2, "total_return": 0.12, "sharpe": 1.1},
                "quality": {
                    "ok": True,
                    "start_date": "2023-01-01",
                    "end_date": "2026-01-01",
                },
                "folds": [
                    {
                        "model_version": "fold-a",
                        "metrics": {
                            "test_ic_mean": 0.04,
                            "test_top_bottom_spread": 0.01,
                            "factor_coverage_mean": 0.2,
                        },
                    },
                    {
                        "model_version": "fold-b",
                        "metrics": {
                            "test_ic_mean": -0.02,
                            "test_top_bottom_spread": 0.03,
                            "factor_coverage_mean": 0.4,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(walkforward_path)),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    report = service.generate_report(notify=False, sync=False)

    metrics = report["research_metrics"]
    assert metrics["status"] == "OK"
    assert metrics["test_ic_mean"] == 0.01
    assert metrics["test_ic_positive_ratio"] == 0.5
    assert metrics["test_top_bottom_spread_mean"] == 0.02
    assert metrics["factor_coverage_mean"] == 0.3
    assert metrics["last_model_version"] == "fold-b"
    assert "样本外测试 IC" in (tmp_path / "reports" / f"{report['report_date']}.md").read_text(
        encoding="utf-8"
    )


def test_industry_line_items_cover_the_shared_disclosure_keys() -> None:
    """Both engines publish the same keys, so the daily report can render either.

    The live engine used to pin the book at ~12% exposure whenever the industry
    classification was missing; the disclosure makes that decision visible
    instead of silent, and it must line up with the research backtester's shape.
    """

    lines = dict(
        ReportGenerator._industry_lines(
            {
                "industry_cap_applied": False,
                "industry_cap_reason": (
                    "行业覆盖仅 0.0%（未知行业 5/5），行业上限无法分散风险，已停用"
                ),
                "sector_coverage": 0.0,
                "distinct_known_sectors": 0,
                "sector_source": "UNAVAILABLE:sector",
            }
        )
    )
    assert lines["行业上限是否执行"] == "否"
    assert lines["行业覆盖"] == "0.00%"
    assert lines["可识别行业数"] == "0"
    assert lines["行业字段来源"] == "UNAVAILABLE:sector"
    assert "已停用" in lines["判定依据"]

    fallback = dict(ReportGenerator._industry_lines({}))
    assert "未披露" in fallback["行业上限"]


def test_industry_disclosure_is_rendered_in_the_close_report(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(tmp_path / "missing.json")),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    report = service.generate_report(notify=False, sync=False)

    assert "industry_disclosure" in report
    markdown = (tmp_path / "reports" / f"{report['report_date']}.md").read_text(encoding="utf-8")
    assert "组合构建披露" in markdown
    assert "行业上限是否执行" in markdown

    html = (tmp_path / "reports" / f"{report['report_date']}.html").read_text(encoding="utf-8")
    assert "组合构建披露" in html


def test_security_status_line_items_cover_the_shared_disclosure_keys() -> None:
    """A status filter that runs on one path only changes the tradeable universe.

    Nothing else in the report would move, so both the presence and the absence
    of the filter have to be readable from the same key set.
    """

    lines = dict(
        ReportGenerator._security_status_lines(
            {
                "st_filter_applied": True,
                "suspension_filter_applied": True,
                "security_status_source": "security_master",
                "security_status_basis": "名称前缀 + is_suspended（当前快照）",
                "special_treatment_excluded": 7,
                "suspension_excluded": 3,
                "universe_note": "研究池含 ST，故两条路径可交易域不同",
            }
        )
    )
    assert lines["ST 过滤是否执行"] == "是"
    assert lines["停牌过滤是否执行"] == "是"
    assert lines["状态字段来源"] == "security_master"
    assert lines["已排除：名称含 ST"] == "7"
    assert lines["已排除：停牌"] == "3"
    assert "不同" in lines["与研究池的差异"]

    fallback = dict(ReportGenerator._security_status_lines({}))
    assert "未披露" in fallback["交易状态过滤"]


def test_security_status_disclosure_is_rendered_in_the_close_report(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(tmp_path / "missing.json")),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    report = service.generate_report(notify=False, sync=False)

    disclosure = report["security_status_disclosure"]
    assert disclosure["security_status_source"].startswith("UNAVAILABLE")
    # The demo provider has no status columns, so the honest answer is that
    # nothing was filtered -- and that must be visible, not blank.
    assert disclosure["st_filter_applied"] is False
    assert disclosure["universe_note"]

    markdown = (tmp_path / "reports" / f"{report['report_date']}.md").read_text(encoding="utf-8")
    assert "交易状态过滤披露" in markdown
    assert "ST 过滤是否执行" in markdown

    html = (tmp_path / "reports" / f"{report['report_date']}.html").read_text(encoding="utf-8")
    assert "交易状态过滤披露" in html
    assert "ST 过滤是否执行" in html


def _write_walkforward_report(path, *, age_seconds: float = 0.0) -> None:
    path.write_text(
        json.dumps(
            {
                "stats": {"fold_count": 1, "total_return": 0.05, "sharpe": 0.9},
                "quality": {
                    "ok": True,
                    "start_date": "2023-01-01",
                    "end_date": "2026-01-01",
                },
                "folds": [
                    {
                        "model_version": "fold-a",
                        "metrics": {
                            "test_ic_mean": 0.03,
                            "test_top_bottom_spread": 0.01,
                            "factor_coverage_mean": 0.5,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(path, (stamp, stamp))


def test_stale_walkforward_report_is_downgraded_and_disclosed(tmp_path) -> None:
    """A report written before the producing code was last edited used to render
    exactly like a fresh one.  It must instead lose its OK status and say why."""

    walkforward_path = tmp_path / "walkforward.json"
    _write_walkforward_report(walkforward_path)
    # STALE 的契约是"报告早于最新的产出模块"，而不是"报告比墙钟旧 30 天"：
    # 把报告时钟锚定到产出模块的真实 mtime，测试才不会随检出变旧而漂移。
    from quantpaper.provenance import DEFAULT_WALKFORWARD_PRODUCERS
    package_root = Path(quantpaper.provenance.__file__).resolve().parent
    stale_stamp = (
        max(
            (package_root / name).stat().st_mtime
            for name in DEFAULT_WALKFORWARD_PRODUCERS
        )
        - 30 * 86400
    )
    os.utime(walkforward_path, (stale_stamp, stale_stamp))
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(walkforward_path)),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    report = service.generate_report(notify=False, sync=False)
    metrics = report["research_metrics"]

    assert metrics["provenance_verdict"] == "STALE"
    assert metrics["status"] == "STALE"
    assert metrics["newer_producers"]
    assert metrics["report_written_at"]
    assert metrics["newest_producer_at"]
    # The underlying numbers are still carried through -- the point is to label
    # them, not to hide them.
    assert metrics["test_ic_mean"] == 0.03

    markdown = (tmp_path / "reports" / f"{report['report_date']}.md").read_text(encoding="utf-8")
    assert "报告版本比对：STALE" in markdown
    assert "【警告】该报告早于当前代码" in markdown

    html = (tmp_path / "reports" / f"{report['report_date']}.html").read_text(encoding="utf-8")
    assert "报告版本比对：STALE" in html
    assert "【警告】该报告早于当前代码" in html


def test_fresh_walkforward_report_keeps_ok_status(tmp_path) -> None:
    """False-positive check: the guard must not downgrade a report that is at
    least as new as every producer module."""

    walkforward_path = tmp_path / "walkforward.json"
    _write_walkforward_report(walkforward_path)
    service = RuntimeService(
        config=AppConfig(
            model=ModelConfig(walkforward_report_path=str(walkforward_path)),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = service.generate_report(notify=False, sync=False)["research_metrics"]

    assert metrics["provenance_verdict"] == "FRESH"
    assert metrics["status"] == "OK"
    assert metrics["newer_producers"] == []

