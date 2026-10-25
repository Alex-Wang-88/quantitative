from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .broker import PaperBroker
from .domain import DailyReport, RiskEvent, Signal, jsonable, utc_now
from .ledger import Ledger


class ReportGenerator:
    def __init__(
        self, output_dir: str | Path = "data/reports", ledger: Ledger | None = None
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger

    def generate(
        self,
        broker: PaperBroker,
        signals: list[Signal],
        risks: list[RiskEvent],
        report_date: str | None = None,
        model_version: str = "unknown",
        as_of=None,
        run_id: str = "",
        strategy_version: str = "baseline-v0.1",
        model_name: str = "unknown",
        model_metrics: dict[str, float] | None = None,
        factor_coverage: float = 0.0,
        data_status: str = "UNKNOWN",
        research_metrics: dict[str, Any] | None = None,
        industry_disclosure: dict[str, Any] | None = None,
        security_status_disclosure: dict[str, Any] | None = None,
    ) -> DailyReport:
        generated_at = utc_now()
        as_of = as_of or generated_at
        report_date = report_date or as_of.astimezone().date().isoformat()
        account = broker.snapshot(as_of)
        day_fills = [
            fill
            for fill in broker.fills
            if fill.timestamp.astimezone().date().isoformat() == report_date
        ]
        day_orders = [
            order
            for order in broker.orders.values()
            if order.created_at.astimezone().date().isoformat() == report_date
        ]
        notional = sum(fill.quantity * fill.price for fill in day_fills)
        turnover = notional / account.initial_capital if account.initial_capital else 0.0
        status = (
            "DEGRADED"
            if any(item.severity == "HIGH" for item in risks)
            or data_status in {"STALE", "UNAVAILABLE"}
            else "OK"
        )
        summary = (
            f"纸盘净值 {account.total_equity:,.2f}；现金 {account.cash:,.2f}；"
            f"持仓 {len(broker.positions)} 个；成交 {len(day_fills)} 笔。"
        )
        report = DailyReport(
            report_date=report_date,
            generated_at=generated_at,
            status=status,
            model_version=model_version,
            account=account,
            signals_count=len(signals),
            orders_count=len(day_orders),
            fills_count=len(day_fills),
            turnover=round(turnover, 6),
            summary=summary,
            risk_events=risks,
            run_id=run_id,
            strategy_version=strategy_version,
            model_name=model_name,
            model_metrics=dict(model_metrics or {}),
            factor_coverage=round(float(factor_coverage), 6),
            data_status=data_status,
            research_metrics=dict(research_metrics or {}),
            industry_disclosure=dict(industry_disclosure or {}),
            security_status_disclosure=dict(security_status_disclosure or {}),
        )
        payload = jsonable(report)
        (self.output_dir / f"{report_date}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.output_dir / f"{report_date}.md").write_text(self._markdown(report), encoding="utf-8")
        (self.output_dir / f"{report_date}.html").write_text(self._html(report), encoding="utf-8")
        if self.ledger:
            self.ledger.save_report(report_date, report, generated_at)
        return report

    @staticmethod
    def _markdown(report: DailyReport) -> str:
        risks = (
            "\n".join(
                f"- [{risk.severity}] {risk.code}: {risk.message}" for risk in report.risk_events
            )
            or "- 无"
        )
        research = ReportGenerator._research_markdown(report.research_metrics)
        industry = ReportGenerator._industry_markdown(report.industry_disclosure)
        status = ReportGenerator._security_status_markdown(report.security_status_disclosure)
        return f"""# A 股纸盘日报 {report.report_date}

- 状态：{report.status}
- 模型：{report.model_version}
- 模型类型：{report.model_name}
- 基本面覆盖：{report.factor_coverage:.2%}
- 数据状态：{report.data_status}
- 净值：{report.account.total_equity:,.2f}
- 现金：{report.account.cash:,.2f}
- 换手：{report.turnover:.2%}
- 信号：{report.signals_count}
- 订单：{report.orders_count}
- 成交：{report.fills_count}

## 摘要

{report.summary}

## 模型研究评估

{research}

## 组合构建披露

{industry}

## 交易状态过滤披露

{status}

## 风险事件

{risks}
"""

    @staticmethod
    def _html(report: DailyReport) -> str:
        css = (
            "body{font-family:Arial,sans-serif;background:#020617;color:#f8fafc;padding:32px}"
            "section{background:#0e1223;border:1px solid #334155;border-radius:12px;"
            "padding:20px;max-width:720px}"
            "li{margin:8px 0;color:#94a3b8}"
        )
        research = ReportGenerator._research_html(report.research_metrics)
        industry = ReportGenerator._industry_html(report.industry_disclosure)
        status = ReportGenerator._security_status_html(report.security_status_disclosure)
        return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>A股纸盘日报 {report.report_date}</title>
<style>{css}</style>
<section><h1>A股纸盘日报 {report.report_date}</h1><p>{report.summary}</p><ul>
<li>状态：{report.status}</li><li>模型：{report.model_version}</li><li>模型类型：{report.model_name}</li>
<li>基本面覆盖：{report.factor_coverage:.2%}</li><li>数据状态：{report.data_status}</li><li>换手：{report.turnover:.2%}</li>
<li>风险事件：{len(report.risk_events)}</li></ul><h2>模型研究评估</h2><ul>{research}</ul>
<h2>组合构建披露</h2><ul>{industry}</ul>
<h2>交易状态过滤披露</h2><ul>{status}</ul></section></html>"""

    @staticmethod
    def _format_percent(value: Any) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.2%}"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _format_number(value: Any, digits: int = 2) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    @classmethod
    def _research_provenance_lines(cls, metrics: dict[str, Any]) -> list[tuple[str, str]]:
        """Label/value pairs describing whether the cited report matches the code.

        The walk-forward summary is the only research evidence the close report
        quotes.  Without this block, a report written before the producing
        module was last edited is rendered exactly like a fresh one.

        Returns nothing when the research section did not reach the comparison
        at all (missing or unreadable file) -- those cases already say so, and
        five lines of ``UNKNOWN`` would just be noise.
        """

        if not metrics.get("provenance_verdict"):
            return []
        verdict = str(metrics["provenance_verdict"])
        lag = metrics.get("provenance_lag_seconds")
        lines: list[tuple[str, str]] = [
            ("报告版本比对", verdict),
            ("报告写于", str(metrics.get("report_written_at") or "--")),
            ("产出代码最新修改", str(metrics.get("newest_producer_at") or "--")),
        ]
        newer = metrics.get("newer_producers") or []
        if newer:
            lines.append(("报告生成后又被修改的模块", "、".join(str(name) for name in newer)))
        if isinstance(lag, (int, float)) and lag:
            lines.append(("滞后", f"{float(lag):.0f} 秒"))
        lines.append(("比对结论", str(metrics.get("provenance_detail") or "--")))
        return lines

    @classmethod
    def _research_markdown(cls, metrics: dict[str, Any]) -> str:
        if not metrics:
            return "- 未加载 Walk-forward 研究报告"
        lines = [
            f"- 状态：{metrics.get('status', 'UNKNOWN')}",
            f"- 数据区间：{metrics.get('data_start', '--')} 至 {metrics.get('data_end', '--')}",
            f"- 测试折数：{metrics.get('fold_count', '--')}",
            f"- 样本外测试 IC：{cls._format_percent(metrics.get('test_ic_mean'))}",
            f"- 正 IC 折数比例：{cls._format_percent(metrics.get('test_ic_positive_ratio'))}",
            "- Top/Bottom 分组收益差："
            f"{cls._format_percent(metrics.get('test_top_bottom_spread_mean'))}",
            f"- Walk-forward 总收益：{cls._format_percent(metrics.get('total_return'))}",
            f"- Sharpe：{cls._format_number(metrics.get('sharpe'))}",
            f"- 因子覆盖率：{cls._format_percent(metrics.get('factor_coverage_mean'))}",
        ]
        lines.extend(
            f"- {label}：{value}" for label, value in cls._research_provenance_lines(metrics)
        )
        if str(metrics.get("provenance_verdict")) == "STALE":
            lines.append("- 【警告】该报告早于当前代码，引用前必须重跑，不得作为当前模型证据")
        lines.append(f"- 说明：{metrics.get('message', '仅用于研究评估，不自动晋级')}")
        return "\n".join(lines)

    @classmethod
    def _industry_lines(cls, disclosure: dict[str, Any]) -> list[tuple[str, str]]:
        """Shared label/value pairs for the industry-cap disclosure.

        Both engines publish the same five keys, so the live daily report and the
        research backtester can be read side by side instead of the live book
        quietly under-investing whenever the classification is missing.
        """

        if not disclosure:
            return [("行业上限", "未披露（本次未构建组合或无持仓信号）")]
        applied = bool(disclosure.get("industry_cap_applied"))
        return [
            ("行业上限是否执行", "是" if applied else "否"),
            ("行业覆盖", cls._format_percent(disclosure.get("sector_coverage"))),
            ("可识别行业数", str(disclosure.get("distinct_known_sectors", "--"))),
            ("行业字段来源", str(disclosure.get("sector_source", "--"))),
            ("判定依据", str(disclosure.get("industry_cap_reason", "--"))),
        ]

    @classmethod
    def _industry_markdown(cls, disclosure: dict[str, Any]) -> str:
        return "\n".join(f"- {label}：{value}" for label, value in cls._industry_lines(disclosure))

    @classmethod
    def _industry_html(cls, disclosure: dict[str, Any]) -> str:
        return "".join(
            f"<li>{label}：{value}</li>" for label, value in cls._industry_lines(disclosure)
        )

    @classmethod
    def _security_status_lines(cls, disclosure: dict[str, Any]) -> list[tuple[str, str]]:
        """Label/value pairs for the trading-status filter disclosure.

        A status filter that runs on one path and not the other changes which
        names a book can hold without changing any headline number, so its
        absence has to be as visible as its presence.
        """

        if not disclosure:
            return [("交易状态过滤", "未披露")]
        return [
            ("ST 过滤是否执行", "是" if disclosure.get("st_filter_applied") else "否"),
            (
                "停牌过滤是否执行",
                "是" if disclosure.get("suspension_filter_applied") else "否",
            ),
            ("状态字段来源", str(disclosure.get("security_status_source", "--"))),
            ("判定依据", str(disclosure.get("security_status_basis", "--"))),
            (
                "已排除：名称含 ST",
                str(disclosure.get("special_treatment_excluded", 0)),
            ),
            ("已排除：停牌", str(disclosure.get("suspension_excluded", 0))),
            ("与研究池的差异", str(disclosure.get("universe_note", "--"))),
            ("保证金可用证券", str(disclosure.get("marginable_securities", 0))),
            ("融券可用证券", str(disclosure.get("shortable_securities", 0))),
            ("账户能力说明", str(disclosure.get("capability_note", "--"))),
        ]

    @classmethod
    def _security_status_markdown(cls, disclosure: dict[str, Any]) -> str:
        return "\n".join(
            f"- {label}：{value}" for label, value in cls._security_status_lines(disclosure)
        )

    @classmethod
    def _security_status_html(cls, disclosure: dict[str, Any]) -> str:
        return "".join(
            f"<li>{label}：{value}</li>"
            for label, value in cls._security_status_lines(disclosure)
        )

    @classmethod
    def _research_html(cls, metrics: dict[str, Any]) -> str:
        if not metrics:
            return "<li>未加载 Walk-forward 研究报告</li>"
        items = [
            f"<li>状态：{metrics.get('status', 'UNKNOWN')}</li>",
            "<li>数据区间："
            f"{metrics.get('data_start', '--')} 至 {metrics.get('data_end', '--')}</li>",
            f"<li>测试折数：{metrics.get('fold_count', '--')}</li>",
            f"<li>样本外测试 IC：{cls._format_percent(metrics.get('test_ic_mean'))}</li>",
            "<li>正 IC 折数比例："
            f"{cls._format_percent(metrics.get('test_ic_positive_ratio'))}</li>",
            "<li>Top/Bottom 分组收益差："
            f"{cls._format_percent(metrics.get('test_top_bottom_spread_mean'))}</li>",
            f"<li>Walk-forward 总收益：{cls._format_percent(metrics.get('total_return'))}</li>",
            f"<li>Sharpe：{cls._format_number(metrics.get('sharpe'))}</li>",
            f"<li>因子覆盖率：{cls._format_percent(metrics.get('factor_coverage_mean'))}</li>",
        ]
        items.extend(
            f"<li>{label}：{value}</li>"
            for label, value in cls._research_provenance_lines(metrics)
        )
        if str(metrics.get("provenance_verdict")) == "STALE":
            items.append(
                "<li><strong>【警告】该报告早于当前代码，引用前必须重跑，"
                "不得作为当前模型证据</strong></li>"
            )
        items.append(f"<li>说明：{metrics.get('message', '仅用于研究评估，不自动晋级')}</li>")
        return "".join(items)
