"""Compare two segmented walk-forward reports side by side.

The walk-forward report schema changed while the candidate model was being
repaired: the pre-fix report (``candidate-v2``) predates the health gate, so it
has no ``degenerate_folds`` and no per-fold ``test_ic_mean``.  This script reads
both the old and the new schema, lines the periods up by name, and prints a
markdown table plus the promotion decision of each run.

Usage::

    python scripts/compare_walkforward_reports.py \
        --baseline data/reports/benchmark-excess-walkforward.json \
        --candidate data/reports/benchmark-excess-walkforward-v3.json \
        --output data/reports/benchmark-excess-walkforward-comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _period_lookup(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(period["period"]["name"]): period for period in report.get("periods", [])}


def _fold_metric_stats(period: dict[str, Any]) -> dict[str, float]:
    """Aggregate per-fold out-of-sample diagnostics for one period.

    ``model_degenerate`` only exists once the health gate is in place.  When no
    fold carries that key the period predates the gate, so the count is reported
    as ``-1.0`` ("not measured") rather than ``0`` ("measured, all healthy") --
    conflating the two would let an unchecked run look like a clean one.
    """

    folds = period.get("folds") or []
    ic_values: list[float] = []
    unique_values: list[float] = []
    best_iterations: list[float] = []
    degenerate = 0.0
    degenerate_measured = False
    for fold in folds:
        metrics = fold.get("metrics") or {}
        if "test_ic_mean" in metrics:
            ic_values.append(float(metrics["test_ic_mean"]))
        if "validation_unique_predictions_per_day" in metrics:
            unique_values.append(float(metrics["validation_unique_predictions_per_day"]))
        if "validation_best_iteration" in metrics:
            best_iterations.append(float(metrics["validation_best_iteration"]))
        if "model_degenerate" in metrics:
            degenerate_measured = True
            degenerate += float(metrics["model_degenerate"])
    payload: dict[str, float] = {
        "fold_count": float(len(folds)),
        "degenerate_measured": 1.0 if degenerate_measured else 0.0,
        "degenerate_folds": degenerate if degenerate_measured else -1.0,
    }
    if ic_values:
        payload["test_ic_mean"] = sum(ic_values) / len(ic_values)
        payload["test_ic_positive_ratio"] = sum(1 for v in ic_values if v > 0) / len(ic_values)
    if unique_values:
        payload["unique_predictions_per_day_min"] = min(unique_values)
    if best_iterations:
        payload["best_iteration_median"] = sorted(best_iterations)[len(best_iterations) // 2]
    return payload


def _bucket_metric_stats(period: dict[str, Any]) -> dict[str, float]:
    """Per-fold selection quality of the bucket the portfolio actually trades.

    ``test_top_bottom_spread`` covers the top quintile (~1000 names); the book
    holds ``max_positions`` (~30).  Those are two orders of magnitude apart, and
    the narrow one is the only number that says whether a losing strategy needs
    a better model or a better broker.
    """

    folds = period.get("folds") or []
    bucket: list[float] = []
    universe: list[float] = []
    positive: list[float] = []
    quintile: list[float] = []
    sizes: list[float] = []
    for fold in folds:
        metrics = fold.get("metrics") or {}
        if "test_bucket_label_mean" in metrics:
            bucket.append(float(metrics["test_bucket_label_mean"]))
        if "test_universe_label_mean" in metrics:
            universe.append(float(metrics["test_universe_label_mean"]))
        if "test_bucket_label_positive_ratio" in metrics:
            positive.append(float(metrics["test_bucket_label_positive_ratio"]))
        if "test_top_bottom_spread" in metrics:
            quintile.append(float(metrics["test_top_bottom_spread"]))
        if "test_bucket_size" in metrics:
            sizes.append(float(metrics["test_bucket_size"]))
    payload: dict[str, float] = {}
    if sizes:
        payload["bucket_size"] = sizes[-1]
    if bucket:
        payload["bucket_label_mean"] = sum(bucket) / len(bucket)
    if universe:
        payload["universe_label_mean"] = sum(universe) / len(universe)
    if positive:
        payload["bucket_positive_ratio"] = sum(positive) / len(positive)
    if quintile:
        payload["quintile_spread"] = sum(quintile) / len(quintile)
    return payload


def _traded_bucket_stats(period: dict[str, Any]) -> dict[str, float]:
    """Bucket quality of the score the book was actually built from.

    Only present once a run distinguishes "the model's raw ranking" from "the
    ranking that was traded" (a smoothing transform makes them different
    objects).  Reports nothing rather than falling back to the raw number, so a
    reader cannot mistake one for the other.
    """

    stats = period.get("stats") or {}
    payload: dict[str, float] = {}
    if "traded_bucket_label_mean" in stats:
        payload["traded_bucket_label_mean"] = float(stats["traded_bucket_label_mean"])
    if "traded_universe_label_mean" in stats:
        payload["traded_universe_label_mean"] = float(
            stats["traded_universe_label_mean"]
        )
    if "traded_bucket_label_positive_ratio" in stats:
        payload["traded_bucket_positive_ratio"] = float(
            stats["traded_bucket_label_positive_ratio"]
        )
    return payload


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "未测量（旧报告无此字段）"
    return f"{float(value):.2%}"


def _fmt_count(value: Any) -> str:
    if value is None:
        return "未测量（旧报告无此字段）"
    return f"{float(value):.0f}"


def _fmt_money(value: Any) -> str:
    """Absolute frictions on the 1,000,000 CNY notional the suite starts from."""

    if value is None:
        return "未测量（旧报告无此字段）"
    return f"{float(value):,.0f} 元"


def run_label(report: dict[str, Any], stem: str) -> str:
    """Identify a run by its file plus the config it actually recorded.

    ``candidate_profile`` alone is not an identity: several runs share it and
    differ only in the backtest block, so a comparison labelled with the profile
    would show the same name twice and hide which side is which.
    """

    config = report.get("backtest") or {}
    walkforward = report.get("walkforward") or {}
    parts: list[str] = []
    # horizon_days lives in the walkforward block, not backtest, but it is the
    # single field that separates two otherwise-identical runs (same universe,
    # same book, same turnover cap), so it belongs in the run identity too.
    if "horizon_days" in walkforward:
        parts.append(f"标签 {walkforward['horizon_days']} 日")
    if "rebalance_every_n_days" in config:
        parts.append(f"调仓 {config['rebalance_every_n_days']} 日")
    if "max_daily_turnover" in config:
        parts.append(f"双边换手 ≤ {float(config['max_daily_turnover']):.0%}")
    if "max_positions" in config:
        parts.append(f"持仓 {config['max_positions']}")
    return f"`{stem}`" + (f"（{'，'.join(parts)}）" if parts else "（未记录配置）")


def render(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    base_label: str = "baseline",
    cand_label: str = "candidate",
) -> str:
    base_periods = _period_lookup(baseline)
    cand_periods = _period_lookup(candidate)
    order = [name for name in cand_periods] or list(base_periods)

    lines: list[str] = []
    lines.append("# 分段 Walk-forward 对照\n")
    lines.append(f"- 基线：{base_label}")
    lines.append(f"- 候选：{cand_label}")
    lines.append(f"- 标签定义（候选）：{candidate.get('label_definition', 'n/a')}")
    lines.append("")
    lines.append("## 分段超额收益\n")
    lines.append("| 区间 | 基线策略 | 基线基准 | 基线超额 | 候选策略 | 候选基准 | 候选超额 | 超额改善 |")  # noqa: E501
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in order:
        base = base_periods.get(name)
        cand = cand_periods.get(name)
        base_stats = (base or {}).get("stats", {})
        cand_stats = (cand or {}).get("stats", {})
        base_excess = base_stats.get("excess_return")
        cand_excess = cand_stats.get("excess_return")
        delta = (
            None
            if base_excess is None or cand_excess is None
            else float(cand_excess) - float(base_excess)
        )
        lines.append(
            "| {name} | {bs} | {bb} | {be} | {cs} | {cb} | {ce} | {delta} |".format(
                name=name,
                bs=_fmt_pct(base_stats.get("total_return")),
                bb=_fmt_pct(base_stats.get("benchmark_return")),
                be=_fmt_pct(base_excess),
                cs=_fmt_pct(cand_stats.get("total_return")),
                cb=_fmt_pct(cand_stats.get("benchmark_return")),
                ce=_fmt_pct(cand_excess),
                delta=_fmt_pct(delta),
            )
        )

    lines.append("")
    lines.append("## 模型健康度（候选，逐折聚合）\n")
    lines.append("| 区间 | 折数 | 退化折 | 折均 test IC | test IC 为正占比 | 单日最少不同预测数 | 最佳迭代(中位) |")  # noqa: E501
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name in order:
        cand = cand_periods.get(name)
        if cand is None:
            continue
        stats = _fold_metric_stats(cand)
        lines.append(
            "| {name} | {fold_count:.0f} | {degenerate_folds} | {ic} | {pos} | {uniq} | {bestiter} |".format(  # noqa: E501
                name=name,
                fold_count=stats["fold_count"],
                degenerate_folds=(
                    f"{stats['degenerate_folds']:.0f}"
                    if stats["degenerate_measured"] > 0.0
                    else "未检测（旧报告）"
                ),
                ic=(
                    f"{stats['test_ic_mean']:+.4f}"
                    if "test_ic_mean" in stats
                    else "n/a（旧报告无此字段）"
                ),
                pos=(
                    f"{stats['test_ic_positive_ratio']:.0%}"
                    if "test_ic_positive_ratio" in stats
                    else "n/a"
                ),
                uniq=(
                    f"{stats['unique_predictions_per_day_min']:.0f}"
                    if "unique_predictions_per_day_min" in stats
                    else "n/a"
                ),
                bestiter=(
                    f"{stats['best_iteration_median']:.0f}"
                    if "best_iteration_median" in stats
                    else "n/a"
                ),
            )
        )

    lines.append("")
    lines.append("## 实际交易分位的选股质量（候选，逐折聚合）\n")
    lines.append(
        "标签已是「个股收益 − 指数收益」，所以**桶均超额**就是该分位相对基准的超额，"
        "不需要再减全市场。桶大小 = `max_positions`；`前20%分位价差` 是既有口径，"
        "覆盖约 1000 只，比组合实际持有的 30 只宽两个数量级 —— 两个都要看。\n"
    )
    lines.append(
        "**`原始排名` 与 `实际交易` 是两回事**：前者是模型原始打分的 top-N，"
        "后者是组合真正据以下单的那份打分（例如做过平滑）。只报前者会把"
        "「换了打分」误读成「换了执行」—— 两者都列出来，未测量时如实留空。\n"
    )
    lines.append(
        "| 区间 | 桶大小 | 原始排名桶均超额 | **实际交易桶均超额** | 实际交易桶为正的折占比 "
        "| 全市场均超额 | 前20%分位价差 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name in order:
        cand = cand_periods.get(name)
        if cand is None:
            continue
        stats = _bucket_metric_stats(cand)
        traded = _traded_bucket_stats(cand)
        if "bucket_label_mean" not in stats and "traded_bucket_label_mean" not in traded:
            # The quintile spread may be present while the traded-bucket fields
            # are not; the row must not silently print a zero bucket instead.
            lines.append(f"| {name} | 未测量（旧报告无此字段） | — | — | — | — | — |")
            continue
        lines.append(
            "| {name} | {size} | {bucket} | **{traded}** | {traded_pos} | {universe} "
            "| {quintile} |".format(
                name=name,
                size=f"{stats.get('bucket_size', 0.0):.0f}" if stats else "n/a",
                bucket=_fmt_pct(stats.get("bucket_label_mean")),
                traded=_fmt_pct(traded.get("traded_bucket_label_mean")),
                traded_pos=(
                    f"{traded['traded_bucket_positive_ratio']:.0%}"
                    if "traded_bucket_positive_ratio" in traded
                    else "n/a"
                ),
                universe=_fmt_pct(stats.get("universe_label_mean")),
                quintile=_fmt_pct(stats.get("quintile_spread")),
            )
        )
    lines.append("")
    lines.append("## 仓位与换手（候选，已排除建仓日）\n")
    lines.append(
        "建仓日是从全现金补到目标仓位的一次性部署，按约定豁免换手上限；"
        "下列 `最大单日换手` 指建仓日之外的最大值，即真正的换手上限是否被守住。\n"
    )
    lines.append(
        "| 区间 | 建仓日换手 | 建仓后最大单日换手 | 平均仓位 | 最低仓位 | 期末仓位 "
        "| 换手受限调仓日 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name in order:
        cand = cand_periods.get(name)
        if cand is None:
            continue
        stats = cand.get("stats", {})
        lines.append(
            "| {name} | {entry} | {dmax} | {avg} | {low} | {last} | {capped} |".format(
                name=name,
                entry=_fmt_ratio(stats.get("entry_day_turnover")),
                dmax=_fmt_ratio(stats.get("daily_turnover_max")),
                avg=_fmt_ratio(stats.get("invested_ratio_mean")),
                low=_fmt_ratio(stats.get("invested_ratio_min")),
                last=_fmt_ratio(stats.get("invested_ratio_final")),
                capped=_fmt_count(stats.get("rotation_capped_days")),
            )
        )
    lines.append("")
    lines.append("## 成本拖累（候选，累计 / 平均权益）\n")
    lines.append(
        "佣金与印花税按笔记账，滑点固化在成交价里、没有独立字段，因此由**成交名义额**反推。"
        "在 3.5–4.7 倍单边年换手下，摩擦成本与 alpha 同一量级 —— "
        "只报净值无法区分「信号弱」和「券商贵」，必须单独列出来。\n"
    )
    lines.append("| 区间 | 成交名义额/权益 | 佣金+印花税 | 滑点 | **成本拖累合计** |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in order:
        cand = cand_periods.get(name)
        if cand is None:
            continue
        stats = cand.get("stats", {})
        lines.append(
            "| {name} | {notional} | {fees} | {slippage} | **{drag}** |".format(
                name=name,
                notional=_fmt_ratio(stats.get("traded_notional_ratio")),
                fees=_fmt_money(stats.get("fees_paid")),
                slippage=_fmt_money(stats.get("slippage_paid")),
                drag=_fmt_ratio(stats.get("cost_drag_ratio")),
            )
        )
    lines.append("")
    lines.append("## 晋升判定\n")
    for label, report in (("基线", baseline), ("候选", candidate)):
        status = report.get("status") or report.get("promotion") or "n/a"
        reason = report.get("promotion_reason", "")
        degenerate = report.get("degenerate_folds")
        if isinstance(degenerate, (int, float)):
            suffix = f"（退化折 {degenerate:g}）"
        else:
            suffix = "（未检测：旧报告早于健康门）"
        lines.append(f"- **{label}**：`{status}`{suffix}")
        if reason:
            lines.append(f"  - {reason}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="对照两份分段 walk-forward 报告")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    report = render(
        baseline,
        candidate,
        base_label=run_label(baseline, baseline_path.stem),
        cand_label=run_label(candidate, candidate_path.stem),
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"written: {output.resolve()}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
