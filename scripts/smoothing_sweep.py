"""Sweep portfolio-side knobs over a single fit per period.

Training dominates the cost of a walk-forward run: one fit per period takes
minutes while every knob that turns scores into a book costs seconds.  Testing a
grid of settings by re-running the whole script per cell would cost an
afternoon; fitting once and replaying the backtest costs one run.

Despite the filename (kept so the reports already produced stay traceable), this
sweeps **any** portfolio-side knob, not just the score-smoothing window:
``--windows`` (score smoothing), ``--rebalance`` (rebalance interval),
``--hold-buffer`` (incumbent retention) and ``--turnover`` (two-sided cap).  The
cross product is replayed over byte-identical models, which is what makes the
cells comparable.

The ``--windows 1`` / ``--rebalance <config>`` / ``--hold-buffer 0`` /
``--turnover <config>`` cell is the baseline control: it reproduces the published
run exactly, which both anchors the table and serves as the equivalence check for
the fit/backtest split in ``LightGBMWalkForwardBacktester``.

Usage::

    # Is the smoothing result a plateau or a lucky point?
    python scripts/smoothing_sweep.py --windows 1,5,10,20,40,60

    # Can less turnover hold the same basket?
    python scripts/smoothing_sweep.py --windows 20 \
        --rebalance 5,10,20 --hold-buffer 0,0.25,0.5
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

from quantpaper.backtest import BacktestConfig
from quantpaper.config import load_config
from quantpaper.features import FeatureEngine
from quantpaper.ml_backtest import LightGBMWalkForwardBacktester, WalkForwardConfig
from quantpaper.model import LightGBMModel
from quantpaper.period_backtest import BacktestPeriod, parse_periods
from quantpaper.storage import HistoricalStore

sys.path.insert(0, str(Path(__file__).parent))
from walkforward_benchmark_excess import benchmark_return  # noqa: E402

LABEL_MODES = ("benchmark_excess", "cross_sectional")

METRIC_KEYS = (
    "total_return",
    "turnover",
    "one_way_turnover",
    "cost_drag_ratio",
    "invested_ratio_mean",
    "invested_ratio_min",
    "invested_ratio_final",
    "daily_turnover_max",
    "rotation_capped_days",
    # The two bucket families answer different questions and must both be read:
    # ``test_bucket_*`` describes the raw model ranking (identical for every
    # variant, by construction), ``traded_bucket_*`` describes the score the book
    # was actually built from.  A sweep carrying only the first would make any
    # score transform look like a pure execution effect.
    "test_bucket_label_mean",
    "test_top_bottom_spread",
    "traded_bucket_label_mean",
    "traded_universe_label_mean",
    "traded_bucket_label_positive_ratio",
    "sharpe",
    "max_drawdown",
)


def _parse_list(text: str, cast, label: str) -> list:
    values: list = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(cast(chunk))
        except ValueError as exc:
            raise SystemExit(f"{label} 含非法值: {chunk}") from exc
    if not values:
        raise SystemExit(f"{label} 至少需要一个值")
    return sorted(set(values))


def variant_name(window: int, rebalance: int, hold_buffer: float, turnover: float) -> str:
    """Stable, self-describing cell name.

    Every axis is encoded even when it has a single value, so a cell read out of
    a report cannot be mistaken for a run whose other knobs were different.
    """

    return (
        f"w{window}_r{rebalance}"
        f"_b{int(round(hold_buffer * 100)):02d}"
        f"_t{int(round(turnover * 100)):02d}"
    )


def build_variants(
    config,
    windows: list[int],
    rebalances: list[int],
    hold_buffers: list[float],
    turnovers: list[float],
) -> list[tuple[str, BacktestConfig, int]]:
    """Cross product of the portfolio-side axes, all sharing one book size."""

    variants: list[tuple[str, BacktestConfig, int]] = []
    for window in windows:
        for rebalance in rebalances:
            for hold_buffer in hold_buffers:
                for turnover in turnovers:
                    backtest = BacktestConfig(
                        initial_capital=config.account.initial_capital,
                        max_positions=config.portfolio.max_positions,
                        cash_floor=config.account.cash_floor,
                        single_weight_cap=config.portfolio.single_weight_cap,
                        industry_weight_cap=config.portfolio.industry_weight_cap,
                        max_daily_turnover=turnover,
                        max_daily_orders=config.portfolio.max_daily_action_orders,
                        commission_rate=config.execution.commission_rate,
                        stamp_duty_rate=config.execution.stamp_duty_rate,
                        slippage_bps=config.execution.slippage_bps,
                        rebalance_every_n_days=rebalance,
                        hold_buffer_ratio=hold_buffer,
                    )
                    variants.append(
                        (
                            variant_name(window, rebalance, hold_buffer, turnover),
                            backtest,
                            window,
                        )
                    )
    return variants


def run(
    *,
    market_root: Path,
    periods: list[BacktestPeriod],
    benchmark_symbol: str,
    train_days: int,
    test_days: int,
    step_days: int,
    horizon_days: int,
    artifact_dir: Path,
    output: Path,
    feature_profile: str,
    label_mode: str,
    early_stopping_metric: str,
    variants: list[tuple[str, BacktestConfig, int]],
    curves_dir: Path | None = None,
) -> dict[str, object]:
    if label_mode not in LABEL_MODES:
        raise ValueError("label_mode must be one of: " + ", ".join(LABEL_MODES))
    config = load_config()
    store = HistoricalStore(market_root)
    daily = store.read_daily(
        end=max(period.end for period in periods),
        enrich_fundamentals=True,
        a_shares_only=True,
        require_adjusted=True,
        exclude_quality_events=True,
    )
    benchmarks = store.read_benchmarks(
        end=max(period.end for period in periods),
        symbols=[benchmark_symbol],
    )
    engine = FeatureEngine()
    feature_columns = engine.feature_profile(feature_profile)
    walkforward_config = WalkForwardConfig(
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        horizon_days=horizon_days,
        min_training_rows=config.model.min_training_rows,
        validation_fraction=config.model.validation_fraction,
        min_validation_days=config.model.min_validation_days,
        validation_gap_days=config.model.validation_gap_days,
        early_stopping_rounds=50,
        early_stopping_metric=early_stopping_metric,
        # Overridden per variant below; the fit itself never reads it.
        score_smoothing_days=1,
        artifact_dir=artifact_dir,
    )

    report_periods: list[dict[str, object]] = []
    for period in periods:
        print(f"[sweep] {period.name}: {period.start} -> {period.end}", flush=True)
        results = LightGBMWalkForwardBacktester(feature_engine=engine).run_score_variants(
            daily,
            variants=variants,
            walkforward=walkforward_config,
            benchmark_frame=benchmarks,
            benchmark_symbol=benchmark_symbol,
            label_mode=label_mode,
            feature_columns=feature_columns,
            evaluation_start=period.start.isoformat(),
            evaluation_end=period.end.isoformat(),
        )
        rows: dict[str, dict[str, object]] = {}
        curve_frames: list[pd.DataFrame] = []
        for name, result in results.items():
            benchmark, _, _, _ = benchmark_return(
                benchmarks,
                benchmark_symbol,
                str(result.equity_curve.iloc[0]["trade_date"]),
                str(result.equity_curve.iloc[-1]["trade_date"]),
            )
            stats = result.stats
            row = {key: stats.get(key) for key in METRIC_KEYS}
            row["benchmark_return"] = benchmark
            row["excess_return"] = float(stats["total_return"]) - benchmark
            row["trade_count"] = len(result.trades)
            row["actual_start"] = str(result.equity_curve.iloc[0]["trade_date"])
            row["actual_end"] = str(result.equity_curve.iloc[-1]["trade_date"])
            rows[name] = row
            # The daily equity path is what a paired test between two variants
            # needs: the benchmark cancels in the difference, so a per-day series
            # turns an n = 3 comparison across periods into n ~ 650 days.
            if curves_dir is not None:
                curve = result.equity_curve[["trade_date", "equity"]].copy()
                curve.insert(0, "variant", name)
                curve_frames.append(curve)
        if curves_dir is not None and curve_frames:
            curves_dir.mkdir(parents=True, exist_ok=True)
            curves_path = curves_dir / f"{period.name}.csv"
            pd.concat(curve_frames, ignore_index=True).to_csv(curves_path, index=False)
            print(f"[sweep] curves -> {curves_path}", flush=True)
        report_periods.append({"period": period.as_dict(), "variants": rows})
        print(
            json.dumps(
                {
                    "period": period.name,
                    "excess_by_variant": {
                        name: round(float(row["excess_return"]), 5)
                        for name, row in rows.items()
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del results
        gc.collect()

    all_positive = [
        name
        for name in report_periods[0]["variants"]
        if all(
            float(period["variants"][name]["excess_return"]) > 0 for period in report_periods
        )
    ]
    report: dict[str, object] = {
        "market_root": str(market_root.resolve()),
        "benchmark_symbol": benchmark_symbol,
        "label_mode": label_mode,
        "feature_profile": feature_profile,
        "variant_names": [name for name, _, _ in variants],
        "all_periods_positive": all_positive,
        "walkforward": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "horizon_days": horizon_days,
            "validation_gap_days": walkforward_config.effective_validation_gap_days,
            "early_stopping_metric": early_stopping_metric,
            "feature_columns": feature_columns,
        },
        "periods": report_periods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def _pct(value: object, digits: int = 2) -> str:
    if value is None:
        return "未测量"
    return f"{float(value):+.{digits}%}"


def render_markdown(report: dict[str, object]) -> str:
    """Readable grid report.

    Rows are variants, not periods: a grid is read by comparing cells, so the
    variant has to be the row key.
    """

    names = [str(name) for name in report["variant_names"]]
    periods = report["periods"]
    period_names = [str(period["period"]["name"]) for period in periods]
    walkforward = report["walkforward"]
    lines: list[str] = []
    lines.append("# 组合侧旋钮敏感性扫描\n")
    lines.append(
        "- 标签周期 {horizon} 日；防泄漏 gap {gap} 日；早停指标 `{metric}`".format(
            horizon=walkforward["horizon_days"],
            gap=walkforward["validation_gap_days"],
            metric=walkforward["early_stopping_metric"],
        )
    )
    lines.append(
        "- 每个区间**只训练一次**，对同一份打分回放全部变体 ⇒ 逐行可比，"
        "模型是同一批。列名含义：`w` 平滑窗口 / `r` 调仓间隔 / `b` 在位者缓冲(×100) / "
        "`t` 双边换手上限(×100)。"
    )
    lines.append("")

    lines.append("## 超额收益与三段是否同时为正\n")
    lines.append("| 变体 | " + " | ".join(period_names) + " | 三段全正 |")
    lines.append(f"| :--- | {' | '.join('---:' for _ in period_names)} | :---: |")
    all_positive = set(report.get("all_periods_positive") or [])
    for name in names:
        cells = [_pct(period["variants"][name]["excess_return"]) for period in periods]
        lines.append(
            "| `{name}` | {cells} | {verdict} |".format(
                name=name,
                cells=" | ".join(cells),
                verdict="**✓**" if name in all_positive else "✗",
            )
        )

    lines.append("")
    lines.append("## 换手与成本（判断收益是否靠少交易换来）\n")
    header = []
    for period_name in period_names:
        header.extend([f"{period_name} 双边换手", f"{period_name} 成本"])
    lines.append("| 变体 | " + " | ".join(header) + " |")
    lines.append(f"| :--- | {' | '.join('---:' for _ in header)} |")
    for name in names:
        cells = []
        for period in periods:
            row = period["variants"][name]
            cells.append(f"{float(row['turnover']):.2f}")
            cells.append(_pct(row["cost_drag_ratio"]))
        lines.append("| `{name}` | {cells} |".format(name=name, cells=" | ".join(cells)))

    lines.append("")
    lines.append("## 仓位与换手上限（是否仍满足「满仓 + 单日 ≤ 上限」）\n")
    header = []
    for period_name in period_names:
        header.extend([f"{period_name} 平均仓位", f"{period_name} 最大单日换手"])
    lines.append("| 变体 | " + " | ".join(header) + " |")
    lines.append(f"| :--- | {' | '.join('---:' for _ in header)} |")
    for name in names:
        cells = []
        for period in periods:
            row = period["variants"][name]
            cells.append(_pct(row["invested_ratio_mean"], digits=1))
            cells.append(_pct(row["daily_turnover_max"], digits=2))
        lines.append("| `{name}` | {cells} |".format(name=name, cells=" | ".join(cells)))

    lines.append("")
    lines.append("## 实际交易桶的均超额（组合真正下单所用打分）\n")
    lines.append(
        "这一列上升说明收益来自**选股更强**；持平而超额上升说明来自**拿得更住**。\n"
    )
    lines.append("| 变体 | " + " | ".join(period_names) + " |")
    lines.append(f"| :--- | {' | '.join('---:' for _ in period_names)} |")
    for name in names:
        cells = [
            _pct(period["variants"][name]["traded_bucket_label_mean"], digits=3)
            for period in periods
        ]
        lines.append("| `{name}` | {cells} |".format(name=name, cells=" | ".join(cells)))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="组合侧旋钮敏感性扫描（单次训练，多次回测）")
    parser.add_argument("--market-root", default="data/eastmoney_adjusted")
    parser.add_argument("--benchmark-symbol", default="000300")
    parser.add_argument("--period", dest="periods", action="append")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument(
        "--feature-profile",
        choices=("candidate-v2", "candidate-v3"),
        default="candidate-v3",
    )
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="benchmark_excess")
    parser.add_argument(
        "--early-stopping-metric",
        choices=LightGBMModel.EARLY_STOPPING_METRICS,
        default="rank_ic",
    )
    parser.add_argument(
        "--windows",
        default="1",
        help="打分平滑窗口（交易日）列表；1 表示不平滑。用 1,5,10,20,40,60 检验是否为平台",
    )
    parser.add_argument(
        "--rebalance",
        default=None,
        help="调仓间隔（交易日）列表；省略时取 config 的 portfolio.rebalance_every_n_days",
    )
    parser.add_argument(
        "--hold-buffer",
        default="0",
        help="在位者缓冲比例列表；0 表示不留情面地按排名换",
    )
    parser.add_argument(
        "--turnover",
        default=None,
        help="双边单日换手上限列表；省略时取 config 的 portfolio.max_daily_turnover",
    )
    parser.add_argument("--artifact-dir", default="data/models/research/smoothing-sweep")
    parser.add_argument(
        "--curves-dir",
        default=None,
        help="可选：把每个区间的逐日权益曲线以长表落盘（variant, trade_date, equity），"
        "供 scripts/paired_variant_test.py 做配对检验",
    )
    parser.add_argument("--output", default="data/reports/smoothing-sweep.json")
    parser.add_argument(
        "--markdown", default=None, help="可选：把扫描结果另存为可读的 Markdown 报告"
    )
    args = parser.parse_args()
    if min(args.train_days, args.test_days, args.step_days, args.horizon_days) <= 0:
        raise SystemExit("walk-forward 参数必须为正数")

    config = load_config()
    rebalance_default = str(config.portfolio.rebalance_every_n_days)
    turnover_default = str(config.portfolio.max_daily_turnover)
    windows = _parse_list(args.windows, int, "--windows")
    if any(value <= 0 for value in windows):
        raise SystemExit("平滑窗口必须是正数")
    rebalances = _parse_list(
        args.rebalance if args.rebalance is not None else rebalance_default,
        int,
        "--rebalance",
    )
    if any(value <= 0 for value in rebalances):
        raise SystemExit("调仓间隔必须是正数")
    hold_buffers = _parse_list(args.hold_buffer, float, "--hold-buffer")
    if any(value < 0 for value in hold_buffers):
        raise SystemExit("在位者缓冲必须非负")
    turnovers = _parse_list(
        args.turnover if args.turnover is not None else turnover_default,
        float,
        "--turnover",
    )
    if any(value <= 0 for value in turnovers):
        raise SystemExit("换手上限必须是正数")

    variants = build_variants(config, windows, rebalances, hold_buffers, turnovers)
    report = run(
        market_root=Path(args.market_root),
        periods=parse_periods(args.periods),
        benchmark_symbol=args.benchmark_symbol,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        horizon_days=args.horizon_days,
        artifact_dir=Path(args.artifact_dir),
        output=Path(args.output),
        feature_profile=args.feature_profile,
        label_mode=args.label_mode,
        early_stopping_metric=args.early_stopping_metric,
        variants=variants,
        curves_dir=Path(args.curves_dir) if args.curves_dir else None,
    )
    print(f"report: {Path(args.output).resolve()}")
    print(f"variants: {len(variants)} -> {[name for name, _, _ in variants]}")
    print(f"三段同时为正: {report['all_periods_positive'] or '无'}")
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"markdown: {markdown_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
