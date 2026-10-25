"""Run segmented walk-forward validation for a LightGBM research candidate.

The three default periods (2024, 2025, 2026 year-to-date) compare the strategy
against a real index, so the reported ``excess_return`` stays comparable to the
existing M0/M1 baseline reports regardless of which label the model was trained
on.  The promotion decision is deliberately mechanical and mirrors the gate in
``docs/factor-model-training-plan.md``: every period must beat the benchmark,
and no fold may be degenerate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from quantpaper.backtest import BacktestConfig
from quantpaper.config import load_config
from quantpaper.features import FeatureEngine
from quantpaper.ml_backtest import LightGBMWalkForwardBacktester, WalkForwardConfig
from quantpaper.model import LightGBMModel
from quantpaper.period_backtest import BacktestPeriod, parse_periods
from quantpaper.storage import HistoricalStore

LABEL_MODES = ("benchmark_excess", "cross_sectional")


def benchmark_return(
    frame: pd.DataFrame, symbol: str, start: str, end: str
) -> tuple[float, str, float, float]:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    selected = frame[
        frame["benchmark_symbol"].astype(str).eq(symbol)
        & (dates >= pd.Timestamp(start).date())
        & (dates <= pd.Timestamp(end).date())
    ].sort_values("trade_date")
    if selected.empty:
        raise ValueError(f"基准 {symbol} 没有覆盖区间 {start} -> {end}")
    first_close = float(selected.iloc[0]["close"])
    last_close = float(selected.iloc[-1]["close"])
    return (
        last_close / first_close - 1,
        str(selected.iloc[0].get("name") or symbol),
        first_close,
        last_close,
    )


def period_payload(
    period: BacktestPeriod,
    result: object,
    benchmark_frame: pd.DataFrame,
    benchmark_symbol: str,
) -> dict[str, object]:
    # Keep this helper independent of the concrete result dataclass so the
    # report remains easy to serialize and inspect from outside the package.
    walkforward = result
    equity = walkforward.equity_curve
    actual_start = str(equity.iloc[0]["trade_date"])
    actual_end = str(equity.iloc[-1]["trade_date"])
    benchmark, benchmark_name, first_close, last_close = benchmark_return(
        benchmark_frame, benchmark_symbol, actual_start, actual_end
    )
    strategy = float(walkforward.stats.get("total_return", 0.0))
    return {
        "period": period.as_dict(),
        "actual_start": actual_start,
        "actual_end": actual_end,
        "benchmark": {
            "symbol": benchmark_symbol,
            "name": benchmark_name,
            "first_close": first_close,
            "last_close": last_close,
        },
        "stats": {
            **walkforward.stats,
            "benchmark_return": benchmark,
            "excess_return": strategy - benchmark,
            "benchmark_final_equity": 1_000_000.0 * (1 + benchmark),
            "excess_final_equity": 1_000_000.0 * (1 + strategy - benchmark),
        },
        "quality": walkforward.quality.as_dict(),
        "folds": [fold.as_dict() for fold in walkforward.folds],
        "trade_count": len(walkforward.trades),
    }


def decide_promotion(
    report_periods: list[dict[str, object]], degenerate_folds: float
) -> tuple[str, str]:
    """Apply the documented gate mechanically and explain the outcome."""

    if degenerate_folds > 0:
        return (
            "BLOCKED_DEGENERATE",
            f"{int(degenerate_folds)} 个折的模型没有横截面区分度，任何收益数字都不成立",
        )
    excess_values = [
        float(period["stats"]["excess_return"]) for period in report_periods
    ]
    if all(value > 0 for value in excess_values):
        return (
            "PASSED_GATE_PENDING_SHADOW",
            "三段超额收益同时为正；仍需纸盘影子验证后才能替换生产模型",
        )
    return (
        "NOT_PROMOTED",
        "三段超额收益未同时为正："
        + ", ".join(f"{value:+.2%}" for value in excess_values),
    )


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
    score_smoothing_days: int,
    hold_buffer_ratio: float | None,
    rebalance_every_n_days: int | None,
    max_daily_turnover: float | None = None,
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
        score_smoothing_days=score_smoothing_days,
        artifact_dir=artifact_dir,
    )
    backtest_config = BacktestConfig(
        initial_capital=config.account.initial_capital,
        max_positions=config.portfolio.max_positions,
        cash_floor=config.account.cash_floor,
        single_weight_cap=config.portfolio.single_weight_cap,
        industry_weight_cap=config.portfolio.industry_weight_cap,
        max_daily_turnover=(
            config.portfolio.max_daily_turnover
            if max_daily_turnover is None
            else max_daily_turnover
        ),
        max_daily_orders=config.portfolio.max_daily_action_orders,
        commission_rate=config.execution.commission_rate,
        stamp_duty_rate=config.execution.stamp_duty_rate,
        slippage_bps=config.execution.slippage_bps,
        rebalance_every_n_days=(
            config.portfolio.rebalance_every_n_days
            if rebalance_every_n_days is None
            else rebalance_every_n_days
        ),
        hold_buffer_ratio=(
            config.portfolio.hold_buffer_ratio
            if hold_buffer_ratio is None
            else hold_buffer_ratio
        ),
    )
    report_periods: list[dict[str, object]] = []
    degenerate_folds = 0.0
    for period in periods:
        print(f"[walkforward] {period.name}: {period.start} -> {period.end}", flush=True)
        result = LightGBMWalkForwardBacktester(feature_engine=engine).run(
            daily,
            walkforward=walkforward_config,
            backtest=backtest_config,
            benchmark_frame=benchmarks,
            benchmark_symbol=benchmark_symbol,
            label_mode=label_mode,
            feature_columns=feature_columns,
            evaluation_start=period.start.isoformat(),
            evaluation_end=period.end.isoformat(),
        )
        payload = period_payload(period, result, benchmarks, benchmark_symbol)
        report_periods.append(payload)
        degenerate_folds += float(payload["stats"].get("degenerate_folds", 0.0))
        equity_path = output.with_name(f"{output.stem}.{period.name}.equity.csv")
        result.equity_curve.to_csv(equity_path, index=False)
        print(
            json.dumps(
                {
                    "period": period.name,
                    "strategy_return": payload["stats"]["total_return"],
                    "benchmark_return": payload["stats"]["benchmark_return"],
                    "excess_return": payload["stats"]["excess_return"],
                    "folds": payload["stats"]["fold_count"],
                    "degenerate_folds": payload["stats"].get("degenerate_folds"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del result
        gc.collect()
    promotion, promotion_reason = decide_promotion(report_periods, degenerate_folds)
    report: dict[str, object] = {
        "status": promotion,
        "candidate_profile": f"{feature_profile}-{label_mode}",
        "market_root": str(market_root.resolve()),
        "benchmark_symbol": benchmark_symbol,
        "label_mode": label_mode,
        "label_definition": (
            f"future_{horizon_days}d_stock_return - "
            f"future_{horizon_days}d_{benchmark_symbol}_return"
            if label_mode == "benchmark_excess"
            else f"future_{horizon_days}d_stock_return - "
            f"cross_sectional_mean_future_{horizon_days}d_return"
        ),
        "walkforward": {
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "horizon_days": horizon_days,
            "early_stopping_metric": early_stopping_metric,
            "feature_profile": feature_profile,
            "score_smoothing_days": score_smoothing_days,
            "feature_columns": feature_columns,
        },
        "backtest": {
            "rebalance_every_n_days": backtest_config.rebalance_every_n_days,
            "hold_buffer_ratio": backtest_config.hold_buffer_ratio,
            "commission_rate": backtest_config.commission_rate,
            "stamp_duty_rate": backtest_config.stamp_duty_rate,
            "slippage_bps": backtest_config.slippage_bps,
            "max_daily_turnover": backtest_config.max_daily_turnover,
            "max_daily_orders": backtest_config.max_daily_orders,
            # Book sizing belongs here too: without it a report cannot be
            # re-read later without guessing which toml the numbers came from.
            "max_positions": backtest_config.max_positions,
            "cash_floor": backtest_config.cash_floor,
            "single_weight_cap": backtest_config.single_weight_cap,
            "industry_weight_cap": backtest_config.industry_weight_cap,
        },
        "degenerate_folds": degenerate_folds,
        "promotion": promotion,
        "promotion_reason": promotion_reason,
        "periods": report_periods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="分段验证 LightGBM 研究候选")
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
        default="candidate-v2",
        help="candidate-v2 含市场状态特征（旧行为）；candidate-v3 只保留横截面因子",
    )
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="benchmark_excess")
    parser.add_argument(
        "--early-stopping-metric",
        choices=LightGBMModel.EARLY_STOPPING_METRICS,
        default="rank_ic",
    )
    parser.add_argument("--hold-buffer", type=float, default=None)
    parser.add_argument("--rebalance-every", type=int, default=None)
    parser.add_argument(
        "--max-daily-turnover",
        type=float,
        default=None,
        help="双边换手上限；省略时取 config 的 portfolio.max_daily_turnover。"
        "用于把「仓位控制修复」与「调仓频率/上限调整」拆开做归因对照",
    )
    parser.add_argument(
        "--score-smoothing-days",
        type=int,
        default=1,
        help="模型打分的 N 日滚动均值；1 为不平滑（旧行为），用于抑制排名抖动",
    )
    parser.add_argument(
        "--artifact-dir", default="data/models/research/benchmark-excess-walkforward"
    )
    parser.add_argument("--output", default="data/reports/benchmark-excess-walkforward.json")
    args = parser.parse_args()
    if min(args.train_days, args.test_days, args.step_days, args.horizon_days) <= 0:
        raise SystemExit("walk-forward 参数必须为正数")
    if args.hold_buffer is not None and args.hold_buffer < 0:
        raise SystemExit("--hold-buffer must be non-negative")
    if args.rebalance_every is not None and args.rebalance_every <= 0:
        raise SystemExit("--rebalance-every must be positive")
    if args.score_smoothing_days <= 0:
        raise SystemExit("--score-smoothing-days must be positive")
    if args.max_daily_turnover is not None and args.max_daily_turnover <= 0:
        raise SystemExit("--max-daily-turnover must be positive")
    periods = parse_periods(args.periods)
    report = run(
        market_root=Path(args.market_root),
        periods=periods,
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
        score_smoothing_days=args.score_smoothing_days,
        hold_buffer_ratio=args.hold_buffer,
        rebalance_every_n_days=args.rebalance_every,
        max_daily_turnover=args.max_daily_turnover,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason": report["promotion_reason"],
                "degenerate_folds": report["degenerate_folds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
