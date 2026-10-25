"""Reusable exact-window and multi-period factor backtest helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, FactorBacktester
from .features import FeatureEngine
from .storage import HistoricalStore

DEFAULT_BENCHMARK_SYMBOL = "000300"
DEFAULT_PERIODS = (
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026_ytd", "2026-01-01", "2100-01-01"),
)


@dataclass(frozen=True)
class BacktestPeriod:
    name: str
    start: date
    end: date

    @classmethod
    def parse(cls, value: str) -> BacktestPeriod:
        parts = value.split(":")
        if len(parts) != 3 or not all(parts):
            raise ValueError("区间格式必须是 name:YYYY-MM-DD:YYYY-MM-DD")
        name, start_text, end_text = parts
        try:
            start = date.fromisoformat(start_text)
            end = date.fromisoformat(end_text)
        except ValueError as exc:
            raise ValueError(f"区间日期格式必须是 YYYY-MM-DD: {value}") from exc
        if start > end:
            raise ValueError(f"区间开始日期晚于结束日期: {value}")
        return cls(name=name, start=start, end=end)

    @classmethod
    def defaults(cls, today: date | None = None) -> list[BacktestPeriod]:
        current = today or date.today()
        return [
            cls.parse(f"{name}:{start}:{end if name != '2026_ytd' else current.isoformat()}")
            for name, start, end in DEFAULT_PERIODS
        ]

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "requested_start": self.start.isoformat(),
            "requested_end": self.end.isoformat(),
        }


@dataclass
class PeriodBacktestRun:
    period: BacktestPeriod
    warmup_start: date
    result: BacktestResult
    benchmark_symbol: str
    benchmark_name: str
    benchmark_return: float
    excess_return: float
    benchmark_first_close: float
    benchmark_last_close: float

    def as_dict(self) -> dict[str, Any]:
        payload = self.result.as_dict()
        stats = dict(payload["stats"])
        stats.update(
            {
                "benchmark_return": round(self.benchmark_return, 8),
                "excess_return": round(self.excess_return, 8),
                "benchmark_final_equity": round(
                    float(stats.get("initial_capital", 0.0)) * (1 + self.benchmark_return),
                    6,
                ),
                "excess_final_equity": round(
                    float(stats.get("initial_capital", 0.0)) * (1 + self.excess_return),
                    6,
                ),
            }
        )
        equity_dates = self.result.equity_curve.get("trade_date", pd.Series(dtype=str))
        actual_start = str(equity_dates.iloc[0]) if not equity_dates.empty else None
        actual_end = str(equity_dates.iloc[-1]) if not equity_dates.empty else None
        return {
            "period": self.period.as_dict(),
            "actual_start": actual_start,
            "actual_end": actual_end,
            "warmup_start": self.warmup_start.isoformat(),
            "warmup_calendar_days": _count_calendar_days(self.warmup_start, self.period.start),
            "benchmark": {
                "symbol": self.benchmark_symbol,
                "name": self.benchmark_name,
                "first_close": round(self.benchmark_first_close, 6),
                "last_close": round(self.benchmark_last_close, 6),
            },
            "stats": stats,
            "quality": payload["quality"],
            "trades": payload["trades"],
            "equity_curve": payload["equity_curve"],
        }


def parse_periods(values: Iterable[str] | None) -> list[BacktestPeriod]:
    if values:
        periods = [BacktestPeriod.parse(value) for value in values]
    else:
        periods = BacktestPeriod.defaults()
    names: set[str] = set()
    for period in periods:
        if period.name in names:
            raise ValueError(f"区间名称重复: {period.name}")
        names.add(period.name)
    return periods


def _read_calendar(root: Path) -> list[date]:
    path = root / "trading_calendar.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB is required for research calendar") from exc
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            "SELECT DISTINCT CAST(trade_date AS DATE) AS trade_date "
            "FROM read_parquet(?) ORDER BY trade_date",
            [str(path)],
        ).df()
    finally:
        connection.close()
    return [value.date() for value in pd.to_datetime(frame["trade_date"], errors="coerce").dropna()]


def warmup_start_for_period(
    calendar: list[date], start: date, warmup_trading_days: int
) -> date:
    if warmup_trading_days < 0:
        raise ValueError("warmup_trading_days must be non-negative")
    first_available = next(
        (index for index, current in enumerate(calendar) if current >= start), None
    )
    if first_available is None:
        raise ValueError(f"交易日历没有覆盖区间开始日期: {start}")
    return calendar[max(0, first_available - warmup_trading_days)]


def _count_calendar_days(start: date, end: date) -> int:
    return max(0, (end - start).days)


def _benchmark_window(
    benchmarks: pd.DataFrame,
    symbol: str,
    start: date,
    end: date,
) -> tuple[float, float, str]:
    frame = benchmarks[
        (benchmarks["benchmark_symbol"].astype(str) == symbol)
        & (pd.to_datetime(benchmarks["trade_date"]).dt.date >= start)
        & (pd.to_datetime(benchmarks["trade_date"]).dt.date <= end)
    ].sort_values("trade_date")
    if frame.empty:
        raise ValueError(f"基准 {symbol} 没有覆盖区间 {start} -> {end}")
    first = float(frame.iloc[0]["close"])
    last = float(frame.iloc[-1]["close"])
    if first <= 0 or last <= 0:
        raise ValueError(f"基准 {symbol} 区间收盘价无效")
    name = str(frame.iloc[0].get("name") or symbol)
    return first, last, name


def run_period_backtests(
    store: HistoricalStore,
    periods: Iterable[BacktestPeriod],
    *,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    score_column: str = "factor_score",
    backtest_config: BacktestConfig | None = None,
    warmup_trading_days: int = 120,
    exclude_quality_events: bool = True,
) -> list[PeriodBacktestRun]:
    selected_periods = list(periods)
    if not selected_periods:
        raise ValueError("至少需要一个回测区间")
    calendar = _read_calendar(store.root)
    first_period_index = min(
        next(
            index for index, current in enumerate(calendar) if current >= period.start
        )
        for period in selected_periods
    )
    if first_period_index < warmup_trading_days:
        raise ValueError(
            "股票归档在最早回测区间开始前没有足够的因子预热数据；"
            f"需要 {warmup_trading_days} 个交易日，实际只有 {first_period_index} 个。"
            "请缩短 warmup，或选择更晚的区间。"
        )
    warmup_starts = [
        warmup_start_for_period(calendar, period.start, warmup_trading_days)
        for period in selected_periods
    ]
    data_start = min(warmup_starts)
    data_end = max(period.end for period in selected_periods)
    daily = store.read_daily(
        "daily",
        start=data_start,
        end=data_end,
        enrich_fundamentals=True,
        a_shares_only=True,
        require_adjusted=True,
        exclude_quality_events=exclude_quality_events,
    )
    if daily.empty:
        raise ValueError("所选回测窗口没有股票历史数据")
    benchmarks = store.read_benchmarks(
        start=min(period.start for period in selected_periods),
        end=data_end,
        symbols=[benchmark_symbol],
    )
    feature_engine = FeatureEngine()
    all_features = feature_engine.build_features(daily)
    daily_dates = pd.to_datetime(daily["trade_date"], errors="coerce").dt.date
    feature_dates = pd.to_datetime(all_features["trade_date"], errors="coerce").dt.date
    normalized_config = backtest_config or BacktestConfig()
    runs: list[PeriodBacktestRun] = []
    for period, warmup_start in zip(selected_periods, warmup_starts, strict=True):
        daily_period = daily.loc[
            (daily_dates >= period.start) & (daily_dates <= period.end)
        ].copy()
        feature_period = all_features.loc[
            (feature_dates >= period.start) & (feature_dates <= period.end)
        ].copy()
        if daily_period.empty or feature_period.empty:
            raise ValueError(f"区间没有可回测数据: {period.name}")
        result = FactorBacktester(feature_engine=feature_engine).run(
            daily_period,
            config=normalized_config,
            feature_frame=feature_period,
            score_column=score_column,
        )
        first_close, last_close, benchmark_name = _benchmark_window(
            benchmarks, benchmark_symbol, period.start, period.end
        )
        benchmark_return = last_close / first_close - 1
        strategy_return = float(result.stats.get("total_return", 0.0))
        runs.append(
            PeriodBacktestRun(
                period=period,
                warmup_start=warmup_start,
                result=result,
                benchmark_symbol=benchmark_symbol,
                benchmark_name=benchmark_name,
                benchmark_return=benchmark_return,
                excess_return=strategy_return - benchmark_return,
                benchmark_first_close=first_close,
                benchmark_last_close=last_close,
            )
        )
    return runs


def report_as_dict(
    runs: Iterable[PeriodBacktestRun],
    *,
    market_root: str | Path,
    benchmark_symbol: str,
    score_column: str,
    warmup_trading_days: int,
) -> dict[str, Any]:
    selected = list(runs)
    return {
        "provider": "eastmoney_gm",
        "read_only": True,
        "market_root": str(Path(market_root).resolve()),
        "benchmark_symbol": benchmark_symbol,
        "score_column": score_column,
        "warmup_trading_days": warmup_trading_days,
        "periods": [run.as_dict() for run in selected],
    }
