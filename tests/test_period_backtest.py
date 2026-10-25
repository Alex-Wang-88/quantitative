from datetime import date

import pandas as pd
import pytest

from quantpaper.period_backtest import (
    BacktestPeriod,
    parse_periods,
    warmup_start_for_period,
)
from quantpaper.storage import HistoricalStore


def test_period_parser_accepts_explicit_window_and_rejects_duplicate_names() -> None:
    period = BacktestPeriod.parse("bull:2024-01-01:2024-12-31")
    assert period.name == "bull"
    assert period.start == date(2024, 1, 1)
    assert period.end == date(2024, 12, 31)
    with pytest.raises(ValueError, match="区间名称重复"):
        parse_periods(["x:2024-01-01:2024-02-01", "x:2025-01-01:2025-02-01"])


def test_warmup_uses_trading_calendar_without_crossing_archive_start() -> None:
    calendar = list(pd.date_range("2024-01-02", periods=10, freq="B").date)
    assert warmup_start_for_period(calendar, date(2024, 1, 15), 3) == date(2024, 1, 10)
    assert warmup_start_for_period(calendar, date(2024, 1, 3), 120) == date(2024, 1, 2)


def test_historical_store_reads_benchmarks_without_confusing_stock_symbol(tmp_path) -> None:
    store = HistoricalStore(tmp_path / "archive")
    frame = pd.DataFrame(
        [
            {
                "benchmark_symbol": "000300",
                "vendor_symbol": "SHSE.000300",
                "name": "沪深300",
                "trade_date": "2025-01-03",
                "open": 3800,
                "high": 3810,
                "low": 3790,
                "close": 3805,
            },
            {
                "benchmark_symbol": "000001",
                "vendor_symbol": "SHSE.000001",
                "name": "上证指数",
                "trade_date": "2025-01-03",
                "open": 3200,
                "high": 3210,
                "low": 3190,
                "close": 3205,
            },
        ]
    )
    import duckdb

    connection = duckdb.connect()
    try:
        connection.register("benchmarks", frame)
        connection.execute(
            "COPY benchmarks TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(store.root / "benchmarks.parquet")],
        )
    finally:
        connection.close()

    result = store.read_benchmarks(symbols=["000300"])
    assert result["benchmark_symbol"].tolist() == ["000300"]
    assert result["close"].tolist() == [3805]
