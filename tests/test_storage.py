import pandas as pd
import pytest

from quantpaper.storage import HistoricalStore


def test_historical_store_writes_api_frame_and_filters(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": "2025-01-03",
                "open": 10,
                "high": 11,
                "low": 9.8,
                "close": 10.5,
                "volume": 1000,
                "amount": 1_000_000,
            },
            {
                "symbol": "300001",
                "trade_date": "2025-01-03",
                "open": 20,
                "high": 21,
                "low": 19.8,
                "close": 20.5,
                "volume": 2000,
                "amount": 2_000_000,
            },
        ]
    )

    store = HistoricalStore(tmp_path / "market")
    report = store.write_daily(frame)
    assert report.ok
    assert report.rows == 2
    assert report.symbols == 2

    filtered = store.read_daily(start="2025-01-03", symbols=["300001"])
    assert filtered["symbol"].tolist() == ["300001"]
    assert filtered["volume"].tolist() == [2000]


def test_historical_store_rejects_duplicate_or_invalid_rows(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": "2025-01-03",
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "amount": 1_000_000,
            },
            {
                "symbol": "600001",
                "trade_date": "2025-01-03",
                "open": 0,
                "high": 11,
                "low": 9,
                "close": 10,
                "amount": 1_000_000,
            },
        ]
    )
    store = HistoricalStore(tmp_path / "market")
    with pytest.raises(ValueError, match="daily data quality failed"):
        store.write_daily(frame)
