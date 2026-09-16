import pandas as pd
import pytest

from quantpaper.features import FeatureEngine


def test_point_in_time_validation_rejects_future_announcement() -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": "2026-01-05",
                "ann_date": "2026-01-06",
                "close": 10.0,
                "amount": 1_000_000,
            }
        ]
    )
    with pytest.raises(ValueError, match="future financial disclosure"):
        FeatureEngine().build_features(frame)


def test_factor_scores_follow_documented_directions() -> None:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-02")
    for day in range(25):
        trade_date = start + pd.Timedelta(days=day)
        # A is smaller, cheaper and lower-volatility; B has the higher
        # quality, growth, liquidity and risk values.
        rows.extend(
            [
                {
                    "symbol": "000001",
                    "trade_date": trade_date,
                    "close": 10.0,
                    "amount": 10_000_000,
                    "turnover_rate": 1.0,
                    "pe": 10.0,
                    "pb": 1.0,
                    "roe": 0.10,
                    "profit_growth": 0.05,
                    "market_cap": 100_000_000,
                },
                {
                    "symbol": "000002",
                    "trade_date": trade_date,
                    "close": 9.0 if day % 2 else 11.0,
                    "amount": 20_000_000,
                    "turnover_rate": 5.0,
                    "pe": 20.0,
                    "pb": 2.0,
                    "roe": 0.20,
                    "profit_growth": 0.20,
                    "market_cap": 200_000_000,
                },
            ]
        )

    result = FeatureEngine().latest(pd.DataFrame(rows)).set_index("symbol")
    assert result.loc["000001", "value_score"] > result.loc["000002", "value_score"]
    assert result.loc["000002", "quality_score"] > result.loc["000001", "quality_score"]
    assert result.loc["000002", "growth_score"] > result.loc["000001", "growth_score"]
    assert result.loc["000002", "turnover_20"] > result.loc["000001", "turnover_20"]
    assert result.loc["000001", "size_score"] > result.loc["000002", "size_score"]
    assert result.loc["000001", "volatility_20"] > result.loc["000002", "volatility_20"]
    assert result["fundamental_coverage"].eq(1.0).all()


def test_f1_price_volume_candidates_are_point_in_time_research_features() -> None:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-02")
    previous = {"000001": 10.0, "000002": 20.0, "000003": 30.0}
    sectors = {"000001": "电子", "000002": "电子", "000003": "医药"}
    for day in range(75):
        trade_date = start + pd.Timedelta(days=day)
        for index, symbol in enumerate(previous):
            close = previous[symbol] * (1 + 0.001 * (index + 1) + (day % 5) * 0.0002)
            open_price = previous[symbol] * (1 + (index - 1) * 0.001)
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            amount = 1_000_000 * (index + 1) * (1 + (day % 7) * 0.03)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": amount / close,
                    "amount": amount,
                    "sector": sectors[symbol],
                }
            )
            previous[symbol] = close

    engine = FeatureEngine()
    frame = pd.DataFrame(rows)
    built = engine.build_features(frame)
    cutoff = start + pd.Timedelta(days=60)
    truncated = engine.build_features(frame[frame["trade_date"] <= cutoff])

    assert engine.FACTOR_COLUMNS == engine.BASELINE_FACTOR_COLUMNS
    assert set(engine.F1_FACTOR_COLUMNS).issubset(built.columns)
    assert built[engine.F1_FACTOR_COLUMNS].apply(lambda column: column.between(0, 1).all()).all()
    assert built.loc[built["trade_date"] == cutoff, engine.F1_FACTOR_COLUMNS].nunique().max() > 1

    full_cut = built[built["trade_date"] == cutoff].set_index("symbol")
    truncated_cut = truncated[truncated["trade_date"] == cutoff].set_index("symbol")
    pd.testing.assert_frame_equal(
        full_cut[engine.F1_FACTOR_COLUMNS],
        truncated_cut[engine.F1_FACTOR_COLUMNS],
    )
