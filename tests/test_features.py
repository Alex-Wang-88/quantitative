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


def test_benchmark_regime_features_and_excess_labels_are_point_in_time() -> None:
    dates = pd.date_range("2025-01-02", periods=80, freq="D")
    rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates):
        for symbol, drift in (("000001", 0.001), ("000002", 0.0005)):
            close = (10.0 if symbol == "000001" else 20.0) * (1 + drift) ** index
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "amount": 1_000_000,
                }
            )
    benchmark = pd.DataFrame(
        {
            "benchmark_symbol": "000300",
            "trade_date": dates,
            "close": [3000.0 * (1.0004**index) for index in range(len(dates))],
        }
    )

    engine = FeatureEngine()
    daily = pd.DataFrame(rows)
    features = engine.build_features(daily, benchmark_frame=benchmark)
    assert features[engine.BENCHMARK_FEATURE_COLUMNS].iloc[-1].notna().all()

    training = engine.build_training_frame(
        daily,
        horizon_days=5,
        benchmark_frame=benchmark,
        label_mode="benchmark_excess",
        feature_frame=features,
    )
    sample = training.iloc[-1]
    expected = sample["future_return"] - sample["benchmark_future_return"]
    assert sample["label"] == pytest.approx(expected)


def test_cross_sectional_profile_drops_date_constant_market_features() -> None:
    engine = FeatureEngine()
    profile = engine.feature_profile("candidate-v3")

    # Every market-regime column is identical across stocks on a given date,
    # so it can never rank one stock against another.  It must not sit inside
    # the cross-sectional ranking model.
    for column in engine.BENCHMARK_FEATURE_COLUMNS:
        assert column not in profile
    assert "size_data_available" not in profile
    assert set(profile) >= set(engine.CANDIDATE_FACTOR_COLUMNS)
    assert set(engine.feature_profile("candidate-v2")) > set(profile)


def test_feature_profile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown feature profile"):
        FeatureEngine.feature_profile("does-not-exist")
