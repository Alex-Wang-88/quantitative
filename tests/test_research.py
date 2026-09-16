import numpy as np
import pandas as pd

from quantpaper.features import FeatureEngine
from quantpaper.research import FactorResearchEvaluator


def _research_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=90)
    symbols = ["000001", "000002", "000003", "600001", "600002", "600003"]
    sectors = {
        "000001": "电子",
        "000002": "电子",
        "000003": "医药",
        "600001": "医药",
        "600002": "金融",
        "600003": "金融",
    }
    previous = {symbol: 10.0 + index * 3 for index, symbol in enumerate(symbols)}
    rows: list[dict[str, object]] = []
    for day, trade_date in enumerate(dates):
        for index, symbol in enumerate(symbols):
            close = previous[symbol] * (
                1 + 0.0005 * (index + 1) + 0.0003 * ((day + index) % 4 - 1)
            )
            open_price = previous[symbol] * (1 + 0.0005 * (index % 3 - 1))
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": 100_000 + index * 10_000 + day * 100,
                    "amount": 1_000_000 + index * 200_000 + day * 1_000,
                    "market_cap": 100_000_000 + index * 20_000_000,
                    "sector": sectors[symbol],
                }
            )
            previous[symbol] = close
    return pd.DataFrame(rows)


def test_factor_research_evaluates_individual_factors_and_coverage() -> None:
    engine = FeatureEngine()
    result = FactorResearchEvaluator(engine).evaluate(
        _research_frame(),
        factors=["atr_20", "relative_momentum_20", "industry_momentum_20"],
        horizon_days=5,
        neutralization="none",
    )

    assert result.status == "OK"
    assert result.quality.rows == 540
    assert [item.factor for item in result.factors] == [
        "atr_20",
        "relative_momentum_20",
        "industry_momentum_20",
    ]
    assert all(item.coverage > 0 for item in result.factors)
    assert all(item.valid_days > 0 for item in result.factors)
    assert all(np.isfinite(item.mean_rank_ic) for item in result.factors)


def test_factor_research_supports_sector_size_beta_neutralization() -> None:
    result = FactorResearchEvaluator().evaluate(
        _research_frame(),
        factors=["close_position_20", "volume_trend_20"],
        neutralization="sector_size_beta",
    )

    assert result.neutralization == "sector_size_beta"
    assert len(result.factors) == 2
    assert all(np.isfinite(item.mean_top_bottom_spread) for item in result.factors)


def test_factor_research_marks_missing_neutralization_exposures_degraded() -> None:
    frame = _research_frame()
    frame["sector"] = "UNKNOWN"
    result = FactorResearchEvaluator().evaluate(
        frame,
        factors=["atr_20"],
        neutralization="sector",
    )

    assert result.status == "DEGRADED"
    assert result.exposure_coverage["sector"] == 0.0
    assert "UNKNOWN" in result.message
