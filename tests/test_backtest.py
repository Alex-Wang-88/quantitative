import numpy as np
import pandas as pd

from quantpaper.backtest import BacktestConfig, FactorBacktester
from quantpaper.domain import Board, Security
from quantpaper.features import FeatureEngine


def _daily_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=150)
    rows: list[dict[str, object]] = []
    specs = {
        "600001": (10.0, 0.0020, Board.MAIN, "金融"),
        "300001": (20.0, 0.0008, Board.CHINEXT_300, "先进制造"),
        "688001": (30.0, -0.0004, Board.STAR_688, "电子"),
    }
    for index, current_date in enumerate(dates):
        for symbol, (start, drift, _board, sector) in specs.items():
            close = start * (1 + drift) ** index
            open_price = close * 1.001
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": current_date,
                    "open": open_price,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "amount": 10_000_000,
                    "sector": sector,
                    "pe": 20.0 + index % 3,
                    "pb": 2.0,
                    "roe": 0.12,
                    "profit_growth": 0.10,
                }
            )
    return pd.DataFrame(rows)


def test_factor_backtest_executes_next_open_without_same_day_lookahead() -> None:
    securities = [
        Security("600001", "测试主板", "SSE", Board.MAIN, "金融"),
        Security("300001", "测试创业板", "SZSE", Board.CHINEXT_300, "先进制造"),
        Security("688001", "测试科创板", "SSE", Board.STAR_688, "电子"),
    ]
    result = FactorBacktester(securities).run(
        _daily_frame(),
        BacktestConfig(
            max_positions=1,
            single_weight_cap=0.80,
            max_daily_turnover=0.50,
            max_daily_orders=4,
        ),
    )

    assert result.quality.ok
    assert result.trades
    assert all(trade.trade_date > trade.signal_date for trade in result.trades)
    assert result.equity_curve["trade_date"].iloc[0] == "2025-01-02"
    assert result.stats["final_equity"] > 0
    assert np.isfinite(result.stats["sharpe"])
    assert result.as_dict()["equity_curve"]


def test_factor_backtest_can_compare_f1_research_score_without_promoting_it() -> None:
    engine = FeatureEngine()
    features = engine.build_features(_daily_frame())
    result = FactorBacktester(feature_engine=engine).run(
        _daily_frame(),
        BacktestConfig(
            max_positions=1,
            single_weight_cap=0.80,
            max_daily_turnover=0.50,
            max_daily_orders=4,
        ),
        feature_frame=features,
        score_column="f1_factor_score",
    )

    assert result.quality.ok
    assert result.stats["final_equity"] > 0
    assert engine.FACTOR_COLUMNS == engine.BASELINE_FACTOR_COLUMNS
