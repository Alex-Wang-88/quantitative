from __future__ import annotations

from quantpaper.backtest import BacktestConfig
from quantpaper.data import make_demo_provider
from quantpaper.ml_backtest import LightGBMWalkForwardBacktester, WalkForwardConfig


def test_lightgbm_walkforward_keeps_training_before_test_and_reuses_paper_broker(
    tmp_path,
) -> None:
    provider = make_demo_provider(seed=11, quote_steps=2, replay_days=2)
    result = LightGBMWalkForwardBacktester().run(
        provider.get_daily_frame(),
        walkforward=WalkForwardConfig(
            train_days=100,
            test_days=10,
            step_days=10,
            horizon_days=5,
            min_training_rows=100,
            min_validation_days=5,
            artifact_dir=tmp_path,
        ),
        backtest=BacktestConfig(
            max_positions=4,
            single_weight_cap=0.25,
            max_daily_turnover=0.50,
            max_daily_orders=8,
        ),
    )

    assert result.quality.ok
    assert result.folds
    assert result.stats["fold_count"] == len(result.folds)
    assert result.equity_curve.shape[0] > 0
    assert result.stats["final_equity"] > 0
    assert all(fold.train_end < fold.test_start for fold in result.folds)
    assert all(fold.artifact_path for fold in result.folds)
    assert all("test_ic_mean" in fold.metrics for fold in result.folds)
    assert result.as_dict()["folds"]
