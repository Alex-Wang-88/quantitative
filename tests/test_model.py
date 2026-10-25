from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quantpaper.data import make_demo_provider
from quantpaper.features import FeatureEngine
from quantpaper.model import LightGBMModel


def _synthetic_frame(
    *, within_date_signal: bool, n_dates: int = 30, n_symbols: int = 80
) -> pd.DataFrame:
    """Build a frame with (or without) genuine cross-sectional signal.

    ``within_date_signal=False`` makes every feature identical for every stock
    on every date, so no tree can separate two stocks and the model must emit
    one score per date.  That is the failure mode observed in the archived
    candidate-v2 walk-forward folds, reproduced here deterministically.
    """

    generator = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    for date_index in range(n_dates):
        trade_date = pd.Timestamp("2025-01-02") + pd.Timedelta(days=date_index)
        # A date-level component that no feature can explain.
        market = 0.01 * math.sin(date_index / 3.0)
        for symbol_index in range(n_symbols):
            strength = (symbol_index + 1) / n_symbols
            if within_date_signal:
                feature = strength
                stock_component = 0.05 * (strength - 0.5)
            else:
                feature = 0.5
                stock_component = 0.0
            rows.append(
                {
                    "symbol": f"{symbol_index:06d}",
                    "trade_date": trade_date,
                    "feature_a": feature,
                    "feature_b": feature * 0.5 + 0.25,
                    "label": stock_component + market + generator.normal(scale=0.002),
                }
            )
    return pd.DataFrame(rows)


def test_lightgbm_artifact_has_time_ordered_validation_and_can_reload(tmp_path) -> None:
    provider = make_demo_provider(seed=7, quote_steps=2, replay_days=2)
    engine = FeatureEngine()
    training = engine.build_training_frame(provider.get_daily_frame(), horizon_days=5)

    model = LightGBMModel()
    artifact = model.fit(
        training,
        artifact_dir=tmp_path,
        min_training_rows=500,
        validation_fraction=0.2,
        min_validation_days=20,
    )

    assert artifact.training_rows >= 500
    assert artifact.validation_rows > 0
    assert artifact.validation_start is not None
    assert artifact.training_end < artifact.validation_start
    assert "validation_rank_corr" in artifact.metrics
    assert artifact.artifact_path is not None

    loaded = LightGBMModel.load(
        artifact.artifact_path,
        expected_feature_columns=engine.FACTOR_COLUMNS,
    )
    scored = loaded.score(engine.latest(provider.get_daily_frame()))
    assert len(scored) == len(provider.get_security_master())
    assert scored["model_version"].eq(artifact.model_version).all()


def test_daily_rank_ic_mean_matches_rank_agreement() -> None:
    dates = np.repeat(np.arange(4), 50)
    values = np.tile(np.arange(50, dtype=float), 4)

    assert LightGBMModel.daily_rank_ic_mean(dates, values, values) == 1.0
    assert LightGBMModel.daily_rank_ic_mean(dates, values, -values) == -1.0
    # A constant score carries no ranking information at all.
    assert LightGBMModel.daily_rank_ic_mean(dates, values, np.zeros_like(values)) == 0.0


def test_rank_ic_early_stopping_keeps_cross_sectional_dispersion(tmp_path) -> None:
    artifact = LightGBMModel(feature_columns=["feature_a", "feature_b"]).fit(
        _synthetic_frame(within_date_signal=True),
        artifact_dir=tmp_path,
        min_training_rows=500,
        validation_fraction=0.2,
        min_validation_days=5,
        validation_gap_days=0,
        early_stopping_metric="rank_ic",
    )

    assert artifact.health == LightGBMModel.HEALTH_OK
    assert artifact.metrics["validation_unique_predictions_per_day"] > 1
    assert artifact.metrics["validation_ic_mean"] > 0
    assert artifact.training_params["early_stopping_metric"] == "rank_ic"


def test_model_without_ranking_power_is_reported_as_degenerate(tmp_path) -> None:
    artifact = LightGBMModel(feature_columns=["feature_a", "feature_b"]).fit(
        _synthetic_frame(within_date_signal=False),
        artifact_dir=tmp_path,
        min_training_rows=500,
        validation_fraction=0.2,
        min_validation_days=5,
        validation_gap_days=0,
        early_stopping_metric="rank_ic",
    )

    assert artifact.health == LightGBMModel.HEALTH_DEGENERATE
    assert artifact.metrics["validation_unique_predictions_per_day"] <= 1.0
    assert "横截面" in artifact.health_detail


def test_training_configuration_changes_the_model_version(tmp_path) -> None:
    frame = _synthetic_frame(within_date_signal=True)
    common = {
        "artifact_dir": tmp_path,
        "min_training_rows": 500,
        "validation_fraction": 0.2,
        "min_validation_days": 5,
        "validation_gap_days": 0,
    }
    by_metric = {
        metric: LightGBMModel(feature_columns=["feature_a", "feature_b"]).fit(
            frame, early_stopping_metric=metric, **common
        ).model_version
        for metric in LightGBMModel.EARLY_STOPPING_METRICS
    }

    assert len(set(by_metric.values())) == len(by_metric)
