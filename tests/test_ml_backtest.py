from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quantpaper.backtest import BacktestConfig
from quantpaper.data import make_demo_provider
from quantpaper.ml_backtest import (
    LightGBMWalkForwardBacktester,
    WalkForwardConfig,
    smooth_scores,
)
from quantpaper.model import LightGBMModel


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


def test_score_smoothing_is_a_trailing_average_that_cannot_see_the_future() -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    frame = pd.DataFrame(
        [
            {"symbol": "600001", "trade_date": dates[0], "factor_score": 1.0},
            {"symbol": "600001", "trade_date": dates[1], "factor_score": 3.0},
            {"symbol": "600001", "trade_date": dates[2], "factor_score": 100.0},
            {"symbol": "600002", "trade_date": dates[0], "factor_score": 8.0},
            {"symbol": "600002", "trade_date": dates[1], "factor_score": 8.0},
            {"symbol": "600002", "trade_date": dates[2], "factor_score": 8.0},
        ]
    )

    smoothed = smooth_scores(frame, 2).set_index(["symbol", "trade_date"])

    assert smoothed.loc[("600001", dates[0]), "factor_score"] == pytest.approx(1.0)
    assert smoothed.loc[("600001", dates[1]), "factor_score"] == pytest.approx(2.0)
    # A two-day window averages the spike with the prior day; the current day's
    # own value is included, so nothing later can leak in.
    assert smoothed.loc[("600001", dates[2]), "factor_score"] == pytest.approx(51.5)
    assert smoothed.loc[("600002", dates[2]), "factor_score"] == pytest.approx(8.0)
    # A window of one is the identity, i.e. the historical behaviour.
    unchanged = smooth_scores(frame, 1).set_index(["symbol", "trade_date"])
    assert unchanged.loc[("600001", dates[2]), "factor_score"] == pytest.approx(100.0)


def test_walkforward_rejects_a_non_positive_smoothing_window() -> None:
    with pytest.raises(ValueError, match="score_smoothing_days must be positive"):
        WalkForwardConfig(score_smoothing_days=0).validate()


def test_walkforward_records_the_smoothing_window_it_used(tmp_path) -> None:
    provider = make_demo_provider(seed=13, quote_steps=2, replay_days=2)
    result = LightGBMWalkForwardBacktester().run(
        provider.get_daily_frame(),
        walkforward=WalkForwardConfig(
            train_days=100,
            test_days=10,
            step_days=10,
            horizon_days=5,
            min_training_rows=100,
            min_validation_days=5,
            score_smoothing_days=3,
            artifact_dir=tmp_path,
        ),
        backtest=BacktestConfig(
            max_positions=4,
            single_weight_cap=0.25,
            max_daily_turnover=0.50,
            max_daily_orders=8,
        ),
    )

    assert result.stats["score_smoothing_days"] == 3.0
    assert result.stats["final_equity"] > 0


def test_validation_gap_is_derived_from_the_label_horizon() -> None:
    """The validation window must start a full label horizon after training.

    A training row's label looks ``horizon_days`` forward.  With a 20-day label
    and the 5-day default gap, the last fifteen training days would carry labels
    that reach into the validation window -- early stopping would then choose
    the iteration count from data the model had already been fitted to.
    """

    short = WalkForwardConfig(horizon_days=5, validation_gap_days=5)
    assert short.effective_validation_gap_days == 5

    long = WalkForwardConfig(horizon_days=20)
    assert long.effective_validation_gap_days == 20
    # An explicitly wider gap is respected rather than narrowed.
    wider = WalkForwardConfig(horizon_days=20, validation_gap_days=30)
    assert wider.effective_validation_gap_days == 30
    assert long.validate() is None


def test_traded_bucket_label_stats_uses_the_portfolio_sized_bucket() -> None:
    """The traded bucket is the top-N, not the top quintile.

    ``_top_bottom_spread`` looks at 20% of the cross-section -- about a thousand
    names on a real A-share universe -- while the book holds thirty.  A signal
    can order the broad quintiles correctly and still be useless in the tail the
    portfolio is restricted to, so the narrower measurement has to exist and has
    to be the one attached to the bucket size.
    """

    day_one = pd.Timestamp("2026-03-02")
    day_two = pd.Timestamp("2026-03-03")
    labels = [0.10, 0.05, 0.02, 0.00, -0.01, -0.02, -0.05, -0.10, -0.20, -0.30]
    frame = pd.DataFrame(
        {
            "trade_date": [day_one] * 10 + [day_two] * 10,
            "label": labels * 2,
            # Day one ranks the best labels first, day two ranks them last.
            "prediction": list(range(9, -1, -1)) + list(range(10)),
        }
    )

    stats = LightGBMModel._bucket_label_stats(frame, frame["prediction"], top_n=3)

    # Day one: (0.10 + 0.05 + 0.02) / 3.  Day two: (-0.30 - 0.20 - 0.10) / 3.
    assert stats["bucket_label_mean"] == pytest.approx((0.17 / 3 - 0.60 / 3) / 2)
    assert stats["bucket_label_positive_ratio"] == pytest.approx(0.5)
    # The universe mean is the same on both days and is *not* subtracted from
    # the bucket: the label is already benchmark-relative.
    assert stats["universe_label_mean"] == pytest.approx(sum(labels) / 10)
    # A bucket as wide as the cross-section degenerates to the universe mean.
    whole = LightGBMModel._bucket_label_stats(frame, frame["prediction"], top_n=10)
    assert whole["bucket_label_mean"] == pytest.approx(sum(labels) / 10)
    assert LightGBMModel._bucket_label_stats(frame.iloc[0:0], pd.Series(dtype=float), 3) == {
        "bucket_label_mean": 0.0,
        "universe_label_mean": 0.0,
        "bucket_label_positive_ratio": 0.0,
    }


def test_walkforward_discloses_the_quality_of_the_traded_bucket(tmp_path) -> None:
    provider = make_demo_provider(seed=17, quote_steps=2, replay_days=2)
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

    assert result.stats["top_bucket_size"] == 4.0
    assert result.stats["validation_gap_days"] == 5.0
    assert result.stats["bucket_measured_folds"] > 0
    # A fold must carry the bucket metric exactly when it had labelled rows to
    # measure.  Omission is the honest encoding of "not measurable"; writing 0.0
    # would enter the aggregate as "measured, zero excess" and drag it to zero.
    measured = [fold for fold in result.folds if "test_bucket_label_mean" in fold.metrics]
    unmeasured = [fold for fold in result.folds if "test_bucket_label_mean" not in fold.metrics]
    assert len(measured) == int(result.stats["bucket_measured_folds"])
    assert all(fold.metrics["test_labeled_rows"] > 0 for fold in measured)
    assert all(fold.metrics["test_labeled_rows"] == 0 for fold in unmeasured)
    assert all("test_bucket_size" in fold.metrics for fold in measured)
    # The aggregate must be the mean of the per-fold numbers, not a stray zero.
    per_fold = [fold.metrics["test_bucket_label_mean"] for fold in measured]
    assert result.stats["test_bucket_label_mean"] == pytest.approx(
        sum(per_fold) / len(per_fold)
    )


def _sweep_fixture() -> tuple[pd.DataFrame, WalkForwardConfig, BacktestConfig]:
    provider = make_demo_provider(seed=23, quote_steps=2, replay_days=2)
    walkforward = WalkForwardConfig(
        train_days=100,
        test_days=10,
        step_days=10,
        horizon_days=5,
        min_training_rows=100,
        min_validation_days=5,
    )
    backtest = BacktestConfig(
        max_positions=4,
        single_weight_cap=0.25,
        max_daily_turnover=0.50,
        max_daily_orders=8,
    )
    return provider.get_daily_frame(), walkforward, backtest


def test_a_sweep_variant_reproduces_the_single_run_it_replaces(tmp_path) -> None:
    """``run_score_variants`` must equal ``run`` for the same settings.

    The sweep exists only to avoid retraining; if a variant drifted from the
    single-run path the whole sensitivity table would describe a strategy that
    was never measured.  The unsmoothed window doubles as the control, so this
    is the equivalence check for the fit/backtest split.
    """

    daily, walkforward, backtest = _sweep_fixture()
    walkforward = replace(walkforward, artifact_dir=tmp_path / "single")
    direct = LightGBMWalkForwardBacktester().run(
        daily, walkforward=walkforward, backtest=backtest
    )

    swept = LightGBMWalkForwardBacktester().run_score_variants(
        daily,
        variants=[("smooth1", backtest, 1)],
        walkforward=replace(walkforward, artifact_dir=tmp_path / "sweep"),
    )["smooth1"]

    assert swept.stats["total_return"] == pytest.approx(
        direct.stats["total_return"], abs=1e-12
    )
    assert swept.stats["turnover"] == pytest.approx(direct.stats["turnover"], abs=1e-12)
    assert [fold.test_start for fold in swept.folds] == [
        fold.test_start for fold in direct.folds
    ]
    assert len(swept.trades) == len(direct.trades)


def test_a_sweep_scores_once_and_replays_several_windows(tmp_path) -> None:
    daily, walkforward, backtest = _sweep_fixture()

    results = LightGBMWalkForwardBacktester().run_score_variants(
        daily,
        variants=[
            ("smooth1", backtest, 1),
            ("smooth5", backtest, 5),
            ("smooth20", backtest, 20),
        ],
        walkforward=replace(walkforward, artifact_dir=tmp_path),
    )

    assert set(results) == {"smooth1", "smooth5", "smooth20"}
    for name, result in results.items():
        window = int(name.removeprefix("smooth"))
        assert result.stats["score_smoothing_days"] == float(window)
    # Every variant is built on the same folds, so the models -- and therefore
    # the per-fold diagnostics that depend on them -- are shared, not refit.
    assert results["smooth1"].folds is results["smooth20"].folds


def test_a_sweep_refuses_variants_that_disagree_on_book_size(tmp_path) -> None:
    """The fold bucket diagnostic is sized once, at fit time.

    Letting a variant declare a different ``max_positions`` would publish fold
    metrics measured against a book size no variant ever held, so the mismatch
    has to fail loudly instead of silently mislabelling the sweep.
    """

    daily, walkforward, backtest = _sweep_fixture()

    with pytest.raises(ValueError, match="max_positions"):
        LightGBMWalkForwardBacktester().run_score_variants(
            daily,
            variants=[
                ("smooth1", backtest, 1),
                ("smooth20", replace(backtest, max_positions=5), 20),
            ],
            walkforward=replace(walkforward, artifact_dir=tmp_path),
        )


def test_untraded_bucket_equals_the_raw_bucket_only_without_a_transform(tmp_path) -> None:
    """The traded-bucket diagnostic must not go blind to a score transform.

    ``test_bucket_*`` is recorded during the fit on the raw predictions, so it
    is identical for every variant by construction.  Reporting only that would
    make a transform look like a pure execution effect.  With no transform the
    two diagnostics describe the same score and must agree exactly; that
    equality is what licenses reading their difference under smoothing.
    """

    daily, walkforward, backtest = _sweep_fixture()

    results = LightGBMWalkForwardBacktester().run_score_variants(
        daily,
        variants=[("smooth1", backtest, 1), ("smooth5", backtest, 5)],
        walkforward=replace(walkforward, artifact_dir=tmp_path),
    )

    for result in results.values():
        assert "traded_bucket_label_mean" in result.stats
        assert "traded_bucket_label_positive_ratio" in result.stats

    untransformed = results["smooth1"]
    assert untransformed.stats["traded_bucket_label_mean"] == pytest.approx(
        untransformed.stats["test_bucket_label_mean"], abs=1e-12
    )
    assert untransformed.stats["traded_universe_label_mean"] == pytest.approx(
        untransformed.stats["test_universe_label_mean"], abs=1e-12
    )
    # The two aggregates also have to cover the same folds, or their agreement
    # could be a coincidence of two different denominators.
    assert untransformed.stats["traded_bucket_measured_folds"] == (
        untransformed.stats["bucket_measured_folds"]
    )
    # ... while the fitted folds still carry the raw number, untouched.
    assert any("test_bucket_label_mean" in fold.metrics for fold in untransformed.folds)
