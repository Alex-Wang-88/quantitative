from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import BacktestConfig, BacktestResult, FactorBacktester
from .features import FeatureEngine
from .model import LightGBMModel
from .storage import DataQualityReport, HistoricalStore


@dataclass(frozen=True)
class WalkForwardConfig:
    """Time-ordered LightGBM walk-forward evaluation settings."""

    train_days: int = 504
    test_days: int = 21
    step_days: int = 21
    horizon_days: int = 5
    min_training_rows: int = 500
    validation_fraction: float = 0.20
    min_validation_days: int = 20
    validation_gap_days: int = 5
    # Early stopping selects on the same criterion the model is used for:
    # cross-sectional ranking.  L2 stopping halts after a few trees and then
    # emits one score per date, which looks like a valid model in the report
    # but has no stock-selection content.
    early_stopping_rounds: int = 50
    early_stopping_metric: str = "rank_ic"
    # Trailing mean of the model score, in trading days.  ``1`` uses the raw
    # daily score and reproduces the original behaviour.  A shallow model can
    # carry a positive aggregate rank-IC while its individual-name ordering
    # still reshuffles almost completely from one week to the next; the
    # portfolio then tries to replace its whole book every rebalance and a
    # turnover-capped broker can never reach its target weights.  Averaging the
    # score over a window matched to the label horizon damps that noise using
    # information that was already available, so it introduces no look-ahead.
    score_smoothing_days: int = 1
    artifact_dir: str | Path = "data/models/walkforward"

    @property
    def effective_validation_gap_days(self) -> int:
        """Gap actually applied, never shorter than one full label horizon.

        A training row's label looks ``horizon_days`` forward.  If the validation
        window starts fewer than ``horizon_days`` after the last training day,
        the tail of the training set speaks into the window that early stopping
        is judged on -- the model then picks its iteration count using labels it
        has already been fitted to, and the reported IC is optimistic.

        The 5-day defaults happen to be exactly safe (gap == horizon), which is
        why this stayed invisible until the horizon was about to be lengthened.
        Deriving the gap instead of trusting the caller keeps it safe at any
        horizon; the applied value is disclosed in ``stats``.
        """

        return max(self.validation_gap_days, self.horizon_days)

    def validate(self) -> None:
        if self.train_days <= 0:
            raise ValueError("train_days must be positive")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.score_smoothing_days <= 0:
            raise ValueError("score_smoothing_days must be positive")
        if self.min_training_rows <= 0:
            raise ValueError("min_training_rows must be positive")
        if self.validation_gap_days < 0:
            raise ValueError("validation_gap_days must be non-negative")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        if self.early_stopping_metric not in LightGBMModel.EARLY_STOPPING_METRICS:
            raise ValueError(
                "early_stopping_metric must be one of: "
                + ", ".join(LightGBMModel.EARLY_STOPPING_METRICS)
            )


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    training_rows: int
    test_rows: int
    model_version: str
    metrics: dict[str, float]
    artifact_path: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    equity_curve: pd.DataFrame
    trades: list[Any]
    stats: dict[str, Any]
    quality: DataQualityReport
    folds: list[WalkForwardFold]

    def as_dict(self) -> dict[str, Any]:
        equity = self.equity_curve.copy()
        if not equity.empty and "trade_date" in equity:
            equity["trade_date"] = equity["trade_date"].astype(str)
        return {
            "stats": self.stats,
            "quality": self.quality.as_dict(),
            "folds": [fold.as_dict() for fold in self.folds],
            "trades": [asdict(trade) for trade in self.trades],
            "equity_curve": equity.to_dict(orient="records"),
        }


@dataclass
class FittedWalkForward:
    """Everything one scoring pass produces, before any portfolio is built.

    Training a fold costs minutes and predicting it costs seconds, while every
    knob that decides *how the scored universe is turned into a book* --
    score smoothing, rebalance interval, turnover cap, incumbent buffer -- costs
    nothing.  Separating the two lets a sweep reuse one fit instead of paying
    for the models again on every variant, which is the difference between a
    one-run sensitivity check and a multi-hour one.
    """

    normalized: pd.DataFrame
    quality: DataQualityReport
    features: pd.DataFrame
    folds: list[WalkForwardFold]
    first_evaluation_date: Any
    last_evaluation_date: Any
    # ``symbol`` / ``trade_date`` / ``label`` only.  Kept so a variant can
    # re-measure the traded bucket against the score it *actually* submitted
    # orders for: the per-fold diagnostic recorded during the fit is built on the
    # raw predictions, which makes it structurally blind to a transform such as
    # score smoothing.  Without this a smoothed run would report the unsmoothed
    # bucket and invite the false conclusion that "nothing about the ranking
    # changed".
    labels: pd.DataFrame


def smooth_scores(features: pd.DataFrame, window: int) -> pd.DataFrame:
    """Replace ``factor_score`` with its trailing per-symbol mean.

    Only the current day and earlier contribute, so the score used on day ``T``
    never depends on anything after ``T``.  The point is to damp the day-to-day
    reshuffling of the ranking: a shallow model can post a positive aggregate
    rank-IC while its individual-name ordering still flips almost every day,
    which makes a turnover-capped portfolio chase its own target book.
    """

    if window <= 1 or "factor_score" not in features.columns:
        return features
    ordered = features.sort_values(["symbol", "trade_date"])
    ordered["factor_score"] = ordered.groupby("symbol", sort=False)[
        "factor_score"
    ].transform(lambda values: values.rolling(window, min_periods=1).mean())
    return ordered.sort_index()


class LightGBMWalkForwardBacktester:
    """Fit a fresh local model at each time-ordered evaluation fold.

    Features are calculated on the full historical frame, but each model only
    sees labels whose five-day future window ends before the test period.  The
    scored test periods are then passed through the same daily PaperBroker
    backtest used by the transparent factor baseline.
    """

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self.feature_engine = feature_engine or FeatureEngine()

    def fit_folds(
        self,
        daily_frame: pd.DataFrame,
        walkforward: WalkForwardConfig | None = None,
        *,
        benchmark_frame: pd.DataFrame | None = None,
        benchmark_symbol: str = "000300",
        label_mode: str = "cross_sectional",
        feature_columns: list[str] | None = None,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
        max_positions: int = 30,
    ) -> FittedWalkForward:
        """Fit one model per fold and score the test windows.

        ``max_positions`` only shapes the traded-bucket diagnostic recorded on
        each fold (the bucket is the size of the book), so every portfolio
        variant fitted on these folds must agree on it; the sweep entry point
        enforces that rather than quietly labelling folds with the wrong size.
        """

        walkforward = walkforward or WalkForwardConfig()
        walkforward.validate()

        normalized = HistoricalStore.normalize_daily(daily_frame)
        quality = HistoricalStore.quality(normalized)
        if not quality.ok:
            raise ValueError(f"daily data quality failed: {quality.as_dict()}")
        if normalized.empty:
            raise ValueError("daily data is empty")

        features = self.feature_engine.build_features(
            normalized,
            benchmark_frame=benchmark_frame,
            benchmark_symbol=benchmark_symbol,
        )
        feature_dates = pd.to_datetime(features["trade_date"], errors="coerce")
        if feature_dates.isna().any():
            raise ValueError("feature frame contains invalid trade_date values")
        features["trade_date"] = feature_dates.dt.date
        training = self.feature_engine.build_training_frame(
            normalized,
            horizon_days=walkforward.horizon_days,
            benchmark_frame=benchmark_frame,
            benchmark_symbol=benchmark_symbol,
            label_mode=label_mode,
            feature_frame=features,
        )
        training_dates = pd.to_datetime(training["trade_date"], errors="coerce")
        if training_dates.isna().any():
            raise ValueError("training frame contains invalid trade_date values")
        training["trade_date"] = training_dates.dt.date
        dates = sorted(features["trade_date"].unique())
        first_test_index = walkforward.train_days + walkforward.horizon_days
        evaluation_start_date = (
            pd.Timestamp(evaluation_start).date() if evaluation_start else None
        )
        evaluation_end_date = pd.Timestamp(evaluation_end).date() if evaluation_end else None
        if evaluation_start_date is not None:
            requested_index = next(
                (index for index, current in enumerate(dates) if current >= evaluation_start_date),
                len(dates),
            )
            first_test_index = max(first_test_index, requested_index)
        if len(dates) <= first_test_index:
            raise ValueError(
                "not enough trading days for walk-forward evaluation: "
                f"{len(dates)} <= {first_test_index}"
            )

        features["model_score"] = pd.NA
        folds: list[WalkForwardFold] = []
        test_start_index = first_test_index
        fold_number = 1
        while test_start_index < len(dates):
            if evaluation_end_date is not None and dates[test_start_index] > evaluation_end_date:
                break
            test_end_index = min(test_start_index + walkforward.test_days, len(dates))
            if evaluation_end_date is not None:
                evaluation_end_index = next(
                    (
                        index
                        for index, current in enumerate(dates)
                        if current > evaluation_end_date
                    ),
                    len(dates),
                )
                test_end_index = min(test_end_index, evaluation_end_index)
            if test_end_index <= test_start_index:
                break
            train_end_index = test_start_index - walkforward.horizon_days
            train_start_index = max(0, train_end_index - walkforward.train_days + 1)
            train_start = dates[train_start_index]
            train_end = dates[train_end_index]
            test_start = dates[test_start_index]
            test_end = dates[test_end_index - 1]

            train_mask = training["trade_date"].between(train_start, train_end)
            train_rows = training.loc[train_mask].copy()
            test_mask = features["trade_date"].between(test_start, test_end)
            test_rows = features.loc[test_mask].copy()
            if train_rows.empty or test_rows.empty:
                test_start_index += walkforward.step_days
                fold_number += 1
                continue

            model = LightGBMModel(
                feature_columns=list(
                    feature_columns or self.feature_engine.MODEL_FEATURE_COLUMNS
                )
            )
            artifact = model.fit(
                train_rows,
                artifact_dir=walkforward.artifact_dir,
                min_training_rows=walkforward.min_training_rows,
                validation_fraction=walkforward.validation_fraction,
                min_validation_days=walkforward.min_validation_days,
                validation_gap_days=walkforward.effective_validation_gap_days,
                label_name=(
                    "benchmark_excess_return"
                    if label_mode == "benchmark_excess"
                    else "cross_sectional_excess_return"
                ),
                early_stopping_rounds=walkforward.early_stopping_rounds,
                early_stopping_metric=walkforward.early_stopping_metric,
            )
            scored = model.score(test_rows)
            features.loc[test_mask, "model_score"] = scored["score"].to_numpy()
            test_scores = test_rows[["symbol", "trade_date"]].copy()
            test_scores["prediction"] = scored["score"].to_numpy()
            test_labels = training[training["trade_date"].between(test_start, test_end)][
                ["symbol", "trade_date", "label"]
            ]
            test_evaluation = test_scores.merge(
                test_labels, on=["symbol", "trade_date"], how="inner"
            )
            fold_metrics = dict(artifact.metrics)
            fold_metrics["test_labeled_rows"] = float(len(test_evaluation))
            test_ic_values = (
                LightGBMModel._daily_ic_values(
                    test_evaluation, test_evaluation["prediction"]
                )
                if not test_evaluation.empty
                else []
            )
            fold_metrics["test_ic_days"] = float(len(test_ic_values))
            fold_metrics["test_ic_mean"] = (
                float(sum(test_ic_values) / len(test_ic_values))
                if test_ic_values
                else 0.0
            )
            fold_metrics["model_degenerate"] = (
                1.0 if artifact.health == LightGBMModel.HEALTH_DEGENERATE else 0.0
            )
            fold_metrics["test_top_bottom_spread"] = (
                LightGBMModel._top_bottom_spread(
                    test_evaluation, test_evaluation["prediction"]
                )
                if not test_evaluation.empty
                else 0.0
            )
            # The quintile spread above is a much wider bucket than the book, so
            # also measure the bucket that is actually traded.  Without it a
            # losing strategy cannot be attributed to the model or to execution.
            #
            # Recorded only when the fold has a measurable bucket.  A fold whose
            # test window falls entirely inside the label's forward-blind tail
            # has no labels at all, and writing ``0.0`` for it would feed "not
            # measured" into the aggregate as "measured, zero excess" -- quietly
            # dragging every reported bucket mean toward zero.
            if not test_evaluation.empty:
                fold_metrics.update(
                    {
                        f"test_{key}": value
                        for key, value in LightGBMModel._bucket_label_stats(
                            test_evaluation,
                            test_evaluation["prediction"],
                            max_positions,
                        ).items()
                    }
                )
                fold_metrics["test_bucket_size"] = float(max_positions)
            folds.append(
                WalkForwardFold(
                    fold=fold_number,
                    train_start=train_start.isoformat(),
                    train_end=train_end.isoformat(),
                    test_start=test_start.isoformat(),
                    test_end=test_end.isoformat(),
                    training_rows=int(len(train_rows)),
                    test_rows=int(len(test_rows)),
                    model_version=artifact.model_version,
                    metrics=fold_metrics,
                    artifact_path=artifact.artifact_path,
                )
            )
            test_start_index += walkforward.step_days
            fold_number += 1

        if not folds:
            raise ValueError("walk-forward produced no trainable folds")

        first_evaluation_date = pd.Timestamp(folds[0].test_start).date()
        last_evaluation_date = pd.Timestamp(folds[-1].test_end).date()
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.date
        return FittedWalkForward(
            normalized=normalized,
            quality=quality,
            features=features,
            folds=folds,
            first_evaluation_date=first_evaluation_date,
            last_evaluation_date=last_evaluation_date,
            labels=training[["symbol", "trade_date", "label"]],
        )

    def run(
        self,
        daily_frame: pd.DataFrame,
        walkforward: WalkForwardConfig | None = None,
        backtest: BacktestConfig | None = None,
        *,
        benchmark_frame: pd.DataFrame | None = None,
        benchmark_symbol: str = "000300",
        label_mode: str = "cross_sectional",
        feature_columns: list[str] | None = None,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
    ) -> WalkForwardResult:
        walkforward = walkforward or WalkForwardConfig()
        walkforward.validate()
        backtest = backtest or BacktestConfig()
        fitted = self.fit_folds(
            daily_frame,
            walkforward,
            benchmark_frame=benchmark_frame,
            benchmark_symbol=benchmark_symbol,
            label_mode=label_mode,
            feature_columns=feature_columns,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            max_positions=backtest.max_positions,
        )
        return self.run_fitted_variant(
            fitted,
            walkforward=walkforward,
            backtest=backtest,
            label_mode=label_mode,
        )

    def run_fitted_variant(
        self,
        fitted: FittedWalkForward,
        *,
        walkforward: WalkForwardConfig,
        backtest: BacktestConfig,
        label_mode: str = "cross_sectional",
    ) -> WalkForwardResult:
        """Build the book on an already-fitted scoring pass.

        Only portfolio-side settings are read from ``walkforward``
        (``score_smoothing_days``) and the disclosures; the folds, the models
        and the raw scores all come from ``fitted`` untouched.  That is what
        makes two variants comparable: they differ in exactly one knob and
        share a byte-identical model.
        """

        evaluation_daily = fitted.normalized[
            fitted.normalized["trade_date"].between(
                fitted.first_evaluation_date, fitted.last_evaluation_date
            )
        ].copy()
        evaluation_features = fitted.features[
            fitted.features["trade_date"].between(
                fitted.first_evaluation_date, fitted.last_evaluation_date
            )
        ].copy()
        evaluation_features["factor_score"] = pd.to_numeric(
            evaluation_features["model_score"], errors="coerce"
        )
        if walkforward.score_smoothing_days > 1:
            evaluation_features = smooth_scores(
                evaluation_features, walkforward.score_smoothing_days
            )
        result: BacktestResult = FactorBacktester(
            feature_engine=self.feature_engine
        ).run(
            evaluation_daily,
            config=backtest,
            feature_frame=evaluation_features,
            score_column="factor_score",
        )
        stats: dict[str, Any] = dict(result.stats)
        stats["score_smoothing_days"] = float(walkforward.score_smoothing_days)
        stats["fold_count"] = float(len(fitted.folds))
        # A walk-forward run is only interpretable when its folds actually
        # learned a cross-sectional ranking; surface that count directly.
        stats["degenerate_folds"] = float(
            sum(
                1
                for fold in fitted.folds
                if fold.metrics.get("model_degenerate", 0.0) > 0.0
            )
        )
        stats["training_days"] = float(walkforward.train_days)
        stats["test_days_per_fold"] = float(walkforward.test_days)
        stats["horizon_days"] = float(walkforward.horizon_days)
        # Disclose the gap that was actually applied, so the caller can see the
        # horizon requirement was enforced rather than assumed.
        stats["validation_gap_days"] = float(walkforward.effective_validation_gap_days)
        # The decisive selection-quality number: the mean forward excess return
        # of the bucket the portfolio trades.  If this is not positive, no
        # amount of portfolio construction can make the strategy work, and the
        # effort belongs on the signal instead.
        bucket_values = [
            fold.metrics["test_bucket_label_mean"]
            for fold in fitted.folds
            if "test_bucket_label_mean" in fold.metrics
        ]
        universe_values = [
            fold.metrics["test_universe_label_mean"]
            for fold in fitted.folds
            if "test_universe_label_mean" in fold.metrics
        ]
        positive_values = [
            fold.metrics["test_bucket_label_positive_ratio"]
            for fold in fitted.folds
            if "test_bucket_label_positive_ratio" in fold.metrics
        ]
        stats["top_bucket_size"] = float(backtest.max_positions)
        stats["test_bucket_label_mean"] = (
            float(sum(bucket_values) / len(bucket_values)) if bucket_values else 0.0
        )
        stats["test_universe_label_mean"] = (
            float(sum(universe_values) / len(universe_values)) if universe_values else 0.0
        )
        stats["test_bucket_label_positive_ratio"] = (
            float(sum(positive_values) / len(positive_values)) if positive_values else 0.0
        )
        # The metrics above describe the *raw* model ranking.  A variant may not
        # trade that ranking at all (score smoothing is exactly such a
        # transform), and a diagnostic that silently describes a score nobody
        # traded is worse than no diagnostic: it invites "the ranking is
        # unchanged, so the gain must be execution" when the traded ranking is in
        # fact a different one.  Re-measure the traded bucket with the same
        # per-fold windows so the two are directly comparable; with no transform
        # the two must agree exactly, which is the check that this stayed honest.
        traded_bucket, traded_universe, traded_positive = self._traded_bucket_metrics(
            fitted, evaluation_features, backtest.max_positions
        )
        stats["traded_bucket_label_mean"] = (
            float(sum(traded_bucket) / len(traded_bucket)) if traded_bucket else 0.0
        )
        stats["traded_universe_label_mean"] = (
            float(sum(traded_universe) / len(traded_universe)) if traded_universe else 0.0
        )
        stats["traded_bucket_label_positive_ratio"] = (
            float(sum(traded_positive) / len(traded_positive)) if traded_positive else 0.0
        )
        # How many folds actually carried a measurable bucket.  Without it the
        # reader cannot tell a mean over 11 folds from a mean over 8.
        stats["bucket_measured_folds"] = float(len(bucket_values))
        stats["traded_bucket_measured_folds"] = float(len(traded_bucket))
        stats["label_mode"] = label_mode
        stats["evaluation_start"] = fitted.first_evaluation_date.isoformat()
        stats["evaluation_end"] = fitted.last_evaluation_date.isoformat()
        return WalkForwardResult(
            equity_curve=result.equity_curve,
            trades=result.trades,
            stats=stats,
            quality=fitted.quality,
            folds=fitted.folds,
        )

    def _traded_bucket_metrics(
        self,
        fitted: FittedWalkForward,
        evaluation_features: pd.DataFrame,
        max_positions: int,
    ) -> tuple[list[float], list[float], list[float]]:
        """Per-fold bucket quality of the score that was actually traded.

        Walks the fold windows recorded during the fit so each number lines up
        one-for-one with ``test_bucket_*``.  A variant that transforms the score
        (smoothing) must be judged on the transformed ranking; a variant that
        does not must reproduce ``test_bucket_*`` exactly.
        """

        scores = evaluation_features[["symbol", "trade_date", "factor_score"]]
        bucket: list[float] = []
        universe: list[float] = []
        positive: list[float] = []
        for fold in fitted.folds:
            start = pd.Timestamp(fold.test_start).date()
            end = pd.Timestamp(fold.test_end).date()
            window = scores[scores["trade_date"].between(start, end)]
            merged = window.merge(
                fitted.labels, on=["symbol", "trade_date"], how="inner"
            ).dropna(subset=["label", "factor_score"])
            if merged.empty:
                continue
            measured = LightGBMModel._bucket_label_stats(
                merged, merged["factor_score"], max_positions
            )
            bucket.append(measured["bucket_label_mean"])
            universe.append(measured["universe_label_mean"])
            positive.append(measured["bucket_label_positive_ratio"])
        return bucket, universe, positive

    def run_score_variants(
        self,
        daily_frame: pd.DataFrame,
        *,
        variants: list[tuple[str, BacktestConfig, int]],
        walkforward: WalkForwardConfig | None = None,
        benchmark_frame: pd.DataFrame | None = None,
        benchmark_symbol: str = "000300",
        label_mode: str = "cross_sectional",
        feature_columns: list[str] | None = None,
        evaluation_start: str | None = None,
        evaluation_end: str | None = None,
    ) -> dict[str, WalkForwardResult]:
        """Fit once, then run several portfolio variants over the same scores.

        ``variants`` is a list of ``(name, backtest_config, smoothing_days)``.
        Training dominates the cost of a walk-forward run and the portfolio
        knobs cost nothing, so sweeping them this way turns an N-run sweep into
        a single fit plus N backtests -- the difference between a feasible
        hyper-parameter sensitivity check and an afternoon of retraining.

        Every variant must agree on ``max_positions``: the per-fold
        traded-bucket diagnostic is recorded once during the fit, and letting
        variants disagree would label folds with a book size the run never used.
        """

        if not variants:
            raise ValueError("at least one variant is required")
        book_sizes = {variant[1].max_positions for variant in variants}
        if len(book_sizes) > 1:
            raise ValueError(
                "all variants must share max_positions (the fold-level "
                "traded-bucket diagnostic is recorded once, at fit time): "
                + ", ".join(str(size) for size in sorted(book_sizes))
            )

        fitted = self.fit_folds(
            daily_frame,
            walkforward,
            benchmark_frame=benchmark_frame,
            benchmark_symbol=benchmark_symbol,
            label_mode=label_mode,
            feature_columns=feature_columns,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            max_positions=variants[0][1].max_positions,
        )
        base = walkforward or WalkForwardConfig()
        base.validate()
        results: dict[str, WalkForwardResult] = {}
        for name, backtest, smoothing_days in variants:
            variant_walkforward = replace(
                base, score_smoothing_days=max(1, int(smoothing_days))
            )
            results[name] = self.run_fitted_variant(
                fitted,
                walkforward=variant_walkforward,
                backtest=backtest,
                label_mode=label_mode,
            )
        return results
