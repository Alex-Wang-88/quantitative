from __future__ import annotations

from dataclasses import asdict, dataclass
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
    artifact_dir: str | Path = "data/models/walkforward"

    def validate(self) -> None:
        if self.train_days <= 0:
            raise ValueError("train_days must be positive")
        if self.test_days <= 0:
            raise ValueError("test_days must be positive")
        if self.step_days <= 0:
            raise ValueError("step_days must be positive")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if self.min_training_rows <= 0:
            raise ValueError("min_training_rows must be positive")


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
    stats: dict[str, float]
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


class LightGBMWalkForwardBacktester:
    """Fit a fresh local model at each time-ordered evaluation fold.

    Features are calculated on the full historical frame, but each model only
    sees labels whose five-day future window ends before the test period.  The
    scored test periods are then passed through the same daily PaperBroker
    backtest used by the transparent factor baseline.
    """

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self.feature_engine = feature_engine or FeatureEngine()

    def run(
        self,
        daily_frame: pd.DataFrame,
        walkforward: WalkForwardConfig | None = None,
        backtest: BacktestConfig | None = None,
    ) -> WalkForwardResult:
        walkforward = walkforward or WalkForwardConfig()
        walkforward.validate()
        backtest = backtest or BacktestConfig()

        normalized = HistoricalStore.normalize_daily(daily_frame)
        quality = HistoricalStore.quality(normalized)
        if not quality.ok:
            raise ValueError(f"daily data quality failed: {quality.as_dict()}")
        if normalized.empty:
            raise ValueError("daily data is empty")

        features = self.feature_engine.build_features(normalized)
        feature_dates = pd.to_datetime(features["trade_date"], errors="coerce")
        if feature_dates.isna().any():
            raise ValueError("feature frame contains invalid trade_date values")
        features["trade_date"] = feature_dates.dt.date
        training = self.feature_engine.build_training_frame(
            normalized, horizon_days=walkforward.horizon_days
        )
        training_dates = pd.to_datetime(training["trade_date"], errors="coerce")
        if training_dates.isna().any():
            raise ValueError("training frame contains invalid trade_date values")
        training["trade_date"] = training_dates.dt.date
        dates = sorted(features["trade_date"].unique())
        first_test_index = walkforward.train_days + walkforward.horizon_days
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
            test_end_index = min(test_start_index + walkforward.test_days, len(dates))
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

            model = LightGBMModel(feature_columns=list(self.feature_engine.FACTOR_COLUMNS))
            artifact = model.fit(
                train_rows,
                artifact_dir=walkforward.artifact_dir,
                min_training_rows=walkforward.min_training_rows,
                validation_fraction=walkforward.validation_fraction,
                min_validation_days=walkforward.min_validation_days,
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
            fold_metrics["test_ic_mean"] = (
                LightGBMModel._daily_ic(test_evaluation, test_evaluation["prediction"])
                if not test_evaluation.empty
                else 0.0
            )
            fold_metrics["test_top_bottom_spread"] = (
                LightGBMModel._top_bottom_spread(
                    test_evaluation, test_evaluation["prediction"]
                )
                if not test_evaluation.empty
                else 0.0
            )
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
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.date
        evaluation_daily = normalized[normalized["trade_date"] >= first_evaluation_date].copy()
        evaluation_features = features[features["trade_date"] >= first_evaluation_date].copy()
        evaluation_features["factor_score"] = pd.to_numeric(
            evaluation_features["model_score"], errors="coerce"
        )
        result: BacktestResult = FactorBacktester(
            feature_engine=self.feature_engine
        ).run(
            evaluation_daily,
            config=backtest,
            feature_frame=evaluation_features,
            score_column="factor_score",
        )
        stats = dict(result.stats)
        stats["fold_count"] = float(len(folds))
        stats["training_days"] = float(walkforward.train_days)
        stats["test_days_per_fold"] = float(walkforward.test_days)
        stats["horizon_days"] = float(walkforward.horizon_days)
        return WalkForwardResult(
            equity_curve=result.equity_curve,
            trades=result.trades,
            stats=stats,
            quality=quality,
            folds=folds,
        )
