from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import FeatureEngine


@dataclass(frozen=True)
class ModelArtifact:
    """Auditable metadata stored next to a trained local model."""

    model_version: str
    model_name: str
    trained_at: datetime
    training_start: str
    training_end: str
    feature_columns: list[str]
    label: str
    metrics: dict[str, float]
    artifact_path: str | None = None
    validation_start: str | None = None
    validation_end: str | None = None
    training_rows: int = 0
    validation_rows: int = 0
    validation_gap_days: int = 0
    # ``health`` makes a degenerate model impossible to mistake for a valid
    # one.  A model that emits one score per trading date has no
    # cross-sectional ranking power, so downstream selection is effectively
    # random even though every headline metric may still look plausible.
    health: str = "OK"
    health_detail: str = ""
    training_params: dict[str, Any] = field(default_factory=dict)


class FactorBaselineModel:
    """Transparent factor scorer used before a trained model is available."""

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or FeatureEngine.FACTOR_COLUMNS)
        self.model_version = "factor-baseline-v0.1"
        self.artifact: ModelArtifact | None = None

    def fit(self, frame: pd.DataFrame) -> ModelArtifact:
        start = str(frame["trade_date"].min().date()) if not frame.empty else "unknown"
        end = str(frame["trade_date"].max().date()) if not frame.empty else "unknown"
        self.artifact = ModelArtifact(
            model_version=self.model_version,
            model_name="factor_baseline",
            trained_at=datetime.now().astimezone(),
            training_start=start,
            training_end=end,
            feature_columns=self.feature_columns,
            label="factor_score",
            metrics={},
        )
        return self.artifact

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[["symbol"]].copy()
        values = frame[self.feature_columns].mean(axis=1, skipna=True)
        result["score"] = values.fillna(0.5)
        result["reason"] = "因子基线：动量、质量、估值、流动性综合排名"
        result["model_version"] = self.model_version
        return result


class LightGBMModel:
    """Local LightGBM wrapper with time-ordered validation and persistence.

    The model is intentionally small enough for the user's RTX 5060/8GB
    machine.  It is a ranking aid for paper trading, not a direct order
    router.  The validation split is chronological so that a future date is
    never used to fit a prediction for an earlier date.

    The model exists to *rank stocks against each other on the same date*, so
    training selects on the mean daily cross-sectional Spearman IC of the
    validation window (`early_stopping_metric="rank_ic"`) rather than on L2
    error.  L2 is dominated by label noise that no feature can explain, so an
    L2-stopped model regularly halts after one or two trees and then emits a
    single score per date -- that is, a market-timing number with no
    stock-selection content.  Both L2 and rank-IC metrics stay in the
    artifact so the two criteria can always be compared.
    """

    MODEL_NAME = "lightgbm"

    HEALTH_OK = "OK"
    HEALTH_DEGENERATE = "DEGENERATE"
    HEALTH_UNVALIDATED = "UNVALIDATED"
    EARLY_STOPPING_METRICS = ("rank_ic", "l2")
    # A validation window where every date collapses to one prediction has no
    # cross-sectional content at all; anything at or below this threshold is
    # reported as DEGENERATE instead of being silently scored as IC 0.
    MIN_UNIQUE_PREDICTIONS_PER_DAY = 2.0

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.feature_columns = list(feature_columns or FeatureEngine.FACTOR_COLUMNS)
        self.model: Any | None = None
        self.model_version = "untrained"
        self.artifact: ModelArtifact | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        artifact_dir: str | Path = "data/models",
        min_training_rows: int = 50,
        validation_fraction: float = 0.2,
        min_validation_days: int = 20,
        validation_gap_days: int = 5,
        label_name: str = "label",
        early_stopping_rounds: int = 50,
        early_stopping_metric: str = "rank_ic",
    ) -> ModelArtifact:
        if "label" not in frame.columns:
            raise ValueError("training frame must contain label")
        if early_stopping_metric not in self.EARLY_STOPPING_METRICS:
            raise ValueError(
                "early_stopping_metric must be one of: "
                + ", ".join(self.EARLY_STOPPING_METRICS)
            )
        if early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is not installed; install the project dependencies"
            ) from exc

        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"training frame missing features: {sorted(missing)}")

        clean = frame.copy()
        clean["trade_date"] = pd.to_datetime(clean["trade_date"], errors="coerce")
        clean = clean.dropna(subset=self.feature_columns + ["label", "trade_date"])
        clean = clean.sort_values("trade_date").reset_index(drop=True)
        if len(clean) < min_training_rows:
            raise ValueError(
                f"not enough rows to train a local model: {len(clean)} < {min_training_rows}"
            )

        clean["_model_date"] = clean["trade_date"].dt.normalize()
        dates = pd.Index(sorted(clean["_model_date"].unique()))
        validation_fraction = min(max(float(validation_fraction), 0.0), 0.8)
        requested_validation_days = max(
            1, math.ceil(len(dates) * validation_fraction)
        )
        if len(dates) > 2:
            requested_validation_days = max(
                requested_validation_days,
                min(max(1, int(min_validation_days)), len(dates) - 1),
            )
        validation_days = min(requested_validation_days, max(0, len(dates) - 1))
        validation_gap_days = max(0, int(validation_gap_days))
        validation_start_index = len(dates) - validation_days

        if validation_days > 0:
            validation_start = dates[validation_start_index]
            train_end_index = max(0, validation_start_index - validation_gap_days)
            train_end = dates[train_end_index]
            train = clean[clean["_model_date"] < train_end].copy()
            validation = clean[clean["_model_date"] >= validation_start].copy()
        else:
            train = clean.copy()
            validation = clean.iloc[0:0].copy()

        if len(train) < min_training_rows:
            # A small history should still be trainable if the caller has
            # explicitly allowed it.  Keep validation metadata honest and use
            # all rows rather than silently training on a tiny subset.
            train = clean.copy()
            validation = clean.iloc[0:0].copy()

        tree_params: dict[str, Any] = {
            "n_estimators": 240,
            "learning_rate": 0.03,
            "num_leaves": 16,
            "max_depth": -1,
            "min_child_samples": 120,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
            "subsample": 0.90,
            "colsample_bytree": 0.80,
            "random_state": 7,
            "verbosity": -1,
        }
        training_params: dict[str, Any] = {
            **tree_params,
            "early_stopping_rounds": int(early_stopping_rounds),
            "early_stopping_metric": early_stopping_metric,
            "validation_fraction": float(validation_fraction),
            "min_validation_days": int(min_validation_days),
            "validation_gap_days": int(validation_gap_days),
        }
        # The built-in metric is disabled whenever rank IC drives selection so
        # early stopping cannot silently fall back to L2, which is dominated by
        # unexplainable label noise and therefore stops after one tree.
        self.model = LGBMRegressor(
            metric="None" if early_stopping_metric == "rank_ic" else "l2",
            **tree_params,
        )
        fit_kwargs: dict[str, Any] = {}
        if not validation.empty:
            try:
                from lightgbm import early_stopping

                fit_kwargs = {
                    "eval_X": validation[self.feature_columns],
                    "eval_y": validation["label"],
                    "callbacks": [
                        early_stopping(
                            int(early_stopping_rounds),
                            first_metric_only=True,
                            verbose=False,
                        )
                    ],
                }
                if early_stopping_metric == "rank_ic":
                    fit_kwargs["eval_metric"] = [
                        self._rank_ic_metric(validation["trade_date"])
                    ]
            except ImportError:
                # Older LightGBM builds can still train deterministically;
                # the time-separated validation metrics remain available.
                fit_kwargs = {}
        self.model.fit(train[self.feature_columns], train["label"], **fit_kwargs)

        train_prediction = self.model.predict(train[self.feature_columns])
        metrics: dict[str, float] = {
            "train_rank_corr": self._correlation(
                train_prediction, train["label"], method="spearman"
            ),
            "train_rmse": self._rmse(train_prediction, train["label"]),
        }
        if not validation.empty:
            validation_prediction = self.model.predict(validation[self.feature_columns])
            validation_ic_values = self._daily_ic_values(
                validation, validation_prediction
            )
            metrics.update(
                {
                    "validation_rank_corr": self._correlation(
                        validation_prediction,
                        validation["label"],
                        method="spearman",
                    ),
                    "validation_rmse": self._rmse(
                        validation_prediction, validation["label"]
                    ),
                    "validation_ic_mean": (
                        float(sum(validation_ic_values) / len(validation_ic_values))
                        if validation_ic_values
                        else 0.0
                    ),
                    "validation_ic_positive_ratio": (
                        float(
                            sum(value > 0 for value in validation_ic_values)
                            / len(validation_ic_values)
                        )
                        if validation_ic_values
                        else 0.0
                    ),
                    "validation_ic_days": float(len(validation_ic_values)),
                    "validation_top_bottom_spread": self._top_bottom_spread(
                        validation, validation_prediction
                    ),
                    "validation_best_iteration": float(
                        getattr(self.model, "best_iteration_", 0) or 0
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "validation_rank_corr": 0.0,
                    "validation_rmse": 0.0,
                    "validation_ic_mean": 0.0,
                    "validation_ic_positive_ratio": 0.0,
                    "validation_ic_days": 0.0,
                    "validation_top_bottom_spread": 0.0,
                    "validation_best_iteration": 0.0,
                }
            )
        # Honest health gate.  A model that returns one score per trading date
        # carries no stock-selection information, and every headline number in
        # ``metrics`` can still look plausible.  Record it explicitly so a
        # degenerate artifact can never be promoted by accident.
        if not validation.empty:
            validation_unique = self._mean_unique_predictions_per_day(
                validation, validation_prediction
            )
            metrics["validation_unique_predictions_per_day"] = validation_unique
            if validation_unique <= self.MIN_UNIQUE_PREDICTIONS_PER_DAY:
                health = self.HEALTH_DEGENERATE
                health_detail = (
                    "验证集内每个交易日的预测值相同，模型没有横截面选股能力；"
                    "该 artifact 不得用于候选晋级"
                )
            else:
                health = self.HEALTH_OK
                health_detail = "验证集内存在横截面排名区分度"
        else:
            metrics["validation_unique_predictions_per_day"] = 0.0
            health = self.HEALTH_UNVALIDATED
            health_detail = "验证集为空，没有样本外证据"

        if "fundamental_coverage" in clean.columns:
            coverage = pd.to_numeric(clean["fundamental_coverage"], errors="coerce").mean()
            metrics["factor_coverage_mean"] = 0.0 if pd.isna(coverage) else float(coverage)

        raw_version = "|".join(
            [
                str(train["trade_date"].min()),
                str(train["trade_date"].max()),
                str(len(train)),
                str(validation["trade_date"].min() if not validation.empty else "none"),
                str(validation["trade_date"].max() if not validation.empty else "none"),
                ",".join(self.feature_columns),
                label_name,
                # Training configuration is part of the identity of a model.
                # Changing early stopping or the tree parameters must produce a
                # different version instead of silently reusing the old hash.
                json.dumps(training_params, sort_keys=True, ensure_ascii=True),
            ]
        )
        self.model_version = f"factor-lgbm-{hashlib.sha256(raw_version.encode()).hexdigest()[:10]}"
        output_dir = Path(artifact_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"{self.model_version}.pkl"
        training_start = str(pd.to_datetime(train["trade_date"]).min().date())
        training_end = str(pd.to_datetime(train["trade_date"]).max().date())
        validation_start = (
            str(pd.to_datetime(validation["trade_date"]).min().date())
            if not validation.empty
            else None
        )
        validation_end = (
            str(pd.to_datetime(validation["trade_date"]).max().date())
            if not validation.empty
            else None
        )
        self.artifact = ModelArtifact(
            model_version=self.model_version,
            model_name=self.MODEL_NAME,
            trained_at=datetime.now().astimezone(),
            training_start=training_start,
            training_end=training_end,
            feature_columns=self.feature_columns,
            label=label_name,
            metrics=metrics,
            artifact_path=str(artifact_path),
            validation_start=validation_start,
            validation_end=validation_end,
            training_rows=int(len(train)),
            validation_rows=int(len(validation)),
            validation_gap_days=validation_gap_days,
            health=health,
            health_detail=health_detail,
            training_params=training_params,
        )
        with artifact_path.open("wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "artifact": self.artifact,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return self.artifact

    @classmethod
    def load(
        cls,
        artifact_path: str | Path,
        expected_feature_columns: list[str] | None = None,
    ) -> LightGBMModel:
        path = Path(artifact_path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        if isinstance(payload, dict) and "model" in payload:
            estimator = payload["model"]
            artifact = payload.get("artifact")
        else:
            # Compatibility with the first prototype, which stored only the
            # estimator.  New artifacts always contain full metadata.
            estimator = payload
            artifact = None

        feature_columns = list(
            getattr(artifact, "feature_columns", None)
            or expected_feature_columns
            or FeatureEngine.FACTOR_COLUMNS
        )
        if expected_feature_columns and feature_columns != list(expected_feature_columns):
            raise ValueError("model artifact feature schema does not match the current strategy")
        model = cls(feature_columns=feature_columns)
        model.model = estimator
        if isinstance(artifact, ModelArtifact):
            model.artifact = artifact
            model.model_version = artifact.model_version
        else:
            model.model_version = path.stem
        return model

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not trained")
        result = frame[["symbol"]].copy()
        features = frame.reindex(columns=self.feature_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        # Warm-up rows can lack a long lookback.  Neutral imputation keeps the
        # score usable while the factor coverage column makes that condition
        # visible in reports and the UI.
        features = features.replace([float("inf"), float("-inf")], pd.NA).fillna(0.5)
        result["score"] = self.model.predict(features)
        result["reason"] = "LightGBM：按时间切分验证的未来超额收益预测"
        result["model_version"] = self.model_version
        return result

    @staticmethod
    def _correlation(prediction: Any, actual: Any, *, method: str) -> float:
        left = pd.Series(prediction).reset_index(drop=True)
        right = pd.Series(actual).reset_index(drop=True)
        valid = pd.concat([left, right], axis=1).dropna()
        if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
            return 0.0
        left = valid.iloc[:, 0]
        right = valid.iloc[:, 1]
        value = left.corr(right, method=method)
        return 0.0 if pd.isna(value) else float(value)

    @staticmethod
    def _rmse(prediction: Any, actual: Any) -> float:
        error = pd.Series(prediction).reset_index(drop=True) - pd.Series(actual).reset_index(
            drop=True
        )
        return float(error.pow(2).mean() ** 0.5) if len(error) else 0.0

    @classmethod
    def _daily_ic(cls, frame: pd.DataFrame, prediction: Any) -> float:
        values = cls._daily_ic_values(frame, prediction)
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def daily_rank_ic_mean(dates: Any, labels: Any, predictions: Any) -> float:
        """Mean daily cross-sectional Spearman IC.

        Ranking is vectorized instead of calling ``Series.corr`` once per day:
        the early-stopping metric is evaluated on every boosting iteration, and
        a per-day Python loop over a half-million-row validation window is far
        too slow to run a few hundred times.
        """

        frame = pd.DataFrame(
            {
                "_group": pd.factorize(pd.Series(dates))[0],
                "_label": pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy(
                    dtype=float
                ),
                "_prediction": pd.to_numeric(
                    pd.Series(predictions), errors="coerce"
                ).to_numpy(dtype=float),
            }
        ).dropna(subset=["_label", "_prediction"])
        if frame.empty or frame["_group"].nunique() < 2:
            return 0.0
        grouped = frame.groupby("_group", sort=False)
        rank_label = grouped["_label"].rank()
        rank_prediction = grouped["_prediction"].rank()
        centred_label = rank_label - rank_label.groupby(frame["_group"]).transform("mean")
        centred_prediction = rank_prediction - rank_prediction.groupby(
            frame["_group"]
        ).transform("mean")
        numerator = (centred_prediction * centred_label).groupby(frame["_group"]).sum()
        denominator = np.sqrt(
            centred_prediction.pow(2).groupby(frame["_group"]).sum()
            * centred_label.pow(2).groupby(frame["_group"]).sum()
        )
        values = (numerator / denominator.replace(0, np.nan)).dropna()
        return float(values.mean()) if len(values) else 0.0

    @classmethod
    def _rank_ic_metric(cls, dates: Any):
        """Build the LightGBM ``eval_metric`` callable used for early stopping."""

        groups = pd.Series(dates).to_numpy()

        def rank_ic(y_true: Any, y_pred: Any) -> tuple[str, float, bool]:
            value = cls.daily_rank_ic_mean(groups, y_true, y_pred)
            return ("rank_ic", value, True)

        return rank_ic

    @staticmethod
    def _mean_unique_predictions_per_day(frame: pd.DataFrame, prediction: Any) -> float:
        """Mean number of distinct scores a model emits per trading date.

        One means every stock is tied on the same date, so the downstream
        portfolio has no ranking information to act on.
        """

        values = pd.DataFrame(
            {
                "trade_date": pd.Series(frame["trade_date"]).to_numpy(),
                "prediction": np.asarray(prediction, dtype=float).ravel(),
            }
        )
        counts = values.groupby("trade_date")["prediction"].nunique()
        return float(counts.mean()) if len(counts) else 0.0

    @classmethod
    def _daily_ic_values(cls, frame: pd.DataFrame, prediction: Any) -> list[float]:
        values = frame[["trade_date", "label"]].copy()
        values["prediction"] = prediction
        correlations: list[float] = []
        for _, group in values.groupby("trade_date"):
            if len(group) < 2:
                continue
            correlation = cls._correlation(
                group["prediction"], group["label"], method="spearman"
            )
            if correlation != 0.0 or group["prediction"].nunique() > 1:
                correlations.append(correlation)
        return correlations

    @staticmethod
    def _bucket_label_stats(
        frame: pd.DataFrame, prediction: Any, top_n: int
    ) -> dict[str, float]:
        """Forward excess return of the bucket the portfolio actually trades.

        ``_top_bottom_spread`` measures the top *quintile* of the cross-section
        (about a thousand names on a real A-share universe), which is two orders
        of magnitude wider than the book: thirty names out of ~5000 is the top
        0.6%.  A signal can order the broad quintiles correctly and still be
        worthless in the tail the portfolio is restricted to, and that
        distinction decides whether a losing strategy needs a better model or a
        better broker.  So measure the traded bucket itself.

        The label is already benchmark-relative (stock return minus index
        return), so the bucket's raw mean *is* its excess return against the
        index; the universe mean is reported alongside as the "average stock"
        reference, not subtracted from it.
        """

        if frame.empty:
            return {
                "bucket_label_mean": 0.0,
                "universe_label_mean": 0.0,
                "bucket_label_positive_ratio": 0.0,
            }
        values = frame[["trade_date", "label"]].copy()
        values["prediction"] = prediction
        bucket_means: list[float] = []
        universe_means: list[float] = []
        for _, group in values.groupby("trade_date"):
            if group["label"].notna().sum() < 4:
                continue
            ordered = group.dropna(subset=["label"]).sort_values("prediction")
            bucket = ordered.tail(max(1, top_n))
            bucket_means.append(float(bucket["label"].mean()))
            universe_means.append(float(ordered["label"].mean()))
        if not bucket_means:
            return {
                "bucket_label_mean": 0.0,
                "universe_label_mean": 0.0,
                "bucket_label_positive_ratio": 0.0,
            }
        return {
            "bucket_label_mean": float(sum(bucket_means) / len(bucket_means)),
            "universe_label_mean": float(sum(universe_means) / len(universe_means)),
            "bucket_label_positive_ratio": float(
                sum(1 for value in bucket_means if value > 0) / len(bucket_means)
            ),
        }

    @staticmethod
    def _top_bottom_spread(frame: pd.DataFrame, prediction: Any) -> float:
        values = frame[["trade_date", "label"]].copy()
        values["prediction"] = prediction
        spreads: list[float] = []
        for _, group in values.groupby("trade_date"):
            if len(group) < 4:
                continue
            ordered = group.sort_values("prediction")
            bucket = max(1, len(ordered) // 5)
            spreads.append(
                float(
                    ordered["label"].tail(bucket).mean()
                    - ordered["label"].head(bucket).mean()
                )
            )
        return float(sum(spreads) / len(spreads)) if spreads else 0.0
