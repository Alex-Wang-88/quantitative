from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
    """

    MODEL_NAME = "lightgbm"

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
    ) -> ModelArtifact:
        if "label" not in frame.columns:
            raise ValueError("training frame must contain label")
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
        split_index = len(dates) - validation_days

        if validation_days > 0:
            split_date = dates[split_index]
            train = clean[clean["_model_date"] < split_date].copy()
            validation = clean[clean["_model_date"] >= split_date].copy()
        else:
            train = clean.copy()
            validation = clean.iloc[0:0].copy()

        if len(train) < min_training_rows:
            # A small history should still be trainable if the caller has
            # explicitly allowed it.  Keep validation metadata honest and use
            # all rows rather than silently training on a tiny subset.
            train = clean.copy()
            validation = clean.iloc[0:0].copy()

        self.model = LGBMRegressor(
            n_estimators=160,
            learning_rate=0.04,
            num_leaves=24,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=7,
            verbosity=-1,
        )
        self.model.fit(train[self.feature_columns], train["label"])

        train_prediction = self.model.predict(train[self.feature_columns])
        metrics: dict[str, float] = {
            "train_rank_corr": self._correlation(
                train_prediction, train["label"], method="spearman"
            ),
            "train_rmse": self._rmse(train_prediction, train["label"]),
        }
        if not validation.empty:
            validation_prediction = self.model.predict(validation[self.feature_columns])
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
                    "validation_ic_mean": self._daily_ic(
                        validation, validation_prediction
                    ),
                    "validation_top_bottom_spread": self._top_bottom_spread(
                        validation, validation_prediction
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "validation_rank_corr": 0.0,
                    "validation_rmse": 0.0,
                    "validation_ic_mean": 0.0,
                    "validation_top_bottom_spread": 0.0,
                }
            )
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
            label="label",
            metrics=metrics,
            artifact_path=str(artifact_path),
            validation_start=validation_start,
            validation_end=validation_end,
            training_rows=int(len(train)),
            validation_rows=int(len(validation)),
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
        return float(sum(correlations) / len(correlations)) if correlations else 0.0

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
