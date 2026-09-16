from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import FeatureEngine
from .storage import DataQualityReport, HistoricalStore


@dataclass(frozen=True)
class FactorEvaluation:
    """Point-in-time evaluation summary for one factor."""

    factor: str
    name: str
    neutralization: str
    observations: int
    valid_days: int
    coverage: float
    mean_rank_ic: float
    positive_ic_ratio: float
    mean_top_bottom_spread: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorResearchResult:
    quality: DataQualityReport
    data_start: str | None
    data_end: str | None
    horizon_days: int
    neutralization: str
    exposure_coverage: dict[str, float]
    exposure_sources: dict[str, str]
    factors: list[FactorEvaluation]

    @property
    def status(self) -> str:
        if not self.quality.ok:
            return "DEGRADED"
        if self.neutralization == "sector" and self.exposure_coverage.get("sector", 0.0) <= 0:
            return "DEGRADED"
        if self.neutralization == "sector_size_beta":
            if self.exposure_coverage.get("sector", 0.0) <= 0:
                return "DEGRADED"
            if self.exposure_sources.get("size") == "size_score_proxy":
                return "DEGRADED"
        return "OK"

    @property
    def message(self) -> str:
        if not self.quality.ok:
            return "历史行情质量未通过，结果不能用于因子晋级"
        if self.neutralization == "sector" and self.exposure_coverage.get("sector", 0.0) <= 0:
            return "行业字段全部为 UNKNOWN，本次没有真正执行行业中性化"
        if self.neutralization == "sector_size_beta":
            if self.exposure_coverage.get("sector", 0.0) <= 0:
                return "行业字段全部为 UNKNOWN，本次没有真正执行行业中性化"
            if self.exposure_sources.get("size") == "size_score_proxy":
                return "缺少真实市值字段，市值暴露使用成交额代理，不能视为完整中性化"
        return "研究结果仅用于候选比较，不自动替换纸盘模型"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "quality": self.quality.as_dict(),
            "data_start": self.data_start,
            "data_end": self.data_end,
            "horizon_days": self.horizon_days,
            "neutralization": self.neutralization,
            "message": self.message,
            "exposure_coverage": self.exposure_coverage,
            "exposure_sources": self.exposure_sources,
            "factor_count": len(self.factors),
            "factors": [factor.as_dict() for factor in self.factors],
        }


class FactorResearchEvaluator:
    """Evaluate individual factors without changing the paper-trading model.

    Scores are calculated after the close of each date and compared with the
    subsequent excess-return label.  Optional cross-sectional residualization
    makes it possible to check whether a factor is only proxying for sector,
    size, or market beta exposure.
    """

    NEUTRALIZATION_MODES = ("none", "sector", "sector_size_beta")

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self.feature_engine = feature_engine or FeatureEngine()

    def evaluate(
        self,
        daily_frame: pd.DataFrame,
        *,
        factors: list[str] | None = None,
        horizon_days: int = 5,
        neutralization: str = "none",
    ) -> FactorResearchResult:
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if neutralization not in self.NEUTRALIZATION_MODES:
            raise ValueError(
                "neutralization must be one of: " + ", ".join(self.NEUTRALIZATION_MODES)
            )

        selected = list(factors or self.feature_engine.F1_FACTOR_COLUMNS)
        unknown = set(selected) - set(self.feature_engine.RESEARCH_FACTOR_COLUMNS)
        if unknown:
            raise ValueError(f"unknown research factors: {sorted(unknown)}")

        normalized = HistoricalStore.normalize_daily(daily_frame)
        quality = HistoricalStore.quality(normalized)
        if not quality.ok:
            raise ValueError(f"daily data quality failed: {quality.as_dict()}")
        if normalized.empty:
            raise ValueError("daily data is empty")

        features = self.feature_engine.build_features(normalized)
        features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce")
        grouped = features.groupby("symbol", group_keys=False)
        close = pd.to_numeric(features["close"], errors="coerce")
        future_return = grouped["close"].shift(-horizon_days) / close - 1
        benchmark_return = future_return.groupby(features["trade_date"]).transform("mean")
        features["_research_label"] = future_return - benchmark_return
        features["_research_sector"] = self._sector_series(features)
        size, size_source = self._size_series(features)
        beta, beta_source = self._beta_series(features)
        features["_research_size"] = size
        features["_research_beta"] = beta
        exposure_coverage = {
            "sector": float((features["_research_sector"] != "UNKNOWN").mean()),
            "size": float(features["_research_size"].notna().mean()),
            "beta": float(features["_research_beta"].notna().mean()),
        }
        exposure_sources = {
            "sector": "sector" if exposure_coverage["sector"] > 0 else "unknown",
            "size": size_source,
            "beta": beta_source,
        }

        evaluations = [
            self._evaluate_factor(features, factor, horizon_days, neutralization)
            for factor in selected
        ]
        return FactorResearchResult(
            quality=quality,
            data_start=quality.start_date,
            data_end=quality.end_date,
            horizon_days=horizon_days,
            neutralization=neutralization,
            exposure_coverage=exposure_coverage,
            exposure_sources=exposure_sources,
            factors=evaluations,
        )

    def _evaluate_factor(
        self,
        features: pd.DataFrame,
        factor: str,
        horizon_days: int,
        neutralization: str,
    ) -> FactorEvaluation:
        availability_column = f"{factor}_available"
        availability = pd.to_numeric(
            features.get(availability_column, features[factor].notna()), errors="coerce"
        ).fillna(0.0)
        labeled = features["_research_label"].notna()
        coverage = float(availability[labeled].mean()) if labeled.any() else 0.0
        columns = [
            "trade_date",
            factor,
            "_research_label",
            "_research_sector",
            "_research_size",
            "_research_beta",
        ]
        work = features[columns].copy()
        work["_available"] = availability
        work = work[
            work["_available"].gt(0)
            & work[factor].notna()
            & work["_research_label"].notna()
        ].reset_index(drop=True)
        work["_score"] = self._neutralize(work, factor, neutralization)
        work = work.dropna(subset=["_score", "_research_label"])

        rank_ics: list[float] = []
        spreads: list[float] = []
        for _, group in work.groupby("trade_date", sort=True):
            values = group[["_score", "_research_label"]].dropna()
            if len(values) < 4 or values["_score"].nunique() < 2:
                continue
            rank_ic = values["_score"].corr(values["_research_label"], method="spearman")
            if pd.notna(rank_ic):
                rank_ics.append(float(rank_ic))
            ordered = values.sort_values("_score")
            bucket = max(1, len(ordered) // 5)
            spreads.append(
                float(
                    ordered["_research_label"].tail(bucket).mean()
                    - ordered["_research_label"].head(bucket).mean()
                )
            )

        mean_ic = float(np.mean(rank_ics)) if rank_ics else 0.0
        positive_ratio = float(np.mean(np.asarray(rank_ics) > 0)) if rank_ics else 0.0
        mean_spread = float(np.mean(spreads)) if spreads else 0.0
        metadata = self.feature_engine.FACTOR_METADATA.get(factor, {})
        return FactorEvaluation(
            factor=factor,
            name=str(metadata.get("name", factor)),
            neutralization=neutralization,
            observations=int(len(work)),
            valid_days=len(rank_ics),
            coverage=coverage,
            mean_rank_ic=mean_ic,
            positive_ic_ratio=positive_ratio,
            mean_top_bottom_spread=mean_spread,
        )

    @staticmethod
    def _sector_series(frame: pd.DataFrame) -> pd.Series:
        if "sector" not in frame.columns:
            return pd.Series("UNKNOWN", index=frame.index, dtype="string")
        return frame["sector"].fillna("UNKNOWN").astype(str)

    @staticmethod
    def _size_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
        for column in ("market_cap", "market_value", "tot_mv", "a_mv_ex_ltd"):
            if column not in frame.columns:
                continue
            value = pd.to_numeric(frame[column], errors="coerce")
            value = value.where(value > 0)
            if value.notna().any():
                return np.log(value), column
        return pd.to_numeric(frame.get("size_score"), errors="coerce"), "size_score_proxy"

    @staticmethod
    def _beta_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
        if "_f1_beta_raw" in frame.columns:
            return pd.to_numeric(frame["_f1_beta_raw"], errors="coerce"), "rolling_beta_60"
        return pd.to_numeric(frame.get("beta_60"), errors="coerce"), "beta_factor_proxy"

    @classmethod
    def _neutralize(cls, frame: pd.DataFrame, factor: str, mode: str) -> pd.Series:
        values = pd.to_numeric(frame[factor], errors="coerce")
        if mode == "none":
            return values

        residual = pd.Series(np.nan, index=frame.index, dtype="float64")
        for _, positions in frame.groupby("trade_date", sort=False).groups.items():
            group = frame.loc[positions]
            target = pd.to_numeric(group[factor], errors="coerce")
            if mode == "sector":
                sector_mean = target.groupby(group["_research_sector"]).transform("mean")
                residual.loc[positions] = (target - sector_mean).to_numpy()
                continue

            design_parts: list[np.ndarray] = [np.ones((len(group), 1), dtype=float)]
            dummies = pd.get_dummies(group["_research_sector"], dtype=float)
            if dummies.shape[1] > 1:
                design_parts.append(dummies.iloc[:, 1:].to_numpy(dtype=float))
            for column in ("_research_size", "_research_beta"):
                covariate = pd.to_numeric(group[column], errors="coerce")
                if covariate.notna().sum() < 2 or covariate.nunique(dropna=True) < 2:
                    continue
                covariate = covariate.fillna(covariate.median())
                centered = covariate - covariate.mean()
                scale = float(centered.std(ddof=0))
                if scale > 0:
                    design_parts.append((centered / scale).to_numpy(dtype=float)[:, None])

            design = np.column_stack(design_parts)
            target_array = target.to_numpy(dtype=float)
            if len(target_array) <= design.shape[1]:
                fitted = np.full(len(target_array), target.mean())
            else:
                coefficients, *_ = np.linalg.lstsq(design, target_array, rcond=None)
                fitted = design @ coefficients
            residual.loc[positions] = target_array - fitted
        return residual


def save_factor_research_report(result: FactorResearchResult, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output
