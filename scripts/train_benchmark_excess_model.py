"""Train a research-only LightGBM candidate and record its honest health.

This script deliberately writes to ``data/models/research`` and never changes
the artifact used by the paper runner.  It can label each stock by its future
excess return over a real index (``benchmark_excess``) or by its future
cross-sectional excess return (``cross_sectional``).

The two labels are not interchangeable.  Subtracting an index return leaves a
date-level common component behind -- the market's breadth versus the index --
which a tree model can fit without learning anything about ranking stocks, and
a market-regime feature set makes that failure easy.  A cross-sectional label
removes the common component by construction, so the model can only improve by
ranking stocks against each other.  ``--label-mode`` therefore defaults to the
current production-style behaviour but the cross-sectional mode is what the
candidate-v3 profile is designed for.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

from quantpaper.config import load_config
from quantpaper.features import FeatureEngine
from quantpaper.model import LightGBMModel
from quantpaper.storage import HistoricalStore

LABEL_MODES = ("benchmark_excess", "cross_sectional")


def train_candidate(
    *,
    market_root: Path,
    train_end: str,
    benchmark_symbol: str,
    horizon_days: int,
    artifact_dir: Path,
    output: Path,
    feature_profile: str = "candidate-v2",
    label_mode: str = "benchmark_excess",
    early_stopping_metric: str = "rank_ic",
) -> dict[str, object]:
    if label_mode not in LABEL_MODES:
        raise ValueError("label_mode must be one of: " + ", ".join(LABEL_MODES))
    config = load_config()
    store = HistoricalStore(market_root)
    daily = store.read_daily(
        end=train_end,
        enrich_fundamentals=True,
        a_shares_only=True,
        require_adjusted=True,
        exclude_quality_events=True,
    )
    quality = HistoricalStore.quality(daily)
    if not quality.ok:
        raise ValueError(f"daily data quality failed: {quality.as_dict()}")
    benchmarks = store.read_benchmarks(end=train_end, symbols=[benchmark_symbol])
    if benchmarks.empty:
        raise ValueError(f"benchmark archive has no rows for {benchmark_symbol}")

    engine = FeatureEngine()
    feature_columns = engine.feature_profile(feature_profile)
    training = engine.build_training_frame(
        daily,
        horizon_days=horizon_days,
        benchmark_frame=benchmarks,
        benchmark_symbol=benchmark_symbol,
        label_mode=label_mode,
    )
    training_rows = len(training)
    label_summary = {
        "mean": float(training["label"].mean()),
        "std": float(training["label"].std()),
        "positive_ratio": float((training["label"] > 0).mean()),
    }
    del daily, benchmarks
    gc.collect()

    model = LightGBMModel(feature_columns=feature_columns)
    artifact = model.fit(
        training,
        artifact_dir=artifact_dir,
        min_training_rows=config.model.min_training_rows,
        validation_fraction=config.model.validation_fraction,
        min_validation_days=config.model.min_validation_days,
        validation_gap_days=config.model.validation_gap_days,
        label_name=f"{label_mode}_return_{horizon_days}d",
        early_stopping_metric=early_stopping_metric,
    )
    # A degenerate artifact must never look promotable, whatever its headline
    # IC says.  The walk-forward gate still decides, but the status is honest.
    promoted_status = (
        "BLOCKED_DEGENERATE"
        if artifact.health == LightGBMModel.HEALTH_DEGENERATE
        else "RESEARCH_CANDIDATE"
    )
    result: dict[str, object] = {
        "status": promoted_status,
        "candidate_profile": f"{feature_profile}-{label_mode}",
        "source": str(market_root.resolve()),
        "benchmark_symbol": benchmark_symbol,
        "label_mode": label_mode,
        "label_definition": (
            f"future_{horizon_days}d_stock_return - "
            f"future_{horizon_days}d_{benchmark_symbol}_return"
            if label_mode == "benchmark_excess"
            else f"future_{horizon_days}d_stock_return - "
            f"cross_sectional_mean_future_{horizon_days}d_return"
        ),
        "train_end_requested": train_end,
        "data_start": quality.start_date,
        "data_end": quality.end_date,
        "daily_rows": quality.rows,
        "training_rows_before_model_dropna": training_rows,
        "label_summary": label_summary,
        "feature_profile": feature_profile,
        "feature_columns": feature_columns,
        "market_regime_features_in_model": [
            column
            for column in engine.BENCHMARK_FEATURE_COLUMNS
            if column in feature_columns
        ],
        "early_stopping_metric": early_stopping_metric,
        "quality": quality.as_dict(),
        "quality_policy": "exclude_quality_event_windows",
        "artifact": asdict(artifact),
        "promotion": "NOT_PROMOTED",
        "model_health": artifact.health,
        "model_health_detail": artifact.health_detail,
        "note": "候选模型不自动替换纸盘当前模型，需通过分段 walk-forward 门禁。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="训练本地 LightGBM 研究候选")
    parser.add_argument("--market-root", default="data/eastmoney_adjusted")
    parser.add_argument("--train-end", default="2026-09-16")
    parser.add_argument("--benchmark-symbol", default="000300")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--artifact-dir", default="data/models/research")
    parser.add_argument(
        "--feature-profile",
        choices=("candidate-v2", "candidate-v3"),
        default="candidate-v2",
        help="candidate-v2 含市场状态特征（旧行为）；candidate-v3 只保留横截面因子",
    )
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODES,
        default="benchmark_excess",
        help="benchmark_excess 为旧行为；cross_sectional 更适合纯选股模型",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=LightGBMModel.EARLY_STOPPING_METRICS,
        default="rank_ic",
    )
    parser.add_argument(
        "--output", default="data/reports/benchmark-excess-model.json"
    )
    args = parser.parse_args()
    if args.horizon_days <= 0:
        raise SystemExit("--horizon-days must be positive")
    result = train_candidate(
        market_root=Path(args.market_root),
        train_end=args.train_end,
        benchmark_symbol=args.benchmark_symbol,
        horizon_days=args.horizon_days,
        artifact_dir=Path(args.artifact_dir),
        output=Path(args.output),
        feature_profile=args.feature_profile,
        label_mode=args.label_mode,
        early_stopping_metric=args.early_stopping_metric,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"report: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
