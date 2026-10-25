"""Train a local LightGBM candidate from the adjusted Eastmoney archive.

The artifact is written to a research directory by default.  It is not
promoted to the paper runner automatically; promotion remains a separate,
auditable decision after out-of-sample checks.
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


def train_from_archive(
    *,
    market_root: Path,
    train_end: str,
    artifact_dir: Path,
    output: Path,
) -> dict[str, object]:
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

    engine = FeatureEngine()
    training = engine.build_training_frame(
        daily,
        horizon_days=config.model.prediction_horizon_days,
    )
    training_rows = len(training)
    del daily
    gc.collect()

    model = LightGBMModel(feature_columns=list(engine.MODEL_FEATURE_COLUMNS))
    artifact = model.fit(
        training,
        artifact_dir=artifact_dir,
        min_training_rows=config.model.min_training_rows,
        validation_fraction=config.model.validation_fraction,
        min_validation_days=config.model.min_validation_days,
        validation_gap_days=config.model.validation_gap_days,
    )
    result: dict[str, object] = {
        "status": "RESEARCH_CANDIDATE",
        "source": str(market_root.resolve()),
        "train_end_requested": train_end,
        "data_start": quality.start_date,
        "data_end": quality.end_date,
        "daily_rows": quality.rows,
        "training_rows_before_model_dropna": training_rows,
        "quality": quality.as_dict(),
        "quality_policy": "exclude_quality_event_windows",
        "artifact": asdict(artifact),
        "promotion": "NOT_PROMOTED",
        "note": "研究候选模型，不自动替换纸盘当前模型。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="从东方财富历史归档训练本地 LightGBM 候选")
    parser.add_argument("--market-root", default="data/eastmoney_adjusted")
    parser.add_argument("--train-end", default="2026-09-16")
    parser.add_argument("--artifact-dir", default="data/models/research")
    parser.add_argument("--output", default="data/reports/full-universe-adjusted-model.json")
    args = parser.parse_args()
    result = train_from_archive(
        market_root=Path(args.market_root),
        train_end=args.train_end,
        artifact_dir=Path(args.artifact_dir),
        output=Path(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"report: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
