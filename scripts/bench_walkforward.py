"""Walk-forward 并行拟合基准 — `python scripts/bench_walkforward.py`

合成一个 400 标的 × 320 交易日的日线帧（列结构与 make_demo_provider 一致），
分别用 workers=1（顺序、每个 fit 用满核）与 workers=N（线程池、每个 fit 单线程）
跑 ``LightGBMWalkForwardBacktester.fit_folds``，输出前后耗时与加速比。

两条路径都必须产出相同的 fold 序列与逐 fold 指标（n_workers 固定为 1 的两个
配置在数值上应逐位一致；任何不一致都会让基准直接失败，防止"快了但错了"）。

用法：
    python scripts/bench_walkforward.py            # workers=1 vs 8
    python scripts/bench_walkforward.py 12         # workers=1 vs 12
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from quantpaper.data import _make_securities
from quantpaper.ml_backtest import LightGBMWalkForwardBacktester, WalkForwardConfig

SYMBOLS = 800
DAYS = 500
TRAIN_DAYS = 360
TEST_DAYS = 15
STEP_DAYS = 15


def synthetic_daily(seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    template = _make_securities()
    rows: list[dict[str, object]] = []
    prices = {i: round(rng.uniform(8, 80), 2) for i in range(SYMBOLS)}
    start = date(2024, 1, 2)
    trade_dates = [
        start + timedelta(days=offset)
        for offset in range(DAYS * 2)
        if (start + timedelta(days=offset)).weekday() < 5
    ][:DAYS]
    for current in trade_dates:
        for i in range(SYMBOLS):
            previous = prices[i]
            drift = 0.0003 + (i % 7 - 3) * 0.0001
            pct = drift + rng.gauss(0, 0.018)
            close = max(2.0, round(previous * (1 + pct), 2))
            high = round(max(previous, close) * (1 + abs(rng.gauss(0, 0.006))), 2)
            low = round(min(previous, close) * (1 - abs(rng.gauss(0, 0.006))), 2)
            amount = round(rng.uniform(8_000_000, 120_000_000), 2)
            rows.append(
                {
                    "symbol": f"B{i:04d}",
                    "trade_date": current,
                    "open": previous,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": int(amount / max(close, 1) * 100),
                    "amount": amount,
                    "turnover_rate": rng.uniform(0.3, 8.0),
                    "pe": rng.uniform(8, 65),
                    "pb": rng.uniform(0.8, 8),
                    "roe": rng.uniform(0.04, 0.28),
                    "profit_growth": rng.uniform(-0.15, 0.45),
                    "sector": template[i % len(template)].sector,
                }
            )
            prices[i] = close
    return pd.DataFrame(rows)


def config(workers: int, tmp: Path) -> WalkForwardConfig:
    return WalkForwardConfig(
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        step_days=STEP_DAYS,
        horizon_days=5,
        min_training_rows=500,
        min_validation_days=5,
        artifact_dir=tmp,
        workers=workers,
        # 顺序基线也钉在单线程拟合上：唯一的变量才是并发数，两次运行的
        # LightGBM 数值逐位一致，等值断言才有意义。
        threads_per_fit=1,
    )


def fingerprint(fitted) -> dict:
    return {
        fold.fold: (
            fold.train_start,
            fold.test_end,
            round(fold.metrics.get("test_ic_mean", 0.0), 12),
            round(fold.metrics.get("test_top_bottom_spread", 0.0), 12),
            fold.model_version,
        )
        for fold in fitted.folds
    }


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    frame = synthetic_daily()
    print(f"frame: {len(frame)} rows, {frame['symbol'].nunique()} symbols")
    backtester = LightGBMWalkForwardBacktester()
    import tempfile

    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = time.perf_counter()
        sequential = backtester.fit_folds(frame, config(1, tmp))
        sequential_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        parallel = backtester.fit_folds(frame, config(workers, tmp))
        parallel_seconds = time.perf_counter() - t0

    left, right = fingerprint(sequential), fingerprint(parallel)
    if left != right:
        for key in left:
            if left[key] != right.get(key):
                raise SystemExit(
                    f"MISMATCH at fold {key}: sequential={left[key]} parallel={right.get(key)}"
                )
        raise SystemExit("fold count mismatch")

    summary = {
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "folds": len(sequential.folds),
        "workers": workers,
        "sequential_seconds": round(sequential_seconds, 1),
        "parallel_seconds": round(parallel_seconds, 1),
        "speedup": round(sequential_seconds / parallel_seconds, 2),
        "scores_identical": bool(
            sequential.features["model_score"].equals(parallel.features["model_score"])
        ),
    }
    if not summary["scores_identical"]:
        raise SystemExit("model_score columns differ between runs")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
