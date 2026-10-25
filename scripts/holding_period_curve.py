"""Holding-period curve: how long is the model's ranking worth holding?

Every walk-forward report in this project measures returns at **one** horizon --
the label horizon (``horizon_days``, 5 trading days by default).  A report can
therefore say "the top-30 bucket earned +0.47% over the next five days", but it
cannot answer the question a per-name recommendation has to answer: *if I buy
this name today, for how many days is there evidence that holding it is better
than the benchmark?*

This script replays the models a walk-forward run already saved -- no retraining
-- and produces two curves:

*   **alpha decay** -- mean forward *excess* return of the top-N bucket as a
    function of holding period (1..60 trading days);
*   **rank decay** -- Spearman autocorrelation of the ranking itself against lag,
    i.e. how fast the ordering reshuffles.

The second curve is what makes the first interpretable.  If the ranking is
almost uncorrelated with itself after five days, then a 20-day holding period is
not "holding the same signal longer", it is "holding a stale ranking" -- and the
alpha-decay curve should be read as decay of *this* ranking, not of a durable
edge.

Both curves carry a significance gate, and it is not optional.  Overlapping
forward windows are heavily autocorrelated, so a t-statistic computed on every
trading day would be inflated by roughly ``sqrt(horizon)``.  Each horizon is
therefore tested on a **non-overlapping subsample spaced by the horizon**, and
the overlapping count is printed next to it so a reader can see how much
information was discarded to earn the interval.  A horizon whose interval spans
zero is reported as unmeasurable rather than as a small positive number --
the same discipline ``bucket_decay_attribution.py`` documents for
period-over-period bucket changes.

Usage::

    python scripts/holding_period_curve.py \\
        --report data/reports/benchmark-excess-walkforward-v3-smooth20.json

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantpaper.features import FeatureEngine
from quantpaper.ml_backtest import smooth_scores
from quantpaper.model import LightGBMModel
from quantpaper.storage import HistoricalStore

DEFAULT_HORIZONS = (1, 2, 3, 5, 10, 20, 40, 60)
DEFAULT_RANK_LAGS = (1, 2, 3, 5, 10, 20, 40)
MIN_PAIR_ROWS = 30


def _parse_list(text: str, cast, label: str) -> list[int]:
    values: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(cast(chunk))
        except ValueError as exc:
            raise SystemExit(f"{label} 含非法值: {chunk}") from exc
    if not values:
        raise SystemExit(f"{label} 至少需要一个值")
    return sorted(set(values))


def _significance(values: pd.Series, spacing: int) -> dict[str, Any]:
    """Mean of ``values`` plus a t-statistic earned on a non-overlapping subsample.

    ``values`` is indexed by rebalance date and each observation is a forward
    return over ``spacing`` trading days, so consecutive observations share
    ``spacing - 1`` days of their window.  Taking every ``spacing``-th point
    removes that overlap; the price is a smaller sample, which is disclosed.
    """

    clean = values.dropna()
    overlapping = int(clean.shape[0])
    if overlapping == 0:
        return {
            "observations_overlapping": 0,
            "observations_independent": 0,
            "mean_excess": None,
            "std_excess": None,
            "standard_error": None,
            "t_stat": None,
            "positive_ratio": None,
            "measurable": False,
        }
    step = max(1, int(spacing))
    sub = clean.iloc[::step]
    n = int(sub.shape[0])
    mean = float(sub.mean())
    if n < 2:
        return {
            "observations_overlapping": overlapping,
            "observations_independent": n,
            "mean_excess": mean,
            "std_excess": None,
            "standard_error": None,
            "t_stat": None,
            "positive_ratio": None,
            "measurable": False,
        }
    std = float(sub.std(ddof=1))
    se = std / math.sqrt(n)
    t_stat = mean / se if se > 0 else None
    return {
        "observations_overlapping": overlapping,
        "observations_independent": n,
        "mean_excess": mean,
        "std_excess": std,
        "standard_error": se,
        "t_stat": t_stat,
        "positive_ratio": float((sub > 0).mean()),
        "measurable": bool(t_stat is not None and abs(t_stat) >= 2.0),
    }


def forward_excess(
    features: pd.DataFrame,
    *,
    horizon_days: int,
    benchmark_frame: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    """Forward excess return at ``horizon_days``, mirrored off ``build_training_frame``.

    Kept in lockstep with the label definition on purpose: the 5-day point of
    this curve must reproduce the published ``test_bucket_label_mean``, and it
    only does so if the same ``_feature_group`` segmentation and the same
    benchmark mapping are used.
    """

    frame = features[["symbol", "trade_date", "close"]].copy()
    group_key = features.get("_feature_group", features["symbol"])
    frame["future_return"] = (
        frame.groupby(group_key, group_keys=False)["close"].shift(-horizon_days)
        / frame["close"]
        - 1.0
    )
    benchmark_returns = FeatureEngine._future_benchmark_returns(  # noqa: SLF001
        benchmark_frame, horizon_days=horizon_days, benchmark_symbol=benchmark_symbol
    )
    by_date = {
        timestamp.date(): float(value)
        for timestamp, value in benchmark_returns.items()
        if pd.notna(value)
    }
    frame["excess"] = frame["future_return"] - frame["trade_date"].map(by_date)
    return frame.loc[:, ["symbol", "trade_date", "excess"]]


def score_period(
    features: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Replay the saved fold models over their own out-of-sample test windows."""

    pieces: list[pd.DataFrame] = []
    for fold in folds:
        artifact = Path(str(fold["artifact_path"]))
        if not artifact.exists():
            raise SystemExit(f"折 {fold.get('fold')} 的模型文件不存在: {artifact}")
        start = pd.Timestamp(fold["test_start"]).date()
        end = pd.Timestamp(fold["test_end"]).date()
        window = features.loc[features["trade_date"].between(start, end)]
        if window.empty:
            continue
        window = window.reset_index(drop=True)
        model = LightGBMModel.load(artifact, expected_feature_columns=feature_columns)
        scored = model.score(window)
        piece = window.loc[:, ["symbol", "trade_date", "close"]].copy()
        piece["model_score"] = pd.to_numeric(scored["score"], errors="coerce").to_numpy()
        piece["fold"] = int(fold.get("fold") or 0)
        pieces.append(piece)
        del model, scored, window
    if not pieces:
        raise SystemExit("没有任何折被成功打分；检查 report 的 artifact_path")
    return pd.concat(pieces, ignore_index=True)


def fold_mean_aggregate(bucket: pd.Series, fold_by_date: dict[Any, int]) -> dict[str, Any]:
    """Mean over folds of the within-fold mean -- the published aggregation.

    The published ``test_bucket_label_mean`` averages *folds* (each fold
    contributes one number, whatever its length); the curve's headline is a
    pooled *per-date* mean.  The two agree only when every fold covers the same
    number of days, so both are reported and the 5-day raw row is required to
    reproduce the published figure.  Without this the curve would be
    unverifiable against the report it came from.
    """

    frame = bucket.dropna().rename("excess").to_frame()
    if frame.empty:
        return {"fold_count": 0, "fold_mean_excess": None}
    frame["fold"] = [fold_by_date.get(date) for date in frame.index]
    frame = frame.loc[frame["fold"].notna()]
    if frame.empty:
        return {"fold_count": 0, "fold_mean_excess": None}
    per_fold = frame.groupby("fold")["excess"].mean()
    return {
        "fold_count": int(per_fold.shape[0]),
        "fold_mean_excess": float(per_fold.mean()),
    }


def top_set_overlap(scored: pd.DataFrame, *, top_n: int, lags: list[int]) -> dict[str, Any]:
    """Jaccard overlap of the top-N set with itself ``lag`` days later.

    This is the counterpart to :func:`rank_decay` and the two must be read
    together.  Spearman autocorrelation is dominated by the ~5000 names in the
    middle of the cross-section; a book holds thirty.  If the global ordering is
    stable while the top thirty is not, the instability lives exactly where the
    recommendation would be taken from, and a per-name pick is then a draw from
    the least reliable part of the ranking.
    """

    ordered = scored.dropna(subset=["model_score"]).sort_values(
        ["trade_date", "model_score"], ascending=[True, False]
    )
    top = ordered.groupby("trade_date", sort=True).head(top_n)
    sets = top.groupby("trade_date", sort=True)["symbol"].apply(frozenset)
    dates = sets.index.to_list()
    out: dict[str, Any] = {}
    for lag in lags:
        ratios: list[float] = []
        if lag <= 0 or lag >= len(dates):
            out[str(lag)] = {"lag_days": lag, "observations": 0, "mean_jaccard": None}
            continue
        for index in range(len(dates) - lag):
            left = sets.iloc[index]
            right = sets.iloc[index + lag]
            union = left | right
            if union:
                ratios.append(len(left & right) / len(union))
        if not ratios:
            out[str(lag)] = {"lag_days": lag, "observations": 0, "mean_jaccard": None}
            continue
        series = pd.Series(ratios)
        out[str(lag)] = {
            "lag_days": lag,
            "observations": len(ratios),
            "mean_jaccard": float(series.mean()),
            "std_jaccard": float(series.std(ddof=1)) if len(series) > 1 else None,
        }
    return out


def per_fold_detail(
    bucket: pd.Series,
    fold_by_date: dict[Any, int],
    published_by_fold: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold-by-fold bucket mean, next to the figure the source report published.

    A single aggregate gap ("ours differs by 0.12 pp") cannot be diagnosed: it
    could be the aggregation order, the ranked universe, or one contaminated
    fold.  Printing the folds side by side localises it -- and in this project's
    source report it does localise, to a fold whose window had no labelled rows
    and which the old code recorded as ``0.0`` rather than as unmeasurable.
    """

    frame = bucket.dropna().rename("excess").to_frame()
    if frame.empty:
        return []
    frame["fold"] = [fold_by_date.get(date) for date in frame.index]
    frame = frame.loc[frame["fold"].notna()]
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for fold_id, group in frame.groupby("fold", sort=True):
        key = int(fold_id)
        published = published_by_fold.get(key) or {}
        rows.append(
            {
                "fold": key,
                "days": int(group.shape[0]),
                "per_day_mean_excess": float(group["excess"].mean()),
                "published_bucket": published.get("bucket"),
                "published_labeled_rows": published.get("labeled_rows"),
            }
        )
    return rows


def bucket_excess(
    scored: pd.DataFrame,
    excess: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.Series, pd.Series]:
    """Per-rebalance-date mean excess of the top-N names, and of the whole universe.

    Returning the universe alongside the bucket keeps the breadth/selection split
    available: a falling bucket mean accompanied by a falling universe mean is a
    narrower market, not a worse model.
    """

    merged = scored.merge(excess, on=["symbol", "trade_date"], how="inner")
    merged = merged.dropna(subset=["model_score", "excess"])
    if merged.empty:
        empty = pd.Series(dtype="float64")
        return empty, empty
    ordered = merged.sort_values(["trade_date", "model_score"], ascending=[True, False])
    bucket = ordered.groupby("trade_date", sort=True).head(top_n)
    return (
        bucket.groupby("trade_date", sort=True)["excess"].mean(),
        merged.groupby("trade_date", sort=True)["excess"].mean(),
    )


def rank_decay(scored: pd.DataFrame, lags: list[int]) -> dict[str, Any]:
    """Spearman autocorrelation of the daily ranking against lag.

    Implemented as Pearson on row-wise ranks, which is the same number as
    ``DataFrame.corr(method="spearman")`` but avoids ~5k SciPy calls.
    """

    pivot = scored.pivot_table(
        index="trade_date", columns="symbol", values="model_score", aggfunc="mean"
    ).sort_index()
    ranked = pivot.rank(axis=1).to_numpy(dtype="float64")
    dates = pivot.index.to_list()
    out: dict[str, Any] = {}
    for lag in lags:
        correlations: list[float] = []
        if lag <= 0 or lag >= len(dates):
            out[str(lag)] = {"lag_days": lag, "observations": 0, "mean_spearman": None}
            continue
        for index in range(len(dates) - lag):
            left = ranked[index]
            right = ranked[index + lag]
            mask = ~np.isnan(left) & ~np.isnan(right)
            if int(mask.sum()) < MIN_PAIR_ROWS:
                continue
            a = left[mask]
            b = right[mask]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
        if not correlations:
            out[str(lag)] = {"lag_days": lag, "observations": 0, "mean_spearman": None}
            continue
        series = pd.Series(correlations)
        out[str(lag)] = {
            "lag_days": lag,
            "observations": len(correlations),
            "mean_spearman": float(series.mean()),
            "std_spearman": float(series.std(ddof=1)) if len(series) > 1 else None,
            "positive_ratio": float((series > 0).mean()),
        }
    return out


def run(
    *,
    report_path: Path,
    horizons: list[int],
    rank_lags: list[int],
    top_n: int | None,
    smoothing_override: int | None,
    output: Path,
    markdown: Path | None,
    curve_csv: Path | None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    market_root = Path(str(report.get("market_root") or "data/eastmoney_adjusted"))
    benchmark_symbol = str(report.get("benchmark_symbol") or "000300")
    walkforward = report.get("walkforward") or {}
    feature_columns = list(walkforward.get("feature_columns") or [])
    if not feature_columns:
        raise SystemExit("report 缺少 walkforward.feature_columns，无法复现打分")
    smoothing = (
        int(smoothing_override)
        if smoothing_override is not None
        else int(walkforward.get("score_smoothing_days") or 1)
    )
    configured_top_n = int((report.get("backtest") or {}).get("max_positions") or 30)
    top_n = int(top_n or configured_top_n)
    label_horizon = int(walkforward.get("horizon_days") or 5)
    periods = report.get("periods") or []
    if not periods:
        raise SystemExit("report 没有 periods，无法定位折与测试窗口")

    end = max(pd.Timestamp(period["actual_end"]).date() for period in periods)
    print(f"[curve] 读取归档 {market_root}（截至 {end}）", flush=True)
    store = HistoricalStore(market_root)
    daily = store.read_daily(
        end=end,
        enrich_fundamentals=True,
        a_shares_only=True,
        require_adjusted=True,
        exclude_quality_events=True,
    )
    benchmarks = store.read_benchmarks(end=end, symbols=[benchmark_symbol])
    engine = FeatureEngine()
    print(f"[curve] 构建特征 {daily.shape[0]:,} 行", flush=True)
    features = engine.build_features(
        daily, benchmark_frame=benchmarks, benchmark_symbol=benchmark_symbol
    )
    features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.date
    sector_coverage = float(
        (features["sector"].astype(str) != "UNKNOWN").mean()
        if "sector" in features.columns
        else 0.0
    )
    del daily
    gc.collect()

    period_reports: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for period in periods:
        name = str((period.get("period") or {}).get("name") or "unknown")
        print(
            f"[curve] 回放 {name}: {period['actual_start']} -> {period['actual_end']}",
            flush=True,
        )
        scored = score_period(features, list(period.get("folds") or []), feature_columns)
        scored = scored.loc[scored["model_score"].notna()].reset_index(drop=True)
        # Fold index per date, so the published fold-mean aggregation can be
        # reproduced from the pooled per-date series.
        fold_by_date = {
            date: int(fold)
            for date, fold in scored.drop_duplicates("trade_date")[
                ["trade_date", "fold"]
            ].itertuples(index=False)
        }
        published_by_fold = {
            int(item["fold"]): {
                "bucket": (item.get("metrics") or {}).get("test_bucket_label_mean"),
                "labeled_rows": (item.get("metrics") or {}).get("test_labeled_rows"),
            }
            for item in (period.get("folds") or [])
        }
        # Smoothing runs over the period's evaluation window, matching
        # ``run_fitted_variant``; the raw score is kept so the two curves can be
        # read against each other.
        smoothed = scored.loc[:, ["symbol", "trade_date", "close"]].copy()
        smoothing_frame = scored.loc[:, ["symbol", "trade_date", "model_score"]].copy()
        smoothing_frame["factor_score"] = smoothing_frame["model_score"]
        smoothed["model_score"] = smooth_scores(smoothing_frame, smoothing)[
            "factor_score"
        ].to_numpy()
        del smoothing_frame

        # The forward excess for a horizon does not depend on which score is
        # being bucket-sorted, so it is computed once and handed to both
        # variants.  Looping variant-outer would rebuild the 4.5M-row forward
        # return grid twice per horizon for no information gain.
        variants: dict[str, dict[str, Any]] = {
            label: {
                "horizons": {},
                "rank_decay": rank_decay(frame, rank_lags),
                "top_overlap": top_set_overlap(frame, top_n=top_n, lags=rank_lags),
                "scored_rows": int(frame.shape[0]),
                "scored_dates": int(frame["trade_date"].nunique()),
            }
            for label, frame in (("raw", scored), ("smoothed", smoothed))
        }
        for horizon in horizons:
            excess = forward_excess(
                features,
                horizon_days=horizon,
                benchmark_frame=benchmarks,
                benchmark_symbol=benchmark_symbol,
            )
            for label, frame in (("raw", scored), ("smoothed", smoothed)):
                bucket, universe = bucket_excess(frame, excess, top_n=top_n)
                stats = _significance(bucket, horizon)
                stats.update(fold_mean_aggregate(bucket, fold_by_date))
                if int(horizon) == label_horizon:
                    stats["per_fold"] = per_fold_detail(
                        bucket, fold_by_date, published_by_fold
                    )
                clean_universe = universe.dropna()
                stats["universe_mean_excess"] = (
                    float(clean_universe.mean()) if not clean_universe.empty else None
                )
                stats["selection_mean_excess"] = (
                    None
                    if stats["mean_excess"] is None or stats["universe_mean_excess"] is None
                    else stats["mean_excess"] - stats["universe_mean_excess"]
                )
                variants[label]["horizons"][str(horizon)] = stats
                curve_rows.append(
                    {
                        "period": name,
                        "score": label,
                        "horizon_days": horizon,
                        "mean_excess": stats["mean_excess"],
                        "fold_mean_excess": stats.get("fold_mean_excess"),
                        "t_stat": stats["t_stat"],
                        "observations_independent": stats["observations_independent"],
                        "measurable": stats["measurable"],
                    }
                )
                del bucket, universe
            del excess
            gc.collect()
        del scored, smoothed
        gc.collect()
        period_reports.append(
            {
                "period": period.get("period"),
                "actual_start": period["actual_start"],
                "actual_end": period["actual_end"],
                "fold_count": len(period.get("folds") or []),
                "variants": variants,
                "published_test_bucket_label_mean": (period.get("stats") or {}).get(
                    "test_bucket_label_mean"
                ),
                "published_traded_bucket_label_mean": (period.get("stats") or {}).get(
                    "traded_bucket_label_mean"
                ),
            }
        )

    payload: dict[str, Any] = {
        "source_report": str(report_path.resolve()),
        "market_root": str(market_root.resolve()),
        "benchmark_symbol": benchmark_symbol,
        "label_mode": report.get("label_mode"),
        "feature_profile": walkforward.get("feature_profile"),
        "label_horizon_days": walkforward.get("horizon_days"),
        "score_smoothing_days": smoothing,
        "top_n": top_n,
        "horizons": horizons,
        "rank_lags": rank_lags,
        "sector_coverage": sector_coverage,
        "periods": period_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if curve_csv is not None:
        curve_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(curve_rows).to_csv(curve_csv, index=False)
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(render_markdown(payload), encoding="utf-8")
    return payload


def _pct(value: Any, digits: int = 3) -> str:
    if value is None:
        return "未测量"
    return f"{float(value):+.{digits}%}"


def _t(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 持有期曲线：排名能维持多久\n")
    lines.append(
        "- 来源运行 `{report}`（模型与折原封复用，**未重训**）".format(
            report=Path(payload["source_report"]).name
        )
    )
    lines.append(
        "- 标签周期 **{h} 日** · 打分平滑 **{s} 日** · 桶大小 **top-{n}** · "
        "基准 `{b}` · 特征档 `{p}`".format(
            h=payload["label_horizon_days"],
            s=payload["score_smoothing_days"],
            n=payload["top_n"],
            b=payload["benchmark_symbol"],
            p=payload["feature_profile"],
        )
    )
    lines.append(
        "- 前向超额 = 个股 h 日收益 − 基准 h 日收益（与训练标签同式）；"
        "`selection` = 桶均超额 − 全市场均超额。"
    )
    lines.append(
        "- **显著性口径**：每个 horizon 在**每 h 个交易日取一个**的非重叠子样本上检验，"
        "t 统计量才不被重叠窗口放大；重叠观测数并列以显示代价。"
        "**|t| < 2 的 horizon 一律标注「不可测」，不当作正数。**"
    )
    lines.append("")

    for period in payload["periods"]:
        lines.append(
            "## {name}（{start} → {end}，{folds} 折）\n".format(
                name=(period["period"] or {}).get("name"),
                start=period["actual_start"],
                end=period["actual_end"],
                folds=period["fold_count"],
            )
        )
        for label, title in (("raw", "原始打分"), ("smoothed", "平滑打分")):
            block = period["variants"].get(label)
            if block is None:
                continue
            lines.append(f"### {title}\n")
            lines.append(
                "| 持有期 | 桶均超额 | 全市场均超额 | selection | 独立样本 | 重叠样本 | t | 判定 |"
            )
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
            for horizon in payload["horizons"]:
                stats = block["horizons"].get(str(horizon))
                if stats is None:
                    continue
                verdict = "可测" if stats["measurable"] else "**不可测**"
                lines.append(
                    "| {h} 日 | {b} | {u} | {s} | {n} | {o} | {t} | {v} |".format(
                        h=horizon,
                        b=_pct(stats["mean_excess"]),
                        u=_pct(stats["universe_mean_excess"]),
                        s=_pct(stats["selection_mean_excess"]),
                        n=stats["observations_independent"],
                        o=stats["observations_overlapping"],
                        t=_t(stats["t_stat"]),
                        v=verdict,
                    )
                )
            lines.append("")
            lines.append("**排名自相关（Spearman）**\n")
            lines.append("| 滞后 | 平均相关 | 为正占比 | 样本 |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for lag in payload["rank_lags"]:
                item = block["rank_decay"].get(str(lag)) or {}
                mean = item.get("mean_spearman")
                lines.append(
                    "| {lag} 日 | {m} | {p} | {n} |".format(
                        lag=lag,
                        m="未测量" if mean is None else f"{float(mean):.3f}",
                        p=(
                            "未测量"
                            if item.get("positive_ratio") is None
                            else f"{float(item['positive_ratio']):.2f}"
                        ),
                        n=item.get("observations", 0),
                    )
                )
            lines.append("")
            lines.append(
                f"**top-{payload['top_n']} 尾部重叠（Jaccard）——与上一张表对照读**\n"
            )
            lines.append("| 滞后 | 全局排名自相关 | 尾部重叠 | 样本 |")
            lines.append("| ---: | ---: | ---: | ---: |")
            for lag in payload["rank_lags"]:
                rank_item = block["rank_decay"].get(str(lag)) or {}
                overlap_item = (block.get("top_overlap") or {}).get(str(lag)) or {}
                rank_mean = rank_item.get("mean_spearman")
                overlap_mean = overlap_item.get("mean_jaccard")
                lines.append(
                    "| {lag} 日 | {r} | {o} | {n} |".format(
                        lag=lag,
                        r="未测量" if rank_mean is None else f"{float(rank_mean):.3f}",
                        o=(
                            "未测量"
                            if overlap_mean is None
                            else f"{float(overlap_mean):.3f}"
                        ),
                        n=overlap_item.get("observations", 0),
                    )
                )
            lines.append("")
            lines.append(
                "> 全局自相关由横截面中段约 5000 只股票主导，而组合只持有 "
                f"{payload['top_n']} 只。**两张表拉开差距，说明不稳定集中在尾部** —— "
                "而个股推荐恰恰取自尾部。\n"
            )
        lines.append("")
        lines.append("**复现校验（5 日 · 原始打分 · 折均口径）**\n")
        raw_block = period["variants"].get("raw") or {"horizons": {}}
        raw_five = raw_block["horizons"].get("5") or raw_block["horizons"].get(
            str(payload.get("label_horizon_days"))
        ) or {}
        published = period.get("published_test_bucket_label_mean")
        reproduced = raw_five.get("fold_mean_excess")
        if published is None or reproduced is None:
            verdict = "**无法校验**（缺已发布值或折均值为空）"
        else:
            gap = float(reproduced) - float(published)
            verdict = (
                f"差 {gap:+.5%} ⇒ **{'一致' if abs(gap) < 0.001 else '不一致，需查口径'}**"
            )
        lines.append(
            f"| 口径 | 值 |\n| :--- | ---: |\n"
            f"| 已发布 `test_bucket_label_mean` | {_pct(published, digits=5)} |\n"
            f"| 本脚本折均复现 | {_pct(reproduced, digits=5)} |\n"
            f"| 判定 | {verdict} |\n"
        )
        lines.append(
            f"> `traded_bucket_label_mean`（已发布）= "
            f"{_pct(period.get('published_traded_bucket_label_mean'), digits=5)}。"
            "该字段描述**组合真正下单所用**的打分，本脚本的「平滑打分」段即对应它；"
            "两个口径都保留，避免把「选股变强」和「拿得更住」混为一谈。\n"
        )
        per_fold = raw_five.get("per_fold") or []
        if per_fold:
            lines.append("**逐折对照（原始打分 · 5 日）**\n")
            lines.append(
                "| 折 | 本脚本天数 | 本脚本折均 | 已发布折均 | 已发布 `test_labeled_rows` | 判定 |"
            )
            lines.append("| ---: | ---: | ---: | ---: | ---: | :--- |")
            for row in per_fold:
                published_fold = row.get("published_bucket")
                empty = row.get("published_labeled_rows") in (0, 0.0)
                if empty:
                    verdict = "**空折记为 0.0**"
                elif published_fold is None:
                    verdict = "无已发布值"
                else:
                    gap = float(row["per_day_mean_excess"]) - float(published_fold)
                    verdict = "一致" if abs(gap) < 0.002 else f"差 {gap:+.3%}"
                lines.append(
                    "| {f} | {d} | {o} | {p} | {r} | {v} |".format(
                        f=row["fold"],
                        d=row["days"],
                        o=_pct(row["per_day_mean_excess"], digits=4),
                        p=_pct(published_fold, digits=4),
                        r=(
                            "—"
                            if row.get("published_labeled_rows") is None
                            else f"{float(row['published_labeled_rows']):,.0f}"
                        ),
                        v=verdict,
                    )
                )
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="持有期曲线：复用已保存的折模型，测量排名的 alpha 衰减与自相关衰减"
    )
    parser.add_argument(
        "--report",
        default="data/reports/benchmark-excess-walkforward-v3-smooth20.json",
    )
    parser.add_argument(
        "--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS)
    )
    parser.add_argument(
        "--rank-lags", default=",".join(str(lag) for lag in DEFAULT_RANK_LAGS)
    )
    parser.add_argument(
        "--top-n", type=int, default=None, help="桶大小；省略时取报告的 max_positions"
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=None,
        help="覆盖报告的平滑窗口；1 表示不平滑",
    )
    parser.add_argument("--output", default="data/reports/holding-period-curve.json")
    parser.add_argument("--markdown", default="data/reports/holding-period-curve.md")
    parser.add_argument("--curve-csv", default="data/reports/holding-period-curve.csv")
    args = parser.parse_args()

    horizons = _parse_list(args.horizons, int, "--horizons")
    if any(value <= 0 for value in horizons):
        raise SystemExit("持有期必须是正数")
    rank_lags = _parse_list(args.rank_lags, int, "--rank-lags")
    if any(value <= 0 for value in rank_lags):
        raise SystemExit("滞后必须是正数")
    if args.top_n is not None and args.top_n <= 0:
        raise SystemExit("--top-n 必须是正数")

    payload = run(
        report_path=Path(args.report),
        horizons=horizons,
        rank_lags=rank_lags,
        top_n=args.top_n,
        smoothing_override=args.smoothing,
        output=Path(args.output),
        markdown=Path(args.markdown) if args.markdown else None,
        curve_csv=Path(args.curve_csv) if args.curve_csv else None,
    )
    print(f"report: {Path(args.output).resolve()}")
    if args.markdown:
        print(f"markdown: {Path(args.markdown).resolve()}")
    if args.curve_csv:
        print(f"csv: {Path(args.curve_csv).resolve()}")
    for period in payload["periods"]:
        raw = period["variants"]["raw"]["horizons"]
        summary = {}
        for horizon in payload["horizons"]:
            value = raw[str(horizon)]["mean_excess"]
            summary[horizon] = None if value is None else round(value, 5)
        print(
            json.dumps(
                {
                    "period": (period["period"] or {}).get("name"),
                    "raw_bucket_excess_by_horizon": summary,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
