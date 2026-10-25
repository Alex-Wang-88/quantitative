"""Attribute a period-over-period change in the traded-bucket excess.

A walk-forward report answers "did the book's excess fall?" but not **why**.
The traded bucket mean is itself a decomposition, and reading it alone hides
which half moved::

    bucket_mean  =  universe_mean  +  (bucket_mean - universe_mean)
                     ^ breadth          ^ selection

``universe_mean`` is the average stock's excess over the benchmark.  When an
index rally is carried by a few heavyweights the average stock underperforms,
so this term falls even though nothing about the model changed -- a narrower
cross-section.  The residual is what the ranking actually bought; if *that*
falls, the model got worse.

The second half of the script is a significance gate, and it is not optional.
Bucket means are fold averages of an extremely noisy quantity: with ~10 folds
per year and a fold-to-fold standard deviation of 1-3 percentage points, a
half-point change in the mean is well inside one standard error.  Printing the
point estimates without the interval invites reading noise as decay, so every
period carries its t statistic and a pairwise Welch test decides whether a
"drop" is separable from fold noise at all.

Kept separate from the report loader in ``ml_backtest`` on purpose: this reads
the JSON that a run already wrote, so it can be pointed at any historical
report without retraining.

Usage::

    python scripts/bucket_decay_attribution.py \\
        --report data/reports/benchmark-excess-walkforward-v3-position.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Critical values are taken from scipy when it is importable and from this small
# table otherwise, so the script still runs in a bare environment.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048, 30: 2.042, 40: 2.021, 60: 2.000,
}


def _t_critical(df: int, confidence: float = 0.95) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    try:
        from scipy import stats as _stats

        return float(_stats.t.ppf(0.5 + confidence / 2.0, df))
    except Exception:
        keys = sorted(_T95)
        return _T95[min(keys, key=lambda k: abs(k - df))]

# Per-fold metric keys.  ``test_bucket_label_mean`` is measured on the RAW model
# ranking (not the traded score), so this script describes the signal side; the
# report's ``stats.traded_bucket_label_mean`` covers the execution side.
FOLD_KEYS = (
    "test_ic_mean",
    "test_bucket_label_mean",
    "test_universe_label_mean",
    "test_bucket_label_positive_ratio",
    "test_bucket_size",
    "test_labeled_rows",
    "test_top_bottom_spread",
)


def _f(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, kind: str = "pct", digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:+.{digits}%}"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:+.{digits}f}"


def collect(report: dict) -> list[dict]:
    rows: list[dict] = []
    for period in report.get("periods", []):
        name = str(period.get("period", {}).get("name", "?"))
        for fold in period.get("folds", []):
            metrics = fold.get("metrics", {}) or {}
            bucket = _f(metrics, "test_bucket_label_mean")
            universe = _f(metrics, "test_universe_label_mean")
            labeled_rows = _f(metrics, "test_labeled_rows")
            # Defence in depth.  A fold whose test window sits inside the
            # label's forward-blind tail has no labels, so its bucket is
            # *unmeasured*, not zero.  Current code omits the field entirely;
            # reports written before that fix carry an explicit ``0.0``, and
            # treating it as a reading would drag every mean toward zero.  The
            # row count is the ground truth either way.
            reason = None
            if labeled_rows is not None and labeled_rows <= 0:
                if bucket is not None or universe is not None:
                    reason = "空折：报告写了 0.0，但该折无标签样本，不可作为读数"
                bucket = universe = None
            rows.append(
                {
                    "period": name,
                    "fold": int(fold.get("fold", 0)),
                    "test_start": str(fold.get("test_start", "")),
                    "test_end": str(fold.get("test_end", "")),
                    "metrics": {key: _f(metrics, key) for key in FOLD_KEYS},
                    "bucket": bucket,
                    "universe": universe,
                    "unmeasured_reason": reason,
                    # The two components of the bucket reading, split out.  None
                    # propagates: a fold with no labeled rows must not be folded
                    # into an average as a zero.
                    "selection": (
                        None if (bucket is None or universe is None) else bucket - universe
                    ),
                }
            )
    return rows


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return mean, sd


def _significance(values: list[float]) -> dict:
    """One-sample t test against zero, plus the 95% interval.

    The question a bucket mean has to survive is not "is it positive?" but "is
    it far enough from zero, given how much it bounces fold to fold?".  With ten
    folds and a standard deviation several times the mean, it usually is not.
    """

    mean, sd = _mean_sd(values)
    if mean is None or sd is None or sd == 0:
        return {"mean": mean, "sd": sd, "se": None, "t": None, "ci_low": None, "ci_high": None}
    n = len(values)
    se = sd / math.sqrt(n)
    t = mean / se
    crit = _t_critical(n - 1)
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "n": n,
        "positive_folds": sum(1 for value in values if value > 0),
    }


def welch(left: list[float], right: list[float]) -> dict | None:
    """Welch t test for a difference in means between two sets of folds.

    Welch rather than Student because the fold counts and variances differ, and
    because the whole point is that these are noisy small samples: assuming
    equal variance would overstate confidence.
    """

    if len(left) < 2 or len(right) < 2:
        return None
    mean_left, sd_left = _mean_sd(left)
    mean_right, sd_right = _mean_sd(right)
    if sd_left is None or sd_right is None or mean_left is None or mean_right is None:
        return None
    var_left = sd_left**2 / len(left)
    var_right = sd_right**2 / len(right)
    if var_left + var_right == 0:
        return None
    diff = mean_left - mean_right
    se = math.sqrt(var_left + var_right)
    t = diff / se
    df = (var_left + var_right) ** 2 / (
        var_left**2 / (len(left) - 1) + var_right**2 / (len(right) - 1)
    )
    crit = _t_critical(int(df))
    return {
        "diff": diff,
        "se": se,
        "t": t,
        "df": df,
        "crit": crit,
        "significant": abs(t) > crit,
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
    }


def summarise(rows: list[dict]) -> list[dict]:
    """Per-period means, unweighted and row-weighted.

    Both are reported because they answer different questions: the unweighted
    mean treats each fold as one observation (what a rebalance sees), the
    row-weighted mean treats each stock-day as one (what the label sample says).
    """

    periods: list[str] = []
    for row in rows:
        if row["period"] not in periods:
            periods.append(row["period"])

    out: list[dict] = []
    for name in periods:
        group = [row for row in rows if row["period"] == name]
        buckets = [row["bucket"] for row in group if row["bucket"] is not None]
        universes = [row["universe"] for row in group if row["universe"] is not None]
        selections = [row["selection"] for row in group if row["selection"] is not None]
        ics = [
            row["metrics"]["test_ic_mean"]
            for row in group
            if row["metrics"]["test_ic_mean"] is not None
        ]

        weighted_bucket = weighted_universe = None
        pairs = [
            (row["bucket"], row["universe"], row["metrics"]["test_labeled_rows"])
            for row in group
            if row["bucket"] is not None
            and row["universe"] is not None
            and row["metrics"]["test_labeled_rows"]
        ]
        if pairs:
            denom = sum(rows_ for _, _, rows_ in pairs)
            weighted_bucket = sum(b * rows_ for b, _, rows_ in pairs) / denom
            weighted_universe = sum(u * rows_ for _, u, rows_ in pairs) / denom

        out.append(
            {
                "period": name,
                "folds": len(group),
                "bucket_measured_folds": len(buckets),
                "unmeasured_folds": [
                    {"fold": row["fold"], "reason": row["unmeasured_reason"]}
                    for row in group
                    if row["unmeasured_reason"]
                ],
                "bucket_mean": sum(buckets) / len(buckets) if buckets else None,
                "universe_mean": sum(universes) / len(universes) if universes else None,
                "selection_mean": sum(selections) / len(selections) if selections else None,
                "ic_mean": sum(ics) / len(ics) if ics else None,
                "bucket_stats": _significance(buckets),
                "selection_stats": _significance(selections),
                "weighted_bucket_mean": weighted_bucket,
                "weighted_universe_mean": weighted_universe,
                "weighted_selection_mean": (
                    None
                    if (weighted_bucket is None or weighted_universe is None)
                    else weighted_bucket - weighted_universe
                ),
                # Kept so the pairwise tests can run on the same fold values.
                "_buckets": buckets,
                "_selections": selections,
            }
        )
    return out


def pairwise(summary: list[dict]) -> list[dict]:
    """Welch tests for each consecutive period pair, on bucket and selection."""

    out: list[dict] = []
    for left, right in zip(summary, summary[1:]):
        for label, key in (("bucket", "_buckets"), ("selection", "_selections")):
            test = welch(right[key], left[key])
            out.append(
                {
                    "comparison": f"{right['period']} - {left['period']}",
                    "component": label,
                    "shift": None if test is None else test["diff"],
                    "se": None if test is None else test["se"],
                    "t": None if test is None else test["t"],
                    "significant": None if test is None else test["significant"],
                    "ci_low": None if test is None else test["ci_low"],
                    "ci_high": None if test is None else test["ci_high"],
                }
            )
    return out


def render(rows: list[dict], summary: list[dict]) -> str:
    lines: list[str] = []
    lines.append(
        f"{'period':<10} {'fold':>4} {'test_start':<11} {'test_end':<11} {'rows':>9} "
        f"{'IC':>9} {'bucket':>10} {'univ':>10} {'sel':>10} {'pos':>5} {'n':>4}"
    )
    lines.append("-" * 108)
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"{row['period']:<10} {row['fold']:>4} {row['test_start']:<11} {row['test_end']:<11} "
            f"{_fmt(m['test_labeled_rows'], 'int'):>9} "
            f"{_fmt(m['test_ic_mean'], 'f'):>9} "
            f"{_fmt(row['bucket']):>10} "
            f"{_fmt(row['universe']):>10} "
            f"{_fmt(row['selection']):>10} "
            f"{_fmt(m['test_bucket_label_positive_ratio'], 'f', 2):>5} "
            f"{_fmt(m['test_bucket_size'], 'int'):>4}"
        )
    lines.append("")
    lines.append("=" * 108)
    lines.append(
        "bucket = 模型 top-N 的基准相对收益；univ = 全市场平均的基准相对收益（市场宽度）；"
        "sel = bucket - univ（选股贡献）"
    )
    lines.append(
        f"{'period':<10} {'folds':>5} {'meas':>4} {'bucket':>10} {'univ':>10} {'sel':>10} "
        f"{'IC':>9} {'折间σ':>10} {'加权bucket':>11} {'加权sel':>10}"
    )
    for entry in summary:
        lines.append(
            f"{entry['period']:<10} {entry['folds']:>5} {entry['bucket_measured_folds']:>4} "
            f"{_fmt(entry['bucket_mean']):>10} {_fmt(entry['universe_mean']):>10} "
            f"{_fmt(entry['selection_mean']):>10} {_fmt(entry['ic_mean'], 'f'):>9} "
            f"{_fmt(entry['bucket_stats']['sd']):>10} "
            f"{_fmt(entry['weighted_bucket_mean']):>11} "
            f"{_fmt(entry['weighted_selection_mean']):>10}"
        )

    lines.append("")
    lines.append("=" * 108)
    lines.append("## 显著性：均值是否可与其自身折间噪声区分")
    lines.append(
        "折间 σ 是同一套参数在不同月份的波动。当 σ / √n 接近或超过均值本身，"
        "「这一年桶质量如何」就还没有被数据支持。"
    )
    lines.append(
        f"{'period':<10} {'组件':<10} {'均值':>10} {'折间σ':>10} {'SE':>10} {'t':>7} "
        f"{'95% 区间':>24} {'正折':>8}"
    )
    for entry in summary:
        for label, key in (("bucket", "bucket_stats"), ("selection", "selection_stats")):
            s = entry[key]
            ci = (
                f"[{s['ci_low']:+.4%}, {s['ci_high']:+.4%}]"
                if s.get("ci_low") is not None
                else "n/a"
            )
            pos = (
                f"{s['positive_folds']}/{s['n']}"
                if s.get("positive_folds") is not None
                else "n/a"
            )
            t_text = f"{s['t']:+.2f}" if s.get("t") is not None else "n/a"
            lines.append(
                f"{entry['period']:<10} {label:<10} {_fmt(s['mean']):>10} "
                f"{_fmt(s['sd']):>10} {_fmt(s['se']):>10} "
                f"{t_text:>7} {ci:>24} {pos:>8}"
            )

    lines.append("")
    lines.append("## 逐段对比（Welch t 检验，按折）")
    lines.append(
        f"{'对比':<24} {'组件':<10} {'差值':>10} {'SE':>10} {'t':>7} {'显著':>6} {'95% 区间':>24}"
    )
    for test in pairwise(summary):
        ci = (
            f"[{test['ci_low']:+.4%}, {test['ci_high']:+.4%}]"
            if test.get("ci_low") is not None
            else "n/a"
        )
        verdict = "n/a"
        if test.get("significant") is not None:
            verdict = "**是**" if test["significant"] else "否"
        t_text = f"{test['t']:+.2f}" if test.get("t") is not None else "n/a"
        lines.append(
            f"{test['comparison']:<24} {test['component']:<10} {_fmt(test['shift']):>10} "
            f"{_fmt(test['se']):>10} {t_text:>7} {verdict:>6} {ci:>24}"
        )
    return "\n".join(lines)


def render_markdown(rows: list[dict], summary: list[dict], report_path: str) -> str:
    lines: list[str] = []
    lines.append("# 桶超额逐年归因：市场宽度 vs 选股\n")
    lines.append(f"- 来源报告：`{report_path}`")
    lines.append(
        "- 分解式：`bucket = univ + sel`。`univ` 是全市场相对基准的平均收益（市场宽度），"
        "`sel` 是模型 top-N 相对全市场的超额（选股贡献）。"
    )
    lines.append(
        "- 桶为**原始模型排名**的 top-N（组合规模），非五分位；标签本身是基准相对收益，"
        "故桶均值即桶的基准相对超额。"
    )
    unmeasured = [
        (entry["period"], item)
        for entry in summary
        for item in entry.get("unmeasured_folds") or []
    ]
    if unmeasured:
        lines.append("")
        lines.append("> **不可测折（不得当作 0 计入均值）**：")
        for period_name, item in unmeasured:
            lines.append(f"> - `{period_name}` 第 {item['fold']} 折：{item['reason']}")
    lines.append("")

    lines.append("## 逐年分解\n")
    lines.append(
        "| 区间 | 折数 | 可测折 | bucket | univ（宽度） | sel（选股） | IC 均值 | "
        "加权 bucket | 加权 sel |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for entry in summary:
        lines.append(
            "| `{p}` | {folds} | {meas} | {bucket} | {univ} | {sel} | {ic} | {wb} | {ws} |".format(
                p=entry["period"],
                folds=entry["folds"],
                meas=entry["bucket_measured_folds"],
                bucket=_fmt(entry["bucket_mean"]),
                univ=_fmt(entry["universe_mean"]),
                sel=_fmt(entry["selection_mean"]),
                ic=_fmt(entry["ic_mean"], "f"),
                wb=_fmt(entry["weighted_bucket_mean"]),
                ws=_fmt(entry["weighted_selection_mean"]),
            )
        )
    lines.append("")

    lines.append("## 显著性：均值能否与其自身折间噪声区分\n")
    lines.append(
        "折间 σ 是同一套参数在不同月份的波动。当 `σ/√n` 与均值同量级时，"
        "「这一年桶质量如何」尚未被数据支持——此时不应据此下结论。\n"
    )
    lines.append("| 区间 | 组件 | 均值 | 折间 σ | SE | t | 95% 区间 | 正折 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: |")
    for entry in summary:
        for label, key in (("bucket", "bucket_stats"), ("selection", "selection_stats")):
            s = entry[key]
            ci = (
                f"[{s['ci_low']:+.4%}, {s['ci_high']:+.4%}]"
                if s.get("ci_low") is not None
                else "n/a"
            )
            pos = (
                f"{s['positive_folds']}/{s['n']}"
                if s.get("positive_folds") is not None
                else "n/a"
            )
            t_text = f"{s['t']:+.2f}" if s.get("t") is not None else "n/a"
            lines.append(
                f"| `{entry['period']}` | {label} | {_fmt(s['mean'])} | {_fmt(s['sd'])} | "
                f"{_fmt(s['se'])} | {t_text} | {ci} | {pos} |"
            )
    lines.append("")

    lines.append("## 逐段对比（Welch t 检验，按折）\n")
    lines.append("| 对比 | 组件 | 差值 | SE | t | 是否显著 | 95% 区间 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | :---: | :--- |")
    for test in pairwise(summary):
        ci = (
            f"[{test['ci_low']:+.4%}, {test['ci_high']:+.4%}]"
            if test.get("ci_low") is not None
            else "n/a"
        )
        verdict = "n/a"
        if test.get("significant") is not None:
            verdict = "**是**" if test["significant"] else "否"
        t_text = f"{test['t']:+.2f}" if test.get("t") is not None else "n/a"
        lines.append(
            f"| `{test['comparison']}` | {test['component']} | {_fmt(test['shift'])} | "
            f"{_fmt(test['se'])} | {t_text} | {verdict} | {ci} |"
        )
    lines.append("")

    lines.append("## 逐折明细\n")
    lines.append(
        "| 区间 | 折 | 测试起 | 测试止 | 已标注行 | IC | bucket | univ | sel | "
        "桶内正比例 | 桶规模 |"
    )
    lines.append("| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        m = row["metrics"]
        lines.append(
            f"| `{row['period']}` | {row['fold']} | {row['test_start']} | {row['test_end']} | "
            f"{_fmt(m['test_labeled_rows'], 'int')} | {_fmt(m['test_ic_mean'], 'f')} | "
            f"{_fmt(row['bucket'])} | {_fmt(row['universe'])} | {_fmt(row['selection'])} | "
            f"{_fmt(m['test_bucket_label_positive_ratio'], 'f', 2)} | "
            f"{_fmt(m['test_bucket_size'], 'int')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="把桶超额逐年变化拆成「市场宽度」与「选股」两部分")
    parser.add_argument("--report", required=True, help="walk-forward 报告 JSON 路径")
    parser.add_argument("--json-out", default=None, help="可选：把分解结果另存为 JSON")
    parser.add_argument("--markdown", default=None, help="可选：把分解结果另存为可读的 Markdown")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"报告不存在: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = collect(report)
    if not rows:
        raise SystemExit("报告里没有逐折记录，无法分解")
    summary = summarise(rows)
    print(render(rows, summary))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        public = [
            {key: value for key, value in entry.items() if not key.startswith("_")}
            for entry in summary
        ]
        out.write_text(
            json.dumps(
                {
                    "report": str(path.resolve()),
                    "periods": public,
                    "pairwise": pairwise(summary),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\njson: {out.resolve()}")
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown(rows, summary, str(path.resolve())), encoding="utf-8"
        )
        print(f"markdown: {markdown_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
