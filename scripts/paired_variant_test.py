"""Paired daily test between two portfolio variants replayed over one fit.

A 3-period sweep compares two variants with n = 3 observations, which cannot
separate a 3 pp cumulative difference from noise.  But both variants were
replayed over the **same fitted scores on the same days**, which makes their
daily returns a *paired* series: the benchmark cancels in the difference, and
the sample size becomes the number of trading days (~650) instead of the number
of periods (3).

The test is therefore: is the mean of ``d_t = log(1 + r_a) - log(1 + r_b)``
distinguishable from zero?

**Log returns, not simple returns.** Arithmetic daily mean differences and
geometric cumulative differences can have *opposite signs*: a variant with more
volatility can compound higher while averaging less per day. Testing the mean of
simple-return differences would then test "equal average day", not "equal
terminal wealth". Log differences telescope exactly --
``sum(d_t) = log(1 + R_a) - log(1 + R_b)`` -- so the null being tested is the one
that matters, and the daily total reconciles with the period-level cumulative
figure instead of contradicting it.

Three things make a naive t test wrong here, and all three are handled:

1. **Autocorrelation.** Positions persist for days, so the difference series is
   strongly autocorrelated; the naive standard error understates uncertainty.
   Report a Newey-West (Bartlett) HAC standard error and a *circular block*
   bootstrap, both of which are valid under autocorrelation.
2. **Concentration.** A few episodes can carry the whole difference.  Report how
   the mean changes once the five largest absolute days are removed.
3. **Pooling.** Concatenating the three periods would let a bootstrap block
   straddle a period boundary, so blocks are drawn *within* each period.

The bootstrap p value re-centres the series at zero before resampling, which is
the correct way to test H0: mean = 0 rather than reading the confidence interval
as if it were a test.

Input is the long-format daily equity curves written by
``scripts/smoothing_sweep.py --curves-dir``.

Usage::

    python scripts/paired_variant_test.py \\
        --curves-dir data/reports/curves/portfolio-sweep \\
        --variant-a w20_r5_b00_t20 --variant-b w20_r10_b50_t20 \\
        --markdown data/reports/paired-variant-r5-vs-r10.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Small t table so the script has no hard scipy dependency; scipy is used when
# importable for degrees of freedom outside the table.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
    60: 2.000, 120: 1.980,
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
        return 1.96 if df >= 120 else _T95[max(k for k in _T95 if k <= 120)]


DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
DEFAULT_BLOCK = 20
DEFAULT_SEED = 20260918
TRADING_DAYS_PER_YEAR = 242


def load_pair(
    curves_dir: Path, variant_a: str, variant_b: str
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Read every ``<period>.csv`` and align the two variants on shared dates.

    Only dates present for *both* variants enter the comparison: a variant that
    starts trading a day earlier would otherwise inject a one-sided return.
    """

    files = sorted(curves_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"没有找到曲线文件: {curves_dir}")
    aligned: dict[str, pd.DataFrame] = {}
    for path in files:
        frame = pd.read_csv(path)
        missing = {"variant", "trade_date", "equity"} - set(frame.columns)
        if missing:
            raise SystemExit(f"{path.name} 缺少列: {', '.join(sorted(missing))}")
        frame = frame[frame["variant"].isin([variant_a, variant_b])]
        if frame.empty:
            continue
        wide = frame.pivot(index="trade_date", columns="variant", values="equity")
        if not {variant_a, variant_b} <= set(wide.columns):
            continue
        wide = wide.loc[:, [variant_a, variant_b]].dropna().sort_index()
        if len(wide) < 3:
            continue
        aligned[path.stem] = wide
    if not aligned:
        raise SystemExit(f"没有区间同时包含 {variant_a} 与 {variant_b}")
    return list(aligned), aligned


def _returns(equity: pd.Series) -> np.ndarray:
    """Daily **log** returns.

    Log rather than simple: the differences telescope, so the sample mean times
    the number of days is exactly the period's log-wealth difference.  Simple
    returns would make the daily average and the compounded total disagree in
    sign whenever the two variants differ in volatility.
    """

    values = pd.to_numeric(equity, errors="coerce").to_numpy(dtype=float)
    return np.log(values[1:] / values[:-1])


def hac_se(values: np.ndarray, lag: int) -> float:
    """Newey-West standard error of the mean, Bartlett kernel.

    ``lag = 0`` degenerates to the naive iid standard error, which is included
    as a comparison precisely because it is too small when the series is
    autocorrelated.
    """

    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    deviation = x - x.mean()
    total = float(deviation @ deviation) / n
    for j in range(1, min(lag, n - 1) + 1):
        gamma = float(deviation[j:] @ deviation[:-j]) / n
        total += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    return math.sqrt(max(total, 0.0) / n)


def block_bootstrap(
    series_list: list[np.ndarray],
    block: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Circular block bootstrap means, blocks drawn **within** each series.

    Returns an array of ``iterations`` resampled means.  Blocks never cross a
    period boundary, so a seam between two periods cannot manufacture a
    spurious mean shift.
    """

    generator = np.random.default_rng(seed)
    offsets = np.arange(block)
    parts: list[np.ndarray] = []
    for values in series_list:
        x = np.asarray(values, dtype=float)
        n = x.size
        if n == 0:
            continue
        count = math.ceil(n / block)
        starts = generator.integers(0, n, size=(iterations, count))
        index = (starts[:, :, None] + offsets[None, None, :]) % n
        parts.append(x[index.reshape(iterations, -1)[:, :n]])
    if not parts:
        return np.zeros(iterations)
    samples = np.concatenate(parts, axis=1)
    return samples.mean(axis=1)


def analyse(series_list: list[np.ndarray], *, hac_lags: list[int], block: int,
            iterations: int, seed: int) -> dict:
    pooled = np.concatenate(series_list) if series_list else np.zeros(0)
    n = pooled.size
    if n < 3:
        raise SystemExit("样本不足，无法检验")
    mean = float(pooled.mean())
    per_period_means = [float(x.mean()) for x in series_list if x.size]
    # The iid reference is the lag-0 case of the same estimator, so the gap
    # between this row and the HAC rows is *only* the autocovariance correction
    # -- not a difference in degrees-of-freedom convention.
    naive = hac_se(pooled, 0)

    # A degenerate series (a constant difference) has zero standard error; the
    # t statistic is then undefined rather than astronomically large, so report
    # it as such instead of printing 5e13.
    def _t(se: float) -> float | None:
        if se is None or math.isnan(se) or abs(se) < 1e-12:
            return None
        return mean / se

    hac = {}
    for lag in hac_lags:
        se = hac_se(pooled, lag)
        hac[str(lag)] = {"se": se, "t": _t(se)}

    # Resolving power.  Pairing by day makes the standard error *credible*, it
    # does not make it small: the daily edge is a few basis points against a
    # standard error of the same order.  State the smallest annualised edge this
    # sample could have detected at 80% power, so a "not significant" verdict is
    # read as "too small to see" rather than "no difference".
    widest_lag = _widest_lag(hac)
    se_ref = hac[widest_lag]["se"] if widest_lag else naive
    mde_daily = (1.959964 + 0.8416212) * se_ref  # two-sided 5%, 80% power
    mde_annual = mde_daily * TRADING_DAYS_PER_YEAR

    samples = block_bootstrap(series_list, block, iterations, seed)
    ci_low, ci_high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))

    # Hypothesis test needs the null distribution: re-centre at zero, resample,
    # then ask how often a |mean| at least as large as observed arises by chance.
    centred = [x - x.mean() for x in series_list if x.size]
    null_samples = block_bootstrap(centred, block, iterations, seed + 1)
    p_value = float(np.mean(np.abs(null_samples) >= abs(mean)))

    order = np.argsort(-np.abs(pooled))
    trim = min(5, n - 1)
    trimmed_mean = float(np.delete(pooled, order[:trim]).mean())

    # Period-level test on the same quantity, for calibration rather than for
    # contrast.  For a random walk the t statistic is invariant to the sampling
    # frequency, so the daily version buys a *credible variance estimate*
    # (n-1 ~ 640 df) rather than extra power.  Printing both is what shows that:
    # when they disagree by much, the period-level sd is a 2-df artefact.
    k = len(per_period_means)
    period_test: dict | None = None
    if k >= 2:
        period_mean = sum(per_period_means) / k
        period_sd = statistics.stdev(per_period_means)
        period_se = period_sd / math.sqrt(k)
        period_t = period_mean / period_se if abs(period_se) > 1e-12 else None
        critical = _t_critical(k - 1)
        period_test = {
            "periods": k,
            "mean": period_mean,
            "sd": period_sd,
            "se": period_se,
            "t": period_t,
            "critical": critical,
            "significant": None if period_t is None else bool(abs(period_t) > critical),
        }

    return {
        "days": n,
        "periods": len(series_list),
        "mean_daily": mean,
        "period_balanced_mean": sum(per_period_means) / len(per_period_means),
        "sum_daily": mean * n,
        "cumulative_log": math.expm1(float(pooled.sum())),
        "per_period_means": per_period_means,
        "period_test": period_test,
        "naive_se": naive,
        "naive_t": _t(naive),
        "hac": hac,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_p": p_value,
        "block": block,
        "iterations": iterations,
        "trimmed_days": trim,
        "trimmed_mean_daily": trimmed_mean,
        "trimmed_sum_daily": trimmed_mean * n,
        "largest_day": float(pooled[order[0]]) if n else None,
        "mde_daily": mde_daily,
        "mde_annual": mde_annual,
        "mde_se_lag": widest_lag,
    }


def _pct(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}%}"


def _widest_lag(hac: dict) -> str | None:
    """The lag whose HAC standard error is largest.

    Selecting by the *largest standard error* (rather than by the numerically or
    lexically largest lag) is what makes the quoted t statistic and resolving
    power the conservative ones -- a smaller standard error would flatter the
    variant being tested.
    """

    keys = [k for k, v in hac.items() if v.get("se") is not None]
    if not keys:
        return None
    return max(keys, key=lambda k: hac[k]["se"])


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def render(
    variant_a: str,
    variant_b: str,
    per_period: dict[str, dict],
    total_diff: dict[str, float],
    pooled: dict,
    periods_order: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# 配对日频检验：两个变体是否真的不同\n")
    lines.append(f"- A = `{variant_a}`，B = `{variant_b}`")
    lines.append(
        "- 两个变体在**同一批拟合打分、同一批交易日**上回放 ⇒ 日收益可**配对**，"
        "基准在差值里约掉，样本量从 3 个区间变成 ~650 个交易日。"
    )
    lines.append(
        "- ⚠️ 日频差值序列**自相关很重**（持仓连续多日），朴素 t 的标准误会偏小，"
        "故同时给出 **Newey-West HAC** 标准误与**分块自助**区间；"
        "自助的 p 值用**零均值重抽**，而不是拿置信区间当检验。"
    )
    lines.append(
        "  朴素那一行就是同一估计量的 **lag = 0** 情形，所以它与 HAC 各行的差距**只来自"
        "自相关修正**，不含自由度口径差异。"
    )
    lines.append("")

    lines.append("## 逐区间：累计收益之差（与扫描报告可对账）\n")
    lines.append("| 区间 | 交易日 | A 累计 | B 累计 | **A − B** | 日均差 | 区间内 t(HAC) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in periods_order:
        row = per_period[name]
        widest = _widest_lag(row["hac"])
        t_hac = row["hac"][widest]["t"] if widest else None
        lines.append(
            "| `{p}` | {d} | {a} | {b} | **{diff}** | {mean} | {t} |".format(
                p=name,
                d=row["days"],
                a=_pct(row["total_return_a"], 2),
                b=_pct(row["total_return_b"], 2),
                diff=_pct(row["total_return_diff"], 2),
                mean=_pct(row["mean_daily"], 3),
                t=_num(t_hac),
            )
        )
    lines.append("")

    lines.append("## 汇总\n")
    lines.append("| 口径 | 数值 |")
    lines.append("| :--- | ---: |")
    lines.append(f"| 区间数 | {pooled['periods']} |")
    lines.append(f"| 日数 n | {pooled['days']} |")
    lines.append(f"| 日均对数差（A − B） | {_pct(pooled['mean_daily'])} |")
    lines.append(f"| 区间均衡日均对数差 | {_pct(pooled['period_balanced_mean'])} |")
    lines.append(
        f"| Σ 各区间累计收益差（直接相加 Σ(Aᵖ−Bᵖ)） | {_pct(total_diff['sum'], 2)} |"
    )
    lines.append(
        f"| log 汇总 expm1(Σ 逐日对数差) | {_pct(pooled['cumulative_log'], 2)} |"
    )
    lines.append("")
    lines.append(
        "> 两行口径不同（上：把各区间收益差直接相加；下：按对数可加性汇总后还原），"
        "但**必须同号、量级接近**。对数差会望远镜式相消，所以逐日求和精确等于"
        "端点财富之比；两行若符号相反，说明统计量用的是算术差分 —— "
        "那测的是「日均相同」而非「终值相同」。"
    )
    lines.append("")

    lines.append("### 显著性：三把尺子\n")
    lines.append("| 方法 | 标准误 | t | 说明 |")
    lines.append("| :--- | ---: | ---: | :--- |")
    lines.append(
        f"| 朴素（iid，即 lag=0） | {_pct(pooled['naive_se'])} | {_num(pooled['naive_t'])} | "
        "**参考下界**，自相关下偏小 |"
    )
    for lag in sorted(pooled["hac"]):
        entry = pooled["hac"][lag]
        lines.append(
            f"| HAC lag={lag} | {_pct(entry['se'])} | {_num(entry['t'])} | "
            "Newey-West（Bartlett） |"
        )
    lines.append("")
    lines.append(
        f"- **分块自助**（块长 {pooled['block']} 日，{pooled['iterations']:,} 次）："
        f"日均差 95% 区间 "
        f"[{_pct(pooled['bootstrap_ci_low'])}, {_pct(pooled['bootstrap_ci_high'])}]；"
        f"零均值重抽 p = **{pooled['bootstrap_p']:.4f}**"
    )
    verdict = (
        "**显著**（可区分）"
        if pooled["bootstrap_p"] < 0.05
        else "**不显著**（与噪声不可区分）"
    )
    lines.append(f"- 结论（α = 0.05）：{verdict}")
    lines.append(
        f"- **分辨力**：以 HAC lag={pooled['mde_se_lag']} 的标准误为准，"
        f"本样本（n={pooled['days']}）在 80% 功效下**最小可分辨**的年度化差额约 "
        f"**{_pct(pooled['mde_annual'], 1)}/年**"
        f"（日均 {_pct(pooled['mde_daily'], 3)} × {TRADING_DAYS_PER_YEAR} 个交易日）。"
        "比它更小的真实优势，本检验**看不见** —— 所以「不显著」应读作"
        "「小到测不出」，而非「确证无差别」。"
    )
    lines.append("")

    period_test = pooled.get("period_test")
    if period_test:
        widest = _widest_lag(pooled["hac"])
        lines.append("### 与区间级检验对照：日频配对到底买到了什么\n")
        lines.append(
            "随机游走下 t 统计量对抽样频率**不变**，所以日频配对带来的不是额外功效，"
            "而是**可信的自由度**（n−1 ≈ 640，而区间级只有 k−1 = "
            f"{period_test['periods'] - 1}，临界值高达 {period_test['critical']:.2f}）。"
            "两者都列出，差异从哪里来才看得见。\n"
        )
        lines.append("| 口径 | 观测数 | 标准误 | t | 临界值(95%) | 结论 |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | :---: |")
        lines.append(
            f"| 日频配对（HAC lag={widest}） | {pooled['days']} | "
            f"{_pct(pooled['hac'][widest]['se'])} | {_num(pooled['hac'][widest]['t'])} | "
            "1.96 | "
            + ("**显著**" if pooled["bootstrap_p"] < 0.05 else "不显著")
            + " |"
        )
        lines.append(
            f"| 区间级（{period_test['periods']} 个区间均值） | {period_test['periods']} | "
            f"{_pct(period_test['se'])} | {_num(period_test['t'])} | "
            f"{period_test['critical']:.2f} | "
            + ("**显著**" if period_test["significant"] else "不显著")
            + " |"
        )
        period_means = ", ".join(_pct(v, 2) for v in pooled["per_period_means"])
        lines.append(f"- 区间均值序列（A − B 的日均差）：{period_means}")
        lines.append("")

    lines.append("### 集中度：差额是否只靠少数几天\n")
    lines.append(
        f"- 去掉绝对值最大的 {pooled['trimmed_days']} 天后，"
        f"日均差 {_pct(pooled['mean_daily'])} → **{_pct(pooled['trimmed_mean_daily'])}**"
        f"（累计 {_pct(pooled['sum_daily'], 2)} → {_pct(pooled['trimmed_sum_daily'], 2)}）"
    )
    lines.append(f"- 单日最大差值：{_pct(pooled['largest_day'])}")
    if abs(pooled["mean_daily"]) < 1e-5:
        lines.append("- 差额本身已≈0，集中度问题无意义。")
    else:
        retained = pooled["trimmed_mean_daily"] / pooled["mean_daily"]
        if pooled["trimmed_mean_daily"] * pooled["mean_daily"] < 0:
            verdict = (
                f"去掉最大的 {pooled['trimmed_days']} 天后差额**变号**"
                f"（{_pct(pooled['mean_daily'])} → {_pct(pooled['trimmed_mean_daily'])}）"
                "⇒ 原差额**由少数几天主导**，不宜据此下结论。"
            )
        elif abs(retained) < 0.5:
            verdict = "差额**由少数几天主导**，不宜据此下结论。"
        else:
            verdict = "差额**不是少数几天造成的**，较稳健。"
        if pooled["trimmed_mean_daily"] * pooled["mean_daily"] >= 0:
            lines.append(f"- 去尾后保留原差额的 **{retained:.0%}** —— " + verdict)
        else:
            lines.append("- " + verdict)
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="两个组合变体的配对日频检验（HAC + 分块自助）")
    parser.add_argument("--curves-dir", required=True, help="逐日权益曲线目录（长表 CSV）")
    parser.add_argument("--variant-a", default="w20_r5_b00_t20")
    parser.add_argument("--variant-b", default="w20_r10_b50_t20")
    parser.add_argument("--hac-lags", default="5,10,20", help="HAC 滞后阶列表")
    parser.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="自助块长（交易日）")
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    hac_lags = [int(chunk) for chunk in str(args.hac_lags).split(",") if chunk.strip()]
    periods_order, aligned = load_pair(
        Path(args.curves_dir), args.variant_a, args.variant_b
    )

    per_period: dict[str, dict] = {}
    series_list: list[np.ndarray] = []
    total_diffs: list[float] = []
    for name in periods_order:
        wide = aligned[name]
        ret_a = _returns(wide[args.variant_a])
        ret_b = _returns(wide[args.variant_b])
        diff = ret_a - ret_b
        series_list.append(diff)
        total_a = float(wide[args.variant_a].iloc[-1] / wide[args.variant_a].iloc[0] - 1.0)
        total_b = float(wide[args.variant_b].iloc[-1] / wide[args.variant_b].iloc[0] - 1.0)
        total_diffs.append(total_a - total_b)
        entry = analyse(
            [diff],
            hac_lags=hac_lags,
            block=args.block,
            iterations=args.iterations,
            seed=args.seed,
        )
        entry["total_return_a"] = total_a
        entry["total_return_b"] = total_b
        entry["total_return_diff"] = total_a - total_b
        per_period[name] = entry

    pooled = analyse(
        series_list,
        hac_lags=hac_lags,
        block=args.block,
        iterations=args.iterations,
        seed=args.seed,
    )
    total_diff = {"sum": sum(total_diffs), "by_period": dict(zip(periods_order, total_diffs))}

    report = {
        "curves_dir": str(Path(args.curves_dir).resolve()),
        "variant_a": args.variant_a,
        "variant_b": args.variant_b,
        "per_period": per_period,
        "pooled": pooled,
        "total_diff": total_diff,
        "settings": {"hac_lags": hac_lags, "block": args.block, "iterations": args.iterations},
    }
    print(
        render(args.variant_a, args.variant_b, per_period, total_diff, pooled, periods_order)
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\njson: {out.resolve()}")
    if args.markdown:
        md = Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(
            render(args.variant_a, args.variant_b, per_period, total_diff, pooled, periods_order),
            encoding="utf-8",
        )
        print(f"markdown: {md.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
