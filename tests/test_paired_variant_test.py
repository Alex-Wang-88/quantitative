"""Tests for the paired daily variant test.

Loaded by path because it is a research script, not part of the library.  What
is pinned down here is the statistics, not the formatting: a 3-period sweep
comparison is replaced by a paired daily one, and the whole value of that
substitution depends on getting three things right —

* the paired alignment (only shared dates enter the difference),
* the standard error under autocorrelation (naive underestimates it),
* the bootstrap resampling **within** periods (a block must not straddle a
  period seam).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_variant_test.py"


def _load():
    spec = importlib.util.spec_from_file_location("paired_variant_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = _load()

FAST = {"hac_lags": [5, 10, 20], "block": 4, "iterations": 400, "seed": 1}


def _curves(directory: Path, frames: dict) -> Path:
    """Write long-format curve CSVs.

    ``frames`` maps period -> variant -> equity levels.  A variant value may be a
    ``(start_offset, levels)`` tuple to give that variant a later first date,
    which is how a misaligned pair is simulated.
    """

    directory.mkdir(parents=True, exist_ok=True)
    for period, variants in frames.items():
        rows = []
        for variant, spec in variants.items():
            offset, values = spec if isinstance(spec, tuple) else (0, spec)
            dates = pd.bdate_range("2024-01-02", periods=len(values) + offset)[offset:]
            rows.append(
                pd.DataFrame(
                    {
                        "variant": variant,
                        "trade_date": dates.strftime("%Y-%m-%d"),
                        "equity": values,
                    }
                )
            )
        pd.concat(rows, ignore_index=True).to_csv(directory / f"{period}.csv", index=False)
    return directory


def _levels(returns: list[float], start: float = 1_000_000.0) -> list[float]:
    values = [start]
    for value in returns:
        values.append(values[-1] * (1.0 + value))
    return values


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #


def test_only_shared_dates_enter_the_difference(tmp_path: Path):
    # B starts one day later, so the first day has no pair and must be dropped:
    # otherwise the difference would be a one-sided return.
    a = _levels([0.001] * 8)
    b = (1, _levels([0.001] * 8))
    directory = _curves(tmp_path / "c", {"2024": {"A": a, "B": b}})
    _, aligned = paired.load_pair(directory, "A", "B")
    wide = aligned["2024"]
    assert len(wide) == 8
    assert list(wide.columns) == ["A", "B"]


def test_missing_variant_is_reported(tmp_path: Path):
    directory = _curves(tmp_path / "c", {"2024": {"A": _levels([0.001] * 5)}})
    with pytest.raises(SystemExit):
        paired.load_pair(directory, "A", "B")


def test_empty_directory_is_reported(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        paired.load_pair(tmp_path / "empty", "A", "B")


# --------------------------------------------------------------------------- #
# standard errors
# --------------------------------------------------------------------------- #


def test_zero_lag_is_the_population_variance_over_n():
    # Pinning the definition matters: the report's "iid" row is the lag-0 case of
    # this same estimator, so its distance to the HAC rows is purely the
    # autocovariance correction.
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 0.01, 500)
    gamma0 = float(((values - values.mean()) ** 2).mean())
    assert paired.hac_se(values, 0) == pytest.approx(
        math.sqrt(gamma0 / values.size), rel=1e-12
    )


def test_reported_iid_error_equals_the_zero_lag_estimator():
    rng = np.random.default_rng(8)
    series = [rng.normal(0.001, 0.006, 400)]
    stats = paired.analyse(series, **FAST)
    assert stats["naive_se"] == pytest.approx(paired.hac_se(series[0], 0), rel=1e-12)


def test_hac_inflates_the_error_when_the_series_is_autocorrelated():
    # A persistent difference is the realistic shape: positions are held for
    # days, so today's difference predicts tomorrow's.  Under that structure the
    # iid standard error is too small and would overstate significance.
    rng = np.random.default_rng(5)
    n = 800
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0.0, 0.004 * math.sqrt(1 - 0.64))
    naive = noise.std(ddof=1) / math.sqrt(n)
    hac = paired.hac_se(noise, 20)
    assert hac > naive * 1.5


def test_hac_matches_naive_on_white_noise_within_tolerance():
    rng = np.random.default_rng(9)
    values = rng.normal(0.0, 0.01, 4000)
    naive = values.std(ddof=1) / math.sqrt(values.size)
    assert paired.hac_se(values, 10) == pytest.approx(naive, rel=0.25)


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #


def test_blocks_do_not_cross_a_period_seam():
    # 9 up days in period 1, 1 down day in period 2, one-day blocks.  Every
    # resample must keep the 9:1 split, so every bootstrap mean is identical.
    # If blocks were drawn from the pooled array instead, the mean would vary.
    series = [np.full(9, 0.01), np.full(1, -0.01)]
    means = paired.block_bootstrap(series, block=1, iterations=200, seed=2)
    assert means.std() == pytest.approx(0.0, abs=1e-15)
    assert means[0] == pytest.approx((9 * 0.01 - 0.01) / 10)


def test_bootstrap_mean_centre_follows_the_sample_mean():
    rng = np.random.default_rng(4)
    series = [rng.normal(0.002, 0.005, 300)]
    means = paired.block_bootstrap(series, block=10, iterations=1500, seed=6)
    assert means.mean() == pytest.approx(series[0].mean(), abs=0.0006)


# --------------------------------------------------------------------------- #
# the verdicts
# --------------------------------------------------------------------------- #


def test_a_persistent_edge_is_detected_and_survives_trimming():
    stats = paired.analyse([np.full(300, 0.0003)], **FAST)
    assert stats["bootstrap_p"] < 0.01
    assert stats["bootstrap_ci_low"] > 0
    assert stats["trimmed_mean_daily"] == pytest.approx(stats["mean_daily"], rel=1e-9)
    assert stats["days"] == 300


def test_an_identical_pair_is_never_called_significant():
    stats = paired.analyse([np.zeros(300)], **FAST)
    assert stats["mean_daily"] == pytest.approx(0.0)
    assert stats["bootstrap_p"] == pytest.approx(1.0, abs=0.05)


def test_a_single_spike_is_not_significant_and_is_flagged_as_concentrated():
    diff = np.zeros(400)
    diff[12] = 0.03
    stats = paired.analyse([diff], **FAST)
    # The whole mean is one day; removing the five largest days must erase it.
    assert stats["bootstrap_p"] > 0.05
    assert stats["trimmed_mean_daily"] == pytest.approx(0.0, abs=1e-12)
    assert stats["largest_day"] == pytest.approx(0.03)


def test_a_degenerate_zero_error_series_reports_no_t_statistic():
    # A constant difference has zero standard deviation; t is undefined, not
    # 5e13, and must not be rendered as an immense significance.
    stats = paired.analyse([np.full(120, 0.001)], **FAST)
    assert stats["naive_t"] is None
    assert stats["hac"]["20"]["t"] is None


def test_too_short_a_sample_is_rejected():
    with pytest.raises(SystemExit):
        paired.analyse([np.array([0.001, 0.002])], **FAST)


def test_pooled_mean_sits_between_the_components():
    stats = paired.analyse(
        [np.full(100, 0.001), np.full(50, -0.001)], **FAST
    )
    assert stats["periods"] == 2
    assert stats["days"] == 150
    assert stats["mean_daily"] == pytest.approx((100 * 0.001 - 50 * 0.001) / 150)
    assert stats["period_balanced_mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# log returns, and why the choice matters
# --------------------------------------------------------------------------- #


def test_daily_returns_are_logarithmic():
    # log(1 + r), not r: the difference of the two series must telescope.
    returns = paired._returns(pd.Series([100.0, 110.0, 99.0]))
    assert returns[0] == pytest.approx(math.log(1.10))
    assert returns[1] == pytest.approx(math.log(0.9))


def test_cumulative_log_reconstructs_the_terminal_wealth_ratio():
    # sum of daily log differences == log(terminal_a / terminal_b), by
    # telescoping.  This is what makes the daily total agree with the
    # period-level cumulative figure instead of contradicting it.
    a = _levels([0.02, -0.01, 0.03, 0.0])
    b = _levels([0.01, 0.015, -0.02, 0.005])
    diff = paired._returns(pd.Series(a)) - paired._returns(pd.Series(b))
    stats = paired.analyse([diff], **FAST)
    expected = (a[-1] / a[0]) / (b[-1] / b[0]) - 1.0
    assert stats["cumulative_log"] == pytest.approx(expected, rel=1e-12)


def test_arithmetic_and_geometric_aggregation_can_disagree_in_sign():
    # The reason log returns are mandatory.  A and B end at *identical* terminal
    # wealth, but B gets there through higher volatility.  Geometric mean <=
    # arithmetic mean, so the more volatile path has the larger average simple
    # return: mean(rA - rB) < 0 even though the terminal difference is exactly 0.
    # Differencing simple returns would therefore test "equal average day" and
    # report a loss where none exists.
    a = [1.0, 1.05, 1.1025]  # +5%, +5%
    b = [1.0, 1.20, 1.1025]  # +20%, -8.125%
    assert a[-1] == pytest.approx(b[-1])  # same terminal wealth
    simple_a = np.diff(a) / np.asarray(a)[:-1]
    simple_b = np.diff(b) / np.asarray(b)[:-1]
    assert (simple_a - simple_b).mean() < 0  # arithmetic says B is better
    log_diff = paired._returns(pd.Series(a)) - paired._returns(pd.Series(b))
    assert log_diff.sum() == pytest.approx(0.0, abs=1e-12)  # log form agrees


# --------------------------------------------------------------------------- #
# resolving power
# --------------------------------------------------------------------------- #


def test_resolving_power_scales_with_the_standard_error():
    rng = np.random.default_rng(11)
    stats = paired.analyse([rng.normal(0.0, 0.008, 500)], **FAST)
    widest = paired._widest_lag(stats["hac"])
    # 2.802 = 1.96 + 0.842, the two-sided 5% / 80% power multiplier.
    assert stats["mde_daily"] == pytest.approx(
        2.8016 * stats["hac"][widest]["se"], rel=1e-3
    )
    assert stats["mde_daily"] > stats["naive_se"]  # based on the HAC error
    assert stats["mde_annual"] == pytest.approx(
        stats["mde_daily"] * paired.TRADING_DAYS_PER_YEAR, rel=1e-12
    )


def test_widest_lag_is_chosen_by_the_largest_standard_error():
    # Not the lexically largest key ("5" > "20" as strings) and not the largest
    # lag: the point is to quote the most conservative error.
    hac = {
        "5": {"se": 0.02, "t": 1.0},
        "20": {"se": 0.01, "t": 2.0},
        "60": {"se": 0.03, "t": 0.7},
    }
    assert paired._widest_lag(hac) == "60"
    assert paired._widest_lag({}) is None
