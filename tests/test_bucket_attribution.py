"""Tests for the bucket-decay attribution script.

The script is not part of the library, so it is loaded by path rather than by
package import.  What is being pinned down here is not formatting but two rules
that were got wrong once already:

* a fold with no labeled rows is **unmeasured**, and writing its bucket as
  ``0.0`` (as older reports do) must not pull the period mean toward zero;
* a bucket mean is only read as a signal after it clears its own fold-to-fold
  standard error -- ten folds of a 1-3 pp noisy quantity rarely do.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bucket_decay_attribution.py"


def _load():
    spec = importlib.util.spec_from_file_location("bucket_decay_attribution", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attrib = _load()


def _fold(number: int, bucket, universe, rows: int, start: str = "2026-01-05") -> dict:
    metrics: dict[str, object] = {"test_labeled_rows": float(rows)}
    if bucket is not None:
        metrics["test_bucket_label_mean"] = bucket
        metrics["test_universe_label_mean"] = universe
        metrics["test_bucket_size"] = 30.0
    else:
        metrics["test_ic_mean"] = 0.0
    return {
        "fold": number,
        "test_start": start,
        "test_end": start,
        "metrics": metrics,
    }


def _report(folds: list[dict], name: str = "2026_ytd") -> dict:
    return {"periods": [{"period": {"name": name}, "folds": folds}]}


def test_empty_fold_is_not_counted_as_a_zero_reading():
    # Fold 3 is the forward-blind tail: no labels, but an older report wrote 0.0
    # into the bucket fields.  Averaging it in would report 1/3 of the truth.
    report = _report(
        [
            _fold(1, 0.02, 0.005, 1000),
            _fold(2, 0.04, 0.010, 1000),
            _fold(3, 0.0, 0.0, 0),
        ]
    )
    summary = attrib.summarise(attrib.collect(report))[0]
    assert summary["bucket_measured_folds"] == 2
    assert summary["folds"] == 3
    assert summary["bucket_mean"] == pytest.approx(0.03)
    assert summary["selection_mean"] == pytest.approx(0.0225)
    unmeasured = summary["unmeasured_folds"]
    assert [item["fold"] for item in unmeasured] == [3]
    assert "无标签样本" in unmeasured[0]["reason"]


def test_empty_fold_without_bucket_fields_is_simply_unmeasured():
    # The post-fix shape: the fields are absent entirely.  Same verdict, and no
    # complaint to raise, because there is nothing misleading to warn about.
    report = _report([_fold(1, 0.02, 0.005, 1000), _fold(2, None, None, 0)])
    summary = attrib.summarise(attrib.collect(report))[0]
    assert summary["bucket_measured_folds"] == 1
    assert summary["bucket_mean"] == pytest.approx(0.02)
    assert summary["unmeasured_folds"] == []


def test_selection_splits_the_bucket_into_breadth_and_ranking():
    report = _report([_fold(1, 0.010, 0.004, 1000)])
    row = attrib.collect(report)[0]
    assert row["universe"] == pytest.approx(0.004)
    assert row["selection"] == pytest.approx(0.006)
    summary = attrib.summarise([row])[0]
    assert summary["universe_mean"] == pytest.approx(0.004)
    assert summary["selection_mean"] == pytest.approx(0.006)


def test_a_noisy_mean_does_not_clear_its_own_standard_error():
    # Point estimate is positive, but the folds swing by several times its size.
    # This is the 2024 / 2026 shape, and the honest verdict is "inconclusive".
    stats = attrib._significance([0.06, -0.05, 0.04, -0.03, 0.02, -0.01, 0.03])
    assert stats["mean"] > 0
    assert stats["se"] is not None and stats["se"] > stats["mean"]
    assert stats["ci_low"] < 0 < stats["ci_high"]
    assert abs(stats["t"]) < attrib._t_critical(stats["n"] - 1)


def test_a_quiet_mean_clears_the_bar():
    stats = attrib._significance([0.010, 0.011, 0.009, 0.010, 0.012, 0.010])
    assert stats["t"] is not None and stats["t"] > attrib._t_critical(stats["n"] - 1)
    assert stats["ci_low"] > 0
    assert stats["positive_folds"] == 6


def test_welch_separates_a_shift_that_swamps_the_noise():
    left = [0.001, 0.002, 0.000, 0.001, 0.002]
    right = [0.050, 0.051, 0.049, 0.050, 0.052]
    test = attrib.welch(right, left)
    assert test is not None
    assert test["significant"] is True
    assert test["diff"] == pytest.approx(0.0492, abs=0.001)


def test_welch_declines_on_a_shift_inside_the_noise():
    left = [0.06, -0.05, 0.04, -0.03, 0.02]
    right = [0.03, -0.02, 0.01, -0.06, 0.05]
    test = attrib.welch(right, left)
    assert test is not None
    assert test["significant"] is False
    assert test["ci_low"] < 0 < test["ci_high"]


def test_missing_metric_fields_stay_unmeasured_rather_than_zero():
    report = _report([{"fold": 1, "test_start": "x", "test_end": "y", "metrics": {}}])
    row = attrib.collect(report)[0]
    assert row["bucket"] is None
    assert row["universe"] is None
    assert row["selection"] is None
    # A single unmeasured fold must not become a 0.0 average either.
    summary = attrib.summarise([row])[0]
    assert summary["bucket_mean"] is None
    assert summary["bucket_measured_folds"] == 0


def test_json_export_drops_internal_fold_arrays(tmp_path: Path):
    report = _report([_fold(1, 0.02, 0.005, 1000), _fold(2, 0.04, 0.010, 1000)])
    rows = attrib.collect(report)
    summary = attrib.summarise(rows)
    assert "_buckets" in summary[0]
    public = [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in summary]
    payload = json.dumps({"periods": public}, default=str)
    assert "_buckets" not in payload
    assert "bucket_mean" in payload
