from __future__ import annotations

from pathlib import Path

import pytest

from quantpaper.domain import (
    has_special_treatment_marker,
    is_target_a_share_symbol,
    security_status_disclosure,
)


def test_target_a_share_filter_includes_configured_boards_and_excludes_other_assets() -> None:
    assert is_target_a_share_symbol("000001")
    assert is_target_a_share_symbol("SZSE.301001")
    assert is_target_a_share_symbol("302132")
    assert is_target_a_share_symbol("SHSE.688001")
    assert is_target_a_share_symbol("689009")
    assert not is_target_a_share_symbol("200001")
    assert not is_target_a_share_symbol("900901")
    assert not is_target_a_share_symbol("830001")
    assert not is_target_a_share_symbol("399001")


def test_special_treatment_marker_covers_all_four_prefix_shapes() -> None:
    # All four shapes occur; a plain startswith(("ST", "*ST")) misses the last.
    assert has_special_treatment_marker("ST星源")
    assert has_special_treatment_marker("*ST美丽")
    assert has_special_treatment_marker("SST前锋")
    assert has_special_treatment_marker("S*ST托普")


def test_special_treatment_marker_is_case_and_whitespace_insensitive() -> None:
    assert has_special_treatment_marker("  st星源 ")
    assert has_special_treatment_marker("s*st托普")


def test_special_treatment_marker_does_not_fire_on_split_share_reform_alone() -> None:
    # "S" marks an un-reformed company, not a special treatment.  It shows up on
    # perfectly healthy names, so peeling it must not turn them into exclusions.
    # The first four are every S-prefixed healthy name in the current archive.
    assert not has_special_treatment_marker("S佳通")
    assert not has_special_treatment_marker("S兰铝")
    assert not has_special_treatment_marker("S山东铝")
    assert not has_special_treatment_marker("S湘火炬")
    assert not has_special_treatment_marker("招商银行")
    assert not has_special_treatment_marker("")


def test_pt_era_designation_is_deliberately_not_matched() -> None:
    """PT (1999-2002) predates the archive; suspension already covers it.

    Pinning this as intended behaviour keeps a later reader from "fixing" it
    into a silent universe change.
    """

    assert not has_special_treatment_marker("PT金田A")
    assert not has_special_treatment_marker("PT中浩A")


def test_security_status_disclosure_states_which_filters_ran() -> None:
    live = security_status_disclosure(
        st_applied=True,
        suspension_applied=True,
        source="security_master",
        basis="名称前缀 + is_suspended（当前快照）",
        special_treatment_excluded=7,
        suspension_excluded=3,
        universe_note="与研究工作池不同",
    )
    assert live["st_filter_applied"] is True
    assert live["suspension_filter_applied"] is True
    assert live["special_treatment_excluded"] == 7
    assert live["suspension_excluded"] == 3

    research = security_status_disclosure(
        st_applied=False,
        suspension_applied=False,
        source="UNAVAILABLE:is_st,suspended",
        basis="归档无该列",
    )
    # The absent filter must be as legible as the present one, with zero counts
    # rather than a missing key.
    assert research["st_filter_applied"] is False
    assert research["suspension_filter_applied"] is False
    assert research["special_treatment_excluded"] == 0
    assert research["universe_note"] == ""


def test_security_status_disclosure_coerces_types_to_stable_shapes() -> None:
    disclosure = security_status_disclosure(
        st_applied=1,  # type: ignore[arg-type]
        suspension_applied=0,  # type: ignore[arg-type]
        source=None,  # type: ignore[arg-type]
        basis=None,  # type: ignore[arg-type]
        special_treatment_excluded="5",  # type: ignore[arg-type]
        suspension_excluded=None,  # type: ignore[arg-type]
    )
    assert disclosure["st_filter_applied"] is True
    assert disclosure["suspension_filter_applied"] is False
    assert disclosure["special_treatment_excluded"] == 5
    assert disclosure["suspension_excluded"] == 0
    assert disclosure["security_status_source"] == ""


ARCHIVE_MASTER = Path("data/eastmoney/instrument_master.parquet")


@pytest.mark.skipif(not ARCHIVE_MASTER.exists(), reason="vendor archive not present")
def test_special_treatment_marker_recall_against_the_real_archive() -> None:
    """The predicate's real-world claim, checked against the 5.5k-name master.

    The archive ships no ``is_st`` column, so the name is the only signal this
    project has.  If the four-shape predicate regressed, these names would slip
    back into a research universe that already trades them.
    """

    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect()
    rows = connection.execute(
        f"select symbol, sec_name from read_parquet('{ARCHIVE_MASTER.as_posix()}')"
    ).fetchall()
    pool = [
        str(name or "")
        for symbol, name in rows
        if is_target_a_share_symbol(str(symbol or ""))
    ]
    matched = {name for name in pool if has_special_treatment_marker(name)}
    legacy = {
        name
        for name in pool
        if name.strip().upper().startswith(("ST", "*ST"))
    }
    # The S*ST shape is the whole reason the predicate is not a two-way startswith.
    assert matched - legacy == {
        "S*ST云大",
        "S*ST佳纸",
        "S*ST国瓷",
        "S*ST托普",
        "S*ST精密",
        "S*ST金荔",
        "S*ST龙昌",
    }
    # Sanity: the predicate must not be so loose that it sweeps the pool.
    assert 0.0 < len(matched) / len(pool) < 0.10

