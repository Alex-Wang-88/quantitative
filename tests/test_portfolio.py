"""Tests for the live portfolio engine's target construction.

The engine feeds the paper-trading runtime, which is a *different* code path from
the research backtester.  That split is exactly why these tests exist: the
industry-cap guard was fixed in ``backtest.py`` and the research runs then reached
~90% invested, while the live book stayed pinned at ~12% because
``PortfolioEngine`` still counted the ``"UNKNOWN"`` placeholder as a sector.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from quantpaper.config import PortfolioConfig
from quantpaper.domain import (
    PLACEHOLDER_SECTORS,
    AccountSnapshot,
    Board,
    Security,
    known_sectors,
    sector_coverage,
)
from quantpaper.portfolio import PortfolioEngine


def _account(equity: float = 1_000_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        account_id="test",
        initial_capital=equity,
        cash=equity,
        financing_debt=0.0,
        financing_interest=0.0,
        short_borrow_value=0.0,
        long_market_value=0.0,
        short_market_value=0.0,
        total_equity=equity,
        collateral_value=equity,
        maintenance_ratio=None,
        available_margin=equity,
        gross_exposure=0.0,
        net_exposure=0.0,
        as_of=datetime(2026, 9, 18, 15, 0, tzinfo=UTC),
    )


def _securities(count: int, sector_of) -> dict[str, Security]:
    return {
        f"{600000 + index:06d}": Security(
            symbol=f"{600000 + index:06d}",
            name=f"股票{index}",
            exchange="SHSE",
            board=Board.MAIN,
            sector=sector_of(index),
        )
        for index in range(count)
    }


def _scores(securities: dict[str, Security]) -> pd.DataFrame:
    symbols = list(securities)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "score": [float(len(symbols) - index) for index in range(len(symbols))],
        }
    )


def _prices(securities: dict[str, Security], price: float = 10.0) -> dict[str, float]:
    return {symbol: price for symbol in securities}


# --------------------------------------------------------------------------- #
# the placeholder-is-not-a-sector rule
# --------------------------------------------------------------------------- #


def test_unknown_industry_does_not_truncate_the_live_book() -> None:
    """The bug that pinned the paper trader at ~12% invested.

    With every name in the ``"UNKNOWN"`` bucket, enforcing a 25% cap caps the book
    at ``0.25 / 0.03 = 8`` names -- 24% invested -- no matter that the config asks
    for 30.  The cap has nothing to diversify when there is no classification, so
    it must stand down and the book must reach its configured size.
    """

    securities = _securities(40, sector_of=lambda index: "UNKNOWN")
    config = PortfolioConfig(max_positions=30, single_weight_cap=0.08)
    engine = PortfolioEngine(config, cash_floor=0.10)

    targets, _ = engine.build_targets(
        _scores(securities), _account(), securities, _prices(securities)
    )

    assert len(targets) == 30
    disclosure = engine.industry_disclosure
    assert disclosure["industry_cap_applied"] is False
    assert disclosure["sector_coverage"] == 0.0
    assert disclosure["sector_source"] == "UNAVAILABLE:sector"


@pytest.mark.parametrize("placeholder", ["UNKNOWN", "unknown", "", "  ", "NAN", "N/A"])
def test_every_placeholder_spelling_is_treated_as_unknown(placeholder: str) -> None:
    securities = _securities(40, sector_of=lambda index: placeholder)
    engine = PortfolioEngine(PortfolioConfig(max_positions=30), cash_floor=0.10)

    targets, _ = engine.build_targets(
        _scores(securities), _account(), securities, _prices(securities)
    )

    assert len(targets) == 30
    assert engine.industry_disclosure["industry_cap_applied"] is False


def test_a_real_classification_still_enforces_the_cap() -> None:
    """The guard must not disable the cap when the data is actually there."""

    securities = _securities(
        40, sector_of=lambda index: "银行" if index % 2 else "电子"
    )
    # base_weight = min(0.03, 0.9/30) = 0.03, cap 0.12 => 4 names per sector.
    config = PortfolioConfig(
        max_positions=30, single_weight_cap=0.03, industry_weight_cap=0.12
    )
    engine = PortfolioEngine(config, cash_floor=0.10)

    targets, _ = engine.build_targets(
        _scores(securities), _account(), securities, _prices(securities)
    )

    assert len(targets) == 8
    disclosure = engine.industry_disclosure
    assert disclosure["industry_cap_applied"] is True
    assert disclosure["distinct_known_sectors"] == 2
    assert disclosure["sector_coverage"] == 1.0
    assert disclosure["sector_source"] == "AVAILABLE:sector"


def test_partial_coverage_stands_the_cap_down() -> None:
    """Below the coverage threshold the placeholder bucket would swallow the book."""

    securities = _securities(
        40, sector_of=lambda index: "银行" if index % 2 else "UNKNOWN"
    )
    engine = PortfolioEngine(
        PortfolioConfig(max_positions=30, single_weight_cap=0.03), cash_floor=0.10
    )

    targets, _ = engine.build_targets(
        _scores(securities), _account(), securities, _prices(securities)
    )

    assert len(targets) == 30
    disclosure = engine.industry_disclosure
    assert disclosure["industry_cap_applied"] is False
    assert disclosure["sector_coverage"] == pytest.approx(0.5)


def test_disclosure_shape_matches_the_research_backtester() -> None:
    # Both paths must publish the same keys, otherwise a comparison between the
    # backtest book and the live book cannot even be tabulated.
    engine = PortfolioEngine(PortfolioConfig(max_positions=30), cash_floor=0.10)
    securities = _securities(40, sector_of=lambda index: "UNKNOWN")
    engine.build_targets(
        _scores(securities), _account(), securities, _prices(securities)
    )

    assert set(engine.industry_disclosure) == {
        "industry_cap_applied",
        "industry_cap_reason",
        "sector_coverage",
        "distinct_known_sectors",
        "sector_source",
    }


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def test_known_sectors_ignores_placeholders() -> None:
    assert known_sectors(["银行", "UNKNOWN", "", None, "电子"]) == {"银行", "电子"}
    assert known_sectors(["UNKNOWN"] * 5) == set()


def test_sector_coverage_counts_only_real_labels() -> None:
    assert sector_coverage([]) == 0.0
    assert sector_coverage(["银行", "UNKNOWN"]) == pytest.approx(0.5)
    assert sector_coverage(["银行", "电子"]) == pytest.approx(1.0)


def test_placeholder_set_is_the_contract() -> None:
    # ``"UNKNOWN"`` is what the eastmoney provider emits when it has no industry
    # field, so it must stay in the set even if others are edited.
    assert "UNKNOWN" in PLACEHOLDER_SECTORS
