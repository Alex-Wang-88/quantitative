from __future__ import annotations

from datetime import UTC, datetime

from quantpaper.config import AccountConfig, PortfolioConfig
from quantpaper.domain import AccountSnapshot, Board, Position
from quantpaper.risk import RiskEngine


def _snapshot(equity: float = 1_000_000.0, cash: float = 500_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        account_id="safe_paper",
        initial_capital=1_000_000.0,
        cash=cash,
        financing_debt=0.0,
        financing_interest=0.0,
        short_borrow_value=0.0,
        long_market_value=equity - cash,
        short_market_value=0.0,
        total_equity=equity,
        collateral_value=equity,
        maintenance_ratio=None,
        available_margin=None,
        gross_exposure=(equity - cash) / equity,
        net_exposure=(equity - cash) / equity,
        as_of=datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
    )


def _positions(count: int) -> dict[str, Position]:
    return {
        f"6000{index:02d}": Position(
            symbol=f"6000{index:02d}",
            name=f"测试{index}",
            board=Board.MAIN,
            sector="测试",
            last_price=10.0,
            long_qty=100,
        )
        for index in range(count)
    }


def _engine(min_positions: int = 20, max_positions: int = 30) -> RiskEngine:
    return RiskEngine(
        AccountConfig(),
        PortfolioConfig(
            min_positions=min_positions, max_positions=max_positions, candidate_pool_size=500
        ),
    )


def _codes(events: list) -> set[str]:
    return {event.code for event in events}


def test_position_count_floor_reports_a_collapsed_book() -> None:
    """``min_positions`` existed in config and was read by nothing.

    Its absence made a book that collapsed to a handful of names look exactly
    like a prudent one -- which is the failure it was written to name.  This is
    the guard that would have caught the live book sitting at four positions.
    """

    events = _engine(min_positions=20).evaluate(_snapshot(), _positions(4), open_order_count=0)

    floor = [event for event in events if event.code == "POSITION_COUNT_FLOOR"]
    assert len(floor) == 1
    assert floor[0].severity == "MEDIUM"
    assert "4" in floor[0].message
    assert "20" in floor[0].message


def test_position_count_floor_is_silent_at_or_above_the_floor() -> None:
    assert "POSITION_COUNT_FLOOR" not in _codes(
        _engine(min_positions=20).evaluate(_snapshot(), _positions(20), open_order_count=0)
    )
    assert "POSITION_COUNT_FLOOR" not in _codes(
        _engine(min_positions=20).evaluate(_snapshot(), _positions(30), open_order_count=0)
    )


def test_position_count_floor_ignores_a_book_that_is_not_built_yet() -> None:
    """Zero positions is a different state, not a breach.

    Reporting it would fire on every session that opens flat, or after a full
    liquidation, which is noise rather than a control.
    """

    assert "POSITION_COUNT_FLOOR" not in _codes(
        _engine(min_positions=20).evaluate(_snapshot(), {}, open_order_count=0)
    )


def test_position_count_floor_is_disabled_at_zero() -> None:
    """``0`` means "no floor", which is what research configs want."""

    assert "POSITION_COUNT_FLOOR" not in _codes(
        _engine(min_positions=0).evaluate(_snapshot(), _positions(1), open_order_count=0)
    )


def test_position_count_ceiling_still_reports_an_overfull_book() -> None:
    assert "POSITION_COUNT" in _codes(
        _engine(min_positions=20).evaluate(_snapshot(), _positions(31), open_order_count=0)
    )
