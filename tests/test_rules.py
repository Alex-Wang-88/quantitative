from datetime import date

from quantpaper.domain import Board, OrderSide, Security, board_for_symbol
from quantpaper.rules import TradingRules


def security(board: Board) -> Security:
    return Security(
        symbol="TEST",
        name="测试",
        exchange="SSE" if board == Board.STAR_688 else "SZSE",
        board=board,
        sector="测试",
    )


def test_board_lots_and_residual_sell() -> None:
    rules = TradingRules()
    assert (
        rules.normalize_quantity(security(Board.MAIN), 250, OrderSide.BUY, date(2026, 1, 5)) == 200
    )
    assert (
        rules.normalize_quantity(security(Board.CHINEXT_300), 250, OrderSide.BUY, date(2026, 1, 5))
        == 200
    )
    assert (
        rules.normalize_quantity(security(Board.STAR_688), 250, OrderSide.BUY, date(2026, 1, 5))
        == 200
    )
    assert (
        rules.normalize_quantity(security(Board.MAIN), 50, OrderSide.SELL, date(2026, 1, 5)) == 50
    )


def test_next_trading_day_skips_weekend() -> None:
    rules = TradingRules()
    assert rules.next_trading_day(date(2026, 1, 9)) == date(2026, 1, 12)


def test_board_classifier_includes_301_chinext_codes() -> None:
    assert board_for_symbol("301108") == Board.CHINEXT_300
    assert board_for_symbol("SZSE.301108") == Board.CHINEXT_300
