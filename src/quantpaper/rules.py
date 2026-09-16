from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .domain import Board, OrderSide, QuoteSnapshot, Security


@dataclass(frozen=True)
class TradingRule:
    board: Board
    effective_from: date
    lot_size: int
    price_limit_pct: float
    t_plus_one: bool = True
    supports_margin: bool = False
    supports_short: bool = False


class TradingRules:
    """Versioned exchange rules used by the simulator.

    The values are intentionally kept in one place so a future exchange-rule
    change does not leak into the strategy or PaperBroker implementation.
    """

    def __init__(self, rules: list[TradingRule] | None = None) -> None:
        self._rules = rules or [
            TradingRule(Board.MAIN, date(1900, 1, 1), 100, 0.10, True, True, True),
            TradingRule(Board.CHINEXT_300, date(1900, 1, 1), 100, 0.10, True, True, True),
            TradingRule(Board.CHINEXT_300, date(2020, 8, 24), 100, 0.20, True, True, True),
            TradingRule(Board.STAR_688, date(2019, 7, 22), 200, 0.20, True, True, True),
        ]

    def for_security(self, security: Security, as_of: date | None = None) -> TradingRule:
        as_of = as_of or date.today()
        candidates = [
            item
            for item in self._rules
            if item.board == security.board and item.effective_from <= as_of
        ]
        if not candidates:
            raise ValueError(f"no trading rule for {security.board} at {as_of}")
        return max(candidates, key=lambda item: item.effective_from)

    def normalize_quantity(
        self,
        security: Security,
        quantity: int,
        side: OrderSide,
        as_of: date | None = None,
    ) -> int:
        rule = self.for_security(security, as_of)
        if quantity <= 0:
            return 0
        if side in {OrderSide.SELL, OrderSide.BUY_TO_COVER}:
            # Residual positions can be closed even when smaller than one lot.
            return quantity
        return quantity - quantity % rule.lot_size

    def price_limits(
        self, previous_close: float, security: Security, as_of: date | None = None
    ) -> tuple[float, float]:
        rule = self.for_security(security, as_of)
        upper = round(previous_close * (1 + rule.price_limit_pct), 2)
        lower = round(previous_close * (1 - rule.price_limit_pct), 2)
        return lower, upper

    def next_trading_day(self, value: date) -> date:
        """Return the next weekday for the first T+1 settlement approximation."""
        candidate = value + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate

    def can_fill(
        self,
        quote: QuoteSnapshot,
        side: OrderSide,
        execution_price: float | None = None,
    ) -> tuple[bool, str | None]:
        if quote.suspended:
            return False, "SUSPENDED"
        price = execution_price if execution_price is not None else quote.last
        if side in {OrderSide.BUY, OrderSide.BORROW_SELL} and (
            quote.is_limit_up
            or (quote.upper_limit is not None and price >= quote.upper_limit - 1e-6)
        ):
            return False, "LIMIT_UP"
        if side in {OrderSide.SELL, OrderSide.BUY_TO_COVER} and (
            quote.is_limit_down
            or (quote.lower_limit is not None and price <= quote.lower_limit + 1e-6)
        ):
            return False, "LIMIT_DOWN"
        return True, None
