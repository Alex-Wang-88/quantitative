from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import replace
from datetime import date, datetime, timedelta
from uuid import uuid4

from .config import AccountConfig, ExecutionConfig
from .domain import (
    AccountSnapshot,
    Board,
    OrderIntent,
    OrderSide,
    OrderStatus,
    PaperFill,
    Position,
    QuoteSnapshot,
    Security,
    utc_now,
)
from .ledger import Ledger
from .rules import TradingRules

logger = logging.getLogger(__name__)


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


class PaperBroker:
    """Deterministic paper broker with A-share rules and auditable accounting.

    This broker deliberately has no real-broker transport. It consumes quote
    snapshots supplied by a provider and writes every state transition to the
    local ledger when one is configured.
    """

    def __init__(
        self,
        securities: Iterable[Security],
        account_config: AccountConfig,
        execution_config: ExecutionConfig,
        rules: TradingRules | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.securities = {item.symbol: item for item in securities}
        self.account_config = account_config
        self.execution_config = execution_config
        self.rules = rules or TradingRules()
        self.ledger = ledger
        self.cash = account_config.initial_capital
        self.financing_debt = 0.0
        self.financing_interest = 0.0
        self.short_borrow_value = 0.0
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, OrderIntent] = {}
        self.fills: list[PaperFill] = []
        self.last_quotes: dict[str, QuoteSnapshot] = {}
        self._pending_long: dict[str, list[tuple[date, int]]] = {}
        self._borrowable = {
            item.symbol: 50_000 if item.shortable else 0 for item in self.securities.values()
        }

    def add_order(self, order: OrderIntent) -> OrderIntent:
        """Add an order and retain rejected orders for auditability."""
        if order.symbol not in self.securities:
            rejected = replace(
                order,
                status=OrderStatus.REJECTED,
                reject_reason="UNKNOWN_SECURITY",
                updated_at=utc_now(),
            )
            self.orders[order.order_id] = rejected
            self._event("order_rejected", rejected, order.order_id)
            return rejected
        if order.quantity <= 0:
            rejected = replace(
                order,
                status=OrderStatus.REJECTED,
                reject_reason="INVALID_QUANTITY",
                updated_at=utc_now(),
            )
            self.orders[order.order_id] = rejected
            self._event("order_rejected", rejected, order.order_id)
            return rejected
        if order.order_id in self.orders:
            raise ValueError(f"duplicate order id: {order.order_id}")
        stored = replace(order, updated_at=order.updated_at or utc_now())
        self.orders[stored.order_id] = stored
        self._event("order_created", stored, stored.order_id)
        self._persist_state()
        return stored

    def confirm_order(
        self,
        order_id: str,
        now: datetime | None = None,
        delay_seconds: int | None = None,
    ) -> OrderIntent:
        now = now or utc_now()
        order = self.orders[order_id]
        if order.status != OrderStatus.DRAFT:
            return order
        if now >= order.expires_at:
            return self._expire(order)
        delay = (
            self.execution_config.manual_confirmation_delay_seconds
            if delay_seconds is None
            else delay_seconds
        )
        confirmed = replace(
            order,
            status=OrderStatus.CONFIRMED,
            confirmed_at=now,
            eligible_at=now + timedelta(seconds=max(0, delay)),
            updated_at=now,
        )
        self.orders[order_id] = confirmed
        self._event("order_confirmed", confirmed, order_id)
        self._persist_state()
        return confirmed

    def reject_order(self, order_id: str, reason: str = "USER_REJECTED") -> OrderIntent:
        order = self.orders[order_id]
        if order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }:
            return order
        rejected = replace(
            order, status=OrderStatus.REJECTED, reject_reason=reason, updated_at=utc_now()
        )
        self.orders[order_id] = rejected
        self._event("order_rejected", rejected, order_id)
        self._persist_state()
        return rejected

    def cancel_order(self, order_id: str, reason: str = "USER_CANCELLED") -> OrderIntent:
        order = self.orders[order_id]
        if order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }:
            return order
        cancelled = replace(
            order, status=OrderStatus.CANCELLED, reject_reason=reason, updated_at=utc_now()
        )
        self.orders[order_id] = cancelled
        self._event("order_cancelled", cancelled, order_id)
        self._persist_state()
        return cancelled

    def modify_order(self, order_id: str, quantity: int) -> OrderIntent:
        order = self.orders[order_id]
        if order.status != OrderStatus.DRAFT:
            raise ValueError("only draft orders can be modified")
        security = self.securities[order.symbol]
        normalized = self.rules.normalize_quantity(
            security, quantity, order.side, order.created_at.date()
        )
        if normalized <= 0:
            raise ValueError("quantity is below the board lot")
        modified = replace(order, quantity=normalized, updated_at=utc_now())
        self.orders[order_id] = modified
        self._event("order_modified", modified, order_id)
        self._persist_state()
        return modified

    def auto_confirm_drafts(self, now: datetime | None = None) -> list[OrderIntent]:
        now = now or utc_now()
        result: list[OrderIntent] = []
        for order in list(self.orders.values()):
            if order.status != OrderStatus.DRAFT:
                continue
            if now >= order.expires_at:
                self._expire(order)
                continue
            result.append(self.confirm_order(order.order_id, now, delay_seconds=0))
        return result

    def process_quotes(
        self, quotes: Iterable[QuoteSnapshot], now: datetime | None = None
    ) -> list[PaperFill]:
        now = now or utc_now()
        quote_list = list(quotes)
        for quote in quote_list:
            self.last_quotes[quote.symbol] = quote
            position = self.positions.get(quote.symbol)
            if position:
                position.last_price = quote.last
                position.last_quote_at = quote.timestamp

        self._release_settled(now.date())
        fills: list[PaperFill] = []
        for order in list(self.orders.values()):
            if (
                order.status in {OrderStatus.DRAFT, OrderStatus.CONFIRMED}
                and now >= order.expires_at
            ):
                self._expire(order)
                continue
            if order.status == OrderStatus.CONFIRMED and now >= order.eligible_at:
                submitted = replace(
                    order, status=OrderStatus.SUBMITTED, submitted_at=now, updated_at=now
                )
                self.orders[order.order_id] = submitted
                self._event("order_submitted", submitted, order.order_id)
            order = self.orders[order.order_id]
            if order.status not in {OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED}:
                continue
            if now >= order.expires_at:
                self._expire(order)
                continue
            quote = self.last_quotes.get(order.symbol)
            if quote is None or quote.timestamp < order.eligible_at:
                continue
            fill = self._try_fill(order, quote, now)
            if fill:
                fills.append(fill)
        self._persist_state()
        return fills

    def accrue_daily_costs(self, days: int = 1, now: datetime | None = None) -> None:
        if days <= 0:
            return
        daily_financing = self.financing_debt * self.execution_config.financing_rate / 252 * days
        short_value = sum(item.short_market_value for item in self.positions.values())
        daily_borrow = short_value * self.execution_config.borrow_rate / 252 * days
        self.financing_interest += daily_financing
        self.short_borrow_value += daily_borrow
        self._event("daily_cost_accrued", {"financing": daily_financing, "borrow": daily_borrow})
        self._persist_state()

    def snapshot(self, now: datetime | None = None) -> AccountSnapshot:
        now = now or utc_now()
        long_value = sum(item.long_market_value for item in self.positions.values())
        short_value = sum(item.short_market_value for item in self.positions.values())
        collateral = max(0.0, self.cash + long_value)
        margin_debt = self.financing_debt + self.financing_interest
        denominator = margin_debt + short_value
        maintenance = collateral / denominator if denominator > 0 else None
        available = max(0.0, collateral - denominator)
        equity = self.cash + long_value - short_value - margin_debt - self.short_borrow_value
        capital_base = max(self.account_config.initial_capital, 1.0)
        gross = (long_value + short_value) / capital_base
        net = (long_value - short_value) / capital_base
        return AccountSnapshot(
            account_id=self.account_config.name,
            initial_capital=self.account_config.initial_capital,
            cash=round(self.cash, 2),
            financing_debt=round(self.financing_debt, 2),
            financing_interest=round(self.financing_interest, 2),
            short_borrow_value=round(self.short_borrow_value, 2),
            long_market_value=round(long_value, 2),
            short_market_value=round(short_value, 2),
            total_equity=round(equity, 2),
            collateral_value=round(collateral, 2),
            maintenance_ratio=None if maintenance is None else round(maintenance, 4),
            available_margin=round(available, 2),
            gross_exposure=round(gross, 4),
            net_exposure=round(net, 4),
            as_of=now,
        )

    def restore_state(self) -> bool:
        """Restore broker state from snapshots; return False for a fresh account."""
        if self.ledger is None:
            return False
        account = self.ledger.load_snapshot("account")
        if not account:
            return False
        self.cash = float(account.get("cash", self.cash))
        self.financing_debt = float(account.get("financing_debt", 0.0))
        self.financing_interest = float(account.get("financing_interest", 0.0))
        self.short_borrow_value = float(account.get("short_borrow_value", 0.0))
        self.positions = {}
        board_migrated = False
        for raw in self.ledger.load_snapshot("positions") or []:
            stored_board = Board(raw["board"])
            security = self.securities.get(raw["symbol"])
            board = security.board if security is not None else stored_board
            board_migrated = board_migrated or board != stored_board
            position = Position(
                symbol=raw["symbol"],
                name=raw["name"],
                board=board,
                sector=raw["sector"],
                last_price=float(raw["last_price"]),
                long_qty=int(raw.get("long_qty", 0)),
                long_available_qty=int(raw.get("long_available_qty", 0)),
                long_pending_qty=int(raw.get("long_pending_qty", 0)),
                long_avg_cost=float(raw.get("long_avg_cost", 0.0)),
                short_qty=int(raw.get("short_qty", 0)),
                short_avg_cost=float(raw.get("short_avg_cost", 0.0)),
                short_borrowed_qty=int(raw.get("short_borrowed_qty", 0)),
                last_quote_at=_parse_datetime(raw.get("last_quote_at")),
            )
            self.positions[position.symbol] = position
        self.orders = {}
        for raw in self.ledger.load_snapshot("orders") or []:
            stored_board = Board(raw["board"])
            security = self.securities.get(raw["symbol"])
            board = security.board if security is not None else stored_board
            board_migrated = board_migrated or board != stored_board
            order = OrderIntent(
                order_id=raw["order_id"],
                signal_id=raw.get("signal_id"),
                symbol=raw["symbol"],
                name=raw["name"],
                board=board,
                sector=raw["sector"],
                side=OrderSide(raw["side"]),
                quantity=int(raw["quantity"]),
                limit_price=raw.get("limit_price"),
                created_at=_parse_datetime(raw["created_at"]) or utc_now(),
                eligible_at=_parse_datetime(raw["eligible_at"]) or utc_now(),
                expires_at=_parse_datetime(raw["expires_at"]) or utc_now(),
                reason=raw["reason"],
                model_version=raw.get("model_version", "unknown"),
                status=OrderStatus(raw.get("status", OrderStatus.DRAFT.value)),
                filled_quantity=int(raw.get("filled_quantity", 0)),
                average_fill_price=raw.get("average_fill_price"),
                confirmed_at=_parse_datetime(raw.get("confirmed_at")),
                submitted_at=_parse_datetime(raw.get("submitted_at")),
                filled_at=_parse_datetime(raw.get("filled_at")),
                reject_reason=raw.get("reject_reason"),
                is_margin_order=bool(raw.get("is_margin_order", False)),
                run_id=raw.get("run_id", ""),
                strategy_version=raw.get("strategy_version", "baseline-v0.1"),
                data_timestamp=_parse_datetime(raw.get("data_timestamp")),
                updated_at=_parse_datetime(raw.get("updated_at")),
            )
            self.orders[order.order_id] = order
        self.fills = []
        for raw in self.ledger.load_snapshot("fills") or []:
            self.fills.append(
                PaperFill(
                    fill_id=raw["fill_id"],
                    order_id=raw["order_id"],
                    symbol=raw["symbol"],
                    side=OrderSide(raw["side"]),
                    quantity=int(raw["quantity"]),
                    price=float(raw["price"]),
                    commission=float(raw.get("commission", 0.0)),
                    stamp_duty=float(raw.get("stamp_duty", 0.0)),
                    financing_cost=float(raw.get("financing_cost", 0.0)),
                    borrow_cost=float(raw.get("borrow_cost", 0.0)),
                    timestamp=_parse_datetime(raw["timestamp"]) or utc_now(),
                    price_source=raw.get("price_source", "unknown"),
                    run_id=raw.get("run_id", ""),
                    strategy_version=raw.get("strategy_version", "baseline-v0.1"),
                    model_version=raw.get("model_version", "unknown"),
                    data_timestamp=_parse_datetime(raw.get("data_timestamp")),
                )
            )
        self._pending_long = {
            symbol: [
                (date.fromisoformat(item["available_after"]), int(item["quantity"]))
                for item in rows
            ]
            for symbol, rows in (self.ledger.load_snapshot("pending_long") or {}).items()
        }
        self._rebuild_borrowable()
        if board_migrated:
            self._persist_state()
        return True

    def _try_fill(
        self, order: OrderIntent, quote: QuoteSnapshot, now: datetime
    ) -> PaperFill | None:
        security = self.securities.get(order.symbol)
        if security is None:
            self._reject(order, "UNKNOWN_SECURITY")
            return None
        if order.side == OrderSide.BORROW_SELL and not security.shortable:
            self._reject(order, "NOT_SHORTABLE")
            return None
        if (
            order.side == OrderSide.BUY
            and not security.marginable
            and not self._has_cash_for_order(order, quote)
        ):
            self._reject(order, "NOT_MARGINABLE")
            return None
        if order.side == OrderSide.BUY_TO_COVER:
            position = self.positions.get(order.symbol)
            if not position or position.short_qty <= 0:
                self._reject(order, "NO_SHORT_POSITION")
                return None
        if order.side == OrderSide.SELL:
            position = self.positions.get(order.symbol)
            if not position or position.long_available_qty <= 0:
                self._reject(order, "T_PLUS_ONE_OR_NO_POSITION")
                return None

        if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}:
            base_price = quote.ask1 or quote.last
        else:
            base_price = quote.bid1 or quote.last
        if order.limit_price is not None:
            if (
                order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}
                and base_price > order.limit_price + 1e-6
            ):
                return None
            if (
                order.side in {OrderSide.SELL, OrderSide.BORROW_SELL}
                and base_price < order.limit_price - 1e-6
            ):
                return None
        slippage = self.execution_config.slippage_bps / 10_000
        if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}:
            price = round(base_price * (1 + slippage), 2)
        else:
            price = round(base_price * (1 - slippage), 2)
        if order.limit_price is not None:
            if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}:
                price = min(price, order.limit_price)
            elif order.side in {OrderSide.SELL, OrderSide.BORROW_SELL}:
                price = max(price, order.limit_price)
        can_fill, _ = self.rules.can_fill(quote, order.side, price)
        if not can_fill:
            return None

        participation_volume = max(
            1, int(quote.bar_volume * self.execution_config.max_participation_pct)
        )
        book_volume = (
            quote.ask1_volume
            if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}
            else quote.bid1_volume
        )
        available_volume = (
            min(book_volume, participation_volume) if book_volume > 0 else participation_volume
        )
        remaining = order.quantity - order.filled_quantity
        quantity = min(remaining, available_volume)
        position = self.positions.get(order.symbol)
        if order.side == OrderSide.SELL and position:
            quantity = min(quantity, position.long_available_qty)
        if order.side == OrderSide.BUY_TO_COVER and position:
            quantity = min(quantity, position.short_qty)
        if order.side == OrderSide.BORROW_SELL:
            quantity = min(quantity, self._borrowable.get(order.symbol, 0))
        quantity = self.rules.normalize_quantity(
            security, quantity, order.side, quote.timestamp.date()
        )
        if quantity <= 0:
            return None

        notional = quantity * price
        commission = round(notional * self.execution_config.commission_rate, 2)
        stamp = (
            round(notional * self.execution_config.stamp_duty_rate, 2)
            if order.side in {OrderSide.SELL, OrderSide.BORROW_SELL}
            else 0.0
        )
        total_cost = notional + commission + stamp
        if order.side in {OrderSide.BUY, OrderSide.BORROW_SELL} and not self._within_gross_limit(
            notional
        ):
            self._reject(order, "MAX_GROSS_EXPOSURE")
            return None
        if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}:
            if total_cost > self.cash + 1e-6:
                needed = total_cost - max(self.cash, 0.0)
                margin_allowed = self.account_config.margin_enabled and (
                    order.side == OrderSide.BUY_TO_COVER or security.marginable
                )
                if not margin_allowed:
                    self._reject(order, "INSUFFICIENT_CASH")
                    return None
                self.financing_debt += needed
                self.cash = max(0.0, self.cash)
            self.cash -= min(self.cash, total_cost)
        else:
            self.cash += notional - commission - stamp

        if position is None:
            position = Position(order.symbol, order.name, order.board, order.sector, price)
            self.positions[order.symbol] = position
        position.last_price = price
        position.last_quote_at = quote.timestamp
        if order.side == OrderSide.BUY:
            old_qty = position.long_qty
            position.long_qty += quantity
            position.long_pending_qty += quantity
            position.long_avg_cost = self._weighted_average(
                position.long_avg_cost, old_qty, price, quantity
            )
            self._pending_long.setdefault(order.symbol, []).append(
                (self.rules.next_trading_day(quote.timestamp.date()), quantity)
            )
        elif order.side == OrderSide.SELL:
            position.long_qty -= quantity
            position.long_available_qty -= quantity
        elif order.side == OrderSide.BORROW_SELL:
            old_qty = position.short_qty
            position.short_qty += quantity
            position.short_borrowed_qty += quantity
            self._borrowable[order.symbol] = max(
                0, self._borrowable.get(order.symbol, 0) - quantity
            )
            position.short_avg_cost = self._weighted_average(
                position.short_avg_cost, old_qty, price, quantity
            )
        else:
            position.short_qty -= quantity
            position.short_borrowed_qty -= quantity
            self._borrowable[order.symbol] = self._borrowable.get(order.symbol, 0) + quantity

        fill = PaperFill(
            fill_id=str(uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            stamp_duty=stamp,
            financing_cost=0.0,
            borrow_cost=0.0,
            timestamp=now,
            price_source=(
                "ask1"
                if order.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER} and quote.ask1
                else "bid1"
                if quote.bid1
                else "last"
            ),
            run_id=order.run_id,
            strategy_version=order.strategy_version,
            model_version=order.model_version,
            data_timestamp=quote.timestamp,
        )
        new_filled = order.filled_quantity + quantity
        average = self._weighted_average(
            order.average_fill_price or 0.0, order.filled_quantity, price, quantity
        )
        status = OrderStatus.FILLED if new_filled >= order.quantity else OrderStatus.PARTIAL_FILLED
        updated = replace(
            order,
            filled_quantity=new_filled,
            average_fill_price=average,
            status=status,
            filled_at=now if status == OrderStatus.FILLED else None,
            updated_at=now,
        )
        self.orders[order.order_id] = updated
        self.fills.append(fill)
        self._event("paper_fill", fill, order.order_id)
        return fill

    def _has_cash_for_order(self, order: OrderIntent, quote: QuoteSnapshot) -> bool:
        price = quote.ask1 or quote.last
        return (
            order.quantity * price * (1 + self.execution_config.commission_rate) <= self.cash + 1e-6
        )

    def _within_gross_limit(self, additional_notional: float) -> bool:
        current = sum(
            position.long_market_value + position.short_market_value
            for position in self.positions.values()
        )
        capital = max(self.account_config.initial_capital, 1.0)
        return (
            current + additional_notional
        ) / capital <= self.account_config.max_gross_exposure + 1e-9

    def _release_settled(self, current_date: date) -> None:
        for symbol, lots in list(self._pending_long.items()):
            position = self.positions.get(symbol)
            if position is None:
                continue
            still_pending: list[tuple[date, int]] = []
            for available_after, quantity in lots:
                if current_date >= available_after:
                    position.long_available_qty += quantity
                    position.long_pending_qty = max(0, position.long_pending_qty - quantity)
                else:
                    still_pending.append((available_after, quantity))
            if still_pending:
                self._pending_long[symbol] = still_pending
            else:
                self._pending_long.pop(symbol, None)

    def _rebuild_borrowable(self) -> None:
        self._borrowable = {
            item.symbol: 50_000 if item.shortable else 0 for item in self.securities.values()
        }
        for symbol, position in self.positions.items():
            self._borrowable[symbol] = max(
                0, self._borrowable.get(symbol, 0) - position.short_borrowed_qty
            )

    def _expire(self, order: OrderIntent) -> OrderIntent:
        expired = replace(order, status=OrderStatus.EXPIRED, updated_at=utc_now())
        self.orders[order.order_id] = expired
        self._event("order_expired", expired, order.order_id)
        return expired

    @staticmethod
    def _weighted_average(
        old_average: float, old_quantity: int, price: float, quantity: int
    ) -> float:
        total = old_quantity + quantity
        return round((old_average * old_quantity + price * quantity) / total, 4) if total else 0.0

    def _reject(self, order: OrderIntent, reason: str) -> None:
        updated = replace(
            order, status=OrderStatus.REJECTED, reject_reason=reason, updated_at=utc_now()
        )
        self.orders[order.order_id] = updated
        self._event("order_rejected", updated, order.order_id)

    def _event(self, event_type: str, payload: object, entity_id: str | None = None) -> None:
        if self.ledger:
            self.ledger.append_event(event_type, payload, entity_id)

    def _persist_state(self) -> None:
        if not self.ledger:
            return
        self.ledger.save_snapshot("account", self.snapshot())
        self.ledger.save_snapshot("positions", list(self.positions.values()))
        self.ledger.save_snapshot("orders", list(self.orders.values()))
        self.ledger.save_snapshot("fills", self.fills[-500:])
        self.ledger.save_snapshot(
            "pending_long",
            {
                symbol: [
                    {"available_after": available_after.isoformat(), "quantity": quantity}
                    for available_after, quantity in lots
                ]
                for symbol, lots in self._pending_long.items()
            },
        )
