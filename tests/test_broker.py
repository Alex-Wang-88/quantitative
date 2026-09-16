from datetime import UTC, datetime, timedelta

from quantpaper.broker import PaperBroker
from quantpaper.config import AccountConfig, ExecutionConfig
from quantpaper.domain import (
    Board,
    OrderIntent,
    OrderSide,
    OrderStatus,
    Position,
    QuoteSnapshot,
    Security,
)
from quantpaper.ledger import Ledger
from quantpaper.rules import TradingRules

UTC = UTC
START = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)


def make_security(
    symbol: str = "600001",
    board: Board = Board.MAIN,
    *,
    marginable: bool = True,
    shortable: bool = False,
) -> Security:
    return Security(
        symbol=symbol,
        name="测试股票",
        exchange="SSE" if symbol.startswith("6") else "SZSE",
        board=board,
        sector="测试",
        marginable=marginable,
        shortable=shortable,
    )


def make_quote(
    symbol: str = "600001",
    when: datetime = START,
    price: float = 10.0,
    *,
    volume: int = 100_000,
    upper: float = 11.0,
    lower: float = 9.0,
) -> QuoteSnapshot:
    return QuoteSnapshot(
        symbol=symbol,
        timestamp=when,
        last=price,
        previous_close=10.0,
        bid1=price - 0.01,
        ask1=price + 0.01,
        bid1_volume=volume,
        ask1_volume=volume,
        bar_volume=volume,
        upper_limit=upper,
        lower_limit=lower,
    )


def make_broker(
    securities: list[Security] | None = None,
    *,
    capital: float = 100_000.0,
    max_gross_exposure: float = 1.0,
    ledger: Ledger | None = None,
) -> PaperBroker:
    return PaperBroker(
        securities or [make_security()],
        AccountConfig(initial_capital=capital, max_gross_exposure=max_gross_exposure),
        ExecutionConfig(
            manual_confirmation_delay_seconds=0,
            slippage_bps=0,
            commission_rate=0,
            stamp_duty_rate=0,
            max_participation_pct=0.10,
        ),
        TradingRules(),
        ledger,
    )


def make_order(
    *,
    order_id: str,
    side: OrderSide = OrderSide.BUY,
    symbol: str = "600001",
    quantity: int = 100,
    created_at: datetime = START,
    expires_after: int = 300,
) -> OrderIntent:
    return OrderIntent(
        order_id=order_id,
        signal_id=None,
        symbol=symbol,
        name="测试股票",
        board=Board.MAIN,
        sector="测试",
        side=side,
        quantity=quantity,
        limit_price=None,
        created_at=created_at,
        eligible_at=created_at,
        expires_at=created_at + timedelta(seconds=expires_after),
        reason="测试",
        model_version="test-model",
    )


def submit_and_fill(broker: PaperBroker, order: OrderIntent, quote: QuoteSnapshot) -> None:
    broker.add_order(order)
    broker.confirm_order(order.order_id, quote.timestamp, delay_seconds=0)
    fills = broker.process_quotes([quote], quote.timestamp + timedelta(seconds=1))
    assert fills


def test_t_plus_one_blocks_same_day_sale_and_releases_next_day() -> None:
    broker = make_broker()
    buy = make_order(order_id="buy-1")
    submit_and_fill(broker, buy, make_quote())
    position = broker.positions["600001"]
    assert position.long_qty == 100
    assert position.long_available_qty == 0
    assert position.long_pending_qty == 100

    sell = make_order(order_id="sell-same-day", side=OrderSide.SELL)
    broker.add_order(sell)
    broker.confirm_order(sell.order_id, START + timedelta(minutes=1), delay_seconds=0)
    broker.process_quotes(
        [make_quote(when=START + timedelta(minutes=1))], START + timedelta(minutes=1, seconds=1)
    )
    assert broker.orders[sell.order_id].status == OrderStatus.REJECTED
    assert broker.orders[sell.order_id].reject_reason == "T_PLUS_ONE_OR_NO_POSITION"

    next_day = datetime(2026, 1, 6, 9, 30, tzinfo=UTC)
    broker.process_quotes([make_quote(when=next_day)], next_day)
    assert broker.positions["600001"].long_available_qty == 100

    sell_next_day = make_order(order_id="sell-next-day", side=OrderSide.SELL, created_at=next_day)
    submit_and_fill(broker, sell_next_day, make_quote(when=next_day, price=10.2))
    assert broker.positions["600001"].long_qty == 0


def test_margin_buy_does_not_make_cash_negative() -> None:
    broker = make_broker(capital=1_000, max_gross_exposure=3.0)
    order = make_order(order_id="margin-buy", quantity=200)
    submit_and_fill(broker, order, make_quote())
    assert broker.cash == 0
    assert broker.financing_debt > 1_000
    assert broker.snapshot().total_equity > 0


def test_safe_account_rejects_gross_exposure_over_one() -> None:
    broker = make_broker(capital=1_000)
    order = make_order(order_id="over-limit", quantity=200)
    broker.add_order(order)
    broker.confirm_order(order.order_id, START, delay_seconds=0)
    broker.process_quotes([make_quote()], START + timedelta(seconds=1))
    assert broker.orders[order.order_id].status == OrderStatus.REJECTED
    assert broker.orders[order.order_id].reject_reason == "MAX_GROSS_EXPOSURE"


def test_partial_fill_respects_participation() -> None:
    broker = make_broker()
    order = make_order(order_id="partial", quantity=300)
    broker.add_order(order)
    broker.confirm_order(order.order_id, START, delay_seconds=0)
    first = make_quote(volume=1_000)
    fills = broker.process_quotes([first], START + timedelta(seconds=1))
    assert fills[0].quantity == 100
    assert broker.orders[order.order_id].status == OrderStatus.PARTIAL_FILLED
    second = make_quote(when=START + timedelta(minutes=1), volume=1_000)
    broker.process_quotes([second], second.timestamp)
    assert broker.orders[order.order_id].filled_quantity == 200


def test_one_minute_quote_fill_records_data_and_execution_times() -> None:
    broker = make_broker()
    order = make_order(order_id="one-minute", quantity=100)
    broker.add_order(order)
    confirmed_at = START + timedelta(seconds=30)
    broker.confirm_order(order.order_id, confirmed_at, delay_seconds=0)

    quote_time = START + timedelta(minutes=1)
    quote = make_quote(when=quote_time, price=10.2, volume=1_000)
    execution_time = quote_time + timedelta(seconds=2)
    fills = broker.process_quotes([quote], execution_time)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.data_timestamp == quote_time
    assert fill.timestamp == execution_time
    assert fill.timestamp > fill.data_timestamp
    assert fill.price_source == "ask1"
    assert broker.orders[order.order_id].status == OrderStatus.FILLED


def test_limit_up_and_expired_order_do_not_fill() -> None:
    broker = make_broker()
    limit_order = make_order(order_id="limit-up")
    broker.add_order(limit_order)
    broker.confirm_order(limit_order.order_id, START, delay_seconds=0)
    quote = make_quote(price=11.0, upper=11.0)
    broker.process_quotes([quote], START + timedelta(seconds=1))
    assert not broker.fills
    assert broker.orders[limit_order.order_id].status == OrderStatus.SUBMITTED

    expired = make_order(order_id="expired", created_at=START, expires_after=1)
    broker.add_order(expired)
    broker.process_quotes(
        [make_quote(when=START + timedelta(seconds=5))], START + timedelta(seconds=5)
    )
    assert broker.orders[expired.order_id].status == OrderStatus.EXPIRED


def test_margin_short_and_cover() -> None:
    symbol = "688001"
    broker = make_broker([make_security(symbol, Board.STAR_688, shortable=True)])
    short = make_order(order_id="short", symbol=symbol, side=OrderSide.BORROW_SELL, quantity=200)
    short = OrderIntent(**{**short.__dict__, "board": Board.STAR_688})
    submit_and_fill(broker, short, make_quote(symbol, price=10.0, upper=12.0, lower=8.0))
    assert broker.positions[symbol].short_qty == 200
    cover = make_order(order_id="cover", symbol=symbol, side=OrderSide.BUY_TO_COVER, quantity=200)
    cover = OrderIntent(**{**cover.__dict__, "board": Board.STAR_688})
    submit_and_fill(broker, cover, make_quote(symbol, price=9.5, upper=12.0, lower=8.0))
    assert broker.positions[symbol].short_qty == 0


def test_ledger_restore_does_not_lose_orders_or_positions(tmp_path) -> None:
    ledger = Ledger(tmp_path / "paper.db")
    broker = make_broker(ledger=ledger)
    order = make_order(order_id="persisted")
    submit_and_fill(broker, order, make_quote())
    restored = make_broker(ledger=ledger)
    assert restored.restore_state()
    assert restored.positions["600001"].long_qty == 100
    assert restored.positions["600001"].long_available_qty == 0
    assert restored.orders[order.order_id].status == OrderStatus.FILLED
    assert len(restored.fills) == 1


def test_ledger_restore_migrates_legacy_301_board_metadata(tmp_path) -> None:
    ledger = Ledger(tmp_path / "paper.db")
    security = make_security("301108", Board.CHINEXT_300)
    source = make_broker([security], ledger=ledger)
    source._persist_state()
    ledger.save_snapshot(
        "positions",
        [
            Position(
                symbol="301108",
                name="测试创业板",
                board=Board.MAIN,
                sector="测试",
                last_price=10.0,
                long_qty=100,
                long_available_qty=100,
                long_avg_cost=10.0,
            )
        ],
    )

    restored = make_broker([security], ledger=ledger)
    assert restored.restore_state()
    assert restored.positions["301108"].board == Board.CHINEXT_300
    persisted = ledger.load_snapshot("positions")
    assert persisted[0]["board"] == Board.CHINEXT_300.value
