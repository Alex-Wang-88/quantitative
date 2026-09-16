from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


class Board(StrEnum):
    MAIN = "MAIN"
    CHINEXT_300 = "CHINEXT_300"
    STAR_688 = "STAR_688"


def board_for_symbol(symbol: str) -> Board:
    """Classify an A-share code, including newer 301 ChiNext listings."""
    raw = str(symbol).strip().upper()
    digits = "".join(character for character in raw if character.isdigit())
    code = digits[-6:] if digits else raw
    if code.startswith("688"):
        return Board.STAR_688
    if code.startswith(("300", "301")):
        return Board.CHINEXT_300
    return Board.MAIN


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    BORROW_SELL = "BORROW_SELL"
    BUY_TO_COVER = "BUY_TO_COVER"


class OrderStatus(StrEnum):
    DRAFT = "DRAFT"
    CONFIRMED = "CONFIRMED"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class RunMode(StrEnum):
    PAPER_AUTO = "paper_auto"
    PAPER_MANUAL = "paper_manual"
    MANUAL_RECORD = "manual_record"


class DataMode(StrEnum):
    REPLAY = "REPLAY"
    LIVE = "LIVE"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class RunnerStatus(StrEnum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str
    exchange: str
    board: Board
    sector: str
    listed_date: str = "2010-01-01"
    is_st: bool = False
    suspended: bool = False
    marginable: bool = False
    shortable: bool = False


@dataclass(frozen=True)
class QuoteSnapshot:
    symbol: str
    timestamp: datetime
    last: float
    previous_close: float
    bid1: float | None = None
    ask1: float | None = None
    bid1_volume: int = 0
    ask1_volume: int = 0
    bar_volume: int = 0
    upper_limit: float | None = None
    lower_limit: float | None = None
    suspended: bool = False
    source: str = "REPLAY"

    @property
    def is_limit_up(self) -> bool:
        return self.upper_limit is not None and self.last >= self.upper_limit - 1e-6

    @property
    def is_limit_down(self) -> bool:
        return self.lower_limit is not None and self.last <= self.lower_limit + 1e-6


@dataclass
class Position:
    symbol: str
    name: str
    board: Board
    sector: str
    last_price: float
    long_qty: int = 0
    long_available_qty: int = 0
    long_pending_qty: int = 0
    long_avg_cost: float = 0.0
    short_qty: int = 0
    short_avg_cost: float = 0.0
    short_borrowed_qty: int = 0
    last_quote_at: datetime | None = None

    @property
    def long_market_value(self) -> float:
        return self.long_qty * self.last_price

    @property
    def short_market_value(self) -> float:
        return self.short_qty * self.last_price

    @property
    def market_value(self) -> float:
        return self.long_market_value - self.short_market_value


@dataclass
class AccountSnapshot:
    account_id: str
    initial_capital: float
    cash: float
    financing_debt: float
    financing_interest: float
    short_borrow_value: float
    long_market_value: float
    short_market_value: float
    total_equity: float
    collateral_value: float
    maintenance_ratio: float | None
    available_margin: float | None
    gross_exposure: float
    net_exposure: float
    as_of: datetime


@dataclass
class Signal:
    signal_id: str
    symbol: str
    name: str
    board: Board
    sector: str
    generated_at: datetime
    data_timestamp: datetime
    score: float
    action: str
    target_weight: float
    current_weight: float
    reason: str
    model_version: str
    status: str = "OPEN"
    run_id: str = ""
    strategy_version: str = "baseline-v0.1"
    updated_at: datetime | None = None


@dataclass
class TargetPosition:
    symbol: str
    target_weight: float
    target_value: float
    target_qty: int
    reason: str


@dataclass
class OrderIntent:
    order_id: str
    signal_id: str | None
    symbol: str
    name: str
    board: Board
    sector: str
    side: OrderSide
    quantity: int
    limit_price: float | None
    created_at: datetime
    eligible_at: datetime
    expires_at: datetime
    reason: str
    model_version: str
    status: OrderStatus = OrderStatus.DRAFT
    filled_quantity: int = 0
    average_fill_price: float | None = None
    confirmed_at: datetime | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    reject_reason: str | None = None
    is_margin_order: bool = False
    run_id: str = ""
    strategy_version: str = "baseline-v0.1"
    data_timestamp: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class PaperFill:
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    stamp_duty: float
    financing_cost: float
    borrow_cost: float
    timestamp: datetime
    price_source: str
    run_id: str = ""
    strategy_version: str = "baseline-v0.1"
    model_version: str = "unknown"
    data_timestamp: datetime | None = None


@dataclass
class RiskEvent:
    event_id: str
    severity: str
    code: str
    message: str
    created_at: datetime
    symbol: str | None = None
    acknowledged: bool = False
    run_id: str = ""


@dataclass
class DataStatus:
    mode: DataMode
    provider: str
    last_update: datetime | None
    symbol_count: int
    latency_seconds: float | None
    message: str


@dataclass
class DailyReport:
    report_date: str
    generated_at: datetime
    status: str
    model_version: str
    account: AccountSnapshot
    signals_count: int
    orders_count: int
    fills_count: int
    turnover: float
    summary: str
    risk_events: list[RiskEvent] = field(default_factory=list)
    run_id: str = ""
    strategy_version: str = "baseline-v0.1"
    model_name: str = "unknown"
    model_metrics: dict[str, float] = field(default_factory=dict)
    factor_coverage: float = 0.0
    data_status: str = "UNKNOWN"
    research_metrics: dict[str, Any] = field(default_factory=dict)


def jsonable(value: Any) -> Any:
    """Convert domain objects into JSON-safe values for the API and reports."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value
