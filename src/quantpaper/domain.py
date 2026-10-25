from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

# Sector labels that carry no real classification.  The eastmoney provider falls
# back to ``"UNKNOWN"`` when no industry field is available, and the research
# archive ships neither an industry field nor an ``industries/`` directory.
# A placeholder is not a sector: treating it as one lumps the entire universe
# into a single bucket, and an industry cap then silently truncates the book
# (0.25 / 0.03 = 8 names = 24% invested) however many positions the
# configuration asks for.  Both the research backtester and the live portfolio
# engine must therefore agree on what "known" means.
PLACEHOLDER_SECTORS = frozenset({"", "UNKNOWN", "NONE", "NAN", "N/A", "NULL"})

# An industry cap may only be enforced when the classification actually covers
# the candidate universe.  Below this share the cap diversifies nothing and only
# truncates the book, so it must be stood down and disclosed instead.
INDUSTRY_CAP_MIN_COVERAGE = 0.95


def known_sectors(values: Iterable[str]) -> set[str]:
    """Return the distinct sectors that carry a real classification."""

    return {
        value.strip().upper()
        for value in values
        if value is not None and str(value).strip().upper() not in PLACEHOLDER_SECTORS
    }


def sector_coverage(values: Iterable[str]) -> float:
    """Share of ``values`` that carry a real (non-placeholder) sector label."""

    materialised = list(values)
    if not materialised:
        return 0.0
    placeholders = sum(
        1
        for value in materialised
        if value is None or str(value).strip().upper() in PLACEHOLDER_SECTORS
    )
    return (len(materialised) - placeholders) / len(materialised)


def has_special_treatment_marker(name: str) -> bool:
    """Whether a security name carries an exchange special-treatment prefix.

    The vendor archive ships no ``is_st`` column, so the name is the only
    special-treatment signal this project has.  The exchanges compose the prefix
    with the split-share-reform marker, which means all four of these shapes
    occur in one archive::

        ST星源     *ST美丽     SST前锋     S*ST托普

    A plain ``startswith(("ST", "*ST"))`` misses the last shape, and seven names
    in the current archive are exactly that.  ``S`` alone is *not* a
    special-treatment marker -- it flags an un-reformed company and appears on
    healthy names such as ``S佳通`` -- so it is peeled only when something
    behind it still looks like a marker.

    The older ``PT`` (particular transfer, 1999-2002) regime is deliberately not
    matched: it was abolished long before the current archive, and the only two
    ``PT`` names it still lists are flagged suspended, so the suspension filter
    already keeps them out.  A test pins that boundary so it stays a decision
    rather than an oversight.

    This identifies a **current** label, not a point-in-time one: the same symbol
    carries the marker only in the periods around its own treatment.  Live
    decisions may use it; historical backtests may not, which is why the two
    paths disclose different policies off this same predicate.
    """

    normalized = str(name or "").strip().upper()
    if not normalized:
        return False
    peeled = normalized.lstrip("*")
    if peeled.startswith("S") and not peeled.startswith("ST"):
        peeled = peeled[1:].lstrip("*")
    return peeled.startswith("ST")


def _as_text(value: Any) -> str:
    """Render a disclosure string, mapping an absent value to empty text.

    ``str(None)`` would otherwise leak the literal ``"None"`` into a report the
    user reads, which looks like a value rather than a missing one.
    """

    return "" if value is None else str(value)


def _as_int(value: Any) -> int:
    """Best-effort integer coercion for disclosure counts.

    Counters reach this from provider state, which may legitimately be ``None``
    before the master has been built.  A disclosure must never be the reason a
    report fails to render, so unknown counts degrade to ``0`` rather than
    raising.
    """

    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def security_status_disclosure(
    *,
    st_applied: bool,
    suspension_applied: bool,
    source: str,
    basis: str,
    special_treatment_excluded: int = 0,
    suspension_excluded: int = 0,
    universe_note: str = "",
    marginable_securities: int = 0,
    shortable_securities: int = 0,
    capability_note: str = "",
) -> dict[str, Any]:
    """Uniform disclosure of what a book could actually do, on both paths.

    The research backtester and the paper runtime are two separate
    implementations of the same decision, so a status filter wired into only one
    of them silently changes which names are tradeable without changing any
    reported number.  Both paths build this dict so the report always states
    which filters actually ran, on what evidence, and what the universe
    difference is.

    It also carries the *capability* counts (marginable / shortable).  Those are
    not filters but they are the same kind of fact: a config can declare a
    capability the code can never exercise, and the only way a reader learns that
    is if the report says so.  ``margin_enabled = true`` while every instrument
    is ``marginable = False`` meant the whole financing and liquidation
    subsystem was decorative and nothing said so.
    """

    return {
        "st_filter_applied": bool(st_applied),
        "suspension_filter_applied": bool(suspension_applied),
        "security_status_source": _as_text(source),
        "security_status_basis": _as_text(basis),
        "special_treatment_excluded": _as_int(special_treatment_excluded),
        "suspension_excluded": _as_int(suspension_excluded),
        "universe_note": _as_text(universe_note),
        "marginable_securities": _as_int(marginable_securities),
        "shortable_securities": _as_int(shortable_securities),
        "capability_note": _as_text(capability_note),
    }


def utc_now() -> datetime:
    return datetime.now(UTC)


class Board(StrEnum):
    MAIN = "MAIN"
    CHINEXT_300 = "CHINEXT_300"
    STAR_688 = "STAR_688"


TARGET_A_SHARE_PREFIXES = (
    "000",
    "001",
    "002",
    "003",
    "300",
    "301",
    "302",
    "600",
    "601",
    "603",
    "605",
    "688",
    "689",
)


def is_target_a_share_symbol(symbol: str) -> bool:
    """Return whether ``symbol`` is an SH/SZ listed A-share code in scope.

    The vendor's stock instrument endpoint can return Shenzhen B shares and
    Shanghai B shares even when SHSE/SZSE and stock filters are supplied.  A
    code-prefix gate keeps those securities, Beijing securities, funds and
    indices out of the research universe.  It intentionally includes 300/301
    and 688/689 so the configured ChiNext and STAR boards are covered.
    """

    raw = str(symbol).strip().upper()
    digits = "".join(character for character in raw if character.isdigit())
    code = digits[-6:] if digits else raw
    return code.startswith(TARGET_A_SHARE_PREFIXES)


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
    # Why the industry concentration cap was or was not enforceable on this
    # report.  Without it a missing classification shows up only as a silently
    # under-invested book.
    industry_disclosure: dict[str, Any] = field(default_factory=dict)
    # Which trading-status filters (special treatment / suspension) actually ran
    # on this report, on what evidence, and how the live universe therefore
    # differs from the research one.
    security_status_disclosure: dict[str, Any] = field(default_factory=dict)


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
