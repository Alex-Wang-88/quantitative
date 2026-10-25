from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .domain import RunMode


@dataclass(frozen=True)
class AccountConfig:
    name: str = "safe_paper"
    initial_capital: float = 1_000_000.0
    max_gross_exposure: float = 1.0
    cash_floor: float = 0.10
    # Margin is only real when a *dated* credit-availability source says a
    # specific account may borrow a specific security, so the live profile
    # declares it off.  ``initial_margin_ratio`` used to sit here at 0.50 and was
    # read by nothing: the broker books the full shortfall as financing debt and
    # never sizes it against a margin requirement, so the field advertised a
    # model that did not exist.
    margin_enabled: bool = True
    maintenance_ratio_warning: float = 1.30
    maintenance_ratio_liquidation: float = 1.10


@dataclass(frozen=True)
class PortfolioConfig:
    # Floor on the number of names the book should carry.  This is a *risk*
    # floor, not a construction target: the portfolio engine still takes the best
    # ``max_positions`` names, and the RiskEngine raises a
    # ``POSITION_COUNT_FLOOR`` event when a live book sits below this.  ``0``
    # disables the floor, which is the honest default for research configs that
    # deliberately run a small book (``max_positions = 6``); the live account
    # sets 20 in ``config/default.toml``.
    #
    # It spent its whole life unread, which made it the one guard that would have
    # caught a collapse from a 30-name book back to a 4-name one -- the exact
    # regression it exists to catch -- while appearing to already exist.
    min_positions: int = 0
    max_positions: int = 30
    candidate_pool_size: int = 500
    single_weight_cap: float = 0.08
    industry_weight_cap: float = 0.25
    # Cap on **two-sided** traded notional per rebalance as a share of equity,
    # i.e. (buys + sells) / equity.  The portfolio is held at its target gross
    # exposure at all times; this only limits how much of the book may rotate,
    # so the per-leg slice is half of it (0.10 here) and realised one-sided
    # turnover lands in the 10%-20% band.  The session that funds the book from
    # all-cash to target is a deployment, not a rotation, and is exempt.
    max_daily_turnover: float = 0.20
    # Must be >= ``max_positions``.  Ten orders a day cannot build or maintain a
    # thirty-name book, and the shortfall surfaces as permanently under-invested
    # positions rather than as a visible error.
    max_daily_action_orders: int = 40
    # Research-backtest knobs.  ``rebalance_every_n_days`` and
    # ``hold_buffer_ratio`` trade signal freshness for lower turnover, which
    # matters because the 5-day label is currently paired with daily
    # re-ranking.  The default matches the label horizon: re-ranking more often
    # than the signal horizon is over-trading by construction.
    rebalance_every_n_days: int = 5
    hold_buffer_ratio: float = 0.0

    def __post_init__(self) -> None:
        # A floor above the ceiling is not a stricter portfolio, it is an
        # unsatisfiable one.  This used to be accepted silently, because nothing
        # read the floor at all.
        if self.min_positions < 0:
            raise ValueError("min_positions must be non-negative; 0 disables the floor")
        if self.min_positions > self.max_positions:
            raise ValueError(
                f"min_positions ({self.min_positions}) must not exceed "
                f"max_positions ({self.max_positions})"
            )


@dataclass(frozen=True)
class ExecutionConfig:
    mode: RunMode = RunMode.PAPER_AUTO
    quote_interval_seconds: int = 60
    manual_confirmation_delay_seconds: int = 60
    order_expiry_seconds: int = 300
    order_reissue_cooldown_seconds: int = 300
    session_rebalance_enabled: bool = True
    session_rebalance: str = "PM_OPEN"
    session_rebalance_window_minutes: int = 15
    session_rebalance_retry_seconds: int = 60
    close_report_enabled: bool = True
    close_report_time: str = "15:05"
    max_participation_pct: float = 0.10
    slippage_bps: float = 5.0
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    financing_rate: float = 0.08
    borrow_rate: float = 0.10


@dataclass(frozen=True)
class ModelConfig:
    name: str = "factor_lgbm"
    prediction_horizon_days: int = 5
    retrain_weekday: int = 4
    min_training_rows: int = 500
    validation_fraction: float = 0.20
    min_validation_days: int = 20
    validation_gap_days: int = 5
    artifact_dir: str = "data/models"
    walkforward_report_path: str = "data/reports/lightgbm-walkforward-3y.json"
    auto_retrain: bool = True


@dataclass(frozen=True)
class DataConfig:
    mode: str = "live"
    historical_provider: str = "unconfigured"
    realtime_provider: str = "unconfigured"
    universe: str = ""
    stale_after_seconds: int = 180
    history_lookback_days: int = 260
    # Research uses the full eligible universe by default.  Intraday quotes
    # remain limited to a smaller default pool; RuntimeService can override it
    # with the model's daily top candidates and current positions.
    research_universe_limit: int = 0
    realtime_universe_limit: int = 500
    fundamentals_universe_limit: int = 500
    eastmoney_fundamentals_enabled: bool = True
    # Industry is a separately permissioned endpoint. Keep it opt-in until
    # a small live capability probe has passed.
    eastmoney_industry_enabled: bool = False
    eastmoney_sdk_python: str = ".venv-eastmoney\\Scripts\\python.exe"
    eastmoney_bridge_script: str = "scripts/eastmoney_readonly_bridge.py"
    eastmoney_token_env: str = "QUANTPAPER_EASTMONEY_TOKEN"
    eastmoney_terminal_adapter_enabled: bool = True
    eastmoney_terminal_feed_path: str = "data/runtime/eastmoney-terminal-feed.json"


@dataclass(frozen=True)
class NotificationConfig:
    enabled: bool = False


@dataclass(frozen=True)
class AppConfig:
    account: AccountConfig = AccountConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    execution: ExecutionConfig = ExecutionConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    notifications: NotificationConfig = NotificationConfig()


def _coerce_mode(value: str) -> RunMode:
    try:
        return RunMode(value)
    except ValueError as exc:
        raise ValueError(f"unsupported execution mode: {value}") from exc


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or os.environ.get("QUANTPAPER_CONFIG", "config/default.toml"))
    if not path.exists():
        return AppConfig()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    account = raw.get("account", {})
    portfolio = raw.get("portfolio", {})
    execution = raw.get("execution", {})
    model = raw.get("model", {})
    data = raw.get("data", {})
    notifications = raw.get("notifications", {})
    return AppConfig(
        account=AccountConfig(**{**AccountConfig().__dict__, **account}),
        portfolio=PortfolioConfig(**{**PortfolioConfig().__dict__, **portfolio}),
        execution=ExecutionConfig(
            **{
                **ExecutionConfig().__dict__,
                **execution,
                "mode": _coerce_mode(execution.get("mode", "paper_auto")),
            }
        ),
        model=ModelConfig(**{**ModelConfig().__dict__, **model}),
        data=DataConfig(**{**DataConfig().__dict__, **data}),
        notifications=NotificationConfig(**{**NotificationConfig().__dict__, **notifications}),
    )
