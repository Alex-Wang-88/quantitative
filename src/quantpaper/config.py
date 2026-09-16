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
    margin_enabled: bool = True
    maintenance_ratio_warning: float = 1.30
    maintenance_ratio_liquidation: float = 1.10
    initial_margin_ratio: float = 0.50


@dataclass(frozen=True)
class PortfolioConfig:
    min_positions: int = 20
    max_positions: int = 30
    candidate_pool_size: int = 500
    single_weight_cap: float = 0.08
    industry_weight_cap: float = 0.25
    max_daily_turnover: float = 0.15
    max_daily_action_orders: int = 10


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
    artifact_dir: str = "data/models"
    walkforward_report_path: str = "data/reports/lightgbm-walkforward-3y.json"
    auto_retrain: bool = True


@dataclass(frozen=True)
class DataConfig:
    mode: str = "live"
    historical_provider: str = "unconfigured"
    realtime_provider: str = "unconfigured"
    universe: str = ""
    intraday_interval: str = "1m"
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
