from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from math import sqrt
from typing import Any

import pandas as pd

from .domain import Board, OrderSide, QuoteSnapshot, Security, board_for_symbol
from .features import FeatureEngine
from .rules import TradingRules
from .storage import DataQualityReport, HistoricalStore


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the first long-only factor research baseline."""

    initial_capital: float = 1_000_000.0
    max_positions: int = 30
    cash_floor: float = 0.10
    single_weight_cap: float = 0.08
    industry_weight_cap: float = 0.25
    max_daily_turnover: float = 0.15
    max_daily_orders: int = 10
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    slippage_bps: float = 5.0
    rebalance_every_n_days: int = 1


@dataclass(frozen=True)
class BacktestTrade:
    signal_date: str
    trade_date: str
    symbol: str
    side: str
    quantity: int
    price: float
    notional: float
    commission: float
    stamp_duty: float
    reason: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: list[BacktestTrade]
    stats: dict[str, float]
    quality: DataQualityReport

    def as_dict(self) -> dict[str, Any]:
        equity = self.equity_curve.copy()
        if not equity.empty and "trade_date" in equity:
            equity["trade_date"] = equity["trade_date"].astype(str)
        return {
            "stats": self.stats,
            "quality": self.quality.as_dict(),
            "trades": [asdict(trade) for trade in self.trades],
            "equity_curve": equity.to_dict(orient="records"),
        }


class FactorBacktester:
    """Run a point-in-time factor strategy against daily OHLCV data.

    A score observed after the close of day T can only create an order for
    the open of day T+1.  The implementation keeps that boundary explicit by
    building targets from the signal date and executing them in the next
    iteration of the trading calendar.
    """

    def __init__(
        self,
        securities: Iterable[Security] | Mapping[str, Security] | None = None,
        rules: TradingRules | None = None,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        if securities is None:
            self.securities: dict[str, Security] = {}
        elif isinstance(securities, Mapping):
            self.securities = dict(securities)
        else:
            self.securities = {security.symbol: security for security in securities}
        self.rules = rules or TradingRules()
        self.feature_engine = feature_engine or FeatureEngine()

    def run(
        self,
        daily_frame: pd.DataFrame,
        config: BacktestConfig | None = None,
        feature_frame: pd.DataFrame | None = None,
        score_column: str = "factor_score",
    ) -> BacktestResult:
        config = config or BacktestConfig()
        if config.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if config.max_positions <= 0 or config.max_daily_orders <= 0:
            raise ValueError("position and order limits must be positive")
        if not 0 <= config.cash_floor < 1:
            raise ValueError("cash_floor must be in [0, 1)")
        if config.single_weight_cap <= 0 or config.industry_weight_cap <= 0:
            raise ValueError("weight caps must be positive")
        if config.rebalance_every_n_days <= 0:
            raise ValueError("rebalance_every_n_days must be positive")

        normalized = HistoricalStore.normalize_daily(daily_frame)
        quality = HistoricalStore.quality(normalized)
        if not quality.ok:
            raise ValueError(f"daily data quality failed: {quality.as_dict()}")
        if normalized.empty:
            raise ValueError("daily data is empty")

        securities = self._resolve_securities(normalized)
        features = (
            self.feature_engine.build_features(normalized)
            if feature_frame is None
            else feature_frame.copy()
        )
        required_score_columns = {"symbol", "trade_date", score_column}
        missing_score_columns = required_score_columns - set(features.columns)
        if missing_score_columns:
            raise ValueError(
                f"feature frame missing scored backtest columns: {sorted(missing_score_columns)}"
            )
        features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce")
        if features["trade_date"].isna().any():
            raise ValueError("feature frame contains invalid trade_date values")
        features["trade_date"] = pd.to_datetime(features["trade_date"]).dt.date
        dates = sorted(features["trade_date"].unique())
        rows_by_date = {
            current_date: group.set_index("symbol")
            for current_date, group in normalized.assign(
                trade_date=pd.to_datetime(normalized["trade_date"]).dt.date
            ).groupby("trade_date", sort=True)
        }
        scores_by_date = {
            current_date: group.set_index("symbol")
            for current_date, group in features.groupby("trade_date", sort=True)
        }

        cash = float(config.initial_capital)
        positions: dict[str, int] = {}
        last_prices: dict[str, float] = {}
        scheduled_targets: dict[date, dict[str, tuple[float, str, str]]] = {}
        trades: list[BacktestTrade] = []
        equity_rows: list[dict[str, Any]] = []
        peak_equity = float(config.initial_capital)

        for date_index, current_date in enumerate(dates):
            day_rows = rows_by_date[current_date]
            previous_prices = dict(last_prices)
            prices_at_open = self._prices(day_rows, "open", last_prices)
            prices_at_close = self._prices(day_rows, "close", last_prices)
            for symbol, price in prices_at_close.items():
                if price > 0:
                    last_prices[symbol] = price

            target_plan = scheduled_targets.pop(current_date, {})
            turnover = 0.0
            if target_plan:
                cash, turnover, day_trades = self._execute_targets(
                    current_date=current_date,
                    day_rows=day_rows,
                    target_plan=target_plan,
                    positions=positions,
                    cash=cash,
                    prices_at_open=prices_at_open,
                    last_prices=previous_prices,
                    securities=securities,
                    config=config,
                )
                trades.extend(day_trades)

            mark_value = cash + sum(
                quantity * prices_at_close.get(symbol, last_prices.get(symbol, 0.0))
                for symbol, quantity in positions.items()
            )
            peak_equity = max(peak_equity, mark_value)
            previous_equity = (
                equity_rows[-1]["equity"] if equity_rows else float(config.initial_capital)
            )
            equity_rows.append(
                {
                    "trade_date": current_date.isoformat(),
                    "cash": round(cash, 6),
                    "long_market_value": round(mark_value - cash, 6),
                    "equity": round(mark_value, 6),
                    "daily_return": (mark_value / previous_equity - 1) if previous_equity else 0.0,
                    "drawdown": mark_value / peak_equity - 1 if peak_equity else 0.0,
                    "turnover": turnover / mark_value if mark_value else 0.0,
                    "turnover_value": round(turnover, 6),
                    "position_count": sum(quantity > 0 for quantity in positions.values()),
                }
            )

            if date_index < len(dates) - 1 and self._should_rebalance(
                date_index, config.rebalance_every_n_days
            ):
                score_rows = scores_by_date[current_date]
                signal_equity = mark_value
                plan = self._build_target_plan(
                    score_rows=score_rows,
                    current_date=current_date,
                    signal_equity=signal_equity,
                    positions=positions,
                    securities=securities,
                    config=config,
                    score_column=score_column,
                )
                next_date = dates[date_index + 1]
                scheduled_targets[next_date] = plan

        equity_curve = pd.DataFrame(equity_rows)
        stats = self._stats(equity_curve, trades, config.initial_capital)
        return BacktestResult(equity_curve, trades, stats, quality)

    def _resolve_securities(self, frame: pd.DataFrame) -> dict[str, Security]:
        result = dict(self.securities)
        for row in frame.itertuples(index=False):
            symbol = str(row.symbol)
            if symbol in result:
                continue
            board = self._board_for_symbol(symbol)
            result[symbol] = Security(
                symbol=symbol,
                name=symbol,
                exchange="SSE" if symbol.startswith(("6", "68")) else "SZSE",
                board=board,
                sector=str(getattr(row, "sector", "UNKNOWN")),
                is_st=self._flag(getattr(row, "is_st", False)),
                suspended=self._flag(getattr(row, "suspended", False)),
            )
        return result

    @staticmethod
    def _board_for_symbol(symbol: str) -> Board:
        return board_for_symbol(symbol)

    @staticmethod
    def _prices(
        day_rows: pd.DataFrame, column: str, fallback: Mapping[str, float]
    ) -> dict[str, float]:
        result = dict(fallback)
        if column in day_rows:
            for symbol, value in day_rows[column].items():
                if pd.notna(value) and float(value) > 0:
                    result[str(symbol)] = float(value)
        return result

    def _build_target_plan(
        self,
        score_rows: pd.DataFrame,
        current_date: date,
        signal_equity: float,
        positions: Mapping[str, int],
        securities: Mapping[str, Security],
        config: BacktestConfig,
        score_column: str = "factor_score",
    ) -> dict[str, tuple[float, str, str]]:
        if signal_equity <= 0 or score_rows.empty:
            return {}
        candidates = score_rows.copy()
        candidates = candidates[candidates.index.astype(str).isin(securities)]
        candidates = candidates[
            ~candidates.index.map(
                lambda symbol: securities[str(symbol)].is_st or securities[str(symbol)].suspended
            )
        ]
        candidates = candidates[
            pd.to_numeric(candidates[score_column], errors="coerce").notna()
        ]
        candidates = candidates.sort_values(score_column, ascending=False)
        candidate_count = min(config.max_positions, len(candidates))
        if candidate_count <= 0:
            return {
                symbol: (0.0, "退出当前候选池", current_date.isoformat()) for symbol in positions
            }

        deployable_weight = 1.0 - config.cash_floor
        base_weight = min(
            config.single_weight_cap,
            config.industry_weight_cap,
            deployable_weight / candidate_count,
        )
        sector_weights: dict[str, float] = {}
        selected: dict[str, tuple[float, str, str]] = {}
        for symbol, row in candidates.iterrows():
            symbol = str(symbol)
            if len(selected) >= config.max_positions:
                break
            sector = securities[symbol].sector
            if sector_weights.get(sector, 0.0) + base_weight > config.industry_weight_cap + 1e-9:
                continue
            reason = self._reason(row, score_column=score_column)
            selected[symbol] = (base_weight, reason, current_date.isoformat())
            sector_weights[sector] = sector_weights.get(sector, 0.0) + base_weight

        for symbol in positions:
            if symbol not in selected:
                selected[symbol] = (0.0, "退出当前候选池", current_date.isoformat())
        return selected

    @staticmethod
    def _reason(row: pd.Series, *, score_column: str = "factor_score") -> str:
        score = float(row.get(score_column, 0.0))
        components = [
            ("动量", row.get("momentum_20")),
            ("波动", row.get("volatility_20")),
            ("估值", row.get("value_score")),
        ]
        strongest = max(
            ((name, float(value)) for name, value in components if pd.notna(value)),
            key=lambda item: abs(item[1]),
            default=("综合", 0.0),
        )
        return f"因子综合分 {score:.3f}；主要贡献 {strongest[0]}"

    def _execute_targets(
        self,
        current_date: date,
        day_rows: pd.DataFrame,
        target_plan: Mapping[str, tuple[float, str, str]],
        positions: dict[str, int],
        cash: float,
        prices_at_open: Mapping[str, float],
        last_prices: Mapping[str, float],
        securities: Mapping[str, Security],
        config: BacktestConfig,
    ) -> tuple[float, float, list[BacktestTrade]]:
        equity_before = cash + sum(
            quantity * prices_at_open.get(symbol, last_prices.get(symbol, 0.0))
            for symbol, quantity in positions.items()
        )
        turnover_budget = max(0.0, equity_before * config.max_daily_turnover)
        trades: list[BacktestTrade] = []
        remaining_budget = turnover_budget

        sell_candidates: list[tuple[str, int, str, str]] = []
        buy_candidates: list[tuple[str, int, str, str]] = []
        for symbol, (target_weight, reason, signal_date) in target_plan.items():
            security = securities[symbol]
            open_price = prices_at_open.get(symbol)
            if not open_price or open_price <= 0:
                continue
            target_quantity = self.rules.normalize_quantity(
                security,
                int(equity_before * target_weight / open_price),
                OrderSide.BUY,
                current_date,
            )
            current_quantity = positions.get(symbol, 0)
            delta = target_quantity - current_quantity
            if delta < 0:
                sell_candidates.append((symbol, -delta, reason, signal_date))
            elif delta > 0:
                buy_candidates.append((symbol, delta, reason, signal_date))

        sell_candidates.sort(key=lambda item: item[1], reverse=True)
        buy_candidates.sort(key=lambda item: item[1], reverse=True)
        for side, candidates in (
            (OrderSide.SELL, sell_candidates),
            (OrderSide.BUY, buy_candidates),
        ):
            for symbol, quantity, reason, signal_date in candidates:
                if len(trades) >= config.max_daily_orders or remaining_budget <= 0:
                    break
                security = securities[symbol]
                row = day_rows.loc[symbol] if symbol in day_rows.index else None
                if row is None or self._blocked(row, security):
                    continue
                open_price = prices_at_open.get(symbol)
                if not open_price or open_price <= 0:
                    continue
                execution_price = open_price * (
                    1 + config.slippage_bps / 10_000
                    if side == OrderSide.BUY
                    else 1 - config.slippage_bps / 10_000
                )
                if not self._can_fill(
                    current_date=current_date,
                    symbol=symbol,
                    security=security,
                    row=row,
                    side=side,
                    execution_price=execution_price,
                    previous_close=last_prices.get(symbol, open_price),
                ):
                    continue
                allowed = min(quantity, int(remaining_budget / execution_price))
                if side == OrderSide.BUY:
                    cash_available = max(0.0, cash - equity_before * config.cash_floor)
                    gross_per_share = execution_price * (1 + config.commission_rate)
                    allowed = min(allowed, int(cash_available / gross_per_share))
                allowed = self.rules.normalize_quantity(security, allowed, side, current_date)
                if allowed <= 0:
                    continue
                notional = allowed * execution_price
                commission = notional * config.commission_rate
                stamp_duty = notional * config.stamp_duty_rate if side == OrderSide.SELL else 0.0
                if side == OrderSide.BUY:
                    cash -= notional + commission
                    positions[symbol] = positions.get(symbol, 0) + allowed
                else:
                    cash += notional - commission - stamp_duty
                    positions[symbol] = positions.get(symbol, 0) - allowed
                    if positions[symbol] <= 0:
                        positions.pop(symbol, None)
                remaining_budget -= notional
                trades.append(
                    BacktestTrade(
                        signal_date=signal_date,
                        trade_date=current_date.isoformat(),
                        symbol=symbol,
                        side=side.value,
                        quantity=allowed,
                        price=round(execution_price, 6),
                        notional=round(notional, 6),
                        commission=round(commission, 6),
                        stamp_duty=round(stamp_duty, 6),
                        reason=reason,
                    )
                )
        return cash, turnover_budget - remaining_budget, trades

    def _can_fill(
        self,
        current_date: date,
        symbol: str,
        security: Security,
        row: pd.Series,
        side: OrderSide,
        execution_price: float,
        previous_close: float,
    ) -> bool:
        lower, upper = self.rules.price_limits(previous_close, security, current_date)
        quote = QuoteSnapshot(
            symbol=symbol,
            timestamp=pd.Timestamp(current_date).to_pydatetime(),
            last=float(row["open"]),
            previous_close=previous_close,
            upper_limit=upper,
            lower_limit=lower,
            suspended=self._blocked(row, security),
            source="HISTORICAL_OPEN",
        )
        if self._blocked(row, security):
            return False
        can_fill, _ = self.rules.can_fill(quote, side, execution_price)
        return can_fill

    @staticmethod
    def _blocked(row: pd.Series, security: Security) -> bool:
        return (
            security.is_st
            or security.suspended
            or FactorBacktester._flag(row.get("is_st", False))
            or FactorBacktester._flag(row.get("suspended", False))
        )

    @staticmethod
    def _flag(value: Any) -> bool:
        return False if pd.isna(value) else bool(value)

    @staticmethod
    def _should_rebalance(date_index: int, every_n_days: int) -> bool:
        return date_index % every_n_days == 0

    @staticmethod
    def _stats(
        equity_curve: pd.DataFrame,
        trades: list[BacktestTrade],
        initial_capital: float,
    ) -> dict[str, float]:
        if equity_curve.empty:
            return {
                "initial_capital": initial_capital,
                "final_equity": initial_capital,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "turnover": 0.0,
                "trade_count": 0.0,
            }
        final_equity = float(equity_curve.iloc[-1]["equity"])
        total_return = final_equity / initial_capital - 1
        periods = max(1, len(equity_curve) - 1)
        annualized_return = (final_equity / initial_capital) ** (252 / periods) - 1
        returns = pd.to_numeric(equity_curve["daily_return"], errors="coerce").dropna()
        volatility = float(returns.std(ddof=1) * sqrt(252)) if len(returns) > 1 else 0.0
        sharpe = (
            float(returns.mean() / returns.std(ddof=1) * sqrt(252))
            if len(returns) > 1 and returns.std(ddof=1)
            else 0.0
        )
        total_turnover = float(equity_curve["turnover_value"].sum())
        average_equity = float(equity_curve["equity"].mean()) or initial_capital
        return {
            "initial_capital": round(initial_capital, 6),
            "final_equity": round(final_equity, 6),
            "total_return": round(total_return, 8),
            "annualized_return": round(float(annualized_return), 8),
            "annualized_volatility": round(volatility, 8),
            "sharpe": round(sharpe, 8),
            "max_drawdown": round(float(equity_curve["drawdown"].min()), 8),
            "turnover": round(total_turnover / average_equity, 8),
            "trade_count": float(len(trades)),
        }
