from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from math import ceil, sqrt
from typing import Any

import pandas as pd

from .domain import (
    Board,
    OrderSide,
    QuoteSnapshot,
    Security,
    board_for_symbol,
    known_sectors,
    sector_coverage,
    security_status_disclosure,
)
from .features import FeatureEngine
from .rules import TradingRules
from .storage import DataQualityReport, HistoricalStore

# ``PLACEHOLDER_SECTORS`` / ``known_sectors`` live in ``domain`` so the research
# backtester and the live portfolio engine cannot drift apart on what counts as a
# known sector.  A drift there silently truncates one of them: treating the
# ``"UNKNOWN"`` placeholder as a sector lumps the whole universe into one bucket
# and the industry cap then caps the book at ``industry_weight_cap /
# base_weight`` names (0.25 / 0.03 = 8 = 24% invested) however many positions the
# configuration asks for.


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the first long-only factor research baseline."""

    initial_capital: float = 1_000_000.0
    min_positions: int = 0
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
    # An incumbent holding keeps its slot until its rank falls outside
    # ``max_positions * (1 + hold_buffer_ratio)``.  The label horizon is five
    # trading days while the portfolio is re-ranked daily, so without a buffer
    # the book churns on rank noise and transaction costs consume the alpha.
    # ``0.0`` reproduces the original top-N behaviour exactly.
    hold_buffer_ratio: float = 0.0


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
    stats: dict[str, Any]
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
        # Filled by ``_build_target_plan`` so ``run`` can disclose whether the
        # industry cap was actually enforceable on this dataset.
        self._industry_disclosure: dict[str, Any] = {
            "industry_cap_applied": False,
            "industry_cap_reason": "no target plan was built",
            "sector_coverage": 0.0,
            "distinct_known_sectors": 0,
            "sector_source": "UNAVAILABLE:sector",
        }
        # Rotation that the turnover budget refused to fund.  A silent refusal
        # would look identical to "the model wanted nothing to change"; the
        # count is published in ``stats`` so a capped rotation is visible.
        self._rotation_capped_days = 0

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
        if config.hold_buffer_ratio < 0:
            raise ValueError("hold_buffer_ratio must be non-negative")
        self._peak_budget_rate = 0.0
        self._rotation_capped_days = 0

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
            # The turnover cap is budgeted out of the equity that was available
            # when the orders were sized, so the realised ratio has to be
            # measured against the same base.  Dividing by the *closing* equity
            # instead let a falling market inflate the reported ratio past the
            # cap it had actually respected (measured: 21.0% reported against a
            # 20% cap on a down day).
            equity_at_open = cash + sum(
                quantity * prices_at_open.get(symbol, last_prices.get(symbol, 0.0))
                for symbol, quantity in positions.items()
            )
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
                    "equity_at_open": round(equity_at_open, 6),
                    "turnover": turnover / equity_at_open if equity_at_open else 0.0,
                    # ``turnover`` above is two-sided: it counts the sell leg and
                    # the buy leg that funds it.  The convention most A-share
                    # readers mean by 换手率 is one-sided, so publish half of it
                    # too instead of leaving the reader to guess which one the
                    # configured cap refers to.
                    "one_way_turnover": (turnover / equity_at_open / 2.0)
                    if equity_at_open
                    else 0.0,
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
        stats: dict[str, Any] = self._stats(equity_curve, trades, config.initial_capital)
        stats.update(self._filter_disclosure(normalized))
        stats.update(self._industry_disclosure)
        stats["peak_daily_turnover_budget"] = round(
            float(getattr(self, "_peak_budget_rate", 0.0)), 6
        )
        stats["rotation_capped_days"] = float(getattr(self, "_rotation_capped_days", 0))
        # The floor is a risk control, so research has to publish whether it was
        # respected -- otherwise only the live book can ever fail it, and a
        # research run holding two names looks the same as a prudent one.  The
        # minimum is taken over days the book holds something, so building up
        # from cash does not read as a breach.
        held_days = (
            equity_curve["position_count"] > 0
            if "position_count" in equity_curve.columns
            else pd.Series(dtype=bool)
        )
        stats["min_positions"] = int(config.min_positions)
        stats["position_count_min"] = (
            int(equity_curve.loc[held_days, "position_count"].min())
            if bool(held_days.any())
            else 0
        )
        stats["one_way_turnover"] = round(float(stats["turnover"]) / 2.0, 8)
        # Cost drag has to be a first-class number.  At 3.5-4.7x one-way annual
        # turnover the frictions are the same order of magnitude as the alpha,
        # so a report showing only the net return cannot say whether a shortfall
        # came from the signal or from the broker.  Commission and stamp duty are
        # booked per trade; slippage is baked into the execution price and has no
        # field of its own, so it is reconstructed from the traded notional
        # instead of being silently omitted.
        average_equity = (
            float(pd.to_numeric(equity_curve["equity_at_open"], errors="coerce").mean())
            if not equity_curve.empty
            else 0.0
        )
        if not average_equity or pd.isna(average_equity):
            average_equity = config.initial_capital
        traded_notional = sum(float(trade.notional) for trade in trades)
        fees_paid = sum(
            float(trade.commission) + float(trade.stamp_duty) for trade in trades
        )
        slippage_paid = traded_notional * config.slippage_bps / 10_000
        stats["traded_notional_ratio"] = round(traded_notional / average_equity, 8)
        stats["fees_paid"] = round(fees_paid, 6)
        stats["slippage_paid"] = round(slippage_paid, 6)
        stats["cost_drag_ratio"] = round((fees_paid + slippage_paid) / average_equity, 8)
        stats["rebalance_every_n_days"] = float(config.rebalance_every_n_days)
        stats["hold_buffer_ratio"] = float(config.hold_buffer_ratio)
        return BacktestResult(equity_curve, trades, stats, quality)

    @staticmethod
    def _filter_disclosure(frame: pd.DataFrame) -> dict[str, Any]:
        """State which trading filters the input data could actually support.

        The archived A-share history in this project carries neither a
        point-in-time ST flag nor a suspension flag.  ``_blocked`` therefore
        evaluates to ``False`` for every row and ST names keep the ordinary
        board price limit even though the exchange applies +/-5%.  Saying so
        explicitly in the report is the only honest option until a
        point-in-time status source is archived; silently reporting nothing
        would make the backtest look stricter than it is.

        This goes through the same builder as the live provider so both paths
        publish one key set.  A hand-rolled dict here previously carried three
        keys where the live side carried seven, which is how two disclosures of
        the same decision drift apart.
        """

        st_present = "is_st" in frame.columns and bool(frame["is_st"].notna().any())
        suspended_present = "suspended" in frame.columns and bool(
            frame["suspended"].notna().any()
        )
        missing = [
            name
            for name, present in (("is_st", st_present), ("suspended", suspended_present))
            if not present
        ]
        return security_status_disclosure(
            st_applied=st_present,
            suspension_applied=suspended_present,
            source="input_columns" if not missing else "UNAVAILABLE:" + ",".join(missing),
            basis=(
                "研究路径不使用 ST 名称前缀，也不用当前快照状态：归档按证券代码前缀"
                "选取股票池，按当前名称或当前停牌状态过滤会构成幸存者偏差"
                if missing
                else "输入帧自带状态列，按帧内取值过滤"
            ),
            universe_note=(
                "无过滤：ST 与停牌证券均可进入候选池，且 ST 股按普通涨跌停（非 ±5%）"
                "撮合，因此本路径的可交易域比线上纸盘更宽"
                if missing
                else "输入帧自带状态列"
            ),
            capability_note=(
                "研究路径是纯多头、不建模保证金与融券：没有 dated 的信用可用性数据，"
                "任何融资/融券收益都只能是假设。与线上一致的一点是两者都不会用到它们"
            ),
        )

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
        # Break score ties by symbol.  A shallow model produces many equal
        # predictions, and a non-deterministic tie order makes the "top N" set
        # flip from day to day: the book then churns names it never intended to
        # sell, and the turnover budget is spent undoing noise.
        candidates = candidates.assign(
            _rank_symbol=candidates.index.astype(str)
        ).sort_values([score_column, "_rank_symbol"], ascending=[False, True])
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
        ranked_symbols = [str(symbol) for symbol in candidates.index]
        position_of = {symbol: index for index, symbol in enumerate(ranked_symbols)}

        # The industry cap is a risk control that needs a real classification.
        # When the archive carries no industry field every candidate resolves to
        # the ``"UNKNOWN"`` placeholder, so the cap would not diversify anything:
        # it would just divide the target book by ``industry_weight_cap /
        # base_weight`` and cap the whole portfolio at that many names.  Enforce
        # it only when the classification actually covers the candidate
        # universe, and record why when it does not.
        sector_values = [securities[symbol].sector for symbol in ranked_symbols]
        distinct_sectors = known_sectors(sector_values)
        coverage = sector_coverage(sector_values)
        placeholder_count = len(sector_values) - int(round(coverage * len(sector_values)))
        industry_cap_applied = coverage >= 0.95 and bool(distinct_sectors)
        if industry_cap_applied:
            industry_cap_reason = (
                f"行业覆盖 {coverage:.1%}，{len(distinct_sectors)} 个行业，"
                "上限可执行"
            )
        else:
            industry_cap_reason = (
                f"行业覆盖仅 {coverage:.1%}"
                f"（未知行业 {placeholder_count}/{len(sector_values)}），"
                "行业上限无法分散风险，已停用"
            )
        self._industry_disclosure = {
            "industry_cap_applied": industry_cap_applied,
            "industry_cap_reason": industry_cap_reason,
            "sector_coverage": round(coverage, 6),
            "distinct_known_sectors": len(distinct_sectors),
            "sector_source": (
                "AVAILABLE:sector" if industry_cap_applied else "UNAVAILABLE:sector"
            ),
        }

        # Entry and exit use different ranks: an incumbent is only dropped once
        # it leaves the wider exit band.  ``hold_buffer_ratio == 0`` collapses
        # the band onto the entry rank and reproduces the original behaviour.
        hold_limit = int(ceil(config.max_positions * (1.0 + config.hold_buffer_ratio)))
        chosen: list[str] = []
        for symbol in ranked_symbols[:hold_limit]:
            if symbol in positions:
                chosen.append(symbol)
        for symbol in ranked_symbols:
            if len(chosen) >= config.max_positions:
                break
            if symbol not in chosen:
                chosen.append(symbol)

        selected: dict[str, tuple[float, str, str]] = {}
        sector_weights: dict[str, float] = {}
        for symbol in chosen:
            if len(selected) >= config.max_positions:
                break
            sector = securities[symbol].sector
            if (
                industry_cap_applied
                and sector_weights.get(sector, 0.0) + base_weight
                > config.industry_weight_cap + 1e-9
            ):
                continue
            row = candidates.iloc[position_of[symbol]]
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
        long_value = sum(
            quantity * prices_at_open.get(symbol, last_prices.get(symbol, 0.0))
            for symbol, quantity in positions.items()
        )
        equity_before = cash + long_value

        # Two different things are being throttled and they must not share one
        # number.  *Holding the target book* is the objective: the plan already
        # says the portfolio should carry ``1 - cash_floor`` of equity, so idle
        # cash is a failure, not a conservative choice.  *Rotating* the book is
        # what ``max_daily_turnover`` limits.  Budgeting both out of a single cap
        # left the book permanently un-invested (measured: ~15% invested against
        # a 90% target), so the deficit to the target gross exposure is reserved
        # for the buy leg first, and only what is left is available to rotation.
        target_gross = sum(
            max(0.0, weight) for weight, _, _ in target_plan.values()
        )
        current_gross = long_value / equity_before if equity_before > 0 else 0.0
        deployment_gap = max(0.0, target_gross - current_gross)
        budget_rate = max(config.max_daily_turnover, deployment_gap)
        self._peak_budget_rate = max(getattr(self, "_peak_budget_rate", 0.0), budget_rate)
        turnover_budget = max(0.0, equity_before * budget_rate)
        # Maintenance trades must never be rationed below the configured maximum
        # book size: ten orders a day cannot keep a thirty-name book at its
        # target weights.
        order_limit = max(config.max_daily_orders, config.max_positions)
        trades: list[BacktestTrade] = []

        # Sells come in two flavours and they must not be treated alike.  An
        # *exit* removes a name from the book, so it is the only kind of sell
        # that frees a slot for a new name; a *trim* only shrinks a name that is
        # staying.  Exits are queued first and, further down, are only ever
        # executed in full.  A half-finished exit leaves a stub holding that
        # still occupies one of ``max_positions`` slots while no longer being
        # wanted, no replacement can be admitted, and the book bleeds down
        # session after session -- measured at 90% -> 78% invested under a
        # rotation cap before this rule existed.
        sell_candidates: list[tuple[str, int, str, str, bool]] = []
        buy_candidates: list[tuple[str, int, str, str, bool]] = []
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
                sell_candidates.append(
                    (symbol, -delta, reason, signal_date, target_weight <= 0)
                )
            elif delta > 0:
                buy_candidates.append((symbol, delta, reason, signal_date, False))

        sell_candidates.sort(key=lambda item: (not item[4], -item[1]))
        buy_candidates.sort(key=lambda item: -item[1])

        # The deficit gets its own budget line and is reserved for the buy leg,
        # because buying is the only way the book returns to its target
        # exposure.  What remains is split evenly, so a rotation cannot let the
        # sell leg consume everything, dump the book into cash, and force it to
        # be rebuilt on the next session.
        #
        # When the book is more than one cap away from its target, deployment
        # therefore takes the whole budget and rotation gets nothing.  That is
        # intended, not an oversight: selling while under-invested widens the
        # deficit it is supposed to close.  ``deployment_gap`` persists, so the
        # buy leg keeps its priority on the next rebalance as well -- the earlier
        # "exempt the first session only" version dropped the priority after one
        # day and the book could never be topped back up.
        deployment_reserve = min(deployment_gap * equity_before, turnover_budget)
        rotation_budget = max(0.0, turnover_budget - deployment_reserve)
        sell_budget = rotation_budget / 2.0
        buy_budget = deployment_reserve + rotation_budget / 2.0
        sell_consumed = 0.0
        consumed_budget = 0.0
        declined_exit = False

        for side, candidates in (
            (OrderSide.SELL, sell_candidates),
            (OrderSide.BUY, buy_candidates),
        ):
            if side == OrderSide.BUY:
                # Buys may also spend whatever the sell leg left on the table,
                # but never more than the distance to the target gross
                # exposure: the book is maintained at its target, not pushed
                # past it.
                gross_after_sells = sum(
                    quantity * prices_at_open.get(symbol, last_prices.get(symbol, 0.0))
                    for symbol, quantity in positions.items()
                )
                buy_allowance = max(
                    0.0, target_gross * equity_before - gross_after_sells
                )
                remaining_budget = min(
                    buy_budget + max(0.0, sell_budget - sell_consumed),
                    buy_allowance,
                )
            else:
                remaining_budget = sell_budget
            for symbol, quantity, reason, signal_date, is_exit in candidates:
                if len(trades) >= order_limit or remaining_budget <= 0:
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
                if side == OrderSide.SELL and is_exit:
                    # An exit that the rotation budget cannot fund in full is
                    # skipped rather than half-filled.  ``continue`` and not
                    # ``break``: a smaller exit further down the queue still
                    # gets its turn.
                    if quantity * execution_price > remaining_budget:
                        declined_exit = True
                        continue
                # Never open a new name that would push the book past
                # ``max_positions``.  The sell leg runs first, so exits that did
                # execute have already reduced the count; the remaining incumbents
                # hold their slots and wait for their turn to be sold.
                if (
                    side == OrderSide.BUY
                    and symbol not in positions
                    and len(positions) >= config.max_positions
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
                consumed_budget += notional
                if side == OrderSide.SELL:
                    sell_consumed += notional
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
        if declined_exit:
            self._rotation_capped_days += 1
        return cash, consumed_budget, trades

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
                "entry_day_turnover": 0.0,
                "daily_turnover_max": 0.0,
                "invested_ratio_mean": 0.0,
                "invested_ratio_min": 0.0,
                "invested_ratio_final": 0.0,
                "traded_notional_ratio": 0.0,
                "fees_paid": 0.0,
                "slippage_paid": 0.0,
                "cost_drag_ratio": 0.0,
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
        # Same base as the per-day ratio: see the note in ``run``.
        average_equity = (
            float(pd.to_numeric(equity_curve["equity_at_open"], errors="coerce").mean())
            or initial_capital
        )
        # Position control has to be readable off the report, not inferred from
        # the equity curve by hand: "the book is built once and then held" is the
        # difference between a real portfolio and a slow liquidation, and the two
        # look identical in a return number.
        invested = pd.to_numeric(equity_curve["long_market_value"], errors="coerce") / (
            pd.to_numeric(equity_curve["equity"], errors="coerce").replace(0.0, float("nan"))
        )
        held = equity_curve["position_count"] > 0
        entry_label = int(held.idxmax()) if bool(held.any()) else None
        active = invested[held] if bool(held.any()) else pd.Series([0.0])
        daily_turnover = pd.to_numeric(equity_curve["turnover"], errors="coerce")
        # The session that funds the book from all-cash is a deployment, not a
        # rotation, and is deliberately exempt from the rotation cap -- so it is
        # excluded here instead of silently dominating the maximum.
        rotation_turnover = (
            daily_turnover.drop(index=entry_label)
            if entry_label is not None and entry_label in daily_turnover.index
            else daily_turnover
        )
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
            "entry_day_turnover": round(
                float(daily_turnover.loc[entry_label]) if entry_label is not None else 0.0, 8
            ),
            "daily_turnover_max": round(float(rotation_turnover.max()), 8),
            "invested_ratio_mean": round(float(active.mean()), 8),
            "invested_ratio_min": round(float(active.min()), 8),
            "invested_ratio_final": round(float(invested.iloc[-1]), 8),
        }
