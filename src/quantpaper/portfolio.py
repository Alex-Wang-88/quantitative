from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any
from uuid import uuid4

import pandas as pd

from .config import PortfolioConfig
from .domain import (
    INDUSTRY_CAP_MIN_COVERAGE,
    AccountSnapshot,
    OrderIntent,
    OrderSide,
    OrderStatus,
    Position,
    QuoteSnapshot,
    Security,
    Signal,
    TargetPosition,
    known_sectors,
    sector_coverage,
    utc_now,
)
from .rules import TradingRules


@dataclass
class PortfolioPlan:
    targets: list[TargetPosition]
    signals: list[Signal]
    orders: list[OrderIntent]


class PortfolioEngine:
    def __init__(
        self,
        config: PortfolioConfig,
        rules: TradingRules | None = None,
        cash_floor: float = 0.10,
    ) -> None:
        self.config = config
        self.rules = rules or TradingRules()
        self.cash_floor = cash_floor
        # Whether the industry cap was enforceable on the most recent plan, and
        # why.  Mirrors the research backtester's disclosure so the two paths can
        # be compared instead of silently disagreeing about exposure.
        self._industry_disclosure: dict[str, Any] = {
            "industry_cap_applied": False,
            "industry_cap_reason": "no target plan was built",
            "sector_coverage": 0.0,
            "distinct_known_sectors": 0,
            "sector_source": "UNAVAILABLE:sector",
        }

    @property
    def industry_disclosure(self) -> dict[str, Any]:
        return dict(self._industry_disclosure)

    def _describe_industry_cap(self, security_labels: list[str]) -> bool:
        """Decide whether the industry cap is enforceable, and disclose why.

        The cap needs a real classification to diversify anything.  With no
        industry field every candidate resolves to the ``"UNKNOWN"`` placeholder,
        so the cap degenerates into ``industry_weight_cap / base_weight`` names and
        pins the book at that exposure however many positions the configuration
        asks for -- 0.25 / 0.03 = 8 names = 24% invested.  Enforce it only when the
        classification actually covers the candidates, and record the decision
        rather than under-investing in silence.
        """

        distinct = known_sectors(security_labels)
        coverage = sector_coverage(security_labels)
        placeholder_count = len(security_labels) - int(
            round(coverage * len(security_labels))
        )
        applied = coverage >= INDUSTRY_CAP_MIN_COVERAGE and bool(distinct)
        if applied:
            reason = (
                f"行业覆盖 {coverage:.1%}，{len(distinct)} 个行业，上限可执行"
            )
        else:
            reason = (
                f"行业覆盖仅 {coverage:.1%}"
                f"（未知行业 {placeholder_count}/{len(security_labels)}），"
                "行业上限无法分散风险，已停用"
            )
        self._industry_disclosure = {
            "industry_cap_applied": applied,
            "industry_cap_reason": reason,
            "sector_coverage": round(coverage, 6),
            "distinct_known_sectors": len(distinct),
            "sector_source": "AVAILABLE:sector" if applied else "UNAVAILABLE:sector",
        }
        return applied

    def build_targets(
        self,
        scores: pd.DataFrame,
        account: AccountSnapshot,
        securities: dict[str, Security],
        prices: dict[str, float],
        positions: dict[str, Position] | None = None,
        data_timestamp=None,
        model_version: str = "unknown",
    ) -> tuple[list[TargetPosition], list[Signal]]:
        eligible = scores[scores["symbol"].isin(securities)].copy()
        eligible = eligible.sort_values("score", ascending=False)
        eligible = eligible[
            eligible["symbol"].map(
                lambda symbol: not securities[symbol].is_st and not securities[symbol].suspended
            )
        ]
        if eligible.empty:
            return [], []
        account_equity = max(account.total_equity, 0.0)
        if account_equity <= 0:
            return [], []
        deployable = account_equity * (1 - self.cash_floor)
        count = max(1, min(self.config.max_positions, len(eligible)))
        base_weight = min(self.config.single_weight_cap, deployable / account_equity / count)
        industry_cap_applied = self._describe_industry_cap(
            [securities[symbol].sector for symbol in eligible["symbol"]]
        )
        targets: list[TargetPosition] = []
        signals: list[Signal] = []
        positions = positions or {}
        current_by_symbol = {
            symbol: position.long_market_value / account_equity
            for symbol, position in positions.items()
            if account_equity > 0
        }
        sector_weights: dict[str, float] = {}
        selected: set[str] = set()
        for row in eligible.itertuples(index=False):
            if len(selected) >= self.config.max_positions:
                break
            security = securities[row.symbol]
            if (
                industry_cap_applied
                and sector_weights.get(security.sector, 0.0) + base_weight
                > self.config.industry_weight_cap + 1e-9
            ):
                continue
            target_weight = base_weight
            target_value = account.total_equity * target_weight
            price = prices.get(row.symbol, 0.0)
            target_qty = self.rules.normalize_quantity(
                security,
                int(target_value / price) if price else 0,
                OrderSide.BUY,
                data_timestamp.date() if hasattr(data_timestamp, "date") else date.today(),
            )
            reason = str(getattr(row, "reason", "模型综合评分"))
            targets.append(
                TargetPosition(row.symbol, target_weight, target_value, target_qty, reason)
            )
            signals.append(
                Signal(
                    signal_id=str(uuid4()),
                    symbol=row.symbol,
                    name=security.name,
                    board=security.board,
                    sector=security.sector,
                    generated_at=utc_now(),
                    data_timestamp=data_timestamp or utc_now(),
                    score=float(row.score),
                    action="BUY" if target_qty else "WATCH",
                    target_weight=target_weight,
                    current_weight=current_by_symbol.get(row.symbol, 0.0),
                    reason=reason,
                    model_version=model_version,
                )
            )
            selected.add(row.symbol)
            sector_weights[security.sector] = (
                sector_weights.get(security.sector, 0.0) + target_weight
            )

        for symbol, position in positions.items():
            if symbol not in securities or symbol in selected or position.long_qty <= 0:
                continue
            security = securities[symbol]
            reason = "退出当前候选池，目标仓位归零"
            targets.append(TargetPosition(symbol, 0.0, 0.0, 0, reason))
            signals.append(
                Signal(
                    signal_id=str(uuid4()),
                    symbol=symbol,
                    name=security.name,
                    board=security.board,
                    sector=security.sector,
                    generated_at=utc_now(),
                    data_timestamp=data_timestamp or utc_now(),
                    score=0.0,
                    action="SELL",
                    target_weight=0.0,
                    current_weight=current_by_symbol.get(symbol, 0.0),
                    reason=reason,
                    model_version=model_version,
                )
            )
        return targets, signals


class OrderPlanner:
    def __init__(
        self,
        config: PortfolioConfig,
        rules: TradingRules | None = None,
        order_expiry_seconds: int = 300,
    ) -> None:
        self.config = config
        self.rules = rules or TradingRules()
        self.order_expiry_seconds = order_expiry_seconds

    def plan_orders(
        self,
        targets: list[TargetPosition],
        positions: dict[str, Position],
        securities: dict[str, Security],
        quotes: dict[str, QuoteSnapshot],
        model_version: str,
        created_at=None,
        max_turnover_value: float | None = None,
    ) -> list[OrderIntent]:
        created_at = created_at or utc_now()
        drafts: list[OrderIntent] = []
        for target in targets:
            security = securities[target.symbol]
            position = positions.get(target.symbol)
            current_qty = position.long_qty if position else 0
            delta = target.target_qty - current_qty
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            quantity = self.rules.normalize_quantity(security, abs(delta), side, created_at.date())
            if quantity <= 0:
                continue
            quote = quotes.get(target.symbol)
            limit_price = None
            if quote:
                limit_price = quote.ask1 if side == OrderSide.BUY else quote.bid1
            drafts.append(
                OrderIntent(
                    order_id=str(uuid4()),
                    signal_id=None,
                    symbol=target.symbol,
                    name=security.name,
                    board=security.board,
                    sector=security.sector,
                    side=side,
                    quantity=quantity,
                    limit_price=limit_price,
                    created_at=created_at,
                    eligible_at=created_at,
                    expires_at=created_at + timedelta(seconds=self.order_expiry_seconds),
                    reason=target.reason,
                    model_version=model_version,
                    status=OrderStatus.DRAFT,
                    is_margin_order=side == OrderSide.BUY,
                )
            )
        sells = [item for item in drafts if item.side in {OrderSide.SELL, OrderSide.BUY_TO_COVER}]
        buys = [item for item in drafts if item.side in {OrderSide.BUY, OrderSide.BORROW_SELL}]
        ordered = sorted(sells, key=lambda item: item.quantity, reverse=True) + sorted(
            buys, key=lambda item: item.quantity, reverse=True
        )
        selected: list[OrderIntent] = []
        remaining_turnover = (
            float("inf") if max_turnover_value is None else max(0.0, max_turnover_value)
        )
        for item in ordered:
            if len(selected) >= self.config.max_daily_action_orders:
                break
            quote = quotes.get(item.symbol)
            reference_price = quote.last if quote else item.limit_price
            if not reference_price or remaining_turnover <= 0:
                continue
            allowed_quantity = min(item.quantity, int(remaining_turnover / reference_price))
            security = securities[item.symbol]
            allowed_quantity = self.rules.normalize_quantity(
                security, allowed_quantity, item.side, created_at.date()
            )
            if allowed_quantity <= 0:
                continue
            selected_item = (
                item
                if allowed_quantity == item.quantity
                else replace(item, quantity=allowed_quantity)
            )
            selected.append(selected_item)
            remaining_turnover -= allowed_quantity * reference_price
        return selected
