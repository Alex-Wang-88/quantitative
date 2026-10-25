from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from .config import AccountConfig, PortfolioConfig
from .domain import AccountSnapshot, Position, RiskEvent, utc_now


class RiskEngine:
    def __init__(self, account: AccountConfig, portfolio: PortfolioConfig) -> None:
        self.account = account
        self.portfolio = portfolio

    def evaluate(
        self,
        snapshot: AccountSnapshot,
        positions: dict[str, Position],
        open_order_count: int,
        now: datetime | None = None,
    ) -> list[RiskEvent]:
        now = now or utc_now()
        events: list[RiskEvent] = []
        if snapshot.gross_exposure > self.account.max_gross_exposure + 1e-6:
            events.append(self._event("HIGH", "GROSS_EXPOSURE", "总敞口超过账户上限", now))
        if (
            snapshot.total_equity > 0
            and snapshot.cash / snapshot.total_equity < self.account.cash_floor - 1e-6
        ):
            events.append(self._event("MEDIUM", "CASH_FLOOR", "现金比例低于安全底线", now))
        if snapshot.maintenance_ratio is not None:
            if snapshot.maintenance_ratio < self.account.maintenance_ratio_liquidation:
                events.append(
                    self._event(
                        "HIGH", "MAINTENANCE_RATIO_LIQUIDATION", "维持担保比例进入模拟强平区间", now
                    )
                )
            elif snapshot.maintenance_ratio < self.account.maintenance_ratio_warning:
                events.append(
                    self._event("MEDIUM", "MAINTENANCE_RATIO", "维持担保比例进入模拟风险区间", now)
                )
        if open_order_count > self.portfolio.max_daily_action_orders:
            events.append(self._event("MEDIUM", "ORDER_LIMIT", "待处理订单超过人工操作上限", now))
        if len(positions) > self.portfolio.max_positions:
            events.append(self._event("MEDIUM", "POSITION_COUNT", "持仓数量超过组合上限", now))
        # The symmetric floor.  ``min_positions`` went unread for its entire
        # life, so a book that collapsed back to a handful of names looked
        # exactly like a book that was simply being prudent -- the failure the
        # floor exists to name.  A book with no positions at all is a separate
        # state (nothing built yet, or fully liquidated) and is not reported
        # here, or every session would open with a spurious event.
        if 0 < len(positions) < self.portfolio.min_positions:
            events.append(
                self._event(
                    "MEDIUM",
                    "POSITION_COUNT_FLOOR",
                    f"持仓数量 {len(positions)} 低于组合下限 {self.portfolio.min_positions}",
                    now,
                )
            )
        return events

    @staticmethod
    def _event(severity: str, code: str, message: str, now: datetime) -> RiskEvent:
        return RiskEvent(str(uuid4()), severity, code, message, now)
