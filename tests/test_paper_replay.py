from __future__ import annotations

import asyncio

from quantpaper.config import AppConfig, ExecutionConfig, ModelConfig, NotificationConfig
from quantpaper.data import make_demo_provider
from quantpaper.domain import RunMode
from quantpaper.ledger import Ledger
from quantpaper.runtime import RuntimeService


def test_replay_runs_ten_sessions_without_duplicate_orders(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(
            execution=ExecutionConfig(mode=RunMode.PAPER_AUTO),
            model=ModelConfig(
                artifact_dir=str(tmp_path / "models"),
                walkforward_report_path=str(tmp_path / "missing-walkforward.json"),
            ),
            notifications=NotificationConfig(enabled=False),
        ),
        provider=make_demo_provider(quote_steps=2, replay_days=10),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.report_generator.output_dir = tmp_path / "reports"
    service.report_generator.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_sessions() -> None:
        for _ in range(20):
            await service.tick()

    asyncio.run(run_sessions())

    order_ids = list(service.broker.orders)
    fill_ids = [fill.fill_id for fill in service.broker.fills]
    reports = service.ledger.list_reports(limit=20)

    assert service.runner_status.value == "RUNNING"
    assert service.clock.date().isoformat() == "2026-01-16"
    assert len(reports) >= 9
    assert len(order_ids) == len(set(order_ids))
    assert len(fill_ids) == len(set(fill_ids))
    assert service.broker.snapshot(service.clock).total_equity > 0
