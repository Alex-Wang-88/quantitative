import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from quantpaper.config import AppConfig, ExecutionConfig, ModelConfig, NotificationConfig
from quantpaper.data import ProviderUnavailable, make_demo_provider
from quantpaper.domain import OrderStatus, RunMode
from quantpaper.ledger import Ledger
from quantpaper.runtime import RuntimeService


def test_runtime_restarts_from_ledger_without_seeding_duplicate_orders(tmp_path) -> None:
    config = AppConfig(notifications=NotificationConfig(enabled=False))
    first = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    asyncio.run(first.tick())
    order_count = len(first.broker.orders)
    fill_count = len(first.broker.fills)
    position_count = len(first.broker.positions)

    second = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    assert len(second.broker.orders) == order_count
    assert len(second.broker.fills) == fill_count
    assert len(second.broker.positions) == position_count
    assert second.provider.cursor == 1


def test_runtime_dashboard_exposes_market_context_and_excess_return(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )

    asyncio.run(service.tick())
    market = service.dashboard()["market"]

    assert market["primary_benchmark"] == "000300"
    assert len(market["benchmarks"]) == 3
    assert market["strategy_day_return"] is not None
    assert market["excess_return"] is not None
    assert market["status"] == "REPLAY"


def test_runtime_rebalance_limits_manual_actions(tmp_path) -> None:
    config = AppConfig(notifications=NotificationConfig(enabled=False))
    service = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    result = service.rebalance()
    assert result["signals"]
    assert len(result["orders"]) <= config.portfolio.max_daily_action_orders


def test_runtime_rebalance_suppresses_recent_terminal_reissue(tmp_path) -> None:
    config = AppConfig(
        execution=ExecutionConfig(
            mode=RunMode.PAPER_MANUAL,
            order_reissue_cooldown_seconds=900,
        ),
        notifications=NotificationConfig(enabled=False),
    )
    service = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )

    first = service.rebalance()
    assert first["orders"]
    rejected = first["orders"][0]
    service.reject_order(rejected["order_id"])

    second = service.rebalance()
    assert rejected["symbol"] not in {
        order["symbol"] for order in second["orders"]
    }


def test_session_rebalance_window_is_only_afternoon_open_by_default(tmp_path) -> None:
    config = AppConfig(notifications=NotificationConfig(enabled=False))
    service = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    shanghai = ZoneInfo("Asia/Shanghai")

    assert service._session_open_key(datetime(2026, 9, 15, 12, 59, tzinfo=shanghai)) is None
    assert service._session_open_key(datetime(2026, 9, 15, 13, 0, tzinfo=shanghai)) == (
        "2026-09-15:PM_OPEN"
    )
    assert service._session_open_key(datetime(2026, 9, 15, 13, 14, tzinfo=shanghai)) == (
        "2026-09-15:PM_OPEN"
    )
    assert service._session_open_key(datetime(2026, 9, 15, 13, 15, tzinfo=shanghai)) is None


def test_runtime_status_prefers_newer_runner_data_snapshot(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    newer = service.clock + timedelta(days=10)
    service._shared_data_status = {
        "mode": "LIVE",
        "provider": "eastmoney_gm",
        "last_update": newer.isoformat(),
        "symbol_count": 500,
        "latency_seconds": 1.0,
        "message": "shared runner status",
    }

    assert service.status()["data"]["mode"] == "LIVE"
    assert service.status()["data"]["last_update"] == newer.isoformat()


def test_runtime_degrades_on_market_data_outage_and_recovers(tmp_path, monkeypatch) -> None:
    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )

    asyncio.run(service.tick())
    order_ids_before_outage = set(service.broker.orders)
    original_next_quotes = service.provider.next_quotes

    def raise_outage() -> list:
        raise ProviderUnavailable("模拟行情中断")

    monkeypatch.setattr(service.provider, "next_quotes", raise_outage)
    degraded = asyncio.run(service.tick())

    assert degraded["runner_status"] == "DEGRADED"
    assert degraded["last_error"] == "模拟行情中断"
    assert set(service.broker.orders) == order_ids_before_outage

    monkeypatch.setattr(service.provider, "next_quotes", original_next_quotes)
    recovered = asyncio.run(service.tick())

    assert recovered["runner_status"] == "RUNNING"
    assert recovered["last_error"] is None


def test_ui_process_confirmation_is_seen_by_runner_process(tmp_path) -> None:
    config = AppConfig(
        execution=ExecutionConfig(mode=RunMode.PAPER_MANUAL, manual_confirmation_delay_seconds=0),
        notifications=NotificationConfig(enabled=False),
    )
    db = tmp_path / "paper.db"
    runner = RuntimeService(
        config=config, provider=make_demo_provider(quote_steps=3), ledger=Ledger(db)
    )
    ui = RuntimeService(
        config=config, provider=make_demo_provider(quote_steps=3), ledger=Ledger(db)
    )
    order_id = next(
        order.order_id for order in ui.broker.orders.values() if order.status == OrderStatus.DRAFT
    )
    ui.confirm_order(order_id)
    asyncio.run(runner.tick())
    assert runner.broker.orders[order_id].status in {OrderStatus.FILLED, OrderStatus.PARTIAL_FILLED}


def test_trained_model_is_loaded_by_a_second_runtime_process(tmp_path) -> None:
    config = AppConfig(
        model=ModelConfig(
            artifact_dir=str(tmp_path / "models"),
            min_training_rows=500,
        ),
        notifications=NotificationConfig(enabled=False),
    )
    db = tmp_path / "paper.db"
    first = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(db),
    )
    artifact = first.train_local_model()
    assert first.trained_model is not None

    second = RuntimeService(
        config=config,
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(db),
    )
    assert second.trained_model is not None
    assert second.model_version == artifact["model_version"]
    assert second.status()["model_name"] == "lightgbm"
