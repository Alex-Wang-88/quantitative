import asyncio
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quantpaper.config import AppConfig, ExecutionConfig, ModelConfig, NotificationConfig
from quantpaper.data import ProviderUnavailable, make_demo_provider
from quantpaper.domain import (
    AccountSnapshot,
    OrderStatus,
    RunMode,
    TargetPosition,
)
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


def test_runtime_dashboard_exposes_enriched_rebalance_records(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )

    asyncio.run(service.tick())
    records = service.dashboard()["rebalance_records"]

    assert records
    assert {"timestamp", "symbol", "name", "industry", "side", "quantity"} <= set(
        records[0]
    )
    assert records[0]["quantity"] > 0


def test_runtime_generates_one_close_report_after_configured_time(tmp_path) -> None:
    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    shanghai = ZoneInfo("Asia/Shanghai")

    service._maybe_generate_close_report(datetime(2026, 1, 5, 15, 5, tzinfo=shanghai))
    service._maybe_generate_close_report(datetime(2026, 1, 5, 15, 20, tzinfo=shanghai))

    assert service.last_close_report == "2026-01-05"
    assert [item["report_date"] for item in service.ledger.list_reports()] == ["2026-01-05"]


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


def _account_snapshot(equity: float, initial: float = 1_000_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        account_id="safe_paper",
        initial_capital=initial,
        cash=equity,
        financing_debt=0.0,
        financing_interest=0.0,
        short_borrow_value=0.0,
        long_market_value=0.0,
        short_market_value=0.0,
        total_equity=equity,
        collateral_value=equity,
        maintenance_ratio=None,
        available_margin=None,
        gross_exposure=0.0,
        net_exposure=0.0,
        as_of=datetime(2026, 1, 5, 9, 30, tzinfo=UTC),
    )


def _targets(count: int, weight: float) -> list[TargetPosition]:
    return [
        TargetPosition(f"6000{index:02d}", weight, weight * 1_000_000, 100, "模型综合评分")
        for index in range(count)
    ]


def _runtime(tmp_path) -> RuntimeService:
    """A service with an explicitly empty book.

    ``_turnover_budget_rate`` reads ``broker.positions`` for the *current* gross
    exposure, and the demo provider arrives with a book already seeded, so the
    fixture has to be pinned to cash-only for the target-vs-current arithmetic to
    be predictable.
    """

    service = RuntimeService(
        config=AppConfig(notifications=NotificationConfig(enabled=False)),
        provider=make_demo_provider(quote_steps=3),
        ledger=Ledger(tmp_path / "paper.db"),
    )
    service.broker.positions.clear()
    # Seeded orders would already count against today's budget.
    service.broker.orders.clear()
    return service


def test_turnover_budget_exempts_the_deployment_gap(tmp_path) -> None:
    """Funding the book from cash is deployment, not rotation.

    The research backtester reserves the deficit to the target gross exposure
    before applying the rotation cap.  Without the same exemption the live path
    could only ever invest ``max_daily_turnover`` of a *fixed* base per session,
    so a 90% book took about five sessions where the backtest arrives in one --
    and the resulting low early exposure is indistinguishable, in a report, from
    the construction bug that produced the same symptom.
    """

    service = _runtime(tmp_path)
    # 30 names x 3% = 90% target against a cash-only book.
    rate = service._turnover_budget_rate(_targets(30, 0.03), _account_snapshot(1_000_000.0))

    assert rate == pytest.approx(0.90)


def test_turnover_budget_falls_back_to_the_rotation_cap_when_already_deployed(
    tmp_path,
) -> None:
    service = _runtime(tmp_path)
    # 5 names x 3% = 15% target, i.e. below the 20% rotation cap.
    rate = service._turnover_budget_rate(_targets(5, 0.03), _account_snapshot(1_000_000.0))

    assert rate == pytest.approx(0.20)


def test_turnover_budget_scales_with_current_equity_not_initial_capital(tmp_path) -> None:
    """A budget pinned to ``initial_capital`` stops scaling as the account grows.

    Both paths must measure the same thing or their equity curves cannot be
    compared: the base is equity *now*.
    """

    service = _runtime(tmp_path)
    targets = _targets(5, 0.03)

    at_capital = service._remaining_turnover_budget(
        targets, _account_snapshot(1_000_000.0), {}
    )
    doubled = service._remaining_turnover_budget(
        targets, _account_snapshot(2_000_000.0, initial=1_000_000.0), {}
    )

    assert at_capital == pytest.approx(200_000.0)
    # initial_capital is unchanged at 1,000,000, so the old formula would have
    # returned 200,000 here too.
    assert doubled == pytest.approx(400_000.0)


def test_status_publishes_the_peak_turnover_budget_like_research(tmp_path) -> None:
    service = _runtime(tmp_path)
    service._turnover_budget_rate(_targets(30, 0.03), _account_snapshot(1_000_000.0))

    # Same key name the research backtester publishes, so the two can be diffed.
    assert service.status()["peak_daily_turnover_budget"] == pytest.approx(0.90)
