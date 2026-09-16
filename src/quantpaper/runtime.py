from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from .broker import PaperBroker
from .config import AppConfig, load_config
from .data import MarketDataProvider, ProviderUnavailable, build_provider
from .domain import (
    Board,
    DataMode,
    OrderIntent,
    OrderSide,
    OrderStatus,
    QuoteSnapshot,
    RiskEvent,
    RunMode,
    RunnerStatus,
    Signal,
    jsonable,
    utc_now,
)
from .features import FeatureEngine
from .ledger import Ledger
from .model import FactorBaselineModel, LightGBMModel
from .notifications import NotificationSink, build_notifier
from .portfolio import OrderPlanner, PortfolioEngine
from .report import ReportGenerator
from .risk import RiskEngine
from .rules import TradingRules

logger = logging.getLogger(__name__)
SHANGHAI = ZoneInfo("Asia/Shanghai")
BENCHMARK_SPECS = (
    ("000001", "上证指数"),
    ("399001", "深证成指"),
    ("000300", "沪深300"),
)


class RuntimeService:
    """Application service shared by the API and the long-running runner."""

    def __init__(
        self,
        config: AppConfig | None = None,
        provider: MarketDataProvider | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.config = config or load_config()
        self.provider = provider or build_provider(self.config)
        startup_error: str | None = None
        try:
            self.securities = {item.symbol: item for item in self.provider.get_security_master()}
        except ProviderUnavailable as exc:
            self.securities = {}
            startup_error = str(exc)
            logger.warning("market data provider unavailable during startup: %s", exc)
        self.rules = TradingRules()
        self.ledger = ledger or Ledger()
        self.broker = PaperBroker(
            self.securities.values(),
            self.config.account,
            self.config.execution,
            self.rules,
            self.ledger,
        )
        self.feature_engine = FeatureEngine()
        self.baseline_model = FactorBaselineModel()
        self.trained_model: LightGBMModel | None = None
        self.portfolio_engine = PortfolioEngine(
            self.config.portfolio,
            self.rules,
            cash_floor=self.config.account.cash_floor,
        )
        self.order_planner = OrderPlanner(
            self.config.portfolio,
            self.rules,
            order_expiry_seconds=self.config.execution.order_expiry_seconds,
        )
        self.risk_engine = RiskEngine(self.config.account, self.config.portfolio)
        self.report_generator = ReportGenerator(ledger=self.ledger)
        self.notifier: NotificationSink = build_notifier(self.config.notifications.enabled)
        self.mode = self.config.execution.mode
        self.run_id = str(uuid4())
        self.strategy_version = "baseline-v0.1"
        self.runner_status = RunnerStatus.STOPPED
        self.stop_requested = False
        self.last_heartbeat: datetime | None = None
        self.last_error: str | None = startup_error
        self.last_session_rebalance: str | None = None
        self._shared_data_status: dict[str, Any] | None = None
        self.model_version = "factor-baseline-v0.1"
        self.model_name = "factor_baseline"
        self.model_metrics: dict[str, float] = {}
        self.model_artifact_path: str | None = None
        self.factor_coverage = 0.0
        self.signals: list[Signal] = []
        self.risk_events: list[RiskEvent] = []
        self.manual_records: list[dict[str, Any]] = []
        self._runner_task: asyncio.Task[None] | None = None
        self._tick_lock = asyncio.Lock()
        self.session_date: str | None = None
        self.session_start_equity: float | None = None
        self.equity_history: list[dict[str, Any]] = []
        self.market_overview: dict[str, Any] = self._empty_market_overview()
        self.clock = self._initial_clock()
        if self.broker.restore_state():
            self._restore_runtime()
        elif getattr(self.provider, "name", "") == "replay":
            self._seed_demo_state()
        else:
            # A live account starts empty.  Never populate fictional holdings or
            # signals merely because the real data source is unavailable.
            self._persist_runtime()
        self._load_persisted_model()

    def _initial_clock(self) -> datetime:
        try:
            quotes = self.provider.get_quotes()
            if quotes:
                benchmark_quotes = self._load_benchmark_quotes()
                if benchmark_quotes:
                    self._update_market_overview(benchmark_quotes)
                return max(item.timestamp for item in quotes)
        except ProviderUnavailable as exc:
            self.last_error = str(exc)
        except Exception as exc:
            logger.warning("unable to initialize runtime clock: %s", exc)
        return utc_now()

    def _restore_runtime(self) -> None:
        raw = self.ledger.load_snapshot("runtime") or {}
        if raw.get("mode"):
            try:
                self.mode = RunMode(raw["mode"])
            except ValueError:
                self.mode = self.config.execution.mode
        self.model_version = raw.get("model_version", self.model_version)
        self.model_name = raw.get("model_name", self.model_name)
        raw_metrics = raw.get("model_metrics")
        self.model_metrics = (
            {str(key): float(value) for key, value in raw_metrics.items()}
            if isinstance(raw_metrics, dict)
            else {}
        )
        self.model_artifact_path = raw.get("model_artifact_path")
        try:
            self.factor_coverage = float(raw.get("factor_coverage", self.factor_coverage))
        except (TypeError, ValueError):
            self.factor_coverage = 0.0
        restored_clock = raw.get("clock")
        if restored_clock:
            try:
                self.clock = datetime.fromisoformat(restored_clock)
            except ValueError:
                pass
        heartbeat = raw.get("last_heartbeat")
        if heartbeat:
            try:
                self.last_heartbeat = datetime.fromisoformat(heartbeat)
            except ValueError:
                pass
        if raw.get("runner_status"):
            try:
                self.runner_status = RunnerStatus(raw["runner_status"])
            except ValueError:
                self.runner_status = RunnerStatus.STOPPED
        self.last_error = raw.get("last_error")
        self.last_session_rebalance = raw.get("last_session_rebalance")
        shared_data = raw.get("data")
        self._shared_data_status = shared_data if isinstance(shared_data, dict) else None
        stored_market = raw.get("market")
        if isinstance(stored_market, dict):
            self.market_overview = stored_market
        self.session_date = raw.get("session_date")
        try:
            self.session_start_equity = (
                float(raw["session_start_equity"])
                if raw.get("session_start_equity") is not None
                else None
            )
        except (TypeError, ValueError):
            self.session_start_equity = None
        stored_history = raw.get("equity_history")
        if isinstance(stored_history, list):
            restored_history: list[dict[str, Any]] = []
            for point in stored_history[-120:]:
                if not isinstance(point, dict) or not point.get("as_of"):
                    continue
                try:
                    equity = float(point["equity"])
                except (KeyError, TypeError, ValueError):
                    continue
                restored_history.append(
                    {"as_of": str(point["as_of"]), "equity": round(equity, 2)}
                )
            self.equity_history = restored_history
        self.stop_requested = bool(raw.get("stop_requested", False))
        if hasattr(self.provider, "cursor") and raw.get("replay_cursor") is not None:
            self.provider.cursor = int(raw["replay_cursor"])

        self.signals = []
        for item in self.ledger.load_snapshot("signals") or []:
            try:
                self.signals.append(
                    Signal(
                        signal_id=item["signal_id"],
                        symbol=item["symbol"],
                        name=item["name"],
                        board=Board(item["board"]),
                        sector=item["sector"],
                        generated_at=datetime.fromisoformat(item["generated_at"]),
                        data_timestamp=datetime.fromisoformat(item["data_timestamp"]),
                        score=float(item["score"]),
                        action=item["action"],
                        target_weight=float(item["target_weight"]),
                        current_weight=float(item["current_weight"]),
                        reason=item["reason"],
                        model_version=item["model_version"],
                        status=item.get("status", "OPEN"),
                        run_id=item.get("run_id", ""),
                        strategy_version=item.get("strategy_version", "baseline-v0.1"),
                        updated_at=datetime.fromisoformat(item["updated_at"])
                        if item.get("updated_at")
                        else None,
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed persisted signal")
        self.risk_events = []
        for item in self.ledger.load_snapshot("risk_events") or []:
            try:
                self.risk_events.append(
                    RiskEvent(
                        event_id=item["event_id"],
                        severity=item["severity"],
                        code=item["code"],
                        message=item["message"],
                        created_at=datetime.fromisoformat(item["created_at"]),
                        symbol=item.get("symbol"),
                        acknowledged=bool(item.get("acknowledged", False)),
                        run_id=item.get("run_id", ""),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed persisted risk event")
        self.manual_records = list(self.ledger.load_snapshot("manual_records") or [])[-100:]

    def refresh_from_ledger(self) -> None:
        """Synchronize state with another UI or Runner process sharing the DB."""
        if self.broker.restore_state():
            self._restore_runtime()
        self._load_persisted_model()

    def _model_artifact_root(self) -> Path:
        configured = Path(self.config.model.artifact_dir)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parents[2] / configured

    def _walkforward_report_path(self) -> Path:
        configured = Path(self.config.model.walkforward_report_path)
        if configured.is_absolute():
            return configured
        return Path(__file__).resolve().parents[2] / configured

    def _load_walkforward_metrics(self) -> dict[str, Any]:
        """Load the latest offline research summary for close reports.

        This is deliberately read-only.  A walk-forward result is research
        evidence and never changes the active paper-trading model by itself.
        """
        path = self._walkforward_report_path()
        if not path.exists():
            return {
                "status": "UNAVAILABLE",
                "source": path.name,
                "message": "未找到 Walk-forward 研究报告",
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "DEGRADED",
                "source": path.name,
                "message": f"Walk-forward 报告读取失败: {exc}",
            }
        if not isinstance(payload, dict):
            return {
                "status": "DEGRADED",
                "source": path.name,
                "message": "Walk-forward 报告格式无效",
            }

        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
        folds = payload.get("folds") if isinstance(payload.get("folds"), list) else []

        def numeric(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def fold_values(key: str) -> list[float]:
            values: list[float] = []
            for fold in folds:
                if not isinstance(fold, dict) or not isinstance(fold.get("metrics"), dict):
                    continue
                value = numeric(fold["metrics"].get(key))
                if value is not None:
                    values.append(value)
            return values

        def average(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 6) if values else None

        test_ic = fold_values("test_ic_mean")
        test_spread = fold_values("test_top_bottom_spread")
        coverage = fold_values("factor_coverage_mean")
        last_fold = folds[-1] if folds and isinstance(folds[-1], dict) else {}
        last_model_version = (
            str(last_fold.get("model_version"))
            if last_fold.get("model_version")
            else None
        )
        fold_count = numeric(stats.get("fold_count"))
        return {
            "status": "OK" if bool(quality.get("ok")) else "DEGRADED",
            "source": path.name,
            "data_start": quality.get("start_date"),
            "data_end": quality.get("end_date"),
            "fold_count": int(fold_count) if fold_count is not None else len(folds),
            "last_model_version": last_model_version,
            "test_ic_mean": average(test_ic),
            "test_ic_positive_ratio": (
                round(sum(value > 0 for value in test_ic) / len(test_ic), 6)
                if test_ic
                else None
            ),
            "test_top_bottom_spread_mean": average(test_spread),
            "factor_coverage_mean": average(coverage),
            "total_return": numeric(stats.get("total_return")),
            "annualized_return": numeric(stats.get("annualized_return")),
            "sharpe": numeric(stats.get("sharpe")),
            "max_drawdown": numeric(stats.get("max_drawdown")),
            "turnover": numeric(stats.get("turnover")),
            "trade_count": numeric(stats.get("trade_count")),
            "message": "仅用于研究评估，不自动晋级纸盘模型",
        }

    def _load_persisted_model(self) -> None:
        """Load the same model artifact in the API and Runner processes.

        The UI and Runner intentionally have separate Python processes.  The
        artifact file plus the runtime snapshot is therefore the hand-off
        boundary; an in-memory model must never be the only copy of a trained
        model.
        """
        requested = self.model_artifact_path
        root = self._model_artifact_root()
        path: Path | None = None
        if requested:
            candidate = Path(str(requested))
            path = candidate if candidate.is_absolute() else root.parent.parent / candidate
        else:
            candidates = sorted(
                root.glob("*.pkl"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                path = candidates[0]

        if path is None or not path.exists():
            if requested and self.trained_model is not None:
                self.trained_model = None
                self.model_version = "factor-baseline-v0.1"
                self.model_name = "factor_baseline"
                self.model_metrics = {}
            if requested:
                self.last_error = f"模型文件不存在: {requested}"
            return

        normalized_path = str(path.resolve())
        current_path = (
            str(Path(self.model_artifact_path).resolve())
            if self.model_artifact_path
            else None
        )
        if self.trained_model is not None and current_path == normalized_path:
            return
        try:
            loaded = LightGBMModel.load(
                path,
                expected_feature_columns=list(self.feature_engine.FACTOR_COLUMNS),
            )
        except Exception as exc:
            self.trained_model = None
            self.model_version = "factor-baseline-v0.1"
            self.model_name = "factor_baseline"
            self.model_metrics = {}
            self.last_error = f"模型文件加载失败: {exc}"
            logger.warning("unable to load persisted model %s: %s", path, exc)
            return

        self.trained_model = loaded
        self.model_version = loaded.model_version
        self.model_name = loaded.artifact.model_name if loaded.artifact else "lightgbm"
        self.model_metrics = dict(loaded.artifact.metrics) if loaded.artifact else {}
        self.model_artifact_path = normalized_path
        self.last_error = None

    def _seed_demo_state(self) -> None:
        """Populate an explicit demo paper account so the first UI load is useful."""
        try:
            quotes = {item.symbol: item for item in self.provider.get_quotes()}
        except ProviderUnavailable as exc:
            self.runner_status = RunnerStatus.DEGRADED
            self.last_error = str(exc)
            self._persist_runtime()
            return
        for index, symbol in enumerate(list(self.securities)[:5]):
            security = self.securities[symbol]
            quote = quotes.get(symbol)
            if not quote:
                continue
            lot = self.rules.for_security(security, quote.timestamp.date()).lot_size
            quantity = lot * (4 + index)
            from .domain import Position

            self.broker.positions[symbol] = Position(
                symbol=symbol,
                name=security.name,
                board=security.board,
                sector=security.sector,
                last_price=quote.last,
                long_qty=quantity,
                long_available_qty=quantity,
                long_avg_cost=round(quote.last * (1 - 0.02 + index * 0.005), 2),
                last_quote_at=quote.timestamp,
            )
            self.broker.cash -= quantity * self.broker.positions[symbol].long_avg_cost

        self._seed_demo_signals(quotes)
        self._persist_runtime()

    def _seed_demo_signals(self, quotes: dict[str, Any]) -> None:
        symbols = list(self.securities)[:10]
        for index, symbol in enumerate(symbols):
            security = self.securities[symbol]
            quote = quotes.get(symbol)
            if not quote:
                continue
            action = "BUY" if index < 5 else "HOLD"
            signal = Signal(
                signal_id=f"demo-signal-{index + 1}",
                symbol=symbol,
                name=security.name,
                board=security.board,
                sector=security.sector,
                generated_at=self.clock,
                data_timestamp=quote.timestamp,
                score=round(0.94 - index * 0.037, 4),
                action=action,
                target_weight=0.04 if action == "BUY" else 0.03,
                current_weight=0.0,
                reason="演示信号：动量、质量、估值和流动性综合排名",
                model_version=self.model_version,
                run_id=self.run_id,
                strategy_version=self.strategy_version,
                updated_at=self.clock,
            )
            self.signals.append(signal)
            if action == "BUY" and index < 4:
                lot = self.rules.for_security(security, quote.timestamp.date()).lot_size
                order = OrderIntent(
                    order_id=f"demo-order-{index + 1}",
                    signal_id=signal.signal_id,
                    symbol=symbol,
                    name=security.name,
                    board=security.board,
                    sector=security.sector,
                    side=OrderSide.BUY,
                    quantity=lot * (2 + index),
                    limit_price=quote.ask1,
                    created_at=self.clock,
                    eligible_at=self.clock,
                    expires_at=self.clock + timedelta(minutes=5),
                    reason=signal.reason,
                    model_version=self.model_version,
                    status=OrderStatus.DRAFT,
                    run_id=self.run_id,
                    strategy_version=self.strategy_version,
                    data_timestamp=quote.timestamp,
                    updated_at=self.clock,
                )
                self.broker.add_order(order)

    async def start(self) -> dict[str, Any]:
        self.refresh_from_ledger()
        if self._runner_task and not self._runner_task.done():
            return self.status()
        self.stop_requested = False
        self.runner_status = RunnerStatus.RUNNING
        self.last_error = None
        self._persist_runtime()
        self._runner_task = asyncio.create_task(self._run_loop(), name="quantpaper-runner")
        self.notifier.notify("纸盘 Runner 已启动", f"模式：{self.mode.value}")
        return self.status()

    async def stop(self) -> dict[str, Any]:
        self.refresh_from_ledger()
        self.stop_requested = True
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
        self._runner_task = None
        self.runner_status = RunnerStatus.STOPPED
        self._persist_runtime()
        return self.status()

    async def _run_loop(self) -> None:
        interval = max(1, self.config.execution.quote_interval_seconds)
        while True:
            try:
                await self.tick()
                if self.stop_requested:
                    break
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # keep the runner alive and visible
                self.runner_status = RunnerStatus.ERROR
                self.last_error = str(exc)
                logger.exception("runner tick failed")
                self.notifier.notify("纸盘 Runner 异常", str(exc), "error")
                await asyncio.sleep(interval)

    async def tick(self) -> dict[str, Any]:
        async with self._tick_lock:
            self.refresh_from_ledger()
            if self.stop_requested:
                self.runner_status = RunnerStatus.STOPPED
                self._persist_runtime()
                return self.status()
            try:
                self._prepare_realtime_poll()
                quotes = self.provider.next_quotes()
            except ProviderUnavailable as exc:
                self.runner_status = RunnerStatus.DEGRADED
                self.last_error = str(exc)
                self.last_heartbeat = utc_now()
                self._persist_runtime()
                return self.status()
            if not quotes:
                self.runner_status = RunnerStatus.DEGRADED
                self.last_error = "行情源返回空数据"
                self._persist_runtime()
                return self.status()
            event_time = max(item.timestamp for item in quotes)
            previous_date = self.clock.date()
            account_before_quotes = self.broker.snapshot(self.clock)
            self._ensure_session_baseline(
                event_time.date().isoformat(), account_before_quotes.total_equity
            )
            self.clock = event_time
            if event_time.date() > previous_date:
                self._handle_new_trading_day(previous_date, event_time)
            self._maybe_rebalance_at_session_open(event_time)
            if self.mode == RunMode.PAPER_AUTO:
                self.broker.auto_confirm_drafts(event_time)
            fills = self.broker.process_quotes(quotes, event_time)
            snapshot = self.broker.snapshot(event_time)
            benchmark_quotes = self._load_benchmark_quotes()
            if benchmark_quotes:
                self._update_market_overview(benchmark_quotes)
            else:
                self.market_overview["status"] = "STALE"
                self.market_overview["benchmark_return"] = None
                self.market_overview["excess_return"] = None
                self.market_overview["message"] = "指数行情暂不可用；策略收益仍按账户净值计算"
            self._update_strategy_market_metrics(snapshot)
            self.risk_events = self.risk_engine.evaluate(
                snapshot, self.broker.positions, self._open_order_count(), event_time
            )
            self.last_heartbeat = utc_now()
            self.runner_status = RunnerStatus.RUNNING
            self.last_error = None
            for fill in fills:
                self.notifier.notify(
                    "模拟成交",
                    f"{fill.symbol} {fill.side.value} {fill.quantity} 股 @ {fill.price:.2f}",
                )
            for event in self.risk_events:
                if event.severity == "HIGH":
                    self.notifier.notify("风险事件", event.message, "error")
            self._persist_runtime()
            return self.status()

    def _prepare_realtime_poll(self) -> None:
        setter = getattr(self.provider, "set_realtime_symbols", None)
        if not callable(setter):
            return
        active = set(self.broker.positions)
        active.update(
            order.symbol
            for order in self.broker.orders.values()
            if order.status
            in {
                OrderStatus.DRAFT,
                OrderStatus.CONFIRMED,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIAL_FILLED,
            }
        )
        if not active:
            active.update(list(self.securities)[: max(30, self.config.portfolio.max_positions)])
        setter(active)

    def _handle_new_trading_day(self, previous_date: date, event_time: datetime) -> None:
        weekdays = 0
        cursor = previous_date
        while cursor < event_time.date():
            cursor = date.fromordinal(cursor.toordinal() + 1)
            if cursor.weekday() < 5:
                weekdays += 1
        self.broker.accrue_daily_costs(max(1, weekdays), event_time)
        if (
            self.config.model.auto_retrain
            and previous_date.weekday() == int(self.config.model.retrain_weekday)
        ):
            try:
                self._train_local_model(sync=False)
            except Exception as exc:
                # A failed weekly retrain must not prevent the account from
                # receiving its report or from using the last known model.
                self.last_error = f"周度模型重训失败，继续使用旧模型: {exc}"
                self.notifier.notify("模型重训失败", self.last_error, "warning")
        try:
            self.generate_report(report_date=previous_date.isoformat(), notify=True, sync=False)
            self.rebalance(sync=False)
        except Exception as exc:
            self.last_error = f"新交易日任务失败: {exc}"
            self.runner_status = RunnerStatus.DEGRADED
            self.notifier.notify("收盘任务异常", self.last_error, "error")

    def _maybe_rebalance_at_session_open(self, event_time: datetime) -> None:
        """Generate one paper order plan when the configured session opens.

        The Runner polls continuously, but a daily strategy should not create a
        new order plan on every quote.  The session key is persisted in the
        shared ledger so an API/Runner restart cannot duplicate this trigger.
        Stale or unavailable live data never starts a new order plan.
        """
        if self.mode not in {RunMode.PAPER_AUTO, RunMode.PAPER_MANUAL}:
            return
        execution = self.config.execution
        if not execution.session_rebalance_enabled:
            return
        session_key = self._session_open_key(event_time)
        if not session_key or session_key == self.last_session_rebalance:
            return
        data_mode = self.provider.status().mode
        if data_mode in {DataMode.STALE, DataMode.UNAVAILABLE}:
            self.last_error = f"开盘调仓已暂停：行情状态为 {data_mode.value}"
            self.runner_status = RunnerStatus.DEGRADED
            self.notifier.notify("开盘调仓暂停", self.last_error, "warning")
            return
        self.rebalance(sync=False)
        self.last_session_rebalance = session_key

    def _session_open_key(self, event_time: datetime) -> str | None:
        """Return a date/session key only inside the configured open window."""
        if event_time.tzinfo is None:
            local = event_time.replace(tzinfo=SHANGHAI)
        else:
            local = event_time.astimezone(SHANGHAI)
        configured = str(self.config.execution.session_rebalance).strip().upper()
        if configured == "BOTH":
            sessions = ("AM_OPEN", time(9, 30)), ("PM_OPEN", time(13, 0))
        elif configured == "AM_OPEN":
            sessions = (("AM_OPEN", time(9, 30)),)
        elif configured == "PM_OPEN":
            sessions = (("PM_OPEN", time(13, 0)),)
        else:
            return None
        window_minutes = max(1, int(self.config.execution.session_rebalance_window_minutes))
        for name, opening in sessions:
            start = datetime.combine(local.date(), opening, tzinfo=SHANGHAI)
            if start <= local < start + timedelta(minutes=window_minutes):
                return f"{local.date().isoformat()}:{name}"
        return None

    def set_mode(self, mode: RunMode) -> dict[str, Any]:
        self.refresh_from_ledger()
        self.mode = mode
        self._persist_runtime()
        return self.status()

    def confirm_order(self, order_id: str) -> OrderIntent:
        self.refresh_from_ledger()
        order = self.broker.confirm_order(order_id, self.clock)
        self._persist_runtime()
        return order

    def reject_order(self, order_id: str, reason: str = "USER_REJECTED") -> OrderIntent:
        self.refresh_from_ledger()
        order = self.broker.reject_order(order_id, reason)
        self._persist_runtime()
        return order

    def cancel_order(self, order_id: str, reason: str = "USER_CANCELLED") -> OrderIntent:
        self.refresh_from_ledger()
        order = self.broker.cancel_order(order_id, reason)
        self._persist_runtime()
        return order

    def modify_order(self, order_id: str, quantity: int) -> OrderIntent:
        self.refresh_from_ledger()
        order = self.broker.modify_order(order_id, quantity)
        self._persist_runtime()
        return order

    def rebalance(self, sync: bool = True) -> dict[str, Any]:
        if sync:
            self.refresh_from_ledger()
        daily = self.provider.get_daily_frame()
        provider_status = self.provider.status()
        if provider_status.mode in {DataMode.STALE, DataMode.UNAVAILABLE} and getattr(
            self.provider, "name", ""
        ) != "replay":
            reason = f"调仓已暂停：行情/因子状态为 {provider_status.mode.value}"
            self.last_error = reason
            self.runner_status = RunnerStatus.DEGRADED
            self._persist_runtime()
            return {
                "targets": [],
                "signals": [],
                "orders": [],
                "blocked_reason": reason,
            }
        latest = self.feature_engine.latest(daily)
        if "fundamental_coverage" in latest.columns:
            self.factor_coverage = float(latest["fundamental_coverage"].mean())
        scorer = self.trained_model or self.baseline_model
        scores = scorer.score(latest)
        candidate_limit = int(self.config.portfolio.candidate_pool_size)
        ranked_scores = scores.sort_values("score", ascending=False)
        candidate_scores = (
            ranked_scores
            if candidate_limit <= 0
            else ranked_scores.head(max(candidate_limit, self.config.portfolio.max_positions))
        )
        candidate_symbols = candidate_scores["symbol"].astype(str).tolist()
        candidate_symbols.extend(self.broker.positions)
        setter = getattr(self.provider, "set_realtime_symbols", None)
        if callable(setter):
            setter(candidate_symbols)
        candidate_quotes = self.provider.get_quotes(candidate_symbols)
        prices = {item.symbol: item.last for item in candidate_quotes}
        account = self.broker.snapshot(self.clock)
        targets, signals = self.portfolio_engine.build_targets(
            candidate_scores,
            account,
            self.securities,
            prices,
            positions=self.broker.positions,
            data_timestamp=self.clock,
            model_version=self.model_version,
        )
        quotes = {item.symbol: item for item in candidate_quotes}
        orders = self.order_planner.plan_orders(
            targets,
            self.broker.positions,
            self.securities,
            quotes,
            self.model_version,
            self.clock,
            max_turnover_value=self._remaining_turnover_budget(quotes),
        )
        existing_open_symbols = {
            item.symbol
            for item in self.broker.orders.values()
            if item.status
            in {
                OrderStatus.DRAFT,
                OrderStatus.CONFIRMED,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIAL_FILLED,
            }
        }
        remaining_actions = max(
            0, self.config.portfolio.max_daily_action_orders - self._today_order_count()
        )
        orders = [item for item in orders if item.symbol not in existing_open_symbols][
            :remaining_actions
        ]
        orders = self._filter_order_reissues(orders)
        signal_by_symbol = {item.symbol: item.signal_id for item in signals}
        orders = [
            replace(
                item,
                signal_id=signal_by_symbol.get(item.symbol),
                run_id=self.run_id,
                strategy_version=self.strategy_version,
                data_timestamp=self.clock,
                updated_at=self.clock,
            )
            for item in orders
        ]
        signals = [
            replace(
                item,
                run_id=self.run_id,
                strategy_version=self.strategy_version,
                updated_at=self.clock,
            )
            for item in signals
        ]
        self.signals = signals
        for order in orders:
            self.broker.add_order(order)
        if orders and self.mode == RunMode.PAPER_MANUAL:
            self.notifier.notify("出现待确认订单", f"新增 {len(orders)} 笔纸盘订单")
        self._persist_runtime()
        return {
            "targets": jsonable(targets),
            "signals": jsonable(signals),
            "orders": jsonable(orders),
        }

    def train_local_model(self) -> dict[str, Any]:
        self.refresh_from_ledger()
        return self._train_local_model(sync=False)

    def _train_local_model(self, sync: bool = True) -> dict[str, Any]:
        if sync:
            self.refresh_from_ledger()
        daily = self.provider.get_daily_frame()
        training = self.feature_engine.build_training_frame(
            daily, self.config.model.prediction_horizon_days
        )
        model = LightGBMModel(feature_columns=list(self.feature_engine.FACTOR_COLUMNS))
        artifact = model.fit(
            training,
            artifact_dir=self.config.model.artifact_dir,
            min_training_rows=self.config.model.min_training_rows,
            validation_fraction=self.config.model.validation_fraction,
            min_validation_days=self.config.model.min_validation_days,
        )
        self.trained_model = model
        self.model_version = artifact.model_version
        self.model_name = artifact.model_name
        self.model_metrics = dict(artifact.metrics)
        self.model_artifact_path = artifact.artifact_path
        self.strategy_version = "factor-model-v0.2"
        self._persist_runtime()
        return jsonable(artifact)

    def generate_report(
        self,
        report_date: str | None = None,
        notify: bool = True,
        sync: bool = True,
    ) -> dict[str, Any]:
        if sync:
            self.refresh_from_ledger()
        current_data_status = self._current_data_status()
        data_mode = getattr(current_data_status, "mode", None)
        if data_mode is None and isinstance(current_data_status, dict):
            data_mode = current_data_status.get("mode")
        data_mode = getattr(data_mode, "value", data_mode) or "UNKNOWN"
        report = self.report_generator.generate(
            self.broker,
            self.signals,
            self.risk_events,
            report_date=report_date or self.clock.date().isoformat(),
            model_version=self.model_version,
            as_of=self.clock,
            run_id=self.run_id,
            strategy_version=self.strategy_version,
            model_name=self.model_name,
            model_metrics=self.model_metrics,
            factor_coverage=self.factor_coverage,
            data_status=str(data_mode),
            research_metrics=self._load_walkforward_metrics(),
        )
        if notify:
            self.notifier.notify("收盘报告已生成", report.summary)
        return jsonable(report)

    def record_external_fill(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.refresh_from_ledger()
        record = {"recorded_at": utc_now().isoformat(), **payload}
        self.manual_records.append(record)
        self.ledger.append_event("manual_external_fill", record)
        return record

    def status(self) -> dict[str, Any]:
        data_status = self._current_data_status()
        return {
            "runner_status": self.runner_status.value,
            "stop_requested": self.stop_requested,
            "mode": self.mode.value,
            "run_id": self.run_id,
            "strategy_version": self.strategy_version,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "model_metrics": self.model_metrics,
            "model_artifact_path": self.model_artifact_path,
            "factor_coverage": self.factor_coverage,
            "data": jsonable(data_status),
            "last_heartbeat": jsonable(self.last_heartbeat),
            "clock": jsonable(self.clock),
            "last_error": self.last_error,
            "last_session_rebalance": self.last_session_rebalance,
            "session_date": self.session_date,
            "session_start_equity": self.session_start_equity,
            "market": self.market_overview,
            "notifications_enabled": self.config.notifications.enabled,
            "paper_only": True,
        }

    def _current_data_status(self) -> Any:
        """Prefer a newer Runner observation over the API process cache.

        The UI and Runner intentionally run as separate processes.  The UI's
        provider instance may not poll often enough to be authoritative, while
        the Runner writes its latest provider status into the shared runtime
        snapshot on every tick.
        """
        local_status = self.provider.status()
        shared = self._shared_data_status
        if not isinstance(shared, dict):
            return local_status
        shared_last = self._parse_status_timestamp(shared.get("last_update"))
        local_last = local_status.last_update
        if local_last is not None and local_last.tzinfo is None:
            local_last = local_last.replace(tzinfo=SHANGHAI)
        if shared_last is not None and (local_last is None or shared_last > local_last):
            return shared
        return local_status

    @staticmethod
    def _parse_status_timestamp(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=SHANGHAI)

    def dashboard(self) -> dict[str, Any]:
        self.refresh_from_ledger()
        snapshot = self.broker.snapshot(self.clock)
        market = self._market_overview_for_dashboard(snapshot)
        return {
            "status": self.status(),
            "account": jsonable(snapshot),
            "market": jsonable(market),
            "positions": jsonable(list(self.broker.positions.values())),
            "signals": jsonable(getattr(self, "signals", [])),
            "orders": jsonable(list(self.broker.orders.values())),
            "fills": jsonable(self.broker.fills[-100:]),
            "risk_events": jsonable(self.risk_events),
            "manual_records": self.manual_records[-20:],
            "reports": self.ledger.list_reports(),
            "equity_history": self.equity_history[-120:],
        }

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.ledger.recent_events(limit=limit)

    def _open_order_count(self) -> int:
        return sum(
            1
            for item in self.broker.orders.values()
            if item.status
            in {
                OrderStatus.DRAFT,
                OrderStatus.CONFIRMED,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIAL_FILLED,
            }
        )

    def _today_order_count(self) -> int:
        current_date = self.clock.date()
        return sum(
            1
            for item in self.broker.orders.values()
            if item.created_at.date() == current_date
            and item.status
            not in {OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}
        )

    def _filter_order_reissues(self, orders: list[OrderIntent]) -> list[OrderIntent]:
        """Suppress unchanged retries for a short cooldown after a terminal order.

        A manual rejection, expiry, or broker rejection should not immediately
        reappear on the next scoring/rebalance pass when the symbol, side, and
        reason are unchanged.  The complete order history remains in the
        ledger, and a later pass can retry after the configurable cooldown.
        """
        cooldown_seconds = max(0, self.config.execution.order_reissue_cooldown_seconds)
        if cooldown_seconds == 0:
            return orders
        now = utc_now()
        terminal_statuses = {
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
        recent_attempts: set[tuple[str, str, str]] = set()
        for previous in self.broker.orders.values():
            if previous.status not in terminal_statuses:
                continue
            reference = previous.updated_at or previous.created_at
            try:
                age_seconds = (now - reference).total_seconds()
            except TypeError:
                continue
            if 0 <= age_seconds < cooldown_seconds:
                recent_attempts.add(
                    (previous.symbol, previous.side.value, previous.reason)
                )
        return [
            item
            for item in orders
            if (item.symbol, item.side.value, item.reason) not in recent_attempts
        ]

    def _remaining_turnover_budget(self, quotes: dict[str, Any]) -> float:
        budget = self.config.account.initial_capital * self.config.portfolio.max_daily_turnover
        current_date = self.clock.date()
        used = 0.0
        for item in self.broker.orders.values():
            if item.created_at.date() != current_date:
                continue
            if item.status in {OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED}:
                continue
            quote = quotes.get(item.symbol)
            price = quote.last if quote else item.limit_price
            if price:
                used += item.quantity * price
        return max(0.0, budget - used)

    def _persist_runtime(self) -> None:
        snapshot = self.broker.snapshot(self.clock)
        self._record_equity_history(snapshot)
        payload = {
            **self.status(),
            "replay_cursor": getattr(self.provider, "cursor", None),
            "equity_history": self.equity_history,
        }
        self.ledger.save_snapshot("runtime", payload)
        self.ledger.save_snapshot("signals", self.signals)
        self.ledger.save_snapshot("risk_events", self.risk_events)
        self.ledger.save_snapshot("manual_records", self.manual_records[-100:])

    def _record_equity_history(self, snapshot: Any) -> None:
        """Keep a short persisted series for the dashboard sparkline.

        The old UI used a hard-coded rising polyline, which could contradict a
        losing account.  Recording the actual paper-account equity at each
        persisted runner state makes the direction and the history auditable.
        """
        point = {
            "as_of": self.clock.isoformat(),
            "equity": round(float(snapshot.total_equity), 2),
        }
        if self.equity_history and self.equity_history[-1]["as_of"] == point["as_of"]:
            self.equity_history[-1] = point
        else:
            self.equity_history.append(point)
        self.equity_history = self.equity_history[-120:]

    @staticmethod
    def _empty_market_overview() -> dict[str, Any]:
        return {
            "as_of": None,
            "status": "UNAVAILABLE",
            "provider": "--",
            "message": "尚未取得指数行情",
            "primary_benchmark": "000300",
            "primary_benchmark_name": "沪深300",
            "benchmark_return": None,
            "strategy_day_pnl": None,
            "strategy_day_return": None,
            "excess_return": None,
            "benchmarks": [
                {
                    "symbol": symbol,
                    "name": name,
                    "price": None,
                    "change_pct": None,
                    "source": "",
                    "as_of": None,
                }
                for symbol, name in BENCHMARK_SPECS
            ],
        }

    def _load_benchmark_quotes(self) -> list[QuoteSnapshot]:
        getter = getattr(self.provider, "get_benchmark_quotes", None)
        if not callable(getter):
            return []
        try:
            return list(getter())
        except ProviderUnavailable:
            return []
        except Exception as exc:
            logger.warning("benchmark quote request failed: %s", exc)
            return []

    def _update_market_overview(
        self,
        benchmark_quotes: list[QuoteSnapshot],
    ) -> None:
        by_symbol = {quote.symbol: quote for quote in benchmark_quotes}
        rows: list[dict[str, Any]] = []
        for symbol, name in BENCHMARK_SPECS:
            quote = by_symbol.get(symbol)
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "price": None if quote is None else round(quote.last, 2),
                    "change_pct": (
                        None
                        if quote is None or quote.previous_close <= 0
                        else round(quote.last / quote.previous_close - 1, 6)
                    ),
                    "source": "" if quote is None else quote.source,
                    "as_of": None if quote is None else quote.timestamp,
                }
            )
        primary = next(
            (row for row in rows if row["symbol"] == "000300" and row["change_pct"] is not None),
            next((row for row in rows if row["change_pct"] is not None), None),
        )
        status = self.provider.status()
        mode = getattr(status.mode, "value", status.mode)
        latest = max((quote.timestamp for quote in benchmark_quotes), default=None)
        self.market_overview = {
            **self._empty_market_overview(),
            "as_of": latest,
            "status": str(mode or "UNAVAILABLE"),
            "provider": getattr(status, "provider", self.provider.name),
            "message": (
                "回放数据使用股票池等权收益作为市场代理"
                if any(quote.source == "REPLAY_PROXY" for quote in benchmark_quotes)
                else "指数行情已更新"
            ),
            "benchmark_return": None if primary is None else primary["change_pct"],
            "benchmarks": rows,
        }
        if primary is not None:
            self.market_overview["primary_benchmark"] = primary["symbol"]
            self.market_overview["primary_benchmark_name"] = primary["name"]

    def _ensure_session_baseline(self, session_date: str, equity: float) -> None:
        if self.session_date == session_date and self.session_start_equity is not None:
            return
        self.session_date = session_date
        self.session_start_equity = float(equity)

    def _update_strategy_market_metrics(self, snapshot: Any) -> None:
        if self.session_start_equity is None or self.session_start_equity <= 0:
            return
        pnl = round(snapshot.total_equity - self.session_start_equity, 2)
        strategy_return = snapshot.total_equity / self.session_start_equity - 1
        benchmark_return = self.market_overview.get("benchmark_return")
        excess = (
            strategy_return - float(benchmark_return)
            if benchmark_return is not None
            else None
        )
        self.market_overview["strategy_day_pnl"] = pnl
        self.market_overview["strategy_day_return"] = round(strategy_return, 6)
        self.market_overview["excess_return"] = None if excess is None else round(excess, 6)

    def _market_overview_for_dashboard(self, snapshot: Any) -> dict[str, Any]:
        if self.session_start_equity is None:
            self._ensure_session_baseline(self.clock.date().isoformat(), snapshot.total_equity)
        self._update_strategy_market_metrics(snapshot)
        return self.market_overview
