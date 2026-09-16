from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import uuid
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .data import ProviderUnavailable
from .domain import Board, DataMode, DataStatus, QuoteSnapshot, Security, board_for_symbol
from .rules import TradingRules
from .terminal_feed import TerminalFeedStore

SHANGHAI = ZoneInfo("Asia/Shanghai")


class EastmoneyBridgeClient:
    """Persistent JSON-lines client for the SDK's separate Python runtime."""

    def __init__(
        self,
        python_executable: str = ".venv-eastmoney\\Scripts\\python.exe",
        bridge_script: str = "scripts/eastmoney_readonly_bridge.py",
        token_env: str = "QUANTPAPER_EASTMONEY_TOKEN",
        timeout_seconds: float = 120.0,
    ) -> None:
        self.python_executable = python_executable
        self.bridge_script = bridge_script
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._reader: threading.Thread | None = None
        self._closed = False
        atexit.register(self.close)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _user_environment_value(self, name: str) -> str:
        """Read a user-scoped Windows environment value after GUI setup.

        A desktop app may have been started before the token was configured,
        so its inherited environment can be stale.  This fallback reads only
        the named value and never logs or returns it to the application layer.
        """

        if os.name != "nt":
            return ""
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
            return str(value)
        except (FileNotFoundError, OSError):
            return ""

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        if not environment.get(self.token_env):
            configured = self._user_environment_value(self.token_env)
            if configured:
                environment[self.token_env] = configured
        return environment

    def _start(self) -> None:
        if self._closed:
            raise ProviderUnavailable("Eastmoney bridge client is closed")
        python_path = self._resolve_path(self.python_executable)
        script_path = self._resolve_path(self.bridge_script)
        if not python_path.exists():
            raise ProviderUnavailable(f"Eastmoney SDK Python 不存在: {python_path}")
        if not script_path.exists():
            raise ProviderUnavailable(f"Eastmoney 只读桥接脚本不存在: {script_path}")
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                [str(python_path), str(script_path)],
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
                env=self._environment(),
            )
        except OSError as exc:
            raise ProviderUnavailable(f"Eastmoney 只读桥接进程启动失败: {exc}") from exc
        self._process = process
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(process,),
            name="eastmoney-bridge-reader",
            daemon=True,
        )
        self._reader.start()
        try:
            ready = self._responses.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._stop_process()
            raise ProviderUnavailable("Eastmoney 只读桥接初始化超时") from exc
        if ready is None or not ready.get("ok") or ready.get("event") != "ready":
            self._stop_process()
            message = ready.get("error", "未收到就绪消息") if ready else "桥接进程已退出"
            raise ProviderUnavailable(f"Eastmoney 只读桥接初始化失败: {message}")

    def _read_responses(self, process: subprocess.Popen[str]) -> None:
        stdout = process.stdout
        if stdout is None:
            self._responses.put(None)
            return
        try:
            for line in stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    self._responses.put(value)
        finally:
            self._responses.put(None)

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def request(self, action: str, **payload: Any) -> Any:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._stop_process()
                self._start()
            process = self._process
            if process is None or process.stdin is None:
                raise ProviderUnavailable("Eastmoney 只读桥接进程不可用")
            request_id = uuid.uuid4().hex
            request = {"request_id": request_id, "action": action, **payload}
            try:
                process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                process.stdin.flush()
                response = self._responses.get(timeout=self.timeout_seconds)
            except (BrokenPipeError, OSError, queue.Empty) as exc:
                self._stop_process()
                if isinstance(exc, queue.Empty):
                    raise ProviderUnavailable("Eastmoney 只读桥接请求超时") from exc
                raise ProviderUnavailable("Eastmoney 只读桥接进程连接中断") from exc
            if response is None:
                self._stop_process()
                raise ProviderUnavailable("Eastmoney 只读桥接进程已退出")
            if not response.get("ok"):
                raise ProviderUnavailable(str(response.get("error", "Eastmoney API 请求失败")))
            return response.get("data")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop_process()


class EastmoneyProvider:
    """Read-only Eastmoney market-data adapter.

    The vendor SDK is isolated in :class:`EastmoneyBridgeClient` because its
    dependency pins conflict with the research environment.  This provider
    exposes only normalized market data to the strategy and never exposes an
    order-routing method.
    """

    name = "eastmoney_gm"
    BENCHMARKS = (
        ("000001", "SHSE"),  # 上证指数
        ("399001", "SZSE"),  # 深证成指
        ("000300", "SHSE"),  # 沪深300
    )

    def __init__(
        self,
        symbols: Iterable[str] | None = None,
        stale_after_seconds: int = 180,
        history_lookback_days: int = 260,
        universe_limit: int = 500,
        fundamentals_enabled: bool = True,
        industry_enabled: bool = False,
        research_universe_limit: int | None = None,
        realtime_universe_limit: int | None = None,
        fundamentals_universe_limit: int = 500,
        python_executable: str = ".venv-eastmoney\\Scripts\\python.exe",
        bridge_script: str = "scripts/eastmoney_readonly_bridge.py",
        token_env: str = "QUANTPAPER_EASTMONEY_TOKEN",
        terminal_adapter_enabled: bool = False,
        terminal_feed_path: str = "data/runtime/eastmoney-terminal-feed.json",
        client: EastmoneyBridgeClient | Any | None = None,
    ) -> None:
        configured = list(symbols or self._symbols_from_environment())
        legacy_limit = max(0, int(universe_limit))
        self.research_universe_limit = max(
            0,
            legacy_limit
            if research_universe_limit is None
            else int(research_universe_limit),
        )
        self.realtime_universe_limit = max(
            0,
            legacy_limit
            if realtime_universe_limit is None
            else int(realtime_universe_limit),
        )
        self.fundamentals_universe_limit = max(0, int(fundamentals_universe_limit))
        normalized = [self._normalize_symbol(value) for value in configured]
        self._configured_symbols = normalized
        self.stale_after_seconds = max(1, int(stale_after_seconds))
        self.history_lookback_days = max(20, int(history_lookback_days))
        self.fundamentals_enabled = bool(fundamentals_enabled)
        self.industry_enabled = bool(industry_enabled)
        self._client = client or EastmoneyBridgeClient(
            python_executable=python_executable,
            bridge_script=bridge_script,
            token_env=token_env,
        )
        feed_candidate = Path(terminal_feed_path)
        if not feed_candidate.is_absolute():
            feed_candidate = Path(__file__).resolve().parents[2] / feed_candidate
        self.terminal_adapter_enabled = bool(terminal_adapter_enabled)
        self.terminal_feed_store = TerminalFeedStore(feed_candidate)
        self._securities: dict[str, Security] = {}
        self._instrument_rows: dict[str, dict[str, Any]] = {}
        self._quotes: dict[str, QuoteSnapshot] = {}
        self._cumulative_volume: dict[str, float] = {}
        self._realtime_symbols: list[str] | None = None
        self._last_update: datetime | None = None
        self._last_error: str | None = None
        self._fundamental_warning: str | None = None
        self._daily_cache: pd.DataFrame | None = None
        self._daily_cache_date: date | None = None
        self._realtime_universe_symbols: list[str] = []

    @staticmethod
    def _symbols_from_environment() -> list[str]:
        raw = os.environ.get("QUANTPAPER_EASTMONEY_SYMBOLS", "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _normalize_symbol(value: object) -> str:
        raw = str(value).strip().upper()
        if "." in raw:
            raw = raw.split(".", 1)[1]
        digits = "".join(character for character in raw if character.isdigit())
        return digits[-6:].zfill(6) if digits else raw

    @classmethod
    def _gm_symbol(cls, value: object) -> str:
        symbol = cls._normalize_symbol(value)
        exchange = "SHSE" if symbol.startswith(("6", "68")) else "SZSE"
        return f"{exchange}.{symbol}"

    @classmethod
    def _board(cls, symbol: str) -> Board:
        return board_for_symbol(symbol)

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @staticmethod
    def _bool(value: Any) -> bool:
        return bool(value) and str(value).strip().lower() not in {"0", "false", "none"}

    @staticmethod
    def _timestamp(value: Any, fallback: datetime) -> datetime:
        if value is None or value == "":
            return fallback
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
        except (TypeError, ValueError):
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=SHANGHAI)
        return parsed

    @staticmethod
    def _date_key(values: Any) -> pd.Series:
        """Normalize vendor date values to local, timezone-naive dates."""

        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        return parsed.dt.tz_convert(SHANGHAI).dt.tz_localize(None).dt.normalize()

    def _request(self, action: str, **payload: Any) -> Any:
        try:
            result = self._client.request(action, **payload)
            self._last_error = None
            return result
        except ProviderUnavailable as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"Eastmoney {action} 失败: {exc}"
            raise ProviderUnavailable(self._last_error) from exc

    def _update_instruments(self, rows: Iterable[Mapping[str, Any]]) -> None:
        for row in rows:
            raw_symbol = row.get("symbol")
            if raw_symbol is None:
                continue
            symbol = self._normalize_symbol(raw_symbol)
            if len(symbol) != 6:
                continue
            name = str(row.get("sec_name") or row.get("sec_abbr") or symbol).strip()
            if name.upper().startswith(("ST", "*ST")):
                continue
            self._instrument_rows[symbol] = dict(row)
            exchange = str(row.get("exchange") or "")
            if exchange == "SHSE":
                exchange_name = "SSE"
            elif exchange == "SZSE":
                exchange_name = "SZSE"
            else:
                exchange_name = exchange or ("SSE" if symbol.startswith("6") else "SZSE")
            listed = row.get("listed_date")
            listed_date = str(listed)[:10] if listed else "1900-01-01"
            self._securities[symbol] = Security(
                symbol=symbol,
                name=name,
                exchange=exchange_name,
                board=self._board(symbol),
                sector="UNKNOWN",
                listed_date=listed_date,
                is_st=self._bool(row.get("is_st")),
                suspended=self._bool(row.get("is_suspended")),
                # The instrument metadata's margin_ratio is not proof that a
                # particular account can borrow the security.  Keep both off
                # until a dated credit-availability source is integrated.
                marginable=False,
                shortable=False,
            )

    @staticmethod
    def _spread_symbols(symbols: Iterable[str], limit: int) -> list[str]:
        ordered = sorted(set(symbols))
        if not limit or len(ordered) <= limit:
            return ordered
        if limit == 1:
            return [ordered[0]]
        indexes = [
            round(index * (len(ordered) - 1) / (limit - 1))
            for index in range(limit)
        ]
        return [ordered[index] for index in indexes]

    def get_security_master(self) -> list[Security]:
        if self._securities:
            return list(self._securities.values())
        payload: dict[str, Any] = {
            "limit": self.research_universe_limit,
            "exchanges": ["SHSE", "SZSE"],
            "sec_types": [1],
            "skip_suspended": True,
            "skip_st": True,
        }
        if self._configured_symbols:
            payload["symbols"] = [self._gm_symbol(value) for value in self._configured_symbols]
        rows = self._request("instruments", **payload) or []
        self._update_instruments(rows)
        if not self._securities:
            raise ProviderUnavailable("Eastmoney 股票列表为空")
        self._realtime_universe_symbols = self._spread_symbols(
            self._securities,
            self.realtime_universe_limit,
        )
        return list(self._securities.values())

    def _selected_symbols(self, symbols: Iterable[str] | None) -> list[str]:
        securities = self.get_security_master()
        allowed = {item.symbol for item in securities}
        selected = (
            [self._normalize_symbol(value) for value in symbols]
            if symbols is not None
            else (
                list(self._realtime_symbols)
                if self._realtime_symbols is not None
                else list(self._realtime_universe_symbols)
            )
        )
        if symbols is None and self._realtime_symbols is None and not selected:
            selected = list(allowed)
        selected = [symbol for symbol in selected if symbol in allowed]
        if not selected:
            raise ProviderUnavailable("Eastmoney 股票池没有可查询的代码")
        return selected

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        selected = self._selected_symbols(symbols)
        now = datetime.now(SHANGHAI)
        terminal_quotes = self._terminal_feed_quotes(selected)
        missing = [symbol for symbol in selected if symbol not in terminal_quotes]
        rows: list[Mapping[str, Any]] = []
        if missing:
            try:
                rows = self._request(
                    "quotes", symbols=[self._gm_symbol(symbol) for symbol in missing]
                ) or []
            except ProviderUnavailable:
                if not terminal_quotes:
                    raise
        quotes_by_symbol: dict[str, QuoteSnapshot] = dict(terminal_quotes)
        for row in rows:
            symbol = self._normalize_symbol(row.get("symbol"))
            if symbol not in selected:
                continue
            last = self._float(row.get("price"))
            if last is None or last <= 0:
                continue
            metadata = self._instrument_rows.get(symbol, {})
            previous = self._float(row.get("pre_close")) or self._float(
                metadata.get("pre_close")
            )
            if previous is None or previous <= 0:
                previous = self._quotes.get(
                    symbol, QuoteSnapshot(symbol, now, last, last)
                ).previous_close
            upper = self._float(row.get("upper_limit")) or self._float(metadata.get("upper_limit"))
            lower = self._float(row.get("lower_limit")) or self._float(metadata.get("lower_limit"))
            if upper is None or lower is None:
                lower, upper = TradingRules().price_limits(
                    previous,
                    self._securities[symbol],
                    now.date(),
                )
            cumulative_volume = self._float(row.get("cum_volume"))
            if cumulative_volume is None:
                bar_volume = 0
            else:
                previous_cumulative = self._cumulative_volume.get(symbol)
                bar_volume = (
                    0
                    if previous_cumulative is None or cumulative_volume < previous_cumulative
                    else int(max(0.0, cumulative_volume - previous_cumulative))
                )
                self._cumulative_volume[symbol] = cumulative_volume
            timestamp = self._timestamp(row.get("created_at"), now)
            quotes_by_symbol[symbol] = QuoteSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                last=last,
                previous_close=previous,
                bid1=self._float(row.get("bid1")),
                ask1=self._float(row.get("ask1")),
                bid1_volume=int(self._float(row.get("bid1_volume")) or 0),
                ask1_volume=int(self._float(row.get("ask1_volume")) or 0),
                bar_volume=bar_volume,
                upper_limit=upper,
                lower_limit=lower,
                suspended=self._bool(row.get("is_suspended"))
                or self._securities[symbol].suspended,
                source="EASTMONEY_GM_READONLY",
            )
        if not quotes_by_symbol:
            raise ProviderUnavailable("Eastmoney 实时行情没有可用价格")
        self._last_error = None
        self._quotes = dict(quotes_by_symbol)
        self._last_update = max(quote.timestamp for quote in quotes_by_symbol.values())
        return [quotes_by_symbol[symbol] for symbol in selected if symbol in quotes_by_symbol]

    def _terminal_feed_quotes(self, symbols: Iterable[str]) -> dict[str, QuoteSnapshot]:
        """Convert fresh terminal-adapter rows to normalized quote snapshots."""

        if not self.terminal_adapter_enabled:
            return {}
        selected = [self._normalize_symbol(symbol) for symbol in symbols]
        rows = self.terminal_feed_store.read_rows(
            [self._gm_symbol(symbol) for symbol in selected],
            stale_after_seconds=self.stale_after_seconds,
        )
        quotes: dict[str, QuoteSnapshot] = {}
        now = datetime.now(SHANGHAI)
        for row in rows:
            symbol = self._normalize_symbol(row.get("symbol"))
            if symbol not in selected or symbol not in self._securities:
                continue
            last = self._float(row.get("last"))
            if last is None or last <= 0:
                continue
            metadata = self._instrument_rows.get(symbol, {})
            previous = self._float(row.get("previous_close")) or self._float(
                metadata.get("pre_close")
            )
            if previous is None or previous <= 0:
                previous = self._quotes.get(
                    symbol, QuoteSnapshot(symbol, now, last, last)
                ).previous_close
            upper = self._float(row.get("upper_limit")) or self._float(
                metadata.get("upper_limit")
            )
            lower = self._float(row.get("lower_limit")) or self._float(
                metadata.get("lower_limit")
            )
            if upper is None or lower is None:
                lower, upper = TradingRules().price_limits(
                    previous,
                    self._securities[symbol],
                    now.date(),
                )
            quotes[symbol] = QuoteSnapshot(
                symbol=symbol,
                timestamp=self._timestamp(row.get("timestamp"), now),
                last=last,
                previous_close=previous,
                bid1=self._float(row.get("bid1")),
                ask1=self._float(row.get("ask1")),
                bid1_volume=int(self._float(row.get("bid1_volume")) or 0),
                ask1_volume=int(self._float(row.get("ask1_volume")) or 0),
                bar_volume=int(self._float(row.get("bar_volume")) or 0),
                upper_limit=upper,
                lower_limit=lower,
                suspended=self._securities[symbol].suspended,
                source=str(row.get("source") or "EASTMONEY_TERMINAL_ADAPTER"),
            )
        return quotes

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]:
        """Read index snapshots without adding indices to the stock universe."""
        symbols = [f"{exchange}.{symbol}" for symbol, exchange in self.BENCHMARKS]
        feed_rows = (
            self.terminal_feed_store.read_rows(
                symbols,
                stale_after_seconds=self.stale_after_seconds,
            )
            if self.terminal_adapter_enabled
            else []
        )
        rows_by_symbol = {
            self._normalize_symbol(row.get("symbol")): row for row in feed_rows
        }
        missing = [
            vendor_symbol
            for vendor_symbol in symbols
            if self._normalize_symbol(vendor_symbol) not in rows_by_symbol
        ]
        rows = []
        if missing:
            rows = self._request("quotes", symbols=missing, skip_metadata=True) or []
        now = datetime.now(SHANGHAI)
        quotes: list[QuoteSnapshot] = []
        for row in [*feed_rows, *rows]:
            symbol = self._normalize_symbol(row.get("symbol"))
            if symbol not in {item[0] for item in self.BENCHMARKS}:
                continue
            last = self._float(row.get("price") or row.get("last"))
            previous = self._float(row.get("pre_close") or row.get("previous_close"))
            if last is None or last <= 0 or previous is None or previous <= 0:
                continue
            quotes.append(
                QuoteSnapshot(
                    symbol=symbol,
                    timestamp=self._timestamp(row.get("created_at"), now),
                    last=last,
                    previous_close=previous,
                    bid1=self._float(row.get("bid1")),
                    ask1=self._float(row.get("ask1")),
                    source=str(row.get("source") or "EASTMONEY_GM_READONLY"),
                )
            )
        if not quotes:
            raise ProviderUnavailable("Eastmoney 指数行情没有可用价格")
        return quotes

    def next_quotes(self) -> list[QuoteSnapshot]:
        return self.get_quotes(self._realtime_symbols)

    def set_realtime_symbols(self, symbols: Iterable[str]) -> None:
        allowed = {security.symbol for security in self.get_security_master()}
        normalized = [self._normalize_symbol(value) for value in symbols]
        self._realtime_symbols = [symbol for symbol in normalized if symbol in allowed]

    def get_daily_frame(self) -> pd.DataFrame:
        today = date.today()
        if self._daily_cache is not None and self._daily_cache_date == today:
            return self._daily_cache.copy()
        securities = self.get_security_master()
        if not securities:
            raise ProviderUnavailable("Eastmoney 股票池为空")
        start = today - timedelta(days=max(365, int(self.history_lookback_days * 1.6)))
        rows = self._request(
            "history",
            symbols=[self._gm_symbol(security.symbol) for security in securities],
            frequency="1d",
            start_time=start.isoformat(),
            end_time=today.isoformat(),
            fields=[
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pre_close",
                "eob",
            ],
        ) or []
        if not rows:
            raise ProviderUnavailable("Eastmoney 历史日线返回空数据")
        frame = pd.DataFrame(rows)
        if "symbol" not in frame.columns:
            raise ProviderUnavailable("Eastmoney 历史日线缺少 symbol 字段")
        frame["symbol"] = frame["symbol"].map(self._normalize_symbol)
        date_column = "eob" if "eob" in frame.columns else "bob"
        if date_column not in frame.columns:
            raise ProviderUnavailable("Eastmoney 历史日线缺少交易日期字段")
        frame["trade_date"] = pd.to_datetime(frame[date_column], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount", "pre_close"):
            if column not in frame.columns:
                frame[column] = pd.NA
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if self.fundamentals_enabled:
            try:
                fundamental_securities = [
                    self._securities[symbol]
                    for symbol in self._spread_symbols(
                        self._securities,
                        self.fundamentals_universe_limit,
                    )
                ]
                fundamentals = self._request(
                    "fundamentals_history",
                    symbols=[
                        self._gm_symbol(security.symbol)
                        for security in fundamental_securities
                    ],
                    start_date=start.isoformat(),
                    end_date=today.isoformat(),
                    industry_enabled=self.industry_enabled,
                )
                frame = self._merge_fundamentals(frame, fundamentals)
                errors = fundamentals.get("errors") if isinstance(fundamentals, Mapping) else None
                if errors:
                    first = errors[0]
                    self._fundamental_warning = (
                        f"{len(errors)} 个基础面请求失败，首个失败项: "
                        f"{first.get('symbol', '')}/{first.get('dataset', '')}"
                    )
                else:
                    self._fundamental_warning = None
            except ProviderUnavailable as exc:
                # Technical prices remain usable, but the warning is surfaced
                # in the provider status so a report cannot look complete when
                # valuation or financial fields were unavailable.
                self._fundamental_warning = str(exc)
                self._last_error = None
        frame["sector"] = frame["symbol"].map(
            {symbol: security.sector for symbol, security in self._securities.items()}
        )
        frame = frame.dropna(subset=["trade_date", "close"]).sort_values(
            ["symbol", "trade_date"]
        )
        self._daily_cache = frame.reset_index(drop=True)
        self._daily_cache_date = today
        return self._daily_cache.copy()

    @classmethod
    def _merge_fundamentals(
        cls, price_frame: pd.DataFrame, payload: Any
    ) -> pd.DataFrame:
        """Merge dated valuation and point-in-time financial data.

        The bridge returns daily fields plus financial reports with their
        publication date.  Reports are joined with ``merge_asof`` backwards,
        so a report can only affect dates on or after it became public.
        """
        if not isinstance(payload, Mapping):
            return price_frame
        daily = pd.DataFrame(payload.get("daily") or [])
        reports = pd.DataFrame(payload.get("reports") or [])
        if daily.empty and reports.empty:
            return price_frame

        base = price_frame.copy()
        base["trade_date"] = cls._date_key(base["trade_date"])
        pieces: list[pd.DataFrame] = []
        if not daily.empty:
            daily = cls._normalise_fundamental_frame(daily, date_column="trade_date")
            rename = {
                "pe_ttm": "pe",
                "pb_lyr": "pb",
                "turnrate": "turnover_rate",
                "tot_mv": "market_cap",
                "a_mv_ex_ltd": "float_market_cap",
            }
            for source, target in rename.items():
                if source in daily.columns and target not in daily.columns:
                    daily[target] = daily[source]
            pieces.append(daily)

        if not reports.empty:
            reports = cls._normalise_fundamental_frame(reports, date_column="pub_date")
            if "ann_date" not in reports.columns and "pub_date" in reports.columns:
                reports["ann_date"] = reports["pub_date"]
            if "roe_weight" in reports.columns and "roe" not in reports.columns:
                reports["roe"] = reports["roe_weight"]
            if "roe_avg" in reports.columns and "roe" not in reports.columns:
                reports["roe"] = reports["roe_avg"]
            if "net_prof_pcom_yoy" in reports.columns and "profit_growth" not in reports.columns:
                reports["profit_growth"] = reports["net_prof_pcom_yoy"]
            if "inc_oper_yoy" in reports.columns and "revenue_growth" not in reports.columns:
                reports["revenue_growth"] = reports["inc_oper_yoy"]

        if pieces:
            dated_daily = pd.concat(pieces, ignore_index=True, sort=False)
            dated_daily = dated_daily.dropna(subset=["symbol", "trade_date"])
            dated_daily = dated_daily.groupby(
                ["symbol", "trade_date"], as_index=False, sort=False
            ).first()
            base = base.merge(
                dated_daily,
                on=["symbol", "trade_date"],
                how="left",
                suffixes=("", "_fundamental"),
            )
            for column in list(base.columns):
                if not column.endswith("_fundamental"):
                    continue
                target = column.removesuffix("_fundamental")
                if target in base.columns:
                    base[target] = base[target].where(base[target].notna(), base[column])
                    base = base.drop(columns=[column])

        if not reports.empty and {"symbol", "ann_date"}.issubset(reports.columns):
            report_columns = [
                column
                for column in reports.columns
                if column not in {"symbol", "pub_date"}
            ]
            report_frame = reports[["symbol", *report_columns]].dropna(
                subset=["symbol", "ann_date"]
            )
            report_frame = report_frame.sort_values(["symbol", "ann_date"])
            merged_pieces: list[pd.DataFrame] = []
            for symbol, current in base.groupby("symbol", sort=False):
                current = current.sort_values("trade_date")
                available = report_frame[report_frame["symbol"] == symbol]
                if available.empty:
                    merged_pieces.append(current)
                    continue
                merged = pd.merge_asof(
                    current,
                    available.sort_values("ann_date"),
                    left_on="trade_date",
                    right_on="ann_date",
                    direction="backward",
                    suffixes=("", "_report"),
                )
                for column in list(merged.columns):
                    if not column.endswith("_report"):
                        continue
                    target = column.removesuffix("_report")
                    if target in merged.columns:
                        merged[target] = merged[target].where(
                            merged[target].notna(), merged[column]
                        )
                        merged = merged.drop(columns=[column])
                merged_pieces.append(merged)
            base = pd.concat(merged_pieces, ignore_index=True, sort=False)

        return base

    @classmethod
    def _normalise_fundamental_frame(
        cls, frame: pd.DataFrame, *, date_column: str
    ) -> pd.DataFrame:
        result = frame.copy()
        if "symbol" in result.columns:
            result["symbol"] = result["symbol"].map(cls._normalize_symbol)
        if date_column in result.columns:
            result[date_column] = cls._date_key(result[date_column])
        for column in ("pub_date", "ann_date", "rpt_date"):
            if column in result.columns:
                result[column] = cls._date_key(result[column])
        for column in result.columns:
            if column in {"symbol", date_column, "pub_date", "ann_date", "rpt_date"}:
                continue
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        frame = self.get_daily_frame()
        if symbols is None:
            return frame
        allowed = {self._normalize_symbol(symbol) for symbol in symbols}
        return frame[frame["symbol"].isin(allowed)].reset_index(drop=True)

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        return self.get_quotes(symbols)

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]:
        start = start or date.today() - timedelta(days=365)
        end = end or date.today()
        values = self._request(
            "calendar",
            exchange="SHSE",
            start_date=start.isoformat(),
            end_date=end.isoformat(),
        ) or []
        return [pd.Timestamp(value).date() for value in values]

    def get_trading_rules(self) -> TradingRules:
        return TradingRules()

    def get_margin_snapshot(
        self, symbols: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]:
        allowed = {self._normalize_symbol(symbol) for symbol in symbols} if symbols else None
        return {
            security.symbol: {
                "marginable": False,
                "shortable": False,
                "borrowable_quantity": 0,
                "availability": "UNAVAILABLE",
            }
            for security in self.get_security_master()
            if allowed is None or security.symbol in allowed
        }

    def status(self) -> DataStatus:
        if self._last_update is None:
            mode = DataMode.UNAVAILABLE
            latency = None
        else:
            latency = max(0.0, (datetime.now(SHANGHAI) - self._last_update).total_seconds())
            mode = (
                DataMode.LIVE
                if latency <= self.stale_after_seconds and self._last_error is None
                else DataMode.STALE
            )
            if self._fundamental_warning:
                mode = DataMode.STALE
        message = self._last_error or (
            "东方财富掘金只读行情；不发送真实委托；"
            f"研究池 {len(self._securities)} 只，盘中默认行情池 "
            f"{len(self._realtime_universe_symbols)} 只"
        )
        if self._fundamental_warning:
            message = f"{message}；基础面因子不可用: {self._fundamental_warning}"
        return DataStatus(
            mode=mode,
            provider=self.name,
            last_update=self._last_update,
            symbol_count=len(self._securities),
            latency_seconds=latency,
            message=message,
        )
