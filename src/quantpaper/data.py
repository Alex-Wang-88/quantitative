from __future__ import annotations

import os
import random
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from .domain import Board, DataMode, DataStatus, QuoteSnapshot, Security, board_for_symbol
from .rules import TradingRules


class ProviderUnavailable(RuntimeError):
    """Raised when a configured data source cannot be used."""


class MarketDataProvider(Protocol):
    name: str

    def get_security_master(self) -> list[Security]: ...

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]: ...

    def next_quotes(self) -> list[QuoteSnapshot]: ...

    def get_daily_frame(self) -> pd.DataFrame: ...

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame: ...

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]: ...

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]: ...

    def get_trading_rules(self) -> TradingRules: ...

    def get_margin_snapshot(
        self, symbols: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]: ...

    def status(self) -> DataStatus: ...

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]: ...


@dataclass
class ReplayProvider:
    securities: list[Security]
    quote_frames: list[list[QuoteSnapshot]]
    daily_frame: pd.DataFrame
    name: str = "replay"
    cursor: int = 0
    repeat: bool = False

    def get_security_master(self) -> list[Security]:
        return list(self.securities)

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        if not self.quote_frames:
            return []
        frame = self.quote_frames[min(max(self.cursor - 1, 0), len(self.quote_frames) - 1)]
        allowed = set(symbols) if symbols is not None else None
        return [quote for quote in frame if allowed is None or quote.symbol in allowed]

    def next_quotes(self) -> list[QuoteSnapshot]:
        if not self.quote_frames:
            return []
        if self.cursor >= len(self.quote_frames) and not self.repeat:
            return []
        frame = self.quote_frames[self.cursor % len(self.quote_frames)]
        self.cursor += 1
        return list(frame)

    def get_daily_frame(self) -> pd.DataFrame:
        return self.daily_frame.copy()

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        frame = self.get_daily_frame()
        if symbols is None:
            return frame
        return frame[frame["symbol"].isin(set(symbols))].reset_index(drop=True)

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        return self.get_quotes(symbols)

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]:
        """Build deterministic market proxies from the replay universe.

        Replay data contains individual stocks rather than index constituents.
        These proxies therefore use equal-weight returns for the relevant
        board groups and are labelled as REPLAY by the runtime/UI.
        """
        quotes = self.get_quotes()
        if not quotes:
            return []

        groups = {
            "000001": [quote for quote in quotes if quote.symbol.startswith("6")],
            "399001": [
                quote
                for quote in quotes
                if quote.symbol.startswith(("0", "3"))
            ],
            "000300": quotes,
        }
        bases = {"000001": 3_800.0, "399001": 12_000.0, "000300": 4_000.0}
        latest = max(quote.timestamp for quote in quotes)
        result: list[QuoteSnapshot] = []
        for symbol, members in groups.items():
            if not members:
                continue
            returns = [
                quote.last / quote.previous_close - 1
                for quote in members
                if quote.previous_close > 0
            ]
            if not returns:
                continue
            change = sum(returns) / len(returns)
            previous = bases[symbol]
            result.append(
                QuoteSnapshot(
                    symbol=symbol,
                    timestamp=latest,
                    last=round(previous * (1 + change), 2),
                    previous_close=previous,
                    source="REPLAY_PROXY",
                )
            )
        return result

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]:
        dates = sorted({value.date() for value in pd.to_datetime(self.daily_frame["trade_date"])})
        return [
            value
            for value in dates
            if (start is None or value >= start) and (end is None or value <= end)
        ]

    def get_trading_rules(self) -> TradingRules:
        return TradingRules()

    def get_margin_snapshot(
        self, symbols: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]:
        allowed = set(symbols) if symbols is not None else None
        return {
            security.symbol: {
                "marginable": security.marginable,
                "shortable": security.shortable,
                "borrowable_quantity": 50_000 if security.shortable else 0,
            }
            for security in self.securities
            if allowed is None or security.symbol in allowed
        }

    def status(self) -> DataStatus:
        latest = None
        if self.quote_frames:
            frame = self.quote_frames[min(max(self.cursor - 1, 0), len(self.quote_frames) - 1)]
            if frame:
                latest = max(item.timestamp for item in frame)
        return DataStatus(
            mode=DataMode.REPLAY,
            provider=self.name,
            last_update=latest,
            symbol_count=len(self.securities),
            latency_seconds=None,
            message="本地回放数据；不会发送真实委托",
        )


class UnconfiguredRealtimeProvider:
    name = "unconfigured"

    def get_security_master(self) -> list[Security]:
        return []

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def next_quotes(self) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def get_daily_frame(self) -> pd.DataFrame:
        raise ProviderUnavailable("RealtimeProvider does not provide daily history")

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        raise ProviderUnavailable("RealtimeProvider does not provide daily history")

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def get_trading_rules(self) -> TradingRules:
        return TradingRules()

    def get_margin_snapshot(
        self, symbols: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]:
        raise ProviderUnavailable("RealtimeProvider is not configured")

    def status(self) -> DataStatus:
        return DataStatus(
            mode=DataMode.UNAVAILABLE,
            provider=self.name,
            last_update=None,
            symbol_count=0,
            latency_seconds=None,
            message="未配置实时行情源",
        )


class TushareEodProvider:
    """Small explicit adapter for end-of-day data.

    It is intentionally never used as an intraday source. The token is read
    from the environment and no network request is made at construction time.
    """

    name = "tushare_eod"

    def __init__(self, token: str | None = None) -> None:
        self.token = token or os.environ.get("QUANTPAPER_TUSHARE_TOKEN")
        self._pro = None

    def _client(self):
        if not self.token:
            raise ProviderUnavailable("QUANTPAPER_TUSHARE_TOKEN is not set")
        try:
            import tushare as ts
        except ImportError as exc:
            raise ProviderUnavailable(
                "install the optional 'data' dependencies for Tushare"
            ) from exc
        if self._pro is None:
            self._pro = ts.pro_api(self.token)
        return self._pro

    def get_security_master(self) -> list[Security]:
        pro = self._client()
        frame = pro.stock_basic(
            exchange="", list_status="L", fields="ts_code,name,exchange,list_date"
        )
        result: list[Security] = []
        for row in frame.itertuples(index=False):
            code = str(row.ts_code)
            board = board_for_symbol(code)
            result.append(
                Security(
                    symbol=code,
                    name=str(row.name),
                    exchange=str(row.exchange),
                    board=board,
                    sector="UNKNOWN",
                    listed_date=str(row.list_date),
                )
            )
        return result

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("TushareEodProvider is not an intraday realtime source")

    def next_quotes(self) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("TushareEodProvider is not an intraday realtime source")

    def get_daily_frame(self) -> pd.DataFrame:
        pro = self._client()
        trade_date = date.today().strftime("%Y%m%d")
        return pro.daily(trade_date=trade_date)

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        frame = self.get_daily_frame()
        if symbols is None:
            return frame
        return frame[frame["ts_code"].isin(set(symbols))].reset_index(drop=True)

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("TushareEodProvider is not an intraday realtime source")

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("TushareEodProvider does not expose index snapshots")

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]:
        pro = self._client()
        frame = pro.trade_cal(
            exchange="SSE",
            start_date=start.strftime("%Y%m%d") if start else None,
            end_date=end.strftime("%Y%m%d") if end else None,
            is_open=1,
        )
        return [datetime.strptime(str(value), "%Y%m%d").date() for value in frame["cal_date"]]

    def get_trading_rules(self) -> TradingRules:
        return TradingRules()

    def get_margin_snapshot(
        self, symbols: Iterable[str] | None = None
    ) -> dict[str, dict[str, object]]:
        raise ProviderUnavailable("TushareEodProvider does not expose a live margin snapshot")

    def status(self) -> DataStatus:
        return DataStatus(
            mode=DataMode.UNAVAILABLE if not self.token else DataMode.STALE,
            provider=self.name,
            last_update=None,
            symbol_count=0,
            latency_seconds=None,
            message="盘后数据适配器；需配置 token，不能用于盘中行情",
        )


class IfindProvider:
    """同花顺 iFinD 免费数据接口适配器。

    The SDK is optional and imported lazily.  This provider never fabricates a
    quote: missing credentials, missing SDK, quota errors, or malformed API
    responses are surfaced as ``ProviderUnavailable`` and reflected in
    ``status()``.  The universe can be configured explicitly with
    ``QUANTPAPER_IFIND_SYMBOLS`` or discovered from the iFinD A-share block;
    discovery is capped by the configured candidate-pool size.
    """

    name = "ifind"

    def __init__(
        self,
        symbols: Iterable[str] | None = None,
        stale_after_seconds: int = 180,
        history_lookback_days: int = 260,
        universe_limit: int = 500,
        api: Any | None = None,
    ) -> None:
        configured = list(symbols or self._symbols_from_environment())
        self.universe_limit = max(0, universe_limit)
        normalized = [self._normalize_symbol(value) for value in configured]
        self._configured_symbols = (
            normalized[: self.universe_limit] if self.universe_limit else normalized
        )
        self.stale_after_seconds = stale_after_seconds
        self.history_lookback_days = history_lookback_days
        self._api_override = api
        self._api_client: Any | None = None
        self._logged_in = False
        self._securities: dict[str, Security] = {}
        self._quotes: dict[str, QuoteSnapshot] = {}
        self._cumulative_volume: dict[str, float] = {}
        self._realtime_symbols: list[str] | None = None
        self._last_update: datetime | None = None
        self._last_error: str | None = None
        self._daily_cache: pd.DataFrame | None = None
        self._daily_cache_date: date | None = None

    @staticmethod
    def _symbols_from_environment() -> list[str]:
        raw = os.environ.get("QUANTPAPER_IFIND_SYMBOLS", "")
        return [item.strip() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _normalize_symbol(value: object) -> str:
        raw = str(value).strip().upper()
        if "." in raw:
            raw = raw.split(".", 1)[0]
        digits = "".join(character for character in raw if character.isdigit())
        return digits[-6:].zfill(6) if digits else raw

    @classmethod
    def _ifind_code(cls, symbol: str) -> str:
        symbol = cls._normalize_symbol(symbol)
        exchange = "SH" if symbol.startswith(("6", "68")) else "SZ"
        return f"{symbol}.{exchange}"

    @classmethod
    def _security(cls, symbol: str, name: str | None = None) -> Security:
        symbol = cls._normalize_symbol(symbol)
        board = board_for_symbol(symbol)
        return Security(
            symbol=symbol,
            name=name or symbol,
            exchange="SSE" if symbol.startswith(("6", "68")) else "SZSE",
            board=board,
            sector="UNKNOWN",
            listed_date="1900-01-01",
        )

    def _client(self) -> Any:
        if self._api_override is not None:
            return self._api_override
        if self._api_client is not None and self._logged_in:
            return self._api_client
        try:
            import iFinDPy as api
        except ImportError as exc:
            raise ProviderUnavailable(
                "iFinD SDK 未安装，请执行 pip install iFinDAPI"
            ) from exc
        account = os.environ.get("QUANTPAPER_IFIND_ACCOUNT", "")
        password = os.environ.get("QUANTPAPER_IFIND_PASSWORD", "")
        if not account or not password:
            raise ProviderUnavailable(
                "未配置 QUANTPAPER_IFIND_ACCOUNT/QUANTPAPER_IFIND_PASSWORD"
            )
        try:
            result = api.THS_iFinDLogin(account, password)
        except Exception as exc:
            raise ProviderUnavailable(f"iFinD 登录异常: {exc}") from exc
        if result not in {0, -201, "0", "-201", "SUCCESS", "success"}:
            raise ProviderUnavailable(f"iFinD 登录失败，错误码: {result}")
        self._api_client = api
        self._logged_in = True
        return api

    @staticmethod
    def _check_result(result: Any, method: str) -> Any:
        if isinstance(result, dict):
            errorcode = result.get("errorcode", 0)
            message = result.get("errmsg", "unknown error")
        else:
            errorcode = getattr(result, "errorcode", 0)
            message = getattr(result, "errmsg", "unknown error")
        if str(errorcode) not in {"0", "None"}:
            raise ProviderUnavailable(f"iFinD {method} 失败: {errorcode} {message}")
        return (
            result.get("data", result)
            if isinstance(result, dict)
            else getattr(result, "data", result)
        )

    @staticmethod
    def _as_frame(value: Any) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, dict):
            for key in ("data", "table", "tables"):
                nested = value.get(key)
                if isinstance(nested, pd.DataFrame):
                    return nested.copy()
                if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                    return pd.DataFrame(nested)
                if isinstance(nested, dict):
                    try:
                        return pd.DataFrame(nested)
                    except (TypeError, ValueError):
                        pass
            try:
                return pd.DataFrame(value)
            except (TypeError, ValueError):
                return pd.DataFrame()
        if isinstance(value, list):
            try:
                return pd.DataFrame(value)
            except (TypeError, ValueError):
                return pd.DataFrame()
        return pd.DataFrame()

    @staticmethod
    def _column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
        columns = {str(column).strip().casefold(): str(column) for column in frame.columns}
        for name in names:
            found = columns.get(name.casefold())
            if found:
                return found
        return None

    @classmethod
    def _value(cls, row: pd.Series, names: Iterable[str], default: Any = None) -> Any:
        columns = {str(column).strip().casefold(): column for column in row.index}
        for name in names:
            column = columns.get(name.casefold())
            if column is not None:
                value = row[column]
                if pd.notna(value):
                    return value
        return default

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if pd.notna(number) else None

    @staticmethod
    def _timestamp(value: Any, fallback: datetime) -> datetime:
        if value is None:
            return fallback
        try:
            parsed = pd.Timestamp(value).to_pydatetime()
        except (TypeError, ValueError):
            return fallback
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return parsed

    def get_security_master(self) -> list[Security]:
        if self._securities:
            return list(self._securities.values())
        if self._configured_symbols:
            self._securities = {
                symbol: self._security(symbol) for symbol in self._configured_symbols
            }
            return list(self._securities.values())
        try:
            api = self._client()
            data_pool = getattr(api, "THS_DataPool", None) or getattr(api, "THS_DP", None)
            if data_pool is None:
                raise ProviderUnavailable("iFinD SDK 缺少 THS_DataPool 股票池函数")
            result = self._check_result(
                data_pool(
                    "block",
                    "最新;001005010",
                    "date:Y,security_name:Y,thscode:Y",
                ),
                "THS_DataPool",
            )
            frame = self._as_frame(result)
            code_column = self._column(frame, ("thscode", "THSCODE", "code"))
            name_column = self._column(frame, ("security_name", "SECURITY_NAME", "name"))
            if not code_column:
                raise ProviderUnavailable("iFinD 股票池响应缺少 thscode 字段")
            discovered: dict[str, Security] = {}
            for row in frame.itertuples(index=False):
                row_series = pd.Series(row, index=frame.columns)
                code = self._normalize_symbol(row_series[code_column])
                if len(code) == 6:
                    discovered[code] = self._security(
                        code, str(row_series[name_column]) if name_column else code
                    )
            if self.universe_limit:
                discovered = dict(list(discovered.items())[: self.universe_limit])
            self._securities = discovered
        except ProviderUnavailable as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"iFinD 股票池解析失败: {exc}"
            raise ProviderUnavailable(self._last_error) from exc
        return list(self._securities.values())

    def get_quotes(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        securities = {item.symbol: item for item in self.get_security_master()}
        selected = (
            [self._normalize_symbol(value) for value in symbols]
            if symbols is not None
            else list(securities)
        )
        selected = [symbol for symbol in selected if symbol in securities]
        if not selected:
            raise ProviderUnavailable(
                "iFinD 股票池为空，请配置 data.universe 或 QUANTPAPER_IFIND_SYMBOLS"
            )
        indicators = os.environ.get(
            "QUANTPAPER_IFIND_REALTIME_INDICATORS",
            "open;high;low;latest;preClose;latestAmount;latestVolume;"
            "bid1;ask1;bidSize1;askSize1",
        )
        try:
            api = self._client()
            result_object = api.THS_RQ(
                ",".join(self._ifind_code(symbol) for symbol in selected), indicators, ""
            )
            result = self._check_result(result_object, "THS_RQ")
            frame = self._as_frame(result)
            if (
                frame.empty or self._column(frame, ("thscode", "THSCODE", "code")) is None
            ) and hasattr(api, "THS_Trans2DataFrame"):
                try:
                    frame = api.THS_Trans2DataFrame(result_object)
                except Exception:
                    frame = pd.DataFrame()
            if frame.empty:
                raise ProviderUnavailable("iFinD THS_RQ 返回空行情")
            code_column = self._column(frame, ("thscode", "THSCODE", "code"))
            times = getattr(result_object, "time", None)
            if times is None:
                times = []
            elif not isinstance(times, (list, tuple, pd.Series)):
                times = [times]
            else:
                times = list(times)
            quotes: list[QuoteSnapshot] = []
            now = datetime.now(ZoneInfo("Asia/Shanghai"))
            for position, (_, row) in enumerate(frame.iterrows()):
                row = row.copy()
                raw_code = (
                    row[code_column]
                    if code_column
                    else selected[position]
                    if position < len(selected)
                    else None
                )
                if raw_code is None:
                    continue
                symbol = self._normalize_symbol(raw_code)
                if symbol not in securities:
                    continue
                security = securities[symbol]
                last = self._float(self._value(row, ("latest", "new", "last", "close")))
                if last is None or last <= 0:
                    continue
                previous = self._float(
                    self._value(row, ("preClose", "pre_close", "preclose", "previous_close"))
                ) or self._quotes.get(symbol, QuoteSnapshot(symbol, now, last, last)).previous_close
                timestamp_value = self._value(row, ("time", "timestamp", "datetime"))
                if timestamp_value is None and position < len(times):
                    timestamp_value = times[position]
                timestamp = self._timestamp(timestamp_value, now)
                lower, upper = TradingRules().price_limits(previous, security, timestamp.date())
                bid = self._float(self._value(row, ("bid1", "buy1", "bid")))
                ask = self._float(self._value(row, ("ask1", "sell1", "ask")))
                bid_volume = int(self._float(self._value(row, ("bidSize1", "bid1Volume"))) or 0)
                ask_volume = int(self._float(self._value(row, ("askSize1", "ask1Volume"))) or 0)
                cumulative = self._float(
                    self._value(row, ("latestVolume", "volume", "vol"))
                )
                if cumulative is None:
                    bar_volume = max(bid_volume, ask_volume, 0)
                else:
                    previous_cumulative = self._cumulative_volume.get(symbol)
                    if previous_cumulative is None or cumulative < previous_cumulative:
                        bar_volume = 0
                    else:
                        bar_volume = int(max(0.0, cumulative - previous_cumulative))
                    self._cumulative_volume[symbol] = cumulative
                quotes.append(
                    QuoteSnapshot(
                        symbol=symbol,
                        timestamp=timestamp,
                        last=last,
                        previous_close=previous,
                        bid1=bid,
                        ask1=ask,
                        bid1_volume=bid_volume,
                        ask1_volume=ask_volume,
                        bar_volume=bar_volume,
                        upper_limit=upper,
                        lower_limit=lower,
                        source="IFIND",
                    )
                )
            if not quotes:
                raise ProviderUnavailable("iFinD THS_RQ 没有可用价格")
            self._quotes = {quote.symbol: quote for quote in quotes}
            self._last_update = max(quote.timestamp for quote in quotes)
            self._last_error = None
            return [quote for quote in quotes if quote.symbol in selected]
        except ProviderUnavailable as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"iFinD 行情解析失败: {exc}"
            raise ProviderUnavailable(self._last_error) from exc

    def next_quotes(self) -> list[QuoteSnapshot]:
        return self.get_quotes(self._realtime_symbols)

    def set_realtime_symbols(self, symbols: Iterable[str]) -> None:
        """Limit the high-frequency poll to positions and open orders.

        Daily history and close-time scoring still use the whole candidate pool.
        During the session the paper broker only needs fresh quotes for symbols
        that can be filled or whose positions need marking, which keeps a free
        API quota from being consumed by an unnecessary full-universe poll.
        """
        allowed = {security.symbol for security in self.get_security_master()}
        normalized = [self._normalize_symbol(value) for value in symbols]
        self._realtime_symbols = [symbol for symbol in normalized if symbol in allowed]

    def get_daily_frame(self) -> pd.DataFrame:
        today = date.today()
        if self._daily_cache is not None and self._daily_cache_date == today:
            return self._daily_cache.copy()
        securities = self.get_security_master()
        if not securities:
            raise ProviderUnavailable("iFinD 股票池为空")
        indicators = os.environ.get(
            "QUANTPAPER_IFIND_HISTORY_INDICATORS",
            "open;high;low;close;volume;amount",
        )
        start = today - timedelta(days=max(365, int(self.history_lookback_days * 1.6)))
        frames: list[pd.DataFrame] = []
        try:
            api = self._client()
            for offset in range(0, len(securities), 10):
                codes = [
                    self._ifind_code(security.symbol)
                    for security in securities[offset : offset + 10]
                ]
                result_object = api.THS_HistoryQuotes(
                    ",".join(codes),
                    indicators,
                    "period:D,pricetype:1,rptcategory:0,fqdate:1900-01-01,hb:YSHB",
                    start.isoformat(),
                    today.isoformat(),
                )
                result = self._check_result(result_object, "THS_HistoryQuotes")
                frame = self._as_frame(result)
                if frame.empty and hasattr(api, "THS_Trans2DataFrame"):
                    frame = api.THS_Trans2DataFrame(result_object)
                if not frame.empty:
                    frames.append(frame)
        except ProviderUnavailable as exc:
            self._last_error = str(exc)
            raise
        except Exception as exc:
            self._last_error = f"iFinD 历史行情失败: {exc}"
            raise ProviderUnavailable(self._last_error) from exc
        if not frames:
            raise ProviderUnavailable("iFinD 历史行情返回空数据")
        frame = self._normalize_history(pd.concat(frames, ignore_index=True))
        self._daily_cache = frame
        self._daily_cache_date = today
        return frame.copy()

    def _normalize_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        code_column = self._column(frame, ("thscode", "THSCODE", "code"))
        date_column = self._column(frame, ("time", "date", "trade_date"))
        if not code_column or not date_column:
            raise ProviderUnavailable("iFinD 历史行情缺少代码或日期字段")
        result = pd.DataFrame(
            {
                "symbol": frame[code_column].map(self._normalize_symbol),
                "trade_date": pd.to_datetime(frame[date_column], errors="coerce"),
            }
        )
        aliases = {
            "open": ("open",),
            "high": ("high",),
            "low": ("low",),
            "close": ("close", "latest"),
            "volume": ("volume", "latestVolume", "vol"),
            "amount": ("amount", "latestAmount"),
        }
        for target, candidates in aliases.items():
            source = self._column(frame, candidates)
            result[target] = pd.to_numeric(frame[source], errors="coerce") if source else pd.NA
        for symbol, security in self._securities.items():
            result.loc[result["symbol"] == symbol, "sector"] = security.sector
        return result.dropna(subset=["trade_date"]).reset_index(drop=True)

    def get_daily_bars(self, symbols: Iterable[str] | None = None) -> pd.DataFrame:
        frame = self.get_daily_frame()
        if symbols is None:
            return frame
        allowed = {self._normalize_symbol(symbol) for symbol in symbols}
        return frame[frame["symbol"].isin(allowed)].reset_index(drop=True)

    def get_intraday_bars(self, symbols: Iterable[str] | None = None) -> list[QuoteSnapshot]:
        return self.get_quotes(symbols)

    def get_benchmark_quotes(self) -> list[QuoteSnapshot]:
        raise ProviderUnavailable("IfindProvider index snapshots are not configured")

    def get_trading_calendar(
        self, start: date | None = None, end: date | None = None
    ) -> list[date]:
        start = start or date.today() - timedelta(days=365)
        end = end or date.today()
        try:
            result = self._check_result(
                self._client().THS_DateQuery(
                    "SSE", "dateType:0,period:D,dateFormat:0", start.isoformat(), end.isoformat()
                ),
                "THS_DateQuery",
            )
            if isinstance(result, dict):
                tables = result.get("tables", {})
                values = result.get("time") or (
                    tables.get("time") if isinstance(tables, dict) else []
                )
            else:
                frame = self._as_frame(result)
                date_column = self._column(frame, ("time", "date", "trade_date"))
                values = frame[date_column].tolist() if date_column else []
            return [pd.Timestamp(value).date() for value in values]
        except (ProviderUnavailable, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            raise ProviderUnavailable(self._last_error) from exc

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
            }
            for security in self.get_security_master()
            if allowed is None or security.symbol in allowed
        }

    def status(self) -> DataStatus:
        if self._last_update is None:
            mode = DataMode.UNAVAILABLE
            latency = None
        else:
            latency = max(
                0.0,
                (
                    datetime.now(ZoneInfo("Asia/Shanghai")) - self._last_update
                ).total_seconds(),
            )
            mode = (
                DataMode.LIVE
                if latency <= self.stale_after_seconds and self._last_error is None
                else DataMode.STALE
            )
        return DataStatus(
            mode=mode,
            provider=self.name,
            last_update=self._last_update,
            symbol_count=len(self._securities),
            latency_seconds=latency,
            message=self._last_error or "同花顺 iFinD 实时数据接口；不发送真实委托",
        )


def _make_securities() -> list[Security]:
    sectors = ["电子", "医药", "新能源", "先进制造", "消费", "金融"]
    securities: list[Security] = []
    for index in range(36):
        if index < 24:
            board = Board.MAIN
            prefix = "600" if index % 2 == 0 else "000"
            exchange = "SSE" if index % 2 == 0 else "SZSE"
        elif index < 30:
            board = Board.CHINEXT_300
            prefix = "300"
            exchange = "SZSE"
        else:
            board = Board.STAR_688
            prefix = "688"
            exchange = "SSE"
        symbol = f"{prefix}{index + 1:03d}"
        securities.append(
            Security(
                symbol=symbol,
                name=f"演示{sectors[index % len(sectors)]}{index + 1:02d}",
                exchange=exchange,
                board=board,
                sector=sectors[index % len(sectors)],
                listed_date="2015-01-01",
                marginable=board != Board.MAIN or index % 3 == 0,
                shortable=board != Board.MAIN and index % 2 == 0,
            )
        )
    return securities


def make_demo_provider(
    seed: int = 7, quote_steps: int = 240, replay_days: int = 20
) -> ReplayProvider:
    """Create deterministic demo data for the UI and replay tests."""
    rng = random.Random(seed)
    securities = _make_securities()
    prices = {security.symbol: round(rng.uniform(8, 80), 2) for security in securities}
    daily_rows: list[dict[str, object]] = []
    start = date(2025, 1, 2)
    for day_index in range(260):
        current_date = start + timedelta(days=day_index)
        if current_date.weekday() >= 5:
            continue
        for security in securities:
            previous = prices[security.symbol]
            drift = 0.0003 + (securities.index(security) % 7 - 3) * 0.0001
            pct = drift + rng.gauss(0, 0.018)
            close = max(2.0, round(previous * (1 + pct), 2))
            high = round(max(previous, close) * (1 + abs(rng.gauss(0, 0.006))), 2)
            low = round(min(previous, close) * (1 - abs(rng.gauss(0, 0.006))), 2)
            amount = round(rng.uniform(8_000_000, 120_000_000), 2)
            daily_rows.append(
                {
                    "symbol": security.symbol,
                    "trade_date": current_date,
                    "open": previous,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": int(amount / max(close, 1) * 100),
                    "amount": amount,
                    "turnover_rate": rng.uniform(0.3, 8.0),
                    "pe": rng.uniform(8, 65),
                    "pb": rng.uniform(0.8, 8),
                    "roe": rng.uniform(0.04, 0.28),
                    "profit_growth": rng.uniform(-0.15, 0.45),
                    "sector": security.sector,
                }
            )
            prices[security.symbol] = close

    daily_frame = pd.DataFrame(daily_rows)
    quote_frames: list[list[QuoteSnapshot]] = []
    session_start = datetime(2026, 1, 5, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    live_prices = {security.symbol: prices[security.symbol] for security in securities}
    trading_date = session_start.date()
    for _ in range(max(1, replay_days)):
        while trading_date.weekday() >= 5:
            trading_date += timedelta(days=1)
        day_start = datetime.combine(trading_date, session_start.timetz())
        for step in range(quote_steps):
            timestamp = day_start + timedelta(minutes=step)
            frame: list[QuoteSnapshot] = []
            for index, security in enumerate(securities):
                previous = live_prices[security.symbol]
                pct = rng.gauss(0, 0.0015) + (index % 9 - 4) * 0.00003
                last = max(1.0, round(previous * (1 + pct), 2))
                lower, upper = TradingLimitHelper.limits(previous, security.board)
                frame.append(
                    QuoteSnapshot(
                        symbol=security.symbol,
                        timestamp=timestamp,
                        last=last,
                        previous_close=previous,
                        bid1=round(max(lower, last - 0.01), 2),
                        ask1=round(min(upper, last + 0.01), 2),
                        bid1_volume=rng.randint(500, 80_000),
                        ask1_volume=rng.randint(500, 80_000),
                        bar_volume=rng.randint(10_000, 1_000_000),
                        upper_limit=upper,
                        lower_limit=lower,
                        source="REPLAY",
                    )
                )
                live_prices[security.symbol] = last
            quote_frames.append(frame)
        trading_date += timedelta(days=1)
    return ReplayProvider(securities, quote_frames, daily_frame)


class TradingLimitHelper:
    @staticmethod
    def limits(previous_close: float, board: Board) -> tuple[float, float]:
        pct = 0.20 if board in {Board.CHINEXT_300, Board.STAR_688} else 0.10
        return round(previous_close * (1 - pct), 2), round(previous_close * (1 + pct), 2)


def build_provider(config) -> MarketDataProvider:
    """Build only the provider explicitly selected by configuration.

    A missing live provider is intentionally represented as unavailable rather
    than silently replaced with demo data.
    """
    if str(config.data.mode).lower() == "replay":
        return make_demo_provider()
    universe = [
        value.strip()
        for value in str(getattr(config.data, "universe", "")).split(",")
        if value.strip()
    ]
    realtime_provider = str(getattr(config.data, "realtime_provider", "")).lower()
    historical_provider = str(getattr(config.data, "historical_provider", "")).lower()
    if {realtime_provider, historical_provider} & {
        "eastmoney",
        "eastmoney_gm",
        "eastmoney_readonly",
    }:
        from .eastmoney import EastmoneyProvider

        return EastmoneyProvider(
            symbols=universe,
            stale_after_seconds=int(getattr(config.data, "stale_after_seconds", 180)),
            history_lookback_days=int(getattr(config.data, "history_lookback_days", 260)),
            research_universe_limit=int(
                getattr(config.data, "research_universe_limit", 0)
            ),
            realtime_universe_limit=int(
                getattr(config.data, "realtime_universe_limit", 500)
            ),
            fundamentals_universe_limit=int(
                getattr(config.data, "fundamentals_universe_limit", 500)
            ),
            fundamentals_enabled=bool(
                getattr(config.data, "eastmoney_fundamentals_enabled", True)
            ),
            industry_enabled=bool(
                getattr(config.data, "eastmoney_industry_enabled", False)
            ),
            universe_limit=int(getattr(config.portfolio, "candidate_pool_size", 500)),
            python_executable=str(
                getattr(
                    config.data,
                    "eastmoney_sdk_python",
                    ".venv-eastmoney\\Scripts\\python.exe",
                )
            ),
            bridge_script=str(
                getattr(
                    config.data,
                    "eastmoney_bridge_script",
                    "scripts/eastmoney_readonly_bridge.py",
                )
            ),
            token_env=str(
                getattr(config.data, "eastmoney_token_env", "QUANTPAPER_EASTMONEY_TOKEN")
            ),
            terminal_adapter_enabled=bool(
                getattr(config.data, "eastmoney_terminal_adapter_enabled", True)
            ),
            terminal_feed_path=str(
                getattr(
                    config.data,
                    "eastmoney_terminal_feed_path",
                    "data/runtime/eastmoney-terminal-feed.json",
                )
            ),
        )
    if "ifind" in {realtime_provider, historical_provider}:
        return IfindProvider(
            symbols=universe,
            stale_after_seconds=int(getattr(config.data, "stale_after_seconds", 180)),
            universe_limit=int(getattr(config.portfolio, "candidate_pool_size", 500)),
        )
    if historical_provider in {"tushare", "tushare_eod"}:
        return TushareEodProvider()
    return UnconfiguredRealtimeProvider()
