"""JSON-lines bridge for the Eastmoney Juejin SDK.

This process is deliberately kept outside the main environment because the
SDK pins an older pandas/numpy pair.  It imports the SDK's query functions
only; no order function is called by this bridge.  The main application talks
to this process through stdin/stdout and keeps PaperBroker as its only
execution path.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import math
import os
import sys
from collections.abc import Mapping
from typing import Any

PROVIDER_NAME = "eastmoney_gm"
TOKEN_ENV = "QUANTPAPER_EASTMONEY_TOKEN"


def _json_safe(value: Any) -> Any:
    """Convert SDK values into strict JSON values without leaking secrets."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    return str(value)


def _records(value: Any) -> list[Mapping[str, Any]]:
    """Turn SDK list/dict/DataFrame results into mapping rows."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
            if isinstance(records, list):
                return [row for row in records if isinstance(row, Mapping)]
        except Exception:
            pass
    try:
        return [row for row in value if isinstance(row, Mapping)]
    except TypeError:
        return []


def _normalise_symbol(value: object) -> str:
    raw = str(value).strip().upper()
    if "." in raw:
        return raw
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return raw
    code = digits[-6:].zfill(6)
    exchange = "SHSE" if code.startswith(("6", "68")) else "SZSE"
    return f"{exchange}.{code}"


def _selected_instrument(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "symbol",
        "sec_id",
        "exchange",
        "sec_name",
        "sec_abbr",
        "sec_type",
        "pre_close",
        "upper_limit",
        "lower_limit",
        "is_suspended",
        "listed_date",
        "trade_date",
        "margin_ratio",
        "board",
        "price_tick",
    )
    return {field: _json_safe(row.get(field)) for field in fields if field in row}


def _safe_error(exc: BaseException) -> str:
    message = str(exc).replace(TOKEN_ENV, "TOKEN")
    token = os.environ.get(TOKEN_ENV, "")
    if token:
        message = message.replace(token, "<redacted>")
    return message[:1000] or type(exc).__name__


class ReadOnlyService:
    def __init__(self) -> None:
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not token:
            raise RuntimeError(f"{TOKEN_ENV} is not set")

        # Import only the data/query entry points used below.  gm.api itself
        # exposes trade functions for the vendor, but this bridge never calls
        # them and does not accept any order-related command.
        with contextlib.redirect_stdout(io.StringIO()):
            from gm.api import (  # type: ignore[import-not-found]
                current,
                get_instruments,
                get_trading_dates,
                history,
                set_token,
                stk_get_daily_basic,
                stk_get_daily_mktvalue,
                stk_get_daily_valuation,
                stk_get_finance_deriv,
                stk_get_symbol_industry,
            )
            from gm.csdk.c_sdk import gmi_init  # type: ignore[import-not-found]

            set_token(token)
            init_status = gmi_init()

        if init_status not in (None, 0):
            raise RuntimeError(f"Eastmoney SDK init failed: {init_status}")
        self._current = current
        self._get_instruments = get_instruments
        self._get_trading_dates = get_trading_dates
        self._history = history
        self._stk_get_daily_basic = stk_get_daily_basic
        self._stk_get_daily_mktvalue = stk_get_daily_mktvalue
        self._stk_get_daily_valuation = stk_get_daily_valuation
        self._stk_get_finance_deriv = stk_get_finance_deriv
        self._stk_get_symbol_industry = stk_get_symbol_industry
        self._instrument_cache: dict[str, dict[str, Any]] = {}
        self._init_status = init_status

    def _call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        # Some SDK builds write connection diagnostics to stdout.  Keep the
        # stdout channel reserved for the JSON-lines protocol.
        with contextlib.redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def _load_instruments(self, request: Mapping[str, Any]) -> list[dict[str, Any]]:
        symbols = request.get("symbols")
        normalised = [_normalise_symbol(value) for value in symbols or []]
        exchanges = request.get("exchanges") or ["SHSE", "SZSE"]
        sec_types = request.get("sec_types") or [1]
        limit = max(0, int(request.get("limit") or 0))
        rows = self._call(
            self._get_instruments,
            symbols=normalised or None,
            exchanges=exchanges,
            sec_types=sec_types,
            skip_suspended=bool(request.get("skip_suspended", True)),
            skip_st=bool(request.get("skip_st", True)),
            df=False,
        )
        result: list[dict[str, Any]] = []
        for raw in rows or []:
            item = _selected_instrument(raw)
            symbol = str(item.get("symbol") or "")
            if not symbol:
                continue
            item["symbol"] = _normalise_symbol(symbol)
            self._instrument_cache[item["symbol"]] = item
            result.append(item)
        result.sort(key=lambda item: str(item.get("symbol", "")))
        if limit and len(result) > limit:
            # Deterministic spread across the sorted A-share universe. This is
            # used for an explicitly capped pool; a zero limit keeps the full
            # research universe.
            indexes = (
                [0]
                if limit == 1
                else [
                    round(index * (len(result) - 1) / (limit - 1))
                    for index in range(limit)
                ]
            )
            result = [result[index] for index in indexes]
        return result

    def _ensure_instrument_metadata(self, symbols: list[str]) -> None:
        missing = [symbol for symbol in symbols if symbol not in self._instrument_cache]
        if not missing:
            return
        self._load_instruments(
            {
                "symbols": missing,
                "skip_suspended": False,
                "skip_st": False,
            }
        )

    def _fundamentals_history(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Read dated valuation and point-in-time financial fields.

        The SDK's range functions accept one stock at a time.  This bridge
        keeps that vendor-specific loop outside the main application and
        returns partial data plus explicit errors rather than fabricating
        neutral financial values when an account lacks a data permission.
        """
        symbols = [_normalise_symbol(value) for value in request.get("symbols") or []]
        if not symbols:
            raise ValueError("fundamentals_history requires symbols")
        start_date = str(request.get("start_date") or "")
        end_date = str(request.get("end_date") or "")
        daily: list[dict[str, Any]] = []
        reports: list[dict[str, Any]] = []
        industries: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        industry_enabled = bool(request.get("industry_enabled", False))
        industry_source = str(request.get("industry_source") or "sw2021")
        industry_level = max(1, int(request.get("industry_level") or 1))
        industry_date = str(request.get("industry_date") or end_date or "")

        daily_queries = (
            (
                "valuation",
                self._stk_get_daily_valuation,
                "pe_ttm,pb_lyr,dy_ttm",
            ),
            (
                "mktvalue",
                self._stk_get_daily_mktvalue,
                "tot_mv,a_mv_ex_ltd",
            ),
            ("basic", self._stk_get_daily_basic, "turnrate"),
        )
        for symbol in symbols:
            for name, function, fields in daily_queries:
                try:
                    values = self._call(
                        function,
                        symbol=symbol,
                        fields=fields,
                        start_date=start_date,
                        end_date=end_date,
                        df=False,
                    )
                    for value in _records(values):
                        row = dict(value)
                        row["symbol"] = _normalise_symbol(row.get("symbol") or symbol)
                        daily.append(_json_safe(row))
                except Exception as exc:
                    errors.append({"symbol": symbol, "dataset": name, "error": _safe_error(exc)})

            try:
                values = self._call(
                    self._stk_get_finance_deriv,
                    symbol=symbol,
                    fields="roe_weight,net_prof_pcom_yoy,inc_oper_yoy",
                    start_date=start_date,
                    end_date=end_date,
                    df=False,
                )
                for value in _records(values):
                    row = dict(value)
                    row["symbol"] = _normalise_symbol(row.get("symbol") or symbol)
                    reports.append(_json_safe(row))
            except Exception as exc:
                errors.append(
                    {"symbol": symbol, "dataset": "finance_deriv", "error": _safe_error(exc)}
                )

            if industry_enabled:
                # The industry endpoint is separately permissioned. Keep the
                # requested as-of date explicit so historical callers cannot
                # silently apply a current classification backwards.
                try:
                    values = self._call(
                        self._stk_get_symbol_industry,
                        symbols=symbol,
                        source=industry_source,
                        level=industry_level,
                        date=industry_date,
                    )
                    for value in _records(values):
                        row = dict(value)
                        row["symbol"] = _normalise_symbol(row.get("symbol") or symbol)
                        row["industry_source"] = industry_source
                        row["industry_level"] = industry_level
                        row["industry_as_of"] = industry_date
                        industries.append(_json_safe(row))
                except Exception as exc:
                    errors.append(
                        {
                            "symbol": symbol,
                            "dataset": "symbol_industry",
                            "error": _safe_error(exc),
                        }
                    )
        return {
            "daily": daily,
            "reports": reports,
            "industries": industries,
            "errors": errors,
        }

    def handle(self, request: Mapping[str, Any]) -> Any:
        action = str(request.get("action", "")).strip().lower()
        if action == "health":
            return {
                "provider": PROVIDER_NAME,
                "read_only": True,
                "sdk_init_status": self._init_status,
            }
        if action == "instruments":
            return self._load_instruments(request)
        if action == "quotes":
            symbols = [_normalise_symbol(value) for value in request.get("symbols") or []]
            if not symbols:
                raise ValueError("quotes requires symbols")
            if not bool(request.get("skip_metadata", False)):
                self._ensure_instrument_metadata(symbols)
            rows = self._call(self._current, symbols)
            result: list[dict[str, Any]] = []
            for raw in rows or []:
                item = dict(raw)
                symbol = _normalise_symbol(item.get("symbol", ""))
                if not symbol:
                    continue
                item["symbol"] = symbol
                instrument = self._instrument_cache.get(symbol, {})
                quotes = item.get("quotes") or []
                if quotes:
                    top = quotes[0]
                    item["bid1"] = top.get("bid_p")
                    item["ask1"] = top.get("ask_p")
                    item["bid1_volume"] = top.get("bid_v")
                    item["ask1_volume"] = top.get("ask_v")
                item["pre_close"] = instrument.get("pre_close") or item.get("pre_close")
                item["upper_limit"] = instrument.get("upper_limit") or item.get("upper_limit")
                item["lower_limit"] = instrument.get("lower_limit") or item.get("lower_limit")
                item["is_suspended"] = instrument.get("is_suspended", 0)
                result.append(_json_safe(item))
            return result
        if action == "history":
            symbols = [_normalise_symbol(value) for value in request.get("symbols") or []]
            if not symbols:
                raise ValueError("history requires symbols")
            frequency = str(request.get("frequency", "1d"))
            start_time = str(request.get("start_time"))
            end_time = str(request.get("end_time"))
            fields = request.get("fields") or None
            skip_suspended = bool(request.get("skip_suspended", True))
            fill_missing = request.get("fill_missing")
            adjust = request.get("adjust")
            adjust_end_time = str(request.get("adjust_end_time") or "")
            rows: list[dict[str, Any]] = []
            for symbol in symbols:
                values = self._call(
                    self._history,
                    symbol=symbol,
                    frequency=frequency,
                    start_time=start_time,
                    end_time=end_time,
                    fields=fields,
                    skip_suspended=skip_suspended,
                    fill_missing=fill_missing,
                    adjust=adjust,
                    adjust_end_time=adjust_end_time,
                    df=False,
                )
                rows.extend(_json_safe(values or []))
            return rows
        if action == "fundamentals_history":
            return self._fundamentals_history(request)
        if action == "calendar":
            exchange = str(request.get("exchange", "SHSE"))
            start_date = str(request.get("start_date"))
            end_date = str(request.get("end_date"))
            return _json_safe(self._call(self._get_trading_dates, exchange, start_date, end_date))
        raise ValueError(f"unsupported read-only action: {action}")


def _write(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        service = ReadOnlyService()
    except Exception as exc:
        _write({"ok": False, "error": _safe_error(exc)})
        return 1

    _write(
        {
            "ok": True,
            "event": "ready",
            "provider": PROVIDER_NAME,
            "read_only": True,
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Mapping[str, Any] = {}
        try:
            request = json.loads(line)
            data = service.handle(request)
            _write({"ok": True, "request_id": request.get("request_id"), "data": data})
        except Exception as exc:
            _write(
                {
                    "ok": False,
                    "request_id": request.get("request_id") if "request" in locals() else None,
                    "error": _safe_error(exc),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
