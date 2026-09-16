"""Read-only capability probe for the Eastmoney Juejin SDK.

This intentionally queries only a few recent rows.  It is used before a full
historical refresh so a missing data permission cannot create a misleading
partially enriched cache.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import math
import os
from collections.abc import Mapping
from typing import Any

TOKEN_ENV = "QUANTPAPER_EASTMONEY_TOKEN"
SYMBOLS = ["SHSE.600000", "SZSE.000001", "SZSE.300750", "SHSE.688981"]


def _json_safe(value: Any) -> Any:
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


def _safe_error(exc: BaseException) -> str:
    message = str(exc).replace(TOKEN_ENV, "TOKEN")
    token = os.environ.get(TOKEN_ENV, "")
    if token:
        message = message.replace(token, "<redacted>")
    return message[:1000] or type(exc).__name__


def _emit(payload: Mapping[str, Any]) -> None:
    # gm.api replaces Python's stdout during import on some SDK builds.  Use
    # the OS descriptor so the probe remains machine-readable.
    os.write(1, (json.dumps(payload, ensure_ascii=True, allow_nan=False) + "\n").encode())


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        _emit({"ok": False, "error": f"{TOKEN_ENV} is not set"})
        return 1

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            from gm.api import (  # type: ignore[import-not-found]
                set_token,
                stk_get_daily_basic,
                stk_get_daily_mktvalue,
                stk_get_symbol_industry,
            )
            from gm.csdk.c_sdk import gmi_init  # type: ignore[import-not-found]

            set_token(token)
            init_status = gmi_init()
    except Exception as exc:
        _emit({"ok": False, "stage": "import_or_init", "error": _safe_error(exc)})
        return 1

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=7)
    result: dict[str, Any] = {
        "ok": True,
        "sdk_init_status": init_status,
        "symbols": SYMBOLS,
        "window": {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        "datasets": {},
    }

    queries = (
        (
            "daily_mktvalue",
            stk_get_daily_mktvalue,
            {
                "fields": "tot_mv,a_mv_ex_ltd",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        ),
        (
            "daily_basic",
            stk_get_daily_basic,
            {
                "fields": "tclose,ttl_shr,turnrate",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        ),
    )
    for name, function, extra in queries:
        dataset: dict[str, Any] = {"rows": 0, "symbols": {}, "errors": []}
        for symbol in SYMBOLS:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rows = function(symbol=symbol, df=False, **extra)
                values = [_json_safe(row) for row in (rows or [])]
                dataset["rows"] += len(values)
                dataset["symbols"][symbol] = {
                    "rows": len(values),
                    "fields": (
                        sorted(values[0].keys())
                        if values and isinstance(values[0], Mapping)
                        else []
                    ),
                    "sample": values[0] if values else None,
                }
            except Exception as exc:
                dataset["errors"].append({"symbol": symbol, "error": _safe_error(exc)})
        result["datasets"][name] = dataset

    industry: dict[str, Any] = {"rows": 0, "symbols": {}, "errors": []}
    for symbol in SYMBOLS:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = stk_get_symbol_industry(
                    symbols=symbol,
                    source="sw2021",
                    level=1,
                    date=end_date.isoformat(),
                )
            values = [_json_safe(row) for row in (rows or [])]
            industry["rows"] += len(values)
            industry["symbols"][symbol] = {
                "rows": len(values),
                "fields": (
                    sorted(values[0].keys())
                    if values and isinstance(values[0], Mapping)
                    else []
                ),
                "sample": values[0] if values else None,
            }
        except Exception as exc:
            industry["errors"].append({"symbol": symbol, "error": _safe_error(exc)})
    result["datasets"]["symbol_industry_sw2021"] = industry

    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
