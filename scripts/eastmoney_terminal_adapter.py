"""Read-only Eastmoney terminal adapter for QuantPaper.

Two modes are supported:

* ``--poll`` uses the logged-in Eastmoney SDK service and forwards current
  snapshots to the local paper feed.  This is useful for the background
  supervisor and does not require creating a strategy in the UI.
* ``--strategy`` exposes ``init``/``on_bar`` callbacks for importing this file
  into the Eastmoney Quant terminal.  The callback subscribes to 60-second
  bars and forwards them to QuantPaper.  It intentionally imports no order
  function and cannot submit a broker order.

The local PaperBroker remains the only execution engine.  This adapter is a
market-data bridge, not a broker connector.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

TOKEN_ENV = "QUANTPAPER_EASTMONEY_TOKEN"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/v1/integrations/eastmoney/quotes"
SUBSCRIPTION_ENDPOINT = "http://127.0.0.1:8000/api/v1/integrations/eastmoney/subscription"
DEFAULT_SYMBOLS = ["SHSE.600000", "SZSE.000001", "SZSE.300750", "SHSE.688981"]
BENCHMARK_SYMBOLS = ["SHSE.000001", "SZSE.399001", "SHSE.000300"]
LOG = logging.getLogger("eastmoney-terminal-adapter")
_strategy_symbols: list[str] = []
_strategy_frequency = "60s"


def _read_token() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if token or os.name != "nt":
        return token
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return str(winreg.QueryValueEx(key, TOKEN_ENV)[0]).strip()
    except (FileNotFoundError, OSError):
        return ""


def _normalise_symbol(value: object) -> str:
    raw = str(value or "").strip().upper()
    if "." in raw:
        exchange, code = raw.split(".", 1)
        digits = "".join(character for character in code if character.isdigit())
        if digits:
            return f"{exchange}.{digits[-6:].zfill(6)}"
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        return raw
    code = digits[-6:].zfill(6)
    exchange = "SHSE" if code.startswith(("6", "68")) else "SZSE"
    return f"{exchange}.{code}"


def _iso(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return str(value)


def _number(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _symbols_from_env() -> list[str]:
    raw = os.environ.get("QUANTPAPER_ADAPTER_SYMBOLS", "")
    values = [_normalise_symbol(item) for item in raw.split(",") if item.strip()]
    return list(dict.fromkeys(values))


def _http_json(url: str, payload: Mapping[str, Any] | None = None) -> Any:
    body = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _subscription_symbols() -> list[str]:
    configured = _symbols_from_env()
    if configured:
        return list(dict.fromkeys(configured + BENCHMARK_SYMBOLS))[:500]
    try:
        result = _http_json(SUBSCRIPTION_ENDPOINT)
        values = result.get("symbols", []) if isinstance(result, dict) else []
        symbols = [_normalise_symbol(item) for item in values if item]
        if symbols:
            return list(dict.fromkeys(symbols + BENCHMARK_SYMBOLS))[:500]
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        LOG.warning("subscription endpoint unavailable: %s", exc)
    return list(dict.fromkeys(DEFAULT_SYMBOLS + BENCHMARK_SYMBOLS))


def _post_quotes(rows: Iterable[Mapping[str, Any]]) -> int:
    values = [dict(row) for row in rows if isinstance(row, Mapping)]
    if not values:
        return 0
    endpoint = os.environ.get("QUANTPAPER_TERMINAL_ADAPTER_URL", DEFAULT_ENDPOINT)
    try:
        result = _http_json(endpoint, {"quotes": values})
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        LOG.warning("local paper feed unavailable: %s", exc)
        return 0
    accepted = result.get("accepted", 0) if isinstance(result, dict) else 0
    LOG.info("forwarded %s/%s market rows", accepted, len(values))
    return int(accepted or 0)


def _current_row_to_quote(
    row: Mapping[str, Any],
    previous_volume: float | None,
    fallback_previous_close: float | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    symbol = _normalise_symbol(row.get("symbol"))
    last = _number(row.get("price") or row.get("last"))
    timestamp = _iso(row.get("created_at") or row.get("timestamp"))
    if not symbol or last is None or last <= 0 or not timestamp:
        return None, previous_volume
    cumulative = _number(row.get("cum_volume"))
    bar_volume = 0
    if cumulative is not None and previous_volume is not None and cumulative >= previous_volume:
        bar_volume = int(cumulative - previous_volume)
    next_volume = cumulative if cumulative is not None else previous_volume
    levels = row.get("quotes") or []
    top = levels[0] if levels and isinstance(levels[0], Mapping) else {}
    return (
        {
            "symbol": symbol,
            "timestamp": timestamp,
            "last": last,
            "previous_close": _number(
                row.get("pre_close")
                or row.get("previous_close")
                or row.get("last_close")
                or row.get("prev_close"),
                fallback_previous_close,
            ),
            "bid1": _number(row.get("bid1") or top.get("bid_p")),
            "ask1": _number(row.get("ask1") or top.get("ask_p")),
            "bid1_volume": int(_number(row.get("bid1_volume") or top.get("bid_v"), 0) or 0),
            "ask1_volume": int(_number(row.get("ask1_volume") or top.get("ask_v"), 0) or 0),
            "bar_volume": bar_volume,
            "source": "EASTMONEY_TERMINAL_CURRENT",
            "frequency": "snapshot",
        },
        next_volume,
    )


def _bar_to_quote(row: Mapping[str, Any]) -> dict[str, Any] | None:
    symbol = _normalise_symbol(row.get("symbol"))
    last = _number(row.get("close") or row.get("price") or row.get("last"))
    timestamp = _iso(row.get("eob") or row.get("bob") or row.get("created_at"))
    if not symbol or last is None or last <= 0 or not timestamp:
        return None
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "last": last,
            "previous_close": _number(
                row.get("pre_close")
                or row.get("previous_close")
                or row.get("last_close")
                or row.get("prev_close")
            ),
        "bar_volume": int(_number(row.get("volume") or row.get("cum_volume"), 0) or 0),
        "source": "EASTMONEY_TERMINAL_STRATEGY",
        "frequency": _strategy_frequency,
    }


def _init_sdk() -> tuple[Any, Any]:
    token = _read_token()
    if not token:
        raise RuntimeError(f"{TOKEN_ENV} is not configured")
    with contextlib.redirect_stdout(sys.stderr):
        from gm.api import current, get_instruments, set_token  # type: ignore[import-not-found]
        from gm.csdk.c_sdk import gmi_init  # type: ignore[import-not-found]

        set_token(token)
        status = gmi_init()
    if status not in (None, 0):
        raise RuntimeError(f"Eastmoney SDK init failed: {status}")
    return current, get_instruments


def poll_once(
    current: Any,
    symbols: list[str],
    previous_volumes: dict[str, float],
    previous_closes: Mapping[str, float] | None = None,
) -> int:
    vendor_symbols = list(dict.fromkeys(symbols))
    with contextlib.redirect_stdout(sys.stderr):
        rows = current(vendor_symbols)
    quotes: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, Mapping):
            continue
        symbol = _normalise_symbol(raw.get("symbol"))
        quote, next_volume = _current_row_to_quote(
            raw,
            previous_volumes.get(symbol),
            (previous_closes or {}).get(symbol),
        )
        if quote:
            quotes.append(quote)
        if next_volume is not None:
            previous_volumes[symbol] = next_volume
    return _post_quotes(quotes)


def run_poll(interval_seconds: float, once: bool) -> int:
    current, get_instruments = _init_sdk()
    previous_volumes: dict[str, float] = {}
    previous_closes: dict[str, float] = {}
    symbols: list[str] = []
    last_subscription = 0.0
    while True:
        now = time.monotonic()
        if not symbols or now - last_subscription >= 30:
            symbols = _subscription_symbols()
            last_subscription = now
            try:
                instrument_rows = get_instruments(
                    symbols=symbols,
                    skip_suspended=False,
                    skip_st=False,
                )
                previous_closes = {
                    _normalise_symbol(row.get("symbol")): close
                    for row in instrument_rows or []
                    if isinstance(row, Mapping)
                    and (close := _number(row.get("pre_close"))) is not None
                    and close > 0
                }
            except Exception as exc:
                LOG.warning("previous-close query failed: %s", exc)
            LOG.info("subscribed to %s symbols", len(symbols))
        try:
            poll_once(current, symbols, previous_volumes, previous_closes)
        except Exception as exc:
            LOG.warning("current quote poll failed: %s", exc)
        if once:
            return 0
        time.sleep(max(1.0, interval_seconds))


def init(context: Any) -> None:
    """Eastmoney strategy callback: subscribe and forward bars only."""

    global _strategy_symbols
    _strategy_symbols = _subscription_symbols()
    try:
        from gm.api import subscribe  # type: ignore[import-not-found]

        subscribe(
            symbols=",".join(_strategy_symbols),
            frequency=_strategy_frequency,
            count=1,
        )
        LOG.info("strategy adapter subscribed to %s symbols", len(_strategy_symbols))
    except Exception as exc:
        LOG.exception("strategy subscription failed: %s", exc)
        raise


def on_bar(context: Any, bars: Any) -> None:
    values = bars if isinstance(bars, list) else [bars]
    rows = [_bar_to_quote(item) for item in values if isinstance(item, Mapping)]
    _post_quotes([row for row in rows if row is not None])


def run_strategy(strategy_id: str) -> int:
    token = _read_token()
    if not token:
        raise RuntimeError(f"{TOKEN_ENV} is not configured")
    from gm.api import MODE_LIVE, run  # type: ignore[import-not-found]

    # MODE_LIVE here means live market-data callbacks.  The strategy has no
    # order API calls; local PaperBroker remains the only execution engine.
    run(
        strategy_id=strategy_id,
        filename=__file__,
        mode=MODE_LIVE,
        token=token,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Eastmoney read-only terminal adapter")
    parser.add_argument("--poll", action="store_true", help="poll current snapshots")
    parser.add_argument("--once", action="store_true", help="poll one batch and exit")
    parser.add_argument("--strategy", action="store_true", help="run init/on_bar strategy")
    parser.add_argument(
        "--strategy-id",
        default=os.environ.get("QUANTPAPER_EASTMONEY_STRATEGY_ID", ""),
    )
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.strategy:
        if not args.strategy_id:
            parser.error("--strategy requires --strategy-id when run outside the terminal UI")
        return run_strategy(args.strategy_id)
    # Poll is the safe default for the supervisor; it never places an order.
    return run_poll(args.interval, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
