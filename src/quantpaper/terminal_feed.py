"""File-backed feed shared by the Eastmoney terminal adapter and QuantPaper.

The terminal adapter runs in Eastmoney's SDK environment, while the paper
runner and API run in the project environment.  A small atomic JSON file keeps
those runtimes decoupled and makes the last received quote inspectable.  The
feed is market-data only; it has no order or account methods.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")


def normalize_vendor_symbol(value: object) -> str:
    """Normalize ``SHSE.600000``/``600000`` to the vendor form."""

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


def parse_timestamp(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(SHANGHAI)


class TerminalFeedStore:
    """Atomic latest-quote store for one local adapter feed."""

    def __init__(self, path: str | Path = "data/runtime/eastmoney-terminal-feed.json") -> None:
        candidate = Path(path)
        self.path = candidate if candidate.is_absolute() else Path.cwd() / candidate
        self._lock = RLock()

    def _read_payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {"quotes": {}}
        if not isinstance(value, dict) or not isinstance(value.get("quotes"), dict):
            return {"quotes": {}}
        return value

    def write_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        """Merge rows and replace the file atomically.

        Only the latest row per symbol is kept.  Invalid prices or symbols are
        ignored so a malformed callback cannot poison the paper feed.
        """

        accepted: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            symbol = normalize_vendor_symbol(raw.get("symbol"))
            try:
                last = float(raw.get("last"))
            except (TypeError, ValueError):
                continue
            timestamp = parse_timestamp(raw.get("timestamp"))
            if not symbol or last <= 0 or timestamp is None:
                continue
            item = {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "last": last,
                "previous_close": raw.get("previous_close"),
                "bid1": raw.get("bid1"),
                "ask1": raw.get("ask1"),
                "bid1_volume": raw.get("bid1_volume", 0),
                "ask1_volume": raw.get("ask1_volume", 0),
                "bar_volume": raw.get("bar_volume", 0),
                "upper_limit": raw.get("upper_limit"),
                "lower_limit": raw.get("lower_limit"),
                "source": str(raw.get("source") or "EASTMONEY_TERMINAL_ADAPTER"),
                "frequency": str(raw.get("frequency") or "snapshot"),
                "received_at": datetime.now(SHANGHAI).isoformat(),
            }
            accepted[symbol] = item

        if not accepted:
            return 0
        with self._lock:
            payload = self._read_payload()
            quotes = payload.setdefault("quotes", {})
            if not isinstance(quotes, dict):
                quotes = {}
                payload["quotes"] = quotes
            quotes.update(accepted)
            payload["updated_at"] = datetime.now(SHANGHAI).isoformat()
            payload["source"] = "EASTMONEY_TERMINAL_ADAPTER"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(
                f".{self.path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        return len(accepted)

    def read_rows(
        self,
        symbols: Iterable[str] | None = None,
        stale_after_seconds: int = 180,
    ) -> list[dict[str, Any]]:
        allowed = (
            {normalize_vendor_symbol(symbol) for symbol in symbols}
            if symbols is not None
            else None
        )
        now = datetime.now(SHANGHAI)
        with self._lock:
            payload = self._read_payload()
        rows: list[dict[str, Any]] = []
        for raw in payload.get("quotes", {}).values():
            if not isinstance(raw, dict):
                continue
            symbol = normalize_vendor_symbol(raw.get("symbol"))
            if allowed is not None and symbol not in allowed:
                continue
            timestamp = parse_timestamp(raw.get("timestamp"))
            if timestamp is None:
                continue
            age = (now - timestamp).total_seconds()
            if age < -30 or age > max(1, int(stale_after_seconds)):
                continue
            item = dict(raw)
            item["symbol"] = symbol
            item["timestamp"] = timestamp.isoformat()
            rows.append(item)
        return rows

    def status(self, stale_after_seconds: int = 180) -> dict[str, Any]:
        rows = self.read_rows(stale_after_seconds=stale_after_seconds)
        timestamps = [parse_timestamp(row.get("timestamp")) for row in rows]
        latest = max((value for value in timestamps if value is not None), default=None)
        return {
            "path": str(self.path),
            "active_symbols": len(rows),
            "last_update": latest.isoformat() if latest else None,
            "source": "EASTMONEY_TERMINAL_ADAPTER",
        }
