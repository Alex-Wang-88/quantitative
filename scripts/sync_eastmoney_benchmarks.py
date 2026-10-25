"""Archive read-only Eastmoney benchmark/index daily bars.

The archive is deliberately separate from the A-share stock universe.  For
example, ``000001`` in this file means 上证指数, while ``000001`` in
``daily.parquet`` means 平安银行.  The script only calls the existing
read-only JSON-lines bridge and never exposes an order operation.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from quantpaper.eastmoney import EastmoneyBridgeClient

SHANGHAI = ZoneInfo("Asia/Shanghai")
BAR_FIELDS = [
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pre_close",
    "bob",
    "eob",
]

BENCHMARKS: dict[str, dict[str, str]] = {
    "000001": {"name": "上证指数", "exchange": "SHSE"},
    "399001": {"name": "深证成指", "exchange": "SZSE"},
    "000300": {"name": "沪深300", "exchange": "SHSE"},
    "399006": {"name": "创业板指", "exchange": "SZSE"},
    "000905": {"name": "中证500", "exchange": "SHSE"},
}


def now_iso() -> str:
    return datetime.now(UTC).astimezone(SHANGHAI).isoformat()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式必须是 YYYY-MM-DD: {value}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    connection = duckdb.connect()
    try:
        connection.register("benchmark_frame", frame)
        connection.execute(
            "COPY benchmark_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(temporary)],
        )
    finally:
        connection.close()
    temporary.replace(path)


def normalise_bars(
    rows: list[dict[str, Any]], benchmark_symbol: str, metadata: dict[str, str]
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    date_column = "eob" if "eob" in frame.columns else "bob"
    if date_column not in frame.columns:
        raise RuntimeError(f"{benchmark_symbol} 历史日线缺少 eob/bob")
    frame["trade_date"] = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
    frame["benchmark_symbol"] = benchmark_symbol
    frame["vendor_symbol"] = f"{metadata['exchange']}.{benchmark_symbol}"
    frame["name"] = metadata["name"]
    frame["exchange"] = metadata["exchange"]
    frame["frequency"] = "1d"
    frame["price_adjustment"] = "ADJUST_NONE"
    frame["source"] = "eastmoney_gm"
    for column in ("open", "high", "low", "close", "volume", "amount", "pre_close"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
    ]
    columns = [
        "benchmark_symbol",
        "vendor_symbol",
        "name",
        "exchange",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pre_close",
        "frequency",
        "price_adjustment",
        "source",
    ]
    return (
        frame[columns]
        .sort_values(["benchmark_symbol", "trade_date"])
        .drop_duplicates(["benchmark_symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


def fetch_with_retry(
    client: EastmoneyBridgeClient,
    symbol: str,
    start: date,
    end: date,
    retries: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    metadata = BENCHMARKS[symbol]
    vendor_symbol = f"{metadata['exchange']}.{symbol}"
    payload = {
        "symbols": [vendor_symbol],
        "frequency": "1d",
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "fields": BAR_FIELDS,
        "skip_suspended": False,
        "adjust": 0,
        "adjust_end_time": end.isoformat(),
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.request("history", **payload) or []
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                delay = min(20.0, max(1.0, sleep_seconds * (2 ** (attempt - 1))))
                print(
                    f"[{symbol}] retry {attempt}/{retries - 1} in {delay:.1f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="归档东方财富只读指数历史日线")
    parser.add_argument("--root", default="data/eastmoney_adjusted")
    parser.add_argument("--start", type=parse_date, default=date(1990, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date.today())
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=tuple(BENCHMARKS),
        default=list(BENCHMARKS),
        help="默认归档上证、深成、沪深300、创业板和中证500",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start > args.end:
        raise SystemExit("--start 不能晚于 --end")
    if args.retries < 1 or args.sleep_seconds < 0:
        raise SystemExit("--retries 和 --sleep-seconds 参数无效")
    root = Path(args.root)
    rows: list[pd.DataFrame] = []
    client = EastmoneyBridgeClient(timeout_seconds=args.timeout_seconds)
    try:
        for index, symbol in enumerate(args.symbols, start=1):
            raw = fetch_with_retry(
                client,
                symbol,
                args.start,
                args.end,
                args.retries,
                args.sleep_seconds,
            )
            frame = normalise_bars(raw, symbol, BENCHMARKS[symbol])
            if frame.empty:
                print(
                    f"[{index}/{len(args.symbols)}] {symbol} "
                    f"{BENCHMARKS[symbol]['name']}: 0 rows"
                )
            else:
                rows.append(frame)
                print(
                    f"[{index}/{len(args.symbols)}] {symbol} {BENCHMARKS[symbol]['name']}: "
                    f"{len(frame)} rows ({frame['trade_date'].min().date()} -> "
                    f"{frame['trade_date'].max().date()})",
                    flush=True,
                )
            if args.sleep_seconds > 0 and index < len(args.symbols):
                time.sleep(args.sleep_seconds)
    finally:
        client.close()
    if not rows:
        raise SystemExit("东方财富没有返回任何指数日线")
    combined = pd.concat(rows, ignore_index=True).sort_values(
        ["trade_date", "benchmark_symbol"]
    )
    output = root / "benchmarks.parquet"
    write_parquet(combined, output)
    manifest = {
        "provider": "eastmoney_gm",
        "read_only": True,
        "source": "EastmoneyBridgeClient.history",
        "updated_at": now_iso(),
        "window": {"start_date": args.start.isoformat(), "end_date": args.end.isoformat()},
        "symbols": [
            {
                "benchmark_symbol": symbol,
                **BENCHMARKS[symbol],
                "rows": int((combined["benchmark_symbol"] == symbol).sum()),
            }
            for symbol in args.symbols
        ],
        "rows": int(len(combined)),
        "path": str(output.resolve()),
    }
    atomic_json(root / "benchmark_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
