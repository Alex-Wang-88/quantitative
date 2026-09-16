"""Download read-only Eastmoney history into resumable local Parquet archives.

The script deliberately talks to the existing JSON-lines read-only bridge. It
does not import or call any order API. Each security is checkpointed
individually so a terminal restart or a network interruption does not discard
an already downloaded history.

The archive is intentionally separate from ``data/market/daily.parquet`` until
the complete run is inspected. Use ``--compact-only`` to create consolidated
Parquet files after a run.
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
FUNDAMENTALS_NORMALIZATION_VERSION = 2


def now_iso() -> str:
    return datetime.now(UTC).astimezone(SHANGHAI).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    connection = duckdb.connect()
    try:
        connection.register("download_frame", frame)
        connection.execute(
            "COPY download_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(temporary)],
        )
    finally:
        connection.close()
    temporary.replace(path)


def compact_parquet(
    source_glob: str,
    output: Path,
    order_columns: tuple[str, ...] = ("symbol", "trade_date"),
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE compact_source AS SELECT * FROM read_parquet(?)",
            [source_glob],
        )
        result = connection.execute("SELECT COUNT(*) FROM compact_source").fetchone()
        rows = int(result[0]) if result else 0
        if rows:
            columns = set(
                row[0]
                for row in connection.execute("DESCRIBE compact_source").fetchall()
            )
            order_by = [column for column in order_columns if column in columns]
            order_sql = f" ORDER BY {', '.join(order_by)}" if order_by else ""
            connection.execute(
                f"COPY (SELECT * FROM compact_source{order_sql}) "
                "TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(temporary)],
            )
        else:
            return 0
    finally:
        connection.close()
    temporary.replace(output)
    return rows


def code_from_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if "." in raw:
        raw = raw.split(".", 1)[1]
    digits = "".join(character for character in raw if character.isdigit())
    return digits[-6:].zfill(6) if digits else raw


def exchange_from_symbol(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if "." in raw:
        return raw.split(".", 1)[0]
    code = code_from_symbol(raw)
    return "SHSE" if code.startswith(("6", "68")) else "SZSE"


def vendor_symbol(value: Any) -> str:
    return f"{exchange_from_symbol(value)}.{code_from_symbol(value)}"


def normalise_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return parsed.dt.tz_convert(SHANGHAI).dt.tz_localize(None).dt.normalize()


def normalise_instruments(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("东方财富股票主表为空")
    frame["vendor_symbol"] = frame["symbol"].map(vendor_symbol)
    frame["symbol"] = frame["symbol"].map(code_from_symbol)
    frame["exchange"] = frame["vendor_symbol"].map(exchange_from_symbol)
    if "listed_date" in frame:
        frame["listed_date"] = pd.to_datetime(frame["listed_date"], errors="coerce")
    return frame.sort_values("symbol").drop_duplicates("symbol").reset_index(drop=True)


def normalise_bars(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "eob" not in frame and "bob" not in frame:
        raise RuntimeError(f"{symbol} 历史日线缺少 eob/bob")
    date_column = "eob" if "eob" in frame else "bob"
    frame["trade_date"] = normalise_dates(frame, date_column)
    if "symbol" not in frame:
        frame["symbol"] = symbol
    frame["vendor_symbol"] = frame["symbol"].map(vendor_symbol)
    frame["symbol"] = frame["symbol"].map(code_from_symbol)
    frame["exchange"] = frame["vendor_symbol"].map(exchange_from_symbol)
    frame["frequency"] = "1d"
    for column in ("open", "high", "low", "close", "volume", "amount", "pre_close"):
        if column not in frame:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close", "amount"])
    frame = frame[
        (frame["open"] > 0)
        & (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["amount"] >= 0)
    ]
    return (
        frame.sort_values(["symbol", "trade_date"])
        .drop_duplicates(["symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )


def normalise_fundamental_frame(
    rows: list[dict[str, Any]], *, date_column: str
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "symbol" in frame:
        frame["vendor_symbol"] = frame["symbol"].map(vendor_symbol)
        frame["symbol"] = frame["symbol"].map(code_from_symbol)
        frame["exchange"] = frame["vendor_symbol"].map(exchange_from_symbol)
    if date_column in frame:
        frame[date_column] = normalise_dates(frame, date_column)
    for column in ("pub_date", "ann_date", "rpt_date"):
        if column in frame:
            frame[column] = normalise_dates(frame, column)
    protected = {
        "symbol",
        "vendor_symbol",
        "exchange",
        date_column,
        "pub_date",
        "ann_date",
        "rpt_date",
    }
    for column in frame.columns:
        if column not in protected:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    keys = [key for key in ("symbol", date_column) if key in frame]
    if keys:
        frame = (
            frame.dropna(subset=keys)
            .sort_values(keys)
            .groupby(keys, as_index=False, sort=False)
            .first()
        )
    return frame.reset_index(drop=True)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日期格式必须是 YYYY-MM-DD: {value}") from exc


def load_manifest(path: Path, root: Path, start: date, end: date) -> dict[str, Any]:
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "provider": "eastmoney_gm",
        "read_only": True,
        "archive_root": str(root.resolve()),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "window": {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "universe": {},
        "calendar": {},
        "datasets": {},
    }


def dataset_state(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    datasets = manifest.setdefault("datasets", {})
    state = datasets.setdefault(
        name,
        {
            "completed_symbols": [],
            "failed_symbols": {},
            "rows": 0,
            "updated_at": now_iso(),
        },
    )
    state.setdefault("completed_symbols", [])
    state.setdefault("failed_symbols", {})
    state.setdefault("rows", 0)
    return state


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    atomic_json(path, manifest)


def fetch_metadata(
    client: EastmoneyBridgeClient,
    manifest: dict[str, Any],
    root: Path,
    start: date,
    end: date,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    print("[metadata] 获取当前 A 股股票主表...", flush=True)
    rows = client.request(
        "instruments",
        limit=0,
        exchanges=["SHSE", "SZSE"],
        sec_types=[1],
        skip_suspended=False,
        skip_st=False,
    ) or []
    frame = normalise_instruments(rows)
    write_parquet(frame, root / "instrument_master.parquet")
    manifest["universe"] = {
        "count": int(len(frame)),
        "path": str((root / "instrument_master.parquet").resolve()),
        "includes_suspended": True,
        "includes_st": True,
        "updated_at": now_iso(),
    }

    print("[metadata] 获取交易日历...", flush=True)
    calendar = client.request(
        "calendar",
        exchange="SHSE",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
    ) or []
    calendar_frame = pd.DataFrame({"trade_date": pd.to_datetime(calendar, errors="coerce")})
    calendar_frame = calendar_frame.dropna().drop_duplicates().sort_values("trade_date")
    write_parquet(calendar_frame, root / "trading_calendar.parquet")
    manifest["calendar"] = {
        "count": int(len(calendar_frame)),
        "path": str((root / "trading_calendar.parquet").resolve()),
        "updated_at": now_iso(),
    }
    return frame, frame.to_dict(orient="records")


def listed_start(record: dict[str, Any], requested: date) -> date:
    value = record.get("listed_date")
    if value is None or pd.isna(value):
        return requested
    parsed = pd.Timestamp(value).date()
    return max(requested, parsed)


def part_name(record: dict[str, Any]) -> str:
    exchange = exchange_from_symbol(record["vendor_symbol"])
    return f"{exchange}_{code_from_symbol(record['symbol'])}.parquet"


def retry_request(
    client: EastmoneyBridgeClient,
    action: str,
    payload: dict[str, Any],
    retries: int,
    sleep_seconds: float,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.request(action, **payload)
        except Exception as exc:  # bridge converts vendor failures to ProviderUnavailable
            last_error = exc
            if attempt < retries:
                delay = min(30.0, max(1.0, sleep_seconds * (2 ** (attempt - 1))))
                print(
                    f"    retry {attempt}/{retries - 1} in {delay:.1f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def download_daily(
    client: EastmoneyBridgeClient,
    manifest: dict[str, Any],
    manifest_path: Path,
    root: Path,
    records: list[dict[str, Any]],
    start: date,
    end: date,
    retries: int,
    sleep_seconds: float,
    max_symbols: int,
) -> None:
    state = dataset_state(manifest, "daily_bars")
    completed = set(state["completed_symbols"])
    failed = state["failed_symbols"]
    selected = records[:max_symbols] if max_symbols else records
    output_dir = root / "daily_bars"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(selected)
    consecutive_failures = 0
    for index, record in enumerate(selected, start=1):
        symbol = code_from_symbol(record["symbol"])
        part = output_dir / part_name(record)
        if symbol in completed and (part.exists() or state.get("empty_symbols", {}).get(symbol)):
            continue
        try:
            symbol_start = listed_start(record, start)
            rows = retry_request(
                client,
                "history",
                {
                    "symbols": [record["vendor_symbol"]],
                    "frequency": "1d",
                    "start_time": symbol_start.isoformat(),
                    "end_time": end.isoformat(),
                    "fields": BAR_FIELDS,
                    "skip_suspended": False,
                },
                retries,
                sleep_seconds,
            ) or []
            frame = normalise_bars(rows, record["vendor_symbol"])
            if not frame.empty:
                write_parquet(frame, part)
                state["rows"] = int(state.get("rows", 0)) + len(frame)
            else:
                state.setdefault("empty_symbols", {})[symbol] = True
            completed.add(symbol)
            failed.pop(symbol, None)
            consecutive_failures = 0
        except Exception as exc:
            message = str(exc)[:1000]
            failed[symbol] = {"error": message, "updated_at": now_iso()}
            append_jsonl(
                root / "download_errors.jsonl",
                {"dataset": "daily_bars", "symbol": symbol, "error": message, "at": now_iso()},
            )
            consecutive_failures += 1
            print(f"[{index}/{total}] {symbol} FAILED: {message}", flush=True)
            if consecutive_failures >= 5 and any(
                marker in message for marker in ("终端", "连接", "超时", "bridge", "桥接")
            ):
                state["completed_symbols"] = sorted(completed)
                save_manifest(manifest_path, manifest)
                raise RuntimeError(
                    "连续 5 个股票无法访问东方财富接口，已暂停以等待终端恢复"
                ) from exc
        if index % 10 == 0 or index == total:
            state["completed_symbols"] = sorted(completed)
            save_manifest(manifest_path, manifest)
            print(
                f"[daily {index}/{total}] completed={len(completed)} failed={len(failed)}",
                flush=True,
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    state["completed_symbols"] = sorted(completed)
    state["status"] = "complete" if len(completed) >= total else "partial"
    state["finished_at"] = now_iso()
    save_manifest(manifest_path, manifest)


def download_fundamentals(
    client: EastmoneyBridgeClient,
    manifest: dict[str, Any],
    manifest_path: Path,
    root: Path,
    records: list[dict[str, Any]],
    start: date,
    end: date,
    retries: int,
    sleep_seconds: float,
    max_symbols: int,
    industry_enabled: bool,
) -> None:
    state = dataset_state(manifest, "fundamentals")
    if state.get("normalization_version") != FUNDAMENTALS_NORMALIZATION_VERSION:
        # The first implementation wrote one row per vendor endpoint.  Reset
        # only the files produced by this dataset before rebuilding them with
        # non-null same-date fields merged together.
        for directory in ("fundamentals_daily", "fundamentals_reports", "industries"):
            for part in (root / directory).glob("*.parquet"):
                part.unlink()
        for compacted in (
            root / "fundamentals_daily.parquet",
            root / "fundamentals_reports.parquet",
            root / "industries.parquet",
        ):
            if compacted.exists():
                compacted.unlink()
        state.clear()
        state.update(
            {
                "completed_symbols": [],
                "failed_symbols": {},
                "partial_symbols": {},
                "rows": 0,
                "normalization_version": FUNDAMENTALS_NORMALIZATION_VERSION,
                "updated_at": now_iso(),
            }
        )
    completed = set(state["completed_symbols"])
    failed = state["failed_symbols"]
    selected = records[:max_symbols] if max_symbols else records
    total = len(selected)
    for directory in ("fundamentals_daily", "fundamentals_reports", "industries"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    consecutive_failures = 0
    for index, record in enumerate(selected, start=1):
        symbol = code_from_symbol(record["symbol"])
        stem = Path(part_name(record)).stem
        if symbol in completed:
            continue
        try:
            payload = retry_request(
                client,
                "fundamentals_history",
                {
                    "symbols": [record["vendor_symbol"]],
                    "start_date": listed_start(record, start).isoformat(),
                    "end_date": end.isoformat(),
                    "industry_enabled": industry_enabled,
                },
                retries,
                sleep_seconds,
            ) or {}
            daily = normalise_fundamental_frame(
                payload.get("daily") or [], date_column="trade_date"
            )
            reports = normalise_fundamental_frame(
                payload.get("reports") or [], date_column="pub_date"
            )
            industries = normalise_fundamental_frame(
                payload.get("industries") or [], date_column="industry_as_of"
            )
            if not daily.empty:
                write_parquet(daily, root / "fundamentals_daily" / f"{stem}.parquet")
                state["rows"] = int(state.get("rows", 0)) + len(daily)
            if not reports.empty:
                write_parquet(reports, root / "fundamentals_reports" / f"{stem}.parquet")
            if not industries.empty:
                write_parquet(industries, root / "industries" / f"{stem}.parquet")
            errors = payload.get("errors") or []
            if errors:
                state.setdefault("partial_symbols", {})[symbol] = errors
                for item in errors:
                    append_jsonl(
                        root / "download_errors.jsonl",
                        {
                            "dataset": item.get("dataset", "fundamentals"),
                            "symbol": symbol,
                            "error": item.get("error", ""),
                            "at": now_iso(),
                        },
                    )
            completed.add(symbol)
            failed.pop(symbol, None)
            consecutive_failures = 0
        except Exception as exc:
            message = str(exc)[:1000]
            failed[symbol] = {"error": message, "updated_at": now_iso()}
            append_jsonl(
                root / "download_errors.jsonl",
                {"dataset": "fundamentals", "symbol": symbol, "error": message, "at": now_iso()},
            )
            consecutive_failures += 1
            print(f"[{index}/{total}] {symbol} FAILED: {message}", flush=True)
            if consecutive_failures >= 5 and any(
                marker in message for marker in ("终端", "连接", "超时", "bridge", "桥接")
            ):
                state["completed_symbols"] = sorted(completed)
                save_manifest(manifest_path, manifest)
                raise RuntimeError(
                    "连续 5 个股票无法访问东方财富基础面接口，已暂停以等待终端恢复"
                ) from exc
        if index % 10 == 0 or index == total:
            state["completed_symbols"] = sorted(completed)
            save_manifest(manifest_path, manifest)
            print(
                f"[fundamentals {index}/{total}] completed={len(completed)} failed={len(failed)}",
                flush=True,
            )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    state["completed_symbols"] = sorted(completed)
    state["status"] = "complete" if len(completed) >= total else "partial"
    state["finished_at"] = now_iso()
    save_manifest(manifest_path, manifest)


def compact_archive(root: Path, manifest: dict[str, Any]) -> None:
    outputs = {
        "daily_bars": (
            root / "daily_bars" / "*.parquet",
            root / "daily.parquet",
            ("symbol", "trade_date"),
        ),
        "fundamentals_daily": (
            root / "fundamentals_daily" / "*.parquet",
            root / "fundamentals_daily.parquet",
            ("symbol", "trade_date"),
        ),
        "fundamentals_reports": (
            root / "fundamentals_reports" / "*.parquet",
            root / "fundamentals_reports.parquet",
            ("symbol", "pub_date", "ann_date"),
        ),
        "industries": (
            root / "industries" / "*.parquet",
            root / "industries.parquet",
            ("symbol", "industry_as_of"),
        ),
    }
    for dataset, (source, output, order_columns) in outputs.items():
        if not source.parent.exists() or not list(source.parent.glob("*.parquet")):
            continue
        try:
            rows = compact_parquet(source.as_posix(), output, order_columns)
        except Exception as exc:
            print(f"[compact] {dataset} failed: {exc}", flush=True)
            continue
        manifest.setdefault("compacted", {})[dataset] = {
            "rows": rows,
            "path": str(output.resolve()),
            "updated_at": now_iso(),
        }
        print(f"[compact] {dataset}: {rows} rows -> {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从东方财富掘金只读接口归档历史数据")
    parser.add_argument(
        "--dataset",
        choices=("metadata", "daily", "fundamentals", "all"),
        default="daily",
        help="metadata=主表和日历，daily=上市以来日线，fundamentals=历史因子，all=全部",
    )
    parser.add_argument("--root", default="data/eastmoney", help="归档根目录")
    parser.add_argument("--start", type=parse_date, default=date(1990, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date.today())
    parser.add_argument("--max-symbols", type=int, default=0, help="仅用于小范围验证，0 表示全量")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--industry",
        action="store_true",
        help="额外尝试行业接口；无权限时会记录错误",
    )
    parser.add_argument("--compact-only", action="store_true", help="只把已下载分片合并成 Parquet")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start > args.end:
        raise SystemExit("--start 不能晚于 --end")
    if args.max_symbols < 0 or args.retries < 1 or args.sleep_seconds < 0:
        raise SystemExit("--max-symbols、--retries 和 --sleep-seconds 参数无效")

    root = Path(args.root)
    manifest_path = root / "manifest.json"
    manifest = load_manifest(manifest_path, root, args.start, args.end)
    if args.compact_only:
        compact_archive(root, manifest)
        save_manifest(manifest_path, manifest)
        return 0

    client = EastmoneyBridgeClient(timeout_seconds=args.timeout_seconds)
    try:
        frame, records = fetch_metadata(client, manifest, root, args.start, args.end)
        save_manifest(manifest_path, manifest)
        if args.dataset in {"daily", "fundamentals", "all"}:
            print(
                f"[metadata] 当前股票主表 {len(frame)} 只；归档窗口 {args.start} -> {args.end}",
                flush=True,
            )
        if args.dataset in {"daily", "all"}:
            download_daily(
                client,
                manifest,
                manifest_path,
                root,
                records,
                args.start,
                args.end,
                args.retries,
                args.sleep_seconds,
                args.max_symbols,
            )
        if args.dataset in {"fundamentals", "all"}:
            download_fundamentals(
                client,
                manifest,
                manifest_path,
                root,
                records,
                args.start,
                args.end,
                args.retries,
                args.sleep_seconds,
                args.max_symbols,
                args.industry,
            )
        compact_archive(root, manifest)
        save_manifest(manifest_path, manifest)
        print(f"归档完成或已暂停，可从 manifest 继续: {manifest_path.resolve()}", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
