"""Build a traceable research quality-event manifest for an Eastmoney archive.

This script never edits or deletes the raw/adjusted daily archive.  It marks
extreme price jumps and writes a small Parquet manifest consumed by
``HistoricalStore.read_daily(..., exclude_quality_events=True)``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_PRE_EVENT_ROWS = 5
DEFAULT_POST_EVENT_ROWS = 20
ABNORMAL_RETURN_THRESHOLD = 0.30


def _event_query() -> str:
    return """
        WITH daily AS (
            SELECT
                lpad(CAST(symbol AS VARCHAR), 6, '0') AS symbol,
                CAST(trade_date AS DATE) AS event_date,
                close,
                LAG(close) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                ) AS previous_close,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol ORDER BY trade_date
                ) AS event_row_number
            FROM read_parquet(?)
        ), master AS (
            SELECT
                lpad(CAST(symbol AS VARCHAR), 6, '0') AS symbol,
                CAST(listed_date AS DATE) AS listed_date
            FROM read_parquet(?)
        )
        SELECT
            d.symbol,
            d.event_date,
            d.event_row_number,
            d.previous_close,
            d.close,
            d.close / d.previous_close - 1 AS return_pct,
            ABS(d.close / d.previous_close - 1) AS abs_return,
            m.listed_date,
            CASE
                WHEN d.event_row_number <= 5
                 AND m.listed_date IS NOT NULL
                 AND date_diff('day', m.listed_date, d.event_date) BETWEEN 0 AND 30
                    THEN 'IPO_WINDOW'
                ELSE 'UNVERIFIED_PRICE_EVENT'
            END AS event_class,
            CASE
                WHEN d.event_row_number <= 5
                 AND m.listed_date IS NOT NULL
                 AND date_diff('day', m.listed_date, d.event_date) BETWEEN 0 AND 30
                    THEN '上市初期价格发现窗口，不用于普通日频收益训练'
                ELSE '疑似除权、复牌或源数据衔接异常，待公司行为数据核验'
            END AS reason,
            GREATEST(d.event_row_number - ?, 1) AS exclude_start_row_number,
            d.event_row_number + ? AS exclude_end_row_number
        FROM daily d
        LEFT JOIN master m USING (symbol)
        WHERE d.previous_close > 0
          AND ABS(d.close / d.previous_close - 1) > ?
        ORDER BY d.event_date, d.symbol
    """


def _write_events(
    connection: duckdb.DuckDBPyConnection,
    *,
    daily_path: Path,
    master_path: Path,
    output_path: Path,
    pre_event_rows: int,
    post_event_rows: int,
) -> None:
    temporary = output_path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE generated_quality_events AS " + _event_query(),
        [
            str(daily_path),
            str(master_path),
            pre_event_rows,
            post_event_rows,
            ABNORMAL_RETURN_THRESHOLD,
        ],
    )
    connection.execute(
        "COPY generated_quality_events TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(temporary)],
    )
    temporary.replace(output_path)


def build_quality_manifest(
    root: Path,
    *,
    output: Path | None = None,
    report_output: Path | None = None,
    pre_event_rows: int = DEFAULT_PRE_EVENT_ROWS,
    post_event_rows: int = DEFAULT_POST_EVENT_ROWS,
) -> dict[str, Any]:
    daily_path = root / "daily.parquet"
    master_path = root / "instrument_master.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    if not master_path.exists():
        raise FileNotFoundError(master_path)
    if pre_event_rows < 0 or post_event_rows < 0:
        raise ValueError("quality event windows must be non-negative")

    output_path = output or root / "quality_events.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        _write_events(
            connection,
            daily_path=daily_path,
            master_path=master_path,
            output_path=output_path,
            pre_event_rows=pre_event_rows,
            post_event_rows=post_event_rows,
        )
        event_count, event_symbols = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()
        class_rows = connection.execute(
            """
            SELECT event_class, COUNT(*)
            FROM read_parquet(?)
            GROUP BY event_class
            ORDER BY event_class
            """,
            [str(output_path)],
        ).fetchall()
        excluded_rows, excluded_symbols = connection.execute(
            """
            WITH ranked AS (
                SELECT
                    lpad(CAST(symbol AS VARCHAR), 6, '0') AS symbol,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY trade_date
                    ) AS row_number
                FROM read_parquet(?)
            )
            SELECT COUNT(*), COUNT(DISTINCT r.symbol)
            FROM ranked r
            WHERE EXISTS (
                SELECT 1
                FROM read_parquet(?) e
                WHERE e.symbol = r.symbol
                  AND r.row_number BETWEEN e.exclude_start_row_number
                                        AND e.exclude_end_row_number
            )
            """,
            [str(daily_path), str(output_path)],
        ).fetchone()
        daily_rows = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(daily_path)]
        ).fetchone()[0]
    finally:
        connection.close()

    result: dict[str, Any] = {
        "status": "READY" if int(event_count) >= 0 else "DEGRADED",
        "policy": "exclude_quality_event_windows",
        "abnormal_return_threshold": ABNORMAL_RETURN_THRESHOLD,
        "pre_event_rows": pre_event_rows,
        "post_event_rows": post_event_rows,
        "daily_rows": int(daily_rows),
        "quality_event_rows": int(event_count),
        "quality_event_symbols": int(event_symbols),
        "excluded_rows": int(excluded_rows),
        "excluded_symbols": int(excluded_symbols),
        "event_class_counts": {str(row[0]): int(row[1]) for row in class_rows},
        "root": str(root.resolve()),
        "daily_path": str(daily_path.resolve()),
        "quality_events_path": str(output_path.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "note": (
            "The original archive is unchanged. Research reads exclude the event "
            "row, a five-row pre-label buffer, and twenty subsequent rows; "
            "rolling features restart after each event window."
        ),
    }
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="生成东方财富研究质量事件清单")
    parser.add_argument("--root", default="data/eastmoney_adjusted")
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="data/reports/research-quality-adjusted.json")
    parser.add_argument("--pre-event-rows", type=int, default=DEFAULT_PRE_EVENT_ROWS)
    parser.add_argument("--post-event-rows", type=int, default=DEFAULT_POST_EVENT_ROWS)
    args = parser.parse_args()
    result = build_quality_manifest(
        Path(args.root),
        output=Path(args.output) if args.output else None,
        report_output=Path(args.report) if args.report else None,
        pre_event_rows=args.pre_event_rows,
        post_event_rows=args.post_event_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.report:
        print(f"report: {Path(args.report).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
