"""Audit an Eastmoney Parquet archive before research or model training."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

TARGET_A_SHARE_PATTERN = (
    "^(000|001|002|003|300|301|302|600|601|603|605|688|689)"
)


def audit(root: Path) -> dict[str, Any]:
    daily_path = root / "daily.parquet"
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)

    connection = duckdb.connect()
    try:
        schema_rows = connection.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [str(daily_path)]
        ).fetchall()
        has_adjustment_column = any(row[0] == "price_adjustment" for row in schema_rows)
        adjustment_select = (
            "price_adjustment"
            if has_adjustment_column
            else "CAST(NULL AS VARCHAR) AS price_adjustment"
        )
        summary = connection.execute(
            f"""
            WITH daily AS (
                SELECT
                    lpad(CAST(symbol AS VARCHAR), 6, '0') AS symbol,
                    CAST(trade_date AS DATE) AS trade_date,
                    close,
                    {adjustment_select}
                FROM read_parquet(?)
            ), changes AS (
                SELECT
                    symbol,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS previous_close
                FROM daily
            )
            SELECT
                (SELECT COUNT(*) FROM daily) AS rows,
                (SELECT COUNT(DISTINCT symbol) FROM daily) AS symbols,
                (SELECT COUNT(DISTINCT symbol) FROM daily
                 WHERE regexp_matches(symbol, ?)) AS target_a_share_symbols,
                (SELECT MIN(trade_date) FROM daily) AS start_date,
                (SELECT MAX(trade_date) FROM daily) AS end_date,
                (SELECT COUNT(*) FROM changes
                 WHERE previous_close > 0
                   AND ABS(close / previous_close - 1) > 0.30) AS abnormal_return_rows,
                (SELECT MAX(ABS(close / previous_close - 1)) FROM changes
                 WHERE previous_close > 0) AS max_abs_return
            """,
            [str(daily_path), TARGET_A_SHARE_PATTERN],
        ).fetchone()
        columns = [item[0] for item in connection.description]
        values = dict(zip(columns, summary, strict=True))

        mode_rows = (
            connection.execute(
                """
                SELECT DISTINCT UPPER(TRIM(CAST(price_adjustment AS VARCHAR))) AS mode
                FROM read_parquet(?)
                WHERE price_adjustment IS NOT NULL
                  AND TRIM(CAST(price_adjustment AS VARCHAR)) <> ''
                ORDER BY mode
                """,
                [str(daily_path)],
            ).fetchall()
            if has_adjustment_column
            else []
        )
    finally:
        connection.close()

    modes = [str(row[0]) for row in mode_rows]
    adjustment_mode = (
        modes[0] if len(modes) == 1 else "MIXED" if modes else "UNKNOWN"
    )
    values["start_date"] = str(values["start_date"]) if values["start_date"] else None
    values["end_date"] = str(values["end_date"]) if values["end_date"] else None
    values["max_abs_return"] = (
        float(values["max_abs_return"]) if values["max_abs_return"] is not None else None
    )
    values["price_adjustment_modes"] = modes
    values["price_adjustment_mode"] = adjustment_mode
    values["excluded_symbol_count"] = int(values["symbols"]) - int(
        values["target_a_share_symbols"]
    )

    quality_path = root / "quality_events.parquet"
    if quality_path.exists():
        quality_connection = duckdb.connect()
        try:
            quality_event_rows, quality_event_symbols = quality_connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT symbol) FROM read_parquet(?)",
                [str(quality_path)],
            ).fetchone()
            excluded_rows, excluded_symbols = quality_connection.execute(
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
                [str(daily_path), str(quality_path)],
            ).fetchone()
            class_rows = quality_connection.execute(
                """
                SELECT event_class, COUNT(*)
                FROM read_parquet(?)
                GROUP BY event_class
                ORDER BY event_class
                """,
                [str(quality_path)],
            ).fetchall()
        finally:
            quality_connection.close()
        values["research_quality_manifest"] = str(quality_path.resolve())
        values["research_quality_event_rows"] = int(quality_event_rows)
        values["research_quality_event_symbols"] = int(quality_event_symbols)
        values["research_excluded_rows"] = int(excluded_rows)
        values["research_excluded_symbols"] = int(excluded_symbols)
        values["research_quality_event_classes"] = {
            str(row[0]): int(row[1]) for row in class_rows
        }
        values["research_view_status"] = (
            "READY"
            if adjustment_mode in {"ADJUST_PREV", "ADJUST_POST"}
            else "DEGRADED"
        )
    else:
        values["research_quality_manifest"] = None
        values["research_quality_event_rows"] = None
        values["research_quality_event_symbols"] = None
        values["research_excluded_rows"] = None
        values["research_excluded_symbols"] = None
        values["research_quality_event_classes"] = {}
        values["research_view_status"] = "DEGRADED"

    issues: list[str] = []
    if adjustment_mode not in {"ADJUST_PREV", "ADJUST_POST"}:
        issues.append("price_adjustment_unknown_or_mixed")
    if int(values["abnormal_return_rows"]) > 0:
        issues.append("abnormal_return_jumps_over_30pct")
    if not quality_path.exists():
        issues.append("research_quality_manifest_missing")
    values["issues"] = issues
    values["status"] = "OK" if not issues else "DEGRADED"
    values["audited_at"] = datetime.now(UTC).isoformat()
    values["root"] = str(root.resolve())
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="审计东方财富历史归档")
    parser.add_argument("--root", default="data/eastmoney")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = audit(Path(args.root))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"report: {output.resolve()}")
    return 0 if result["status"] == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
