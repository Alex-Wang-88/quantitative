from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    symbols: int
    start_date: str | None
    end_date: str | None
    missing_columns: list[str]
    duplicate_rows: int
    invalid_price_rows: int
    invalid_date_rows: int

    @property
    def ok(self) -> bool:
        return (
            not self.missing_columns
            and not self.duplicate_rows
            and not self.invalid_price_rows
            and not self.invalid_date_rows
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


class HistoricalStore:
    """Parquet/DuckDB store for normalized historical market data."""

    REQUIRED_DAILY_COLUMNS = {"symbol", "trade_date", "open", "high", "low", "close", "amount"}
    REQUIRED_BENCHMARK_COLUMNS = {
        "benchmark_symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
    }
    TARGET_A_SHARE_SQL_PATTERN = (
        "^(000|001|002|003|300|301|302|600|601|603|605|688|689)"
    )
    QUALITY_EVENTS_FILENAME = "quality_events.parquet"
    _DATASET_RE = re.compile(r"^[A-Za-z0-9_-]+$")

    def __init__(self, root: str | Path = "data/market") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def normalize_daily(cls, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        aliases = {
            "ts_code": "symbol",
            "thscode": "symbol",
            "code": "symbol",
            "date": "trade_date",
            "time": "trade_date",
            "vol": "volume",
            "latestVolume": "volume",
            "turnover": "turnover_rate",
            "latestAmount": "amount",
            "latest": "close",
        }
        result = result.rename(
            columns={key: value for key, value in aliases.items() if key in result}
        )
        if "amount" not in result and "成交额" in result:
            result["amount"] = result["成交额"]
        if "trade_date" in result:
            result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
        if "symbol" in result:
            result["symbol"] = result["symbol"].astype("string").str.strip()
        for column in {"open", "high", "low", "close", "amount", "volume", "turnover_rate"}:
            if column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        sort_columns = [column for column in ("trade_date", "symbol") if column in result]
        if sort_columns:
            result = result.sort_values(sort_columns, na_position="last")
        return result.reset_index(drop=True)

    @classmethod
    def quality(cls, frame: pd.DataFrame) -> DataQualityReport:
        missing = sorted(cls.REQUIRED_DAILY_COLUMNS - set(frame.columns))
        if "trade_date" in frame:
            dates = pd.to_datetime(frame["trade_date"], errors="coerce")
            invalid_dates = int(dates.isna().sum())
            start = None if dates.dropna().empty else dates.min().date().isoformat()
            end = None if dates.dropna().empty else dates.max().date().isoformat()
        else:
            invalid_dates = len(frame)
            start = end = None
        duplicate_rows = (
            int(frame.duplicated(["symbol", "trade_date"]).sum())
            if {"symbol", "trade_date"}.issubset(frame.columns)
            else 0
        )
        numeric_columns = [
            column for column in ("open", "high", "low", "close", "amount") if column in frame
        ]
        invalid_numeric = (
            frame[numeric_columns].isna().any(axis=1)
            if numeric_columns
            else pd.Series(False, index=frame.index)
        )
        price_columns = [column for column in ("open", "high", "low", "close") if column in frame]
        invalid_prices = int(
            (invalid_numeric | (frame[price_columns] <= 0).any(axis=1)).sum()
            if price_columns
            else invalid_numeric.sum()
        )
        return DataQualityReport(
            rows=len(frame),
            symbols=int(frame["symbol"].nunique()) if "symbol" in frame else 0,
            start_date=start,
            end_date=end,
            missing_columns=missing,
            duplicate_rows=duplicate_rows,
            invalid_price_rows=invalid_prices,
            invalid_date_rows=invalid_dates,
        )

    def dataset_path(self, dataset_name: str = "daily") -> Path:
        if not self._DATASET_RE.fullmatch(dataset_name):
            raise ValueError("dataset name contains unsupported characters")
        return self.root / f"{dataset_name}.parquet"

    @property
    def quality_events_path(self) -> Path:
        """Path to the generated research quality-event manifest.

        The manifest is deliberately separate from the price archive.  The
        archive remains the source of truth for replay and audit, while
        research commands can opt into the point-in-time-safe view.
        """

        return self.root / self.QUALITY_EVENTS_FILENAME

    def write_daily(self, frame: pd.DataFrame, dataset_name: str = "daily") -> DataQualityReport:
        normalized = self.normalize_daily(frame)
        report = self.quality(normalized)
        if not report.ok:
            raise ValueError(f"daily data quality failed: {report.as_dict()}")
        output = self.dataset_path(dataset_name)
        temporary = output.with_suffix(".tmp.parquet")
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc
        connection = duckdb.connect()
        try:
            connection.register("daily_frame", normalized)
            connection.execute(
                "COPY daily_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
                [str(temporary)],
            )
        finally:
            connection.close()
        temporary.replace(output)
        return report

    def read_daily(
        self,
        dataset_name: str = "daily",
        start: date | str | None = None,
        end: date | str | None = None,
        symbols: list[str] | None = None,
        enrich_fundamentals: bool = False,
        a_shares_only: bool = False,
        require_adjusted: bool = False,
        exclude_quality_events: bool = False,
    ) -> pd.DataFrame:
        if exclude_quality_events and not self.quality_events_path.exists():
            raise ValueError(
                "research quality-event manifest is missing; run "
                "scripts/build_research_quality.py for this archive first"
            )
        if enrich_fundamentals:
            enriched = self._read_eastmoney_bundle(
                dataset_name=dataset_name,
                start=start,
                end=end,
                symbols=symbols,
                a_shares_only=a_shares_only,
                require_adjusted=require_adjusted,
                exclude_quality_events=exclude_quality_events,
            )
            if enriched is not None:
                return self._validate_read_frame(enriched, require_adjusted=require_adjusted)
        path = self.dataset_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(path)
        predicates: list[str] = []
        parameters: list[Any] = [str(path)]
        if start is not None:
            predicates.append("trade_date >= ?")
            parameters.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            predicates.append("trade_date <= ?")
            parameters.append(pd.Timestamp(end).to_pydatetime())
        if a_shares_only:
            predicates.append(
                "regexp_matches(lpad(CAST(symbol AS VARCHAR), 6, '0'), ?)"
            )
            parameters.append(self.TARGET_A_SHARE_SQL_PATTERN)
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            predicates.append(f"symbol IN ({placeholders})")
            parameters.extend(symbols)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        if exclude_quality_events:
            return self._read_quality_filtered(
                path=path,
                where=where,
                parameters=parameters,
                require_adjusted=require_adjusted,
            )
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                "SELECT * FROM read_parquet(?)" + where + " ORDER BY trade_date, symbol",
                parameters,
            ).df()
        finally:
            connection.close()
        return self._validate_read_frame(frame, require_adjusted=require_adjusted)

    def read_benchmarks(
        self,
        dataset_name: str = "benchmarks",
        start: date | str | None = None,
        end: date | str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read the independent Eastmoney benchmark/index archive.

        Index rows intentionally use ``benchmark_symbol`` instead of the stock
        ``symbol`` column.  This prevents an index such as ``000001`` (上证
        指数) from ever being mistaken for stock ``000001`` (平安银行).
        """

        path = self.dataset_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(path)
        predicates: list[str] = []
        parameters: list[Any] = [str(path)]
        if start is not None:
            predicates.append("CAST(trade_date AS DATE) >= ?")
            parameters.append(pd.Timestamp(start).date())
        if end is not None:
            predicates.append("CAST(trade_date AS DATE) <= ?")
            parameters.append(pd.Timestamp(end).date())
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            predicates.append(f"benchmark_symbol IN ({placeholders})")
            parameters.extend(str(symbol).strip() for symbol in symbols)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                "SELECT * FROM read_parquet(?)" + where
                + " ORDER BY trade_date, benchmark_symbol",
                parameters,
            ).df()
        finally:
            connection.close()
        missing = sorted(self.REQUIRED_BENCHMARK_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"benchmark data missing columns: {missing}")
        frame["benchmark_symbol"] = frame["benchmark_symbol"].astype("string").str.strip()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount", "pre_close"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.reset_index(drop=True)

    def _read_eastmoney_bundle(
        self,
        *,
        dataset_name: str,
        start: date | str | None,
        end: date | str | None,
        symbols: list[str] | None,
        a_shares_only: bool,
        require_adjusted: bool,
        exclude_quality_events: bool,
    ) -> pd.DataFrame | None:
        """Read the real Eastmoney archive with point-in-time fundamentals.

        The downloader keeps price bars, daily valuation data and financial
        reports in separate Parquet files.  Research should see one dated
        frame, but it must not use a report before its publication date.  DuckDB
        performs the exact daily join and the as-of report join without first
        materializing three large intermediate pandas frames.

        ``None`` means the store is a normal Parquet root without the optional
        Eastmoney archive files, so callers can retain the ordinary read path.
        """

        if dataset_name != "daily":
            return None
        daily_path = self.dataset_path(dataset_name)
        fundamentals_path = self.root / "fundamentals_daily.parquet"
        reports_path = self.root / "fundamentals_reports.parquet"
        if not daily_path.exists() or not fundamentals_path.exists():
            return None

        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc

        predicates: list[str] = []
        predicate_parameters: list[Any] = []
        if start is not None:
            predicates.append("CAST(trade_date AS DATE) >= ?")
            predicate_parameters.append(pd.Timestamp(start).date())
        if end is not None:
            predicates.append("CAST(trade_date AS DATE) <= ?")
            predicate_parameters.append(pd.Timestamp(end).date())
        if a_shares_only:
            predicates.append(
                "regexp_matches(lpad(CAST(symbol AS VARCHAR), 6, '0'), ?)"
            )
            predicate_parameters.append(self.TARGET_A_SHARE_SQL_PATTERN)
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            predicates.append(f"symbol IN ({placeholders})")
            predicate_parameters.extend(str(symbol) for symbol in symbols)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""

        joins: list[str] = []
        select_columns = ["d.*"]
        if exclude_quality_events:
            parameters: list[Any] = [str(daily_path), str(self.quality_events_path)]
            ctes = [
                """
                daily_source AS (
                    SELECT
                        d.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY d.symbol ORDER BY d.trade_date
                        ) AS _quality_row_number
                    FROM read_parquet(?) d
                )
                """,
                """
                quality_events AS (
                    SELECT
                        symbol,
                        event_row_number,
                        exclude_start_row_number,
                        exclude_end_row_number
                    FROM read_parquet(?)
                )
                """,
                f"""
                marked_daily AS (
                    SELECT
                        d.*,
                        SUM(
                            CASE WHEN e.event_row_number IS NOT NULL THEN 1 ELSE 0 END
                        ) OVER (
                            PARTITION BY d.symbol
                            ORDER BY d._quality_row_number
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS _quality_segment
                    FROM daily_source d
                    LEFT JOIN quality_events e
                      ON d.symbol = e.symbol
                     AND d._quality_row_number = e.event_row_number
                ),
                daily AS (
                    SELECT
                        d.*,
                        CONCAT(CAST(d.symbol AS VARCHAR), '::',
                               CAST(d._quality_segment AS VARCHAR)) AS _feature_group
                    FROM marked_daily d
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM quality_events e
                        WHERE e.symbol = d.symbol
                          AND d._quality_row_number
                              BETWEEN e.exclude_start_row_number
                              AND e.exclude_end_row_number
                    )
                    {"AND " + " AND ".join(predicates) if predicates else ""}
                )
                """,
            ]
        else:
            parameters = [str(daily_path)]
            ctes = [f"daily AS (SELECT * FROM read_parquet(?) {where})"]
        parameters.extend(predicate_parameters)
        parameters.append(str(fundamentals_path))
        ctes.append(
            """
            fundamentals_daily AS (
                SELECT
                    symbol,
                    CAST(trade_date AS DATE) AS trade_date,
                    pe_ttm,
                    pb_lyr,
                    dy_ttm,
                    tot_mv,
                    a_mv_ex_ltd,
                    turnrate
                FROM read_parquet(?)
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY symbol, CAST(trade_date AS DATE)
                    ORDER BY trade_date DESC
                ) = 1
            )
            """
        )
        joins.append(
            """
            LEFT JOIN fundamentals_daily f
              ON d.symbol = f.symbol
             AND CAST(d.trade_date AS DATE) = f.trade_date
            """
        )
        select_columns.extend(
            [
                "f.pe_ttm AS pe",
                "f.pb_lyr AS pb",
                "f.dy_ttm AS dividend_yield",
                "f.tot_mv AS market_cap",
                "f.a_mv_ex_ltd AS float_market_cap",
                "f.turnrate AS turnover_rate",
            ]
        )

        if reports_path.exists():
            parameters.append(str(reports_path))
            ctes.append(
                """
                reports AS (
                    SELECT
                        symbol,
                        CAST(pub_date AS DATE) AS pub_date,
                        CAST(rpt_date AS DATE) AS rpt_date,
                        inc_oper_yoy,
                        roe_weight,
                        net_prof_pcom_yoy
                    FROM read_parquet(?)
                    WHERE pub_date IS NOT NULL
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY symbol, CAST(pub_date AS DATE)
                        ORDER BY rpt_date DESC NULLS LAST
                    ) = 1
                )
                """
            )
            joins.append(
                """
                ASOF LEFT JOIN reports r
                  ON d.symbol = r.symbol
                 AND CAST(d.trade_date AS DATE) >= r.pub_date
                """
            )
            select_columns.extend(
                [
                    "r.roe_weight AS roe",
                    "r.net_prof_pcom_yoy AS profit_growth",
                    "r.inc_oper_yoy AS revenue_growth",
                    "r.pub_date AS ann_date",
                ]
            )

        query = (
            "WITH "
            + ", ".join(ctes)
            + " SELECT "
            + ", ".join(select_columns)
            + " FROM daily d "
            + " ".join(joins)
            + " ORDER BY d.trade_date, d.symbol"
        )
        connection = duckdb.connect()
        try:
            if require_adjusted:
                schema = connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [str(daily_path)]
                ).fetchall()
                if not any(row[0] == "price_adjustment" for row in schema):
                    raise ValueError(
                        "research data must carry one consistent adjusted-price mode; "
                        "received UNKNOWN. Re-download the Eastmoney archive with "
                        "--adjust prev into a new archive root."
                    )
                modes = connection.execute(
                    """
                    SELECT DISTINCT UPPER(TRIM(CAST(price_adjustment AS VARCHAR)))
                    FROM read_parquet(?)
                    WHERE price_adjustment IS NOT NULL
                      AND TRIM(CAST(price_adjustment AS VARCHAR)) <> ''
                    """,
                    [str(daily_path)],
                ).fetchall()
                if len(modes) != 1 or str(modes[0][0]) not in {
                    "ADJUST_PREV",
                    "ADJUST_POST",
                }:
                    mode = (
                        str(modes[0][0])
                        if len(modes) == 1
                        else "MIXED" if modes else "UNKNOWN"
                    )
                    raise ValueError(
                        "research data must carry one consistent adjusted-price mode; "
                        f"received {mode}. Re-download the Eastmoney archive with "
                        "--adjust prev into a new archive root."
                    )
            frame = connection.execute(query, parameters).df()
        finally:
            connection.close()
        return self.normalize_daily(frame)

    def _read_quality_filtered(
        self,
        *,
        path: Path,
        where: str,
        parameters: list[Any],
        require_adjusted: bool,
    ) -> pd.DataFrame:
        """Read a daily dataset while isolating generated quality events.

        The SQL keeps the row number over the complete symbol history, so a
        date-limited read cannot accidentally reinterpret an event as the
        first row of a new window.  ``_feature_group`` is consumed by the
        feature engine to reset rolling calculations after the excluded
        window.
        """

        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc
        quality_path = self.quality_events_path
        query = f"""
            WITH daily_source AS (
                SELECT
                    d.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.symbol ORDER BY d.trade_date
                    ) AS _quality_row_number
                FROM read_parquet(?) d
            ), quality_events AS (
                SELECT
                    symbol,
                    event_row_number,
                    exclude_start_row_number,
                    exclude_end_row_number
                FROM read_parquet(?)
            ), marked_daily AS (
                SELECT
                    d.*,
                    SUM(
                        CASE WHEN e.event_row_number IS NOT NULL THEN 1 ELSE 0 END
                    ) OVER (
                        PARTITION BY d.symbol
                        ORDER BY d._quality_row_number
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS _quality_segment
                FROM daily_source d
                LEFT JOIN quality_events e
                  ON d.symbol = e.symbol
                 AND d._quality_row_number = e.event_row_number
            )
            SELECT
                d.*,
                CONCAT(CAST(d.symbol AS VARCHAR), '::',
                       CAST(d._quality_segment AS VARCHAR)) AS _feature_group
            FROM marked_daily d
            WHERE NOT EXISTS (
                SELECT 1
                FROM quality_events e
                WHERE e.symbol = d.symbol
                  AND d._quality_row_number
                      BETWEEN e.exclude_start_row_number
                      AND e.exclude_end_row_number
            )
            {"AND " + where[7:] if where.startswith(" WHERE ") else where}
            ORDER BY d.trade_date, d.symbol
        """
        # ``parameters`` already starts with the daily path and contains the
        # date/symbol predicate values.  The quality manifest is the second
        # positional parameter in the query.
        query_parameters = [parameters[0], str(quality_path), *parameters[1:]]
        connection = duckdb.connect()
        try:
            if require_adjusted:
                schema = connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
                ).fetchall()
                if not any(row[0] == "price_adjustment" for row in schema):
                    raise ValueError(
                        "research data must carry one consistent adjusted-price mode; "
                        "received UNKNOWN. Re-download the Eastmoney archive with "
                        "--adjust prev into a new archive root."
                    )
            frame = connection.execute(query, query_parameters).df()
        finally:
            connection.close()
        return self._validate_read_frame(frame, require_adjusted=require_adjusted)

    @staticmethod
    def adjustment_mode(frame: pd.DataFrame) -> str:
        """Return the single price-adjustment mode carried by a data frame."""

        if "price_adjustment" not in frame.columns:
            return "UNKNOWN"
        values = sorted(
            {
                str(value).strip().upper()
                for value in frame["price_adjustment"].dropna().tolist()
                if str(value).strip()
            }
        )
        if len(values) != 1:
            return "UNKNOWN" if not values else "MIXED"
        return values[0]

    @classmethod
    def _validate_read_frame(
        cls, frame: pd.DataFrame, *, require_adjusted: bool
    ) -> pd.DataFrame:
        if require_adjusted:
            mode = cls.adjustment_mode(frame)
            if mode not in {"ADJUST_PREV", "ADJUST_POST"}:
                raise ValueError(
                    "research data must carry one consistent adjusted-price mode; "
                    f"received {mode}. Re-download the Eastmoney archive with "
                    "--adjust prev into a new archive root."
                )
        return frame
