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
    ) -> pd.DataFrame:
        path = self.dataset_path(dataset_name)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB is required for Parquet storage") from exc
        predicates: list[str] = []
        parameters: list[Any] = [str(path)]
        if start is not None:
            predicates.append("trade_date >= ?")
            parameters.append(pd.Timestamp(start).to_pydatetime())
        if end is not None:
            predicates.append("trade_date <= ?")
            parameters.append(pd.Timestamp(end).to_pydatetime())
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            predicates.append(f"symbol IN ({placeholders})")
            parameters.extend(symbols)
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        connection = duckdb.connect()
        try:
            return connection.execute(
                "SELECT * FROM read_parquet(?)" + where + " ORDER BY trade_date, symbol",
                parameters,
            ).df()
        finally:
            connection.close()
