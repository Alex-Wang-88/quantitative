from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngine:
    """Point-in-time feature construction for the first research baseline.

    The engine deliberately keeps a small, explainable feature set.  Every
    factor is converted to a cross-sectional score in ``[0, 1]`` where a
    larger value means "more desirable".  This convention is important: the
    portfolio layer can combine the factors without having to know whether a
    raw input such as volatility, PE, or market value is better when it is
    large or small.

    Optional fundamental columns are consumed when the data provider supplies
    them.  Missing optional data is represented by a neutral score and an
    explicit coverage column; it is never silently replaced with fabricated
    financial values.
    """

    BASELINE_FACTOR_COLUMNS = [
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "reversal_5",
        "volatility_20",
        "turnover_20",
        "value_score",
        "quality_score",
        "growth_score",
        "size_score",
    ]

    # F1 is computed and exposed for research, but is deliberately not part
    # of ``FACTOR_COLUMNS`` yet.  This keeps the currently promoted model
    # schema stable until the new factors pass their own out-of-sample and
    # turnover checks.
    F1_FACTOR_COLUMNS = [
        "atr_20",
        "gap_1",
        "close_position_20",
        "volume_trend_20",
        "amount_stability_20",
        "relative_momentum_20",
        "beta_60",
        "industry_momentum_20",
    ]
    FACTOR_COLUMNS = BASELINE_FACTOR_COLUMNS
    RESEARCH_FACTOR_COLUMNS = BASELINE_FACTOR_COLUMNS + F1_FACTOR_COLUMNS

    FACTOR_METADATA = {
        "momentum_5": {
            "name": "5日动量",
            "description": "过去5个交易日收益率，偏好近期上涨",
            "source": "close",
            "direction": "higher_is_better",
        },
        "momentum_20": {
            "name": "20日动量",
            "description": "过去20个交易日收益率，偏好中短期趋势",
            "source": "close",
            "direction": "higher_is_better",
        },
        "momentum_60": {
            "name": "60日动量",
            "description": "过去60个交易日收益率，偏好中期趋势",
            "source": "close",
            "direction": "higher_is_better",
        },
        "momentum_120": {
            "name": "120日动量",
            "description": "过去120个交易日收益率，偏好长期趋势",
            "source": "close",
            "direction": "higher_is_better",
        },
        "reversal_5": {
            "name": "5日反转",
            "description": "5日收益率取反，偏好短期回撤后的相对修复",
            "source": "close",
            "direction": "higher_is_better",
        },
        "volatility_20": {
            "name": "20日低波",
            "description": "20日实现波动率取反，偏好更低的波动",
            "source": "close",
            "direction": "higher_is_better",
        },
        "turnover_20": {
            "name": "20日流动性",
            "description": "优先使用20日平均换手率，没有换手率时使用成交额",
            "source": "turnover_rate/amount",
            "direction": "higher_is_better",
        },
        "value_score": {
            "name": "估值",
            "description": "综合PE、PB，偏好更便宜的股票",
            "source": "pe/pb",
            "direction": "lower_is_better",
        },
        "quality_score": {
            "name": "质量",
            "description": "优先使用ROE，偏好盈利质量更高的公司",
            "source": "roe",
            "direction": "higher_is_better",
        },
        "growth_score": {
            "name": "成长",
            "description": "优先使用利润增速，偏好成长更快的公司",
            "source": "profit_growth",
            "direction": "higher_is_better",
        },
        "size_score": {
            "name": "小市值",
            "description": "优先使用总市值，没有市值时使用成交额代理",
            "source": "market_cap/tot_mv/amount",
            "direction": "lower_is_better",
        },
        "atr_20": {
            "name": "20日低ATR",
            "description": "真实波幅相对收盘价的20日均值取反，控制价格波动风险",
            "source": "open/high/low/close",
            "direction": "higher_is_better",
        },
        "gap_1": {
            "name": "低跳空风险",
            "description": "隔夜开盘跳空绝对幅度取反，偏好更平滑的可成交路径",
            "source": "open/close",
            "direction": "higher_is_better",
        },
        "close_position_20": {
            "name": "20日收盘位置",
            "description": "收盘价位于过去20日高低区间的位置，偏好接近区间上沿",
            "source": "high/low/close",
            "direction": "higher_is_better",
        },
        "volume_trend_20": {
            "name": "量价趋势",
            "description": "5日平均成交量相对20日平均成交量的变化",
            "source": "volume/amount",
            "direction": "higher_is_better",
        },
        "amount_stability_20": {
            "name": "成交额稳定性",
            "description": "20日成交额变异系数取反，偏好流动性更稳定的股票",
            "source": "amount",
            "direction": "higher_is_better",
        },
        "relative_momentum_20": {
            "name": "相对大盘动量",
            "description": "20日个股收益减去股票池等权市场收益",
            "source": "close",
            "direction": "higher_is_better",
        },
        "beta_60": {
            "name": "60日Beta中性",
            "description": "相对股票池等权市场的Beta距离1越近分数越高",
            "source": "close",
            "direction": "higher_is_better",
        },
        "industry_momentum_20": {
            "name": "行业20日动量",
            "description": "行业等权20日收益，缺少行业时使用UNKNOWN分组",
            "source": "sector/close",
            "direction": "higher_is_better",
        },
    }

    COVERAGE_COLUMNS = [
        "value_data_available",
        "quality_data_available",
        "growth_data_available",
        "size_data_available",
    ]

    def build_features(self, daily_frame: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "trade_date", "close", "amount"}
        missing = required - set(daily_frame.columns)
        if missing:
            raise ValueError(f"daily frame missing columns: {sorted(missing)}")
        frame = daily_frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        self.validate_point_in_time(frame)
        group = frame.groupby("symbol", group_keys=False)
        frame["momentum_5"] = group["close"].pct_change(5)
        frame["momentum_20"] = group["close"].pct_change(20)
        frame["momentum_60"] = group["close"].pct_change(60)
        frame["momentum_120"] = group["close"].pct_change(120)
        frame["reversal_5"] = -group["close"].pct_change(5)

        # Store the raw low-volatility signal as the negative of realized
        # volatility.  The final rank can then use the same
        # ``higher_is_better`` convention as every other factor.
        realized_volatility = group["close"].pct_change().transform(
            lambda series: series.rolling(20).std()
        )
        frame["volatility_20"] = -realized_volatility

        turnover = self._first_available(frame, ("turnover_rate", "turnover"))
        turnover_available = turnover.notna() & (turnover > 0)
        amount_roll = group["amount"].transform(lambda series: series.rolling(20).mean())
        turnover_roll = turnover.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(20).mean()
        )
        frame["turnover_20"] = turnover_roll.where(
            turnover_roll.notna() & turnover_available, amount_roll
        )

        pe = self._positive_numeric(frame.get("pe"), frame.index)
        pb = self._positive_numeric(frame.get("pb"), frame.index)
        value_parts = pd.concat(
            [
                self._cross_sectional_score(frame, pe, higher_is_better=False),
                self._cross_sectional_score(frame, pb, higher_is_better=False),
            ],
            axis=1,
        )
        frame["value_data_available"] = pe.notna() | pb.notna()
        frame["value_score"] = value_parts.where(value_parts.notna()).mean(axis=1).fillna(0.5)

        roe = self._first_available(frame, ("roe", "roe_weight_avg", "roe_cut"))
        frame["quality_data_available"] = roe.notna()
        frame["quality_score"] = self._cross_sectional_score(
            frame, roe, higher_is_better=True
        ).fillna(0.5)

        growth = self._first_available(
            frame,
            ("profit_growth", "net_prof_pcom_yoy", "inc_oper_yoy", "revenue_growth"),
        )
        frame["growth_data_available"] = growth.notna()
        frame["growth_score"] = self._cross_sectional_score(
            frame, growth, higher_is_better=True
        ).fillna(0.5)

        market_cap = self._first_available(
            frame, ("market_cap", "market_value", "tot_mv", "a_mv_ex_ltd")
        )
        size_fallback = amount_roll
        size_metric = market_cap.where(market_cap.notna() & (market_cap > 0), size_fallback)
        frame["size_data_available"] = market_cap.notna() & (market_cap > 0)
        frame["size_score"] = self._cross_sectional_score(
            frame, size_metric, higher_is_better=False
        ).fillna(0.5)

        # F1 price/volume candidates.  These use only the current row and
        # trailing windows, so they remain point-in-time when the frame is
        # truncated at any historical date.
        close = self._numeric(frame.get("close"), frame.index)
        previous_close = close.groupby(frame["symbol"], group_keys=False).shift(1)
        open_price = self._numeric(frame.get("open"), frame.index)
        high = self._numeric(frame.get("high"), frame.index)
        low = self._numeric(frame.get("low"), frame.index)
        true_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1, skipna=True)
        atr = true_range.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(20).mean()
        )
        frame["atr_20"] = -(atr / close)

        overnight_gap = (open_price / previous_close) - 1
        frame["gap_1"] = -overnight_gap.abs()

        rolling_high = high.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(20).max()
        )
        rolling_low = low.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(20).min()
        )
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        frame["close_position_20"] = (close - rolling_low) / price_range

        volume = self._numeric(frame.get("volume"), frame.index)
        volume_fallback = frame["amount"] / close.replace(0, np.nan)
        volume = volume.where(volume.notna() & (volume > 0), volume_fallback)
        volume_group = volume.groupby(frame["symbol"], group_keys=False)
        volume_short = volume_group.transform(lambda series: series.rolling(5).mean())
        volume_long = volume_group.transform(lambda series: series.rolling(20).mean())
        frame["volume_trend_20"] = (volume_short / volume_long) - 1

        amount_mean = group["amount"].transform(lambda series: series.rolling(20).mean())
        amount_std = group["amount"].transform(lambda series: series.rolling(20).std())
        frame["amount_stability_20"] = -(amount_std / amount_mean)

        daily_return = close.groupby(frame["symbol"], group_keys=False).pct_change()
        frame["_f1_daily_return"] = daily_return
        market_return = frame.groupby("trade_date")["_f1_daily_return"].transform("mean")
        frame["_f1_market_return"] = market_return
        market_by_date = (
            frame.groupby("trade_date")["_f1_market_return"].first().sort_index()
        )
        market_momentum = self._rolling_compound(market_by_date, 20)
        frame["relative_momentum_20"] = frame["momentum_20"] - frame["trade_date"].map(
            market_momentum
        )

        asset_mean = daily_return.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(60).mean()
        )
        market_mean = market_return.groupby(frame["symbol"], group_keys=False).transform(
            lambda series: series.rolling(60).mean()
        )
        cross_mean = (daily_return * market_return).groupby(
            frame["symbol"], group_keys=False
        ).transform(lambda series: series.rolling(60).mean())
        market_square_mean = (market_return**2).groupby(
            frame["symbol"], group_keys=False
        ).transform(lambda series: series.rolling(60).mean())
        covariance = cross_mean - asset_mean * market_mean
        market_variance = (market_square_mean - market_mean**2).replace(0, np.nan)
        beta = covariance / market_variance
        frame["_f1_beta_raw"] = beta
        frame["beta_60"] = -(beta - 1).abs()

        if "sector" in frame.columns:
            sector = frame["sector"].fillna("UNKNOWN").astype(str)
        else:
            sector = pd.Series("UNKNOWN", index=frame.index, dtype="string")
        sector_frame = frame.assign(_f1_sector=sector).groupby(
            ["_f1_sector", "trade_date"], as_index=False, sort=True
        )["_f1_daily_return"].mean()
        sector_frame["_f1_industry_momentum"] = sector_frame.groupby("_f1_sector", sort=False)[
            "_f1_daily_return"
        ].transform(lambda series: self._rolling_compound(series, 20))
        sector_lookup = sector_frame.set_index(["_f1_sector", "trade_date"])[
            "_f1_industry_momentum"
        ]
        sector_keys = pd.MultiIndex.from_arrays(
            [sector.to_numpy(), frame["trade_date"].to_numpy()]
        )
        frame["industry_momentum_20"] = sector_lookup.reindex(sector_keys).to_numpy()

        for column in self.RESEARCH_FACTOR_COLUMNS:
            frame[f"{column}_available"] = frame[column].notna().astype(float)

        for column in self.RESEARCH_FACTOR_COLUMNS:
            # Fundamental scores are already cross-sectional.  Ranking every
            # factor here gives all factors the same scale and also handles
            # raw momentum/volatility/liquidity columns consistently.
            frame[column] = frame.groupby("trade_date")[column].transform(self._safe_rank)
        frame["factor_score"] = frame[self.FACTOR_COLUMNS].mean(axis=1, skipna=True)
        frame["f1_factor_score"] = frame[self.F1_FACTOR_COLUMNS].mean(axis=1, skipna=True)
        frame["research_factor_score"] = frame[self.RESEARCH_FACTOR_COLUMNS].mean(
            axis=1, skipna=True
        )
        frame["fundamental_coverage"] = frame[self.COVERAGE_COLUMNS].mean(axis=1)
        return frame

    def build_training_frame(
        self, daily_frame: pd.DataFrame, horizon_days: int = 5
    ) -> pd.DataFrame:
        frame = self.build_features(daily_frame)
        group = frame.groupby("symbol", group_keys=False)
        frame["future_return"] = group["close"].shift(-horizon_days) / frame["close"] - 1
        benchmark = frame.groupby("trade_date")["future_return"].transform("mean")
        frame["label"] = frame["future_return"] - benchmark
        return frame.dropna(subset=self.FACTOR_COLUMNS + ["label"]).reset_index(drop=True)

    def latest(self, daily_frame: pd.DataFrame) -> pd.DataFrame:
        frame = self.build_features(daily_frame)
        return (
            frame.sort_values("trade_date")
            .groupby("symbol", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    @staticmethod
    def validate_point_in_time(frame: pd.DataFrame) -> None:
        trade = pd.to_datetime(frame["trade_date"], errors="coerce")
        for column in ("ann_date", "pub_date"):
            if column not in frame.columns:
                continue
            announced = pd.to_datetime(frame[column], errors="coerce")
            if (announced.notna() & (announced > trade)).any():
                raise ValueError(
                    f"future financial disclosure detected in point-in-time dataset: {column}"
                )

    @staticmethod
    def _numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
        if series is None:
            return pd.Series(np.nan, index=index, dtype="float64")
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _positive_numeric(
        cls, series: pd.Series | None, index: pd.Index
    ) -> pd.Series:
        numeric = cls._numeric(series, index)
        return numeric.where(numeric > 0)

    @classmethod
    def _first_available(
        cls, frame: pd.DataFrame, columns: tuple[str, ...]
    ) -> pd.Series:
        result = pd.Series(np.nan, index=frame.index, dtype="float64")
        for column in columns:
            if column not in frame.columns:
                continue
            candidate = cls._numeric(frame[column], frame.index)
            result = result.where(result.notna(), candidate)
        return result

    @classmethod
    def _cross_sectional_score(
        cls, frame: pd.DataFrame, series: pd.Series, *, higher_is_better: bool
    ) -> pd.Series:
        numeric = cls._numeric(series, frame.index)
        ranked = numeric.groupby(frame["trade_date"]).rank(
            pct=True, ascending=higher_is_better
        )
        return ranked.where(numeric.notna())

    @staticmethod
    def _safe_rank(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return numeric.rank(pct=True).fillna(0.5)

    @staticmethod
    def _rolling_compound(series: pd.Series, window: int) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        return (1 + numeric).rolling(window).apply(np.prod, raw=True) - 1
