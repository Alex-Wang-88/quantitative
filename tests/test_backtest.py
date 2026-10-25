import numpy as np
import pandas as pd
import pytest

from quantpaper.backtest import BacktestConfig, FactorBacktester
from quantpaper.domain import Board, Security
from quantpaper.features import FeatureEngine


def _daily_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=150)
    rows: list[dict[str, object]] = []
    specs = {
        "600001": (10.0, 0.0020, Board.MAIN, "金融"),
        "300001": (20.0, 0.0008, Board.CHINEXT_300, "先进制造"),
        "688001": (30.0, -0.0004, Board.STAR_688, "电子"),
    }
    for index, current_date in enumerate(dates):
        for symbol, (start, drift, _board, sector) in specs.items():
            close = start * (1 + drift) ** index
            open_price = close * 1.001
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": current_date,
                    "open": open_price,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "amount": 10_000_000,
                    "sector": sector,
                    "pe": 20.0 + index % 3,
                    "pb": 2.0,
                    "roe": 0.12,
                    "profit_growth": 0.10,
                }
            )
    return pd.DataFrame(rows)


def test_factor_backtest_executes_next_open_without_same_day_lookahead() -> None:
    securities = [
        Security("600001", "测试主板", "SSE", Board.MAIN, "金融"),
        Security("300001", "测试创业板", "SZSE", Board.CHINEXT_300, "先进制造"),
        Security("688001", "测试科创板", "SSE", Board.STAR_688, "电子"),
    ]
    result = FactorBacktester(securities).run(
        _daily_frame(),
        BacktestConfig(
            max_positions=1,
            single_weight_cap=0.80,
            max_daily_turnover=0.50,
            max_daily_orders=4,
        ),
    )

    assert result.quality.ok
    assert result.trades
    assert all(trade.trade_date > trade.signal_date for trade in result.trades)
    assert result.equity_curve["trade_date"].iloc[0] == "2025-01-02"
    assert result.stats["final_equity"] > 0
    assert np.isfinite(result.stats["sharpe"])
    assert result.as_dict()["equity_curve"]


def test_factor_backtest_can_compare_f1_research_score_without_promoting_it() -> None:
    engine = FeatureEngine()
    features = engine.build_features(_daily_frame())
    result = FactorBacktester(feature_engine=engine).run(
        _daily_frame(),
        BacktestConfig(
            max_positions=1,
            single_weight_cap=0.80,
            max_daily_turnover=0.50,
            max_daily_orders=4,
        ),
        feature_frame=features,
        score_column="f1_factor_score",
    )

    assert result.quality.ok
    assert result.stats["final_equity"] > 0
    assert engine.FACTOR_COLUMNS == engine.BASELINE_FACTOR_COLUMNS


def _ranking_fixture(
    n_symbols: int = 12, n_dates: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily prices plus a score column whose top-N set rotates every day."""

    dates = pd.bdate_range("2025-01-02", periods=n_dates)
    price_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    for index, current_date in enumerate(dates):
        for symbol_index in range(n_symbols):
            symbol = f"6000{symbol_index:02d}"
            close = 10.0 + symbol_index * 0.1
            price_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": current_date,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "amount": 10_000_000,
                    "sector": f"S{symbol_index % 3}",
                }
            )
            score_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": current_date,
                    "score": float((symbol_index + index) % n_symbols),
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(score_rows)


def _ranking_config(**overrides: object) -> BacktestConfig:
    base: dict[str, object] = {
        "max_positions": 6,
        "cash_floor": 0.0,
        "single_weight_cap": 0.50,
        "industry_weight_cap": 1.0,
        "max_daily_turnover": 1.0,
        "max_daily_orders": 50,
    }
    base.update(overrides)
    return BacktestConfig(**base)


def test_hold_buffer_cuts_turnover_when_the_ranking_rotates() -> None:
    daily, scores = _ranking_fixture()
    strict = FactorBacktester().run(
        daily, _ranking_config(hold_buffer_ratio=0.0), feature_frame=scores, score_column="score"
    )
    buffered = FactorBacktester().run(
        daily, _ranking_config(hold_buffer_ratio=2.0), feature_frame=scores, score_column="score"
    )

    assert strict.stats["hold_buffer_ratio"] == 0.0
    assert buffered.stats["hold_buffer_ratio"] == 2.0
    assert buffered.stats["turnover"] < strict.stats["turnover"]
    assert buffered.stats["trade_count"] < strict.stats["trade_count"]
    assert buffered.equity_curve["position_count"].max() <= 6


def test_rebalance_interval_reduces_trading() -> None:
    daily, scores = _ranking_fixture()
    daily_rebalance = FactorBacktester().run(
        daily, _ranking_config(rebalance_every_n_days=1), feature_frame=scores, score_column="score"
    )
    slow_rebalance = FactorBacktester().run(
        daily,
        _ranking_config(rebalance_every_n_days=10),
        feature_frame=scores,
        score_column="score",
    )

    assert slow_rebalance.stats["rebalance_every_n_days"] == 10.0
    assert slow_rebalance.stats["trade_count"] < daily_rebalance.stats["trade_count"]


def test_backtest_discloses_which_security_status_filters_ran() -> None:
    daily, scores = _ranking_fixture(n_symbols=6, n_dates=20)
    config = _ranking_config()

    without_status = FactorBacktester().run(
        daily, config, feature_frame=scores, score_column="score"
    )
    assert without_status.stats["st_filter_applied"] is False
    assert without_status.stats["suspension_filter_applied"] is False
    assert without_status.stats["security_status_source"].startswith("UNAVAILABLE")

    with_status = FactorBacktester().run(
        daily.assign(is_st=False, suspended=False),
        config,
        feature_frame=scores,
        score_column="score",
    )
    assert with_status.stats["st_filter_applied"] is True
    assert with_status.stats["suspension_filter_applied"] is True
    assert with_status.stats["security_status_source"] == "input_columns"


def _full_book_config(**overrides: object) -> BacktestConfig:
    """A config whose target book is ``1 - cash_floor`` of equity.

    ``base_weight = min(single_weight_cap, industry_weight_cap,
    deployable/candidate_count) = min(0.09, 1.0, 0.90/10) = 0.09``, so ten names
    carry 90% of equity.  One name is deliberately smaller than the one-leg
    slice of the rotation budget (``max_daily_turnover / 2 = 0.10``): an exit has
    to fit inside that slice to be executed at all, so a book whose names are
    larger than it can never rotate and would (correctly) freeze.
    """

    base: dict[str, object] = {
        "max_positions": 10,
        "cash_floor": 0.10,
        "single_weight_cap": 0.09,
        "industry_weight_cap": 1.0,
        "max_daily_turnover": 0.20,
        "max_daily_orders": 40,
        "rebalance_every_n_days": 5,
    }
    base.update(overrides)
    return BacktestConfig(**base)


def _invested_ratio(result) -> pd.Series:
    curve = result.equity_curve
    return curve["long_market_value"] / curve["equity"]


def test_first_rebalance_builds_the_book_in_one_session() -> None:
    daily, scores = _ranking_fixture()
    result = FactorBacktester().run(
        daily, _full_book_config(), feature_frame=scores, score_column="score"
    )

    invested = _invested_ratio(result)

    # The book is not ramped up two names at a time: the very first session that
    # trades reaches the target exposure, not a fraction of it.
    assert invested.max() >= 0.85
    first_traded = result.equity_curve["position_count"].gt(0).idxmax()
    assert invested.loc[first_traded] >= 0.85
    assert result.equity_curve["position_count"].loc[first_traded] == 10


def test_book_stays_at_its_target_exposure_instead_of_decaying() -> None:
    daily, scores = _ranking_fixture()
    result = FactorBacktester().run(
        daily, _full_book_config(), feature_frame=scores, score_column="score"
    )

    invested = _invested_ratio(result)
    equity = result.equity_curve
    # A cap on rotation must never turn into a slow liquidation.  Every rebalance
    # restores the target gross exposure, so the final level has to hold up
    # against the peak.
    assert invested.iloc[-1] >= 0.85
    assert invested.iloc[-1] >= invested.max() - 0.03
    assert equity["position_count"].iloc[-1] == 10
    # The position cap is respected throughout, not just at the end.
    assert equity["position_count"].max() <= 10
    # ... and the report says so on its own, instead of leaving the reader to
    # derive exposure from the equity curve.
    assert result.stats["invested_ratio_min"] >= 0.85
    assert result.stats["invested_ratio_final"] == pytest.approx(
        float(invested.iloc[-1]), abs=1e-6
    )


def test_realised_turnover_respects_the_configured_band() -> None:
    daily, scores = _ranking_fixture()
    result = FactorBacktester().run(
        daily, _full_book_config(), feature_frame=scores, score_column="score"
    )

    curve = result.equity_curve
    traded = curve[curve["turnover"] > 0].reset_index(drop=True)

    assert not traded.empty
    # The fixture rotates its ranking every day, so every five-session rebalance
    # wants to replace the whole book.  Day one is the only session allowed
    # outside the band: it is the session that funds the book from all-cash to
    # target exposure, which is a deployment, not a rotation.
    first = traded.iloc[0]
    assert first["one_way_turnover"] >= 0.40

    later = traded.iloc[1:]
    assert not later.empty
    assert later["turnover"].max() <= 0.20 + 1e-6
    # The band is used, not merely respected: with the cap binding on every
    # rebalance, realised turnover should sit on it.
    assert later["turnover"].min() >= 0.10 - 1e-6
    # Low frequency: only rebalance sessions trade at all.
    assert len(traded) <= len(curve) / 4
    # The published stat has to agree with the curve, and has to exclude the
    # deployment session rather than let it dominate the maximum.
    assert result.stats["entry_day_turnover"] == pytest.approx(
        float(first["turnover"]), abs=1e-6
    )
    assert result.stats["daily_turnover_max"] == pytest.approx(
        float(later["turnover"].max()), abs=1e-6
    )


def test_book_is_topped_back_up_after_a_deployment_shortfall() -> None:
    """A shortfall that could not be closed in one session stays funded.

    The book is only "always fully invested" if the deficit to the target
    exposure survives the session that failed to close it.  A one-shot "exempt
    the first rebalance" rule fails exactly here: the book would stay short
    forever, and every later rebalance would be throttled to the rotation cap
    while the cash never went back to work.
    """

    dates = pd.bdate_range("2025-01-02", periods=60)
    late = dates[5]
    daily, scores = _ranking_fixture(n_symbols=10)
    early_symbols = ["600000", "600001", "600002"]
    listed_later = (
        ~daily["symbol"].isin(early_symbols) & (daily["trade_date"] < late)
    )
    # The first rebalance can only see three names; the universe widens to ten
    # well after the book has been built on a third of the capital.
    thin = daily[~listed_later]
    thin_scores = scores[scores["symbol"].isin(early_symbols) | (scores["trade_date"] >= late)]

    result = FactorBacktester().run(
        thin, _full_book_config(), feature_frame=thin_scores, score_column="score"
    )

    invested = _invested_ratio(result)
    curve = result.equity_curve
    traded = curve[curve["turnover"] > 0].reset_index(drop=True)
    first_traded = curve["position_count"].gt(0).idxmax()

    assert invested.loc[first_traded] < 0.60
    assert curve["position_count"].loc[first_traded] == 3
    # The deficit survives the session that failed to close it: a *later*
    # rebalance is funded well above the rotation cap, which is only possible
    # because the gap is still open.
    assert traded["turnover"].iloc[1] >= 0.20
    assert invested.iloc[-1] >= 0.85
    assert curve["position_count"].iloc[-1] == 10


def test_names_larger_than_the_rotation_budget_are_left_alone() -> None:
    """A position that cannot be exited in full must not be sliced.

    Six names of 15% of equity against a 20% two-sided rotation cap leaves the
    sell leg only 10%, so no name can be liquidated in a single session.  Slicing
    one would leave a stub that keeps its slot occupied while the name is no
    longer wanted, no replacement could ever be admitted, and the book would
    bleed into cash session after session.  Declining the rotation is the correct
    answer, and it has to be visible in the report.
    """

    daily, scores = _ranking_fixture()
    result = FactorBacktester().run(
        daily,
        _full_book_config(max_positions=6, single_weight_cap=0.20),
        feature_frame=scores,
        score_column="score",
    )

    invested = _invested_ratio(result)
    traded = result.equity_curve[result.equity_curve["turnover"] > 0]

    assert invested.iloc[-1] >= 0.85
    # Nothing traded after the book was built: every later rebalance was refused.
    assert len(traded) == 1
    assert result.stats["rotation_capped_days"] >= 10
    assert result.equity_curve["position_count"].max() <= 6


def test_industry_cap_is_disabled_when_sector_data_is_a_placeholder() -> None:
    """A missing industry field must not silently truncate the book.

    The research archive has no industry column, so every stock resolves to the
    ``"UNKNOWN"`` placeholder.  Enforcing a 25% cap on that single bucket caps
    the portfolio at ``0.25 / base_weight = 8`` names and leaves it 76% in cash
    -- which is what made every earlier excess-return figure meaningless.
    """

    daily, scores = _ranking_fixture(n_symbols=12, n_dates=40)
    config = _ranking_config(
        max_positions=6, single_weight_cap=0.15, industry_weight_cap=0.30
    )

    no_sector_column = FactorBacktester().run(
        daily.drop(columns=["sector"]),
        config,
        feature_frame=scores,
        score_column="score",
    )

    assert no_sector_column.stats["industry_cap_applied"] is False
    assert no_sector_column.stats["sector_source"] == "UNAVAILABLE:sector"
    assert no_sector_column.stats["sector_coverage"] == 0.0
    # Without the cap the plan reaches its configured position count instead of
    # stopping at floor(industry_weight_cap / base_weight) = 2.
    assert no_sector_column.equity_curve["position_count"].max() == 6


def test_industry_cap_still_limits_a_single_real_sector() -> None:
    daily, scores = _ranking_fixture(n_symbols=12, n_dates=40)
    config = _ranking_config(
        max_positions=6, single_weight_cap=0.15, industry_weight_cap=0.30
    )

    single_sector = FactorBacktester().run(
        daily.assign(sector="银行"),
        config,
        feature_frame=scores,
        score_column="score",
    )

    assert single_sector.stats["industry_cap_applied"] is True
    assert single_sector.stats["sector_source"] == "AVAILABLE:sector"
    assert single_sector.stats["distinct_known_sectors"] == 1
    # 0.30 / 0.15 = 2 names, so a genuine single-industry universe is capped.
    assert single_sector.equity_curve["position_count"].max() == 2


def test_industry_cap_publication_is_not_suppressed_by_partial_coverage() -> None:
    daily, scores = _ranking_fixture(n_symbols=12, n_dates=40)
    config = _ranking_config(
        max_positions=6, single_weight_cap=0.15, industry_weight_cap=0.30
    )
    # Half the universe has a real sector and half is a placeholder: the
    # placeholder bucket would still swallow the book, so the cap stays off.
    mixed = daily.assign(sector=["银行" if i % 2 else "UNKNOWN" for i in range(len(daily))])

    result = FactorBacktester().run(mixed, config, feature_frame=scores, score_column="score")

    assert result.stats["industry_cap_applied"] is False
    assert result.stats["sector_coverage"] == 0.5
    assert result.equity_curve["position_count"].max() == 6


STATUS_DISCLOSURE_KEYS = {
    "st_filter_applied",
    "suspension_filter_applied",
    "security_status_source",
    "security_status_basis",
    "special_treatment_excluded",
    "suspension_excluded",
    "universe_note",
    # Capability, not filters -- but the same hazard: a config can declare a
    # capability the code can never exercise, and only the report can say so.
    "marginable_securities",
    "shortable_securities",
    "capability_note",
}


def test_filter_disclosure_publishes_the_shared_key_set() -> None:
    """The research and live paths must describe the same decision identically.

    This path used to hand-roll a three-key dict while the live provider
    published seven, so the same "did you filter ST?" question had two different
    answers depending on which report you opened.  Both now go through the
    shared builder.
    """

    frame = pd.DataFrame({"symbol": ["600001", "300001"], "close": [10.0, 20.0]})
    disclosure = FactorBacktester._filter_disclosure(frame)

    assert set(disclosure) == STATUS_DISCLOSURE_KEYS
    # The archive carries neither column, so the honest answer is "no filter".
    assert disclosure["st_filter_applied"] is False
    assert disclosure["suspension_filter_applied"] is False
    assert disclosure["security_status_source"] == "UNAVAILABLE:is_st,suspended"
    assert "幸存者偏差" in disclosure["security_status_basis"]
    assert disclosure["special_treatment_excluded"] == 0
    # The research path is long-only and models neither capability, so the
    # counts must be zero rather than absent.
    assert disclosure["marginable_securities"] == 0
    assert disclosure["shortable_securities"] == 0
    assert disclosure["capability_note"]


def test_filter_disclosure_reports_availability_when_the_columns_exist() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600001", "300001"],
            "close": [10.0, 20.0],
            "is_st": [False, False],
            "suspended": [False, False],
        }
    )
    disclosure = FactorBacktester._filter_disclosure(frame)

    assert set(disclosure) == STATUS_DISCLOSURE_KEYS
    assert disclosure["st_filter_applied"] is True
    assert disclosure["suspension_filter_applied"] is True
    assert disclosure["security_status_source"] == "input_columns"


def test_run_publishes_the_filter_disclosure_among_its_stats() -> None:
    daily, scores = _ranking_fixture(n_symbols=12, n_dates=40)
    result = FactorBacktester().run(
        daily, _ranking_config(max_positions=6), feature_frame=scores, score_column="score"
    )

    assert STATUS_DISCLOSURE_KEYS <= set(result.stats)
    assert result.stats["st_filter_applied"] is False


def test_cost_drag_is_disclosed_and_matches_the_booked_frictions() -> None:
    """Frictions are the same order as the alpha, so they must be reported.

    Commission and stamp duty are booked per trade; slippage has no field and is
    reconstructed from traded notional.  A report that only carried the net
    return could not tell a weak signal from an expensive broker, and the two
    call for opposite fixes.
    """

    daily, scores = _ranking_fixture()
    config = _full_book_config()
    result = FactorBacktester().run(
        daily, config, feature_frame=scores, score_column="score"
    )

    stats = result.stats
    booked = sum(t.commission + t.stamp_duty for t in result.trades)
    notional = sum(t.notional for t in result.trades)
    average_equity = float(result.equity_curve["equity_at_open"].mean())

    assert stats["fees_paid"] == pytest.approx(booked, rel=1e-6)
    assert stats["slippage_paid"] == pytest.approx(
        notional * config.slippage_bps / 10_000, rel=1e-6
    )
    assert stats["traded_notional_ratio"] == pytest.approx(
        notional / average_equity, rel=1e-6
    )
    # The drag is the sum of two separately rounded components divided by a
    # floating mean, so exact equality is not achievable; the tolerance only has
    # to absorb the rounding of the published fields.
    assert stats["cost_drag_ratio"] == pytest.approx(
        (stats["fees_paid"] + stats["slippage_paid"]) / average_equity, abs=1e-7
    )
    # A book that actually rotates pays something; a silent zero would mean the
    # reconstruction quietly failed.
    assert stats["cost_drag_ratio"] > 0


def test_cost_drag_is_zero_when_nothing_ever_trades() -> None:
    daily, scores = _ranking_fixture()
    # An empty score frame cannot rank anything, so the book is never built and
    # no friction is incurred -- the disclosure must not invent a cost.
    result = FactorBacktester().run(
        daily, _full_book_config(), feature_frame=scores.iloc[0:0], score_column="score"
    )

    assert result.trades == []
    assert result.stats["fees_paid"] == 0.0
    assert result.stats["slippage_paid"] == 0.0
    assert result.stats["cost_drag_ratio"] == 0.0
