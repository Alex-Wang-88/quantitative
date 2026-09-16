from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from quantpaper.config import AppConfig, DataConfig
from quantpaper.data import build_provider
from quantpaper.domain import Board, DataMode
from quantpaper.eastmoney import EastmoneyProvider
from quantpaper.terminal_feed import TerminalFeedStore


class FakeEastmoneyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    def request(self, action: str, **payload: object):
        self.calls.append((action, payload))
        if action == "instruments":
            return [
                {
                    "symbol": "SHSE.600000",
                    "sec_name": "浦发银行",
                    "exchange": "SHSE",
                    "pre_close": 9.4,
                    "upper_limit": 10.34,
                    "lower_limit": 8.46,
                    "is_suspended": 0,
                    "listed_date": "1999-11-10T00:00:00+08:00",
                },
                {
                    "symbol": "SZSE.300750",
                    "sec_name": "宁德时代",
                    "exchange": "SZSE",
                    "pre_close": 337.11,
                    "upper_limit": 404.53,
                    "lower_limit": 269.69,
                    "is_suspended": 0,
                    "listed_date": "2018-06-11T00:00:00+08:00",
                },
                {
                    "symbol": "SHSE.688981",
                    "sec_name": "中芯国际",
                    "exchange": "SHSE",
                    "pre_close": 114.07,
                    "upper_limit": 136.88,
                    "lower_limit": 91.26,
                    "is_suspended": 0,
                    "listed_date": "2020-07-16T00:00:00+08:00",
                },
            ]
        if action == "quotes":
            return [
                {
                    "symbol": "SHSE.600000",
                    "price": 9.41,
                    "created_at": self.timestamp,
                    "cum_volume": 10000,
                    "bid1": 9.40,
                    "ask1": 9.41,
                    "bid1_volume": 1200,
                    "ask1_volume": 1500,
                    "is_suspended": 0,
                },
                {
                    "symbol": "SZSE.300750",
                    "price": 338.0,
                    "created_at": self.timestamp,
                    "cum_volume": 20000,
                    "bid1": 337.9,
                    "ask1": 338.0,
                    "bid1_volume": 800,
                    "ask1_volume": 1000,
                    "is_suspended": 0,
                },
                {
                    "symbol": "SHSE.000001",
                    "price": 3201.25,
                    "pre_close": 3190.0,
                    "created_at": self.timestamp,
                },
                {
                    "symbol": "SZSE.399001",
                    "price": 10120.5,
                    "pre_close": 10080.0,
                    "created_at": self.timestamp,
                },
                {
                    "symbol": "SHSE.000300",
                    "price": 3820.0,
                    "pre_close": 3800.0,
                    "created_at": self.timestamp,
                },
            ]
        if action == "history":
            return [
                {
                    "symbol": "SHSE.600000",
                    "open": 9.13,
                    "high": 9.36,
                    "low": 9.10,
                    "close": 9.35,
                    "volume": 102669612,
                    "amount": 952382709.65,
                    "pre_close": 9.16,
                    "eob": "2026-09-01T00:00:00+08:00",
                },
                {
                    "symbol": "SZSE.300750",
                    "open": 330.0,
                    "high": 340.0,
                    "low": 329.0,
                    "close": 338.0,
                    "volume": 202669612,
                    "amount": 1952382709.65,
                    "pre_close": 331.0,
                    "eob": "2026-09-01T00:00:00+08:00",
                },
            ]
        if action == "fundamentals_history":
            return {
                "daily": [
                    {
                        "symbol": "SHSE.600000",
                        "trade_date": "2026-09-01",
                        "pe_ttm": 6.2,
                        "pb_lyr": 0.8,
                        "tot_mv": 300_000_000_000,
                        "a_mv_ex_ltd": 299_000_000_000,
                        "turnrate": 0.12,
                    }
                ],
                "reports": [],
                "industries": [],
                "errors": [],
            }
        if action == "calendar":
            return ["2026-09-01", "2026-09-02"]
        raise AssertionError(action)


def test_eastmoney_provider_normalizes_live_data() -> None:
    client = FakeEastmoneyClient()
    provider = EastmoneyProvider(
        symbols=["600000", "300750", "688981"],
        universe_limit=10,
        client=client,
    )

    securities = provider.get_security_master()
    assert [security.symbol for security in securities] == ["600000", "300750", "688981"]
    assert securities[1].board == Board.CHINEXT_300
    assert securities[2].board == Board.STAR_688
    assert provider._board("301108") == Board.CHINEXT_300
    assert securities[0].exchange == "SSE"

    quotes = provider.get_quotes(["600000", "300750"])
    assert quotes[0].source == "EASTMONEY_GM_READONLY"
    assert quotes[0].ask1 == 9.41
    assert quotes[0].upper_limit == 10.34
    assert provider.status().mode == DataMode.LIVE

    benchmarks = provider.get_benchmark_quotes()
    assert {item.symbol for item in benchmarks} == {"000001", "399001", "000300"}
    assert next(item for item in benchmarks if item.symbol == "000300").previous_close == 3800.0

    daily = provider.get_daily_frame()
    assert set(daily["symbol"]) == {"600000", "300750"}
    assert daily.loc[daily["symbol"] == "600000", "close"].iloc[0] == 9.35
    assert daily.loc[daily["symbol"] == "600000", "market_cap"].iloc[0] == 300_000_000_000
    assert daily.loc[daily["symbol"] == "600000", "pe"].iloc[0] == 6.2
    assert len(provider.get_trading_calendar()) == 2

    margin = provider.get_margin_snapshot(["600000"])
    assert margin["600000"]["marginable"] is False
    assert any(action == "history" for action, _ in client.calls)


def test_live_config_builds_eastmoney_provider() -> None:
    config = AppConfig(
        data=DataConfig(
            mode="live",
            realtime_provider="eastmoney",
            research_universe_limit=0,
            realtime_universe_limit=500,
        )
    )
    provider = build_provider(config)
    assert isinstance(provider, EastmoneyProvider)
    assert provider.research_universe_limit == 0
    assert provider.realtime_universe_limit == 500


def test_research_universe_and_realtime_quotes_are_separate() -> None:
    client = FakeEastmoneyClient()
    provider = EastmoneyProvider(
        symbols=["600000", "300750", "688981"],
        research_universe_limit=0,
        realtime_universe_limit=1,
        fundamentals_enabled=False,
        client=client,
    )

    assert len(provider.get_security_master()) == 3
    instrument_call = next(payload for action, payload in client.calls if action == "instruments")
    assert instrument_call["limit"] == 0

    # The default quote poll is capped to the realtime pool, while an
    # explicit candidate/position list can still request another research name.
    assert len(provider.get_quotes()) == 1
    provider.set_realtime_symbols(["600000"])
    assert [quote.symbol for quote in provider.get_quotes()] == ["600000"]


def test_terminal_adapter_feed_is_preferred_for_fresh_quotes(tmp_path) -> None:
    client = FakeEastmoneyClient()
    feed_path = tmp_path / "terminal-feed.json"
    provider = EastmoneyProvider(
        symbols=["600000", "300750", "688981"],
        fundamentals_enabled=False,
        terminal_adapter_enabled=True,
        terminal_feed_path=feed_path,
        client=client,
    )
    provider.get_security_master()
    TerminalFeedStore(feed_path).write_rows(
        [
            {
                "symbol": "SHSE.600000",
                "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "last": 9.52,
                "bar_volume": 1234,
                "source": "TEST_TERMINAL",
            }
        ]
    )

    quote = provider.get_quotes(["600000"])[0]
    assert quote.last == 9.52
    assert quote.bar_volume == 1234
    assert quote.source == "TEST_TERMINAL"
    assert not any(action == "quotes" for action, _ in client.calls)


def test_terminal_adapter_feed_is_preferred_for_benchmarks(tmp_path) -> None:
    client = FakeEastmoneyClient()
    feed_path = tmp_path / "terminal-feed.json"
    provider = EastmoneyProvider(
        symbols=["600000", "300750", "688981"],
        fundamentals_enabled=False,
        terminal_adapter_enabled=True,
        terminal_feed_path=feed_path,
        client=client,
    )
    TerminalFeedStore(feed_path).write_rows(
        [
            {
                "symbol": "SHSE.000300",
                "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "last": 3842.5,
                "previous_close": 3800.0,
                "source": "TEST_TERMINAL",
            }
        ]
    )

    benchmarks = provider.get_benchmark_quotes()
    quote = next(item for item in benchmarks if item.symbol == "000300")
    assert quote.last == 3842.5
    assert quote.source == "TEST_TERMINAL"
    benchmark_calls = [payload for action, payload in client.calls if action == "quotes"]
    assert benchmark_calls
    assert all("SHSE.000300" not in payload["symbols"] for payload in benchmark_calls)
