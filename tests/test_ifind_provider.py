from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from quantpaper.config import AppConfig, DataConfig
from quantpaper.data import IfindProvider, build_provider
from quantpaper.domain import Board, DataMode


class FakeIfindApi:
    def __init__(self) -> None:
        self.quote_calls: list[tuple[str, str, str]] = []
        self.history_calls: list[tuple[str, str, str, str, str]] = []

    def THS_RQ(self, codes: str, indicators: str, options: str):
        self.quote_calls.append((codes, indicators, options))
        timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=0).isoformat(
            sep=" "
        )
        return SimpleNamespace(
            errorcode=0,
            data=pd.DataFrame(
                [
                    {
                        "thscode": "600000.SH",
                        "time": timestamp,
                        "latest": 10.5,
                        "preClose": 10.0,
                        "bid1": 10.49,
                        "ask1": 10.51,
                        "bidSize1": 1000,
                        "askSize1": 1200,
                        "latestVolume": 10000,
                    },
                    {
                        "thscode": "300001.SZ",
                        "time": timestamp,
                        "latest": 20.5,
                        "preClose": 20.0,
                        "bid1": 20.49,
                        "ask1": 20.51,
                        "bidSize1": 2000,
                        "askSize1": 2200,
                        "latestVolume": 20000,
                    },
                ]
            ),
        )

    def THS_HistoryQuotes(
        self, codes: str, indicators: str, options: str, start: str, end: str
    ):
        self.history_calls.append((codes, indicators, options, start, end))
        return SimpleNamespace(
            errorcode=0,
            data=pd.DataFrame(
                [
                    {
                        "thscode": "600000.SH",
                        "time": "2026-09-11",
                        "open": 10.0,
                        "high": 10.6,
                        "low": 9.9,
                        "close": 10.5,
                        "volume": 100000,
                        "amount": 1000000,
                    },
                    {
                        "thscode": "300001.SZ",
                        "time": "2026-09-11",
                        "open": 20.0,
                        "high": 20.6,
                        "low": 19.9,
                        "close": 20.5,
                        "volume": 200000,
                        "amount": 4000000,
                    },
                ]
            ),
        )


def test_ifind_provider_normalizes_quotes_and_history() -> None:
    api = FakeIfindApi()
    provider = IfindProvider(symbols=["600000.SH", "300001"], api=api)

    quotes = provider.get_quotes()
    assert [quote.symbol for quote in quotes] == ["600000", "300001"]
    assert quotes[0].source == "IFIND"
    assert quotes[0].upper_limit == 11.0
    assert quotes[1].upper_limit == 24.0
    assert provider.status().mode == DataMode.LIVE
    assert api.quote_calls[0][0] == "600000.SH,300001.SZ"

    provider.set_realtime_symbols(["300001"])
    assert [quote.symbol for quote in provider.next_quotes()] == ["300001"]

    daily = provider.get_daily_frame()
    assert set(daily["symbol"]) == {"600000", "300001"}
    assert daily["close"].tolist() == [10.5, 20.5]
    assert len(api.history_calls) == 1


def test_ifind_provider_classifies_300_and_688() -> None:
    provider = IfindProvider(symbols=["300750", "688981"], api=FakeIfindApi())
    securities = provider.get_security_master()

    assert securities[0].board == Board.CHINEXT_300
    assert securities[1].board == Board.STAR_688
    assert IfindProvider._security("301108").board == Board.CHINEXT_300


def test_live_config_builds_ifind_provider() -> None:
    config = AppConfig(data=DataConfig(mode="live", realtime_provider="ifind"))
    provider = build_provider(config)
    assert isinstance(provider, IfindProvider)
