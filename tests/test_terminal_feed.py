from datetime import datetime

from quantpaper.terminal_feed import TerminalFeedStore


def test_terminal_feed_keeps_fresh_latest_rows(tmp_path) -> None:
    store = TerminalFeedStore(tmp_path / "feed.json")
    accepted = store.write_rows(
        [
            {
                "symbol": "600000",
                "timestamp": datetime.now().astimezone().isoformat(),
                "last": 9.18,
                "bar_volume": 1200,
                "source": "TEST_ADAPTER",
            },
            {"symbol": "bad", "timestamp": "not-a-date", "last": 1.0},
        ]
    )

    assert accepted == 1
    rows = store.read_rows(["SHSE.600000"], stale_after_seconds=60)
    assert len(rows) == 1
    assert rows[0]["symbol"] == "SHSE.600000"
    assert rows[0]["last"] == 9.18
    assert store.status(stale_after_seconds=60)["active_symbols"] == 1
