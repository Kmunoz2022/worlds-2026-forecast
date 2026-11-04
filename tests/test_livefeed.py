from datetime import UTC, datetime

from lolpred.livefeed import ts


def test_feed_timestamps_round_down_to_10_seconds():
    assert ts(datetime(2026, 10, 15, 18, 41, 37, 512000, tzinfo=UTC)) == "2026-10-15T18:41:30Z"
