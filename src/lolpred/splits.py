"""Time-based splits. Nothing here ever shuffles: train is always strictly
earlier than test, so no future information leaks into a prediction.

Sequential models (Elo) run over all games in date order and only log their
pre-game prediction, so they don't need these splits for training; use them
to decide *which* predictions get scored.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    name: str
    train_end: pd.Timestamp  # train on date < train_end
    test_start: pd.Timestamp
    test_end: pd.Timestamp   # test on test_start <= date < test_end
    test_mask_fn: object = None  # optional extra filter on the test window

    def train(self, df):
        return df[df["date"] < self.train_end]

    def test(self, df):
        t = df[(df["date"] >= self.test_start) & (df["date"] < self.test_end)]
        return t[self.test_mask_fn(t)] if self.test_mask_fn else t


def worlds_start(df: pd.DataFrame, season: int) -> pd.Timestamp | None:
    """First cross-league WLDs game of a season (skips regional qualifiers
    that Oracle's Elixir files under WLDs, e.g. LPL vs LPL in 2026)."""
    w = df[(df["league"] == "WLDs") & (df["season"] == season) & df["cross_league"]]
    return w["date"].min() if len(w) else None


def worlds_backtests(df: pd.DataFrame, seasons=(2022, 2023, 2024, 2025)) -> list[Split]:
    """Train on everything before Worlds starts, test on every WLDs game from
    then on (same-region Worlds matchups included; qualifiers come earlier)."""
    out = []
    for y in seasons:
        start = worlds_start(df, y)
        if start is None:
            continue
        end = pd.Timestamp(f"{y + 1}-01-01")
        out.append(Split(f"worlds_{y}", start, start, end, lambda t: t["league"] == "WLDs"))
    return out


def walk_forward(df: pd.DataFrame, start="2023-01-01", end=None, freq="QS") -> list[Split]:
    """Expanding-window folds: train on all games before each period, test on
    the period. Default: quarterly folds from 2023 (2022 is warm-up only)."""
    end = pd.Timestamp(end) if end else df["date"].max() + pd.Timedelta(seconds=1)
    edges = list(pd.date_range(start, end, freq=freq)) + [end]
    return [Split(f"wf_{a:%Y-%m}", a, a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def holdout_2026_worlds(df: pd.DataFrame) -> Split | None:
    """The frozen target. Don't tune anything on this."""
    s = worlds_backtests(df, seasons=(2026,))
    return s[0] if s else None
