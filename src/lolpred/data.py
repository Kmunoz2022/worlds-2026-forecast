"""Load Oracle's Elixir CSVs and build clean game / match / series tables.

Tables produced:
  team_games : one row per team per game (team rows only, position == 'team'),
               with opponent identity and series context joined on.
  matches    : one row per game (blue vs red), handy for sequential rating models.
  series     : one row per reconstructed series (Bo1/Bo3/Bo5), with final score.

Only identity/context columns are "pre-game". Everything else on team_games
(kills, golddiffat15, dragons, ...) is post-game outcome data: use it only
through lagged/rolling features, never as a same-game input.
"""
import numpy as np
import pandas as pd

from .config import EVENT_LEAGUES, MAJOR_INTL, PLAYER_ONLY_COLS, RAW_DIR, RAW_GLOB

SERIES_GAP = pd.Timedelta(hours=12)


def load_team_rows(raw_dir=RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(f"no {RAW_GLOB} in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if df.empty:
            raise ValueError(f"{f.name} is empty")
        frames.append(df[df["position"] == "team"])
    df = pd.concat(frames, ignore_index=True)
    return df.drop(columns=[c for c in PLAYER_ONLY_COLS if c in df.columns])


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (clean team rows, log of what was dropped/fixed)."""
    log = {}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year  # calendar year; raw `year` is a season label

    n = len(df)
    df = df.drop_duplicates(["gameid", "side"])
    log["duplicate_rows_dropped"] = n - len(df)

    # Every game must have exactly one Blue and one Red row, and one winner.
    g = df.groupby("gameid")
    ok = (g["side"].transform("nunique") == 2) & (g["side"].transform("size") == 2) \
        & (g["result"].transform("sum") == 1)
    log["bad_games_dropped"] = sorted(df.loc[~ok, "gameid"].unique().tolist())
    df = df[ok]

    # Missing teamid: reuse the team's id from other rows if the name maps to
    # exactly one id, else fall back to a name-based id.
    name_to_id = df.dropna(subset=["teamid"]).groupby("teamname")["teamid"].agg(
        lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    missing = df["teamid"].isna()
    df.loc[missing, "teamid"] = df.loc[missing, "teamname"].map(name_to_id)
    log["teamid_recovered_from_name"] = int(missing.sum() - df["teamid"].isna().sum())
    still = df["teamid"].isna()
    df.loc[still, "teamid"] = "name:" + df.loc[still, "teamname"]
    log["teamid_name_fallback"] = int(still.sum())

    # Drop columns that are empty on every team row (player-only stats).
    empty = [c for c in df.columns if df[c].isna().all()]
    log["empty_columns_dropped"] = empty
    df = df.drop(columns=empty)

    df = df.sort_values(["date", "gameid", "side"], kind="stable").reset_index(drop=True)
    return df, log


def add_home_league(df: pd.DataFrame) -> pd.DataFrame:
    """Home league = team's most-played non-event league in that calendar year.

    Years with no domestic games borrow the nearest year's home league;
    teams seen only at events get 'EVENT_ONLY:<league>'.
    """
    dom = df[~df["league"].isin(EVENT_LEAGUES)]
    home = dom.groupby(["teamid", "season"])["league"].agg(lambda s: s.value_counts().idxmax())
    grid = df[["teamid", "season"]].drop_duplicates().set_index(["teamid", "season"]).sort_index()
    grid["home_league"] = home
    grid["home_league"] = grid.groupby(level="teamid")["home_league"].transform(
        lambda s: s.ffill().bfill())
    df = df.join(grid, on=["teamid", "season"])
    fallback = df.groupby("teamid")["league"].transform(lambda s: s.value_counts().idxmax())
    df["home_league"] = df["home_league"].fillna("EVENT_ONLY:" + fallback)
    return df


def build_matches(tg: pd.DataFrame) -> pd.DataFrame:
    """One row per game: blue vs red with result from blue's perspective."""
    ctx = ["gameid", "date", "season", "year", "league", "split", "playoffs", "game",
           "patch", "datacompleteness", "gamelength"]
    side_cols = ["teamid", "teamname", "home_league"]
    blue = tg[tg["side"] == "Blue"].set_index("gameid")
    red = tg[tg["side"] == "Red"].set_index("gameid")
    m = blue[[c for c in ctx if c != "gameid"]].copy()
    for c in side_cols:
        m[f"blue_{c}"] = blue[c]
        m[f"red_{c}"] = red[c]
    m["blue_win"] = blue["result"].astype(int)
    m["cross_league"] = m["blue_home_league"] != m["red_home_league"]
    m["major_intl"] = m["league"].isin(MAJOR_INTL)
    return m.reset_index().sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)


def add_series(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct series (no series id in the raw data).

    Consecutive games in the same league between the same two teams form one
    series; a new series starts when `game` resets to 1, doesn't follow on
    (prev + 1), or more than SERIES_GAP passes between games.
    """
    m = m.copy()
    a = m[["blue_teamid", "red_teamid"]].min(axis=1)
    b = m[["blue_teamid", "red_teamid"]].max(axis=1)
    m["team_a"], m["team_b"] = a, b
    m["a_win"] = np.where(m["blue_teamid"] == a, m["blue_win"], 1 - m["blue_win"])

    m = m.sort_values(["league", "team_a", "team_b", "date", "game"], kind="stable")
    same_pair = (m["league"].eq(m["league"].shift()) & m["team_a"].eq(m["team_a"].shift())
                 & m["team_b"].eq(m["team_b"].shift()))
    follows = m["game"].eq(m["game"].shift() + 1) & (m["date"] - m["date"].shift() <= SERIES_GAP)
    new = ~(same_pair & follows)
    m["series_id"] = new.cumsum()

    grp = m.groupby("series_id")
    m["series_game_no"] = grp.cumcount() + 1
    # Series score *before* this game, from team_a's perspective (pre-game safe).
    m["a_series_wins_before"] = grp["a_win"].cumsum() - m["a_win"]
    m["b_series_wins_before"] = (m["series_game_no"] - 1) - m["a_series_wins_before"]

    s = grp.agg(
        league=("league", "first"), start=("date", "min"), end=("date", "max"),
        season=("season", "first"), playoffs=("playoffs", "first"),
        team_a=("team_a", "first"), team_b=("team_b", "first"),
        n_games=("gameid", "size"), a_wins=("a_win", "sum"),
        first_game_no=("game", "min"),
        cross_league=("cross_league", "first"), major_intl=("major_intl", "first"),
    )
    s["b_wins"] = s["n_games"] - s["a_wins"]
    hi = s[["a_wins", "b_wins"]].max(axis=1)
    lo = s[["a_wins", "b_wins"]].min(axis=1)
    # Winner's win count implies format: 1 -> Bo1, 2 -> Bo3, 3 -> Bo5.
    # Tied (Bo2 draws) or series not starting at game 1 are ambiguous.
    s["best_of"] = np.where((hi > lo) & (s["first_game_no"] == 1), 2 * hi - 1, np.nan)
    s["series_winner"] = np.where(s["a_wins"] > s["b_wins"], s["team_a"],
                                  np.where(s["b_wins"] > s["a_wins"], s["team_b"], None))
    s["score"] = hi.astype(str) + "-" + lo.astype(str)

    m = m.join(s["best_of"], on="series_id")
    m = m.sort_values(["date", "gameid"], kind="stable").reset_index(drop=True)
    return m, s.reset_index()


def attach_opponent_and_series(tg: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Add opponent identity and series context to the per-team game table."""
    opp = tg[["gameid", "teamid", "teamname", "home_league"]].rename(columns={
        "teamid": "opp_teamid", "teamname": "opp_teamname", "home_league": "opp_home_league"})
    tg = tg.merge(opp, on="gameid")
    tg = tg[tg["teamid"] != tg["opp_teamid"]]

    sc = m.set_index("gameid")[["series_id", "series_game_no", "best_of", "team_a",
                                "a_series_wins_before", "b_series_wins_before",
                                "cross_league", "major_intl"]]
    tg = tg.join(sc, on="gameid")
    is_a = tg["teamid"] == tg["team_a"]
    tg["series_wins_before"] = np.where(is_a, tg["a_series_wins_before"], tg["b_series_wins_before"])
    tg["opp_series_wins_before"] = np.where(is_a, tg["b_series_wins_before"], tg["a_series_wins_before"])
    tg = tg.drop(columns=["team_a", "a_series_wins_before", "b_series_wins_before"])
    return tg.sort_values(["date", "gameid", "side"], kind="stable").reset_index(drop=True)


def validate(tg: pd.DataFrame, m: pd.DataFrame, s: pd.DataFrame) -> None:
    assert (tg.groupby("gameid").size() == 2).all(), "each game needs exactly 2 team rows"
    assert (tg.groupby("gameid")["result"].sum() == 1).all(), "each game needs one winner"
    assert len(m) == tg["gameid"].nunique()
    assert m["gameid"].is_unique
    assert s["n_games"].sum() == len(m)
    assert (s["n_games"] <= 5).all(), "series longer than Bo5"
    assert tg["teamid"].notna().all() and tg["home_league"].notna().all()
    assert (tg["series_wins_before"] + tg["opp_series_wins_before"] == tg["series_game_no"] - 1).all()


def build_all(raw_dir=RAW_DIR):
    raw = load_team_rows(raw_dir)
    tg, log = clean(raw)
    tg = add_home_league(tg)
    m = build_matches(tg)
    m, s = add_series(m)
    tg = attach_opponent_and_series(tg, m)
    validate(tg, m, s)
    return tg, m, s, log
