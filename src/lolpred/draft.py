"""Champion draft data and meta forecasts (Crystal Ball questions).

Presence = share of games where a champion is picked or banned by either team.

Forecast for an upcoming event:
  1. Sample: recent tier-1 games (major leagues + international events),
     weighted by patch recency (newest patch weight 1, each older patch x
     PATCH_DECAY).
  2. Each champion's rate gets a Beta distribution centred on the weighted
     rate, with strength capped at META_STRENGTH: the event's patch and meta
     will differ from the sample, so even a huge sample isn't certain.
  3. Monte Carlo: per simulated event, draw every champion's rate, then its
     pick/ban counts over the event's number of games; record the leaders.
"""
import numpy as np
import pandas as pd

from .config import LEAGUE_ALIAS, MAJOR_LEAGUES, PROCESSED_DIR, RAW_DIR, RAW_GLOB

TIER1_EVENTS = {"WLDs", "MSI", "EWC", "FST", "DCGI"}
# Chosen on a backtest of Worlds 2022-2025 (see docs/DECISIONS.md): predict each
# Worlds' presence / pick / ban leaders from the 8 weeks before it.
PATCH_DECAY = 1.0          # older tier-1 patches count fully...
NEWEST_PATCH_WEIGHT = 10.0  # ...but each newest-patch game counts 10x
META_STRENGTH = 10.0       # wide uncertainty: the Worlds patch reshuffles the meta
LOOKBACK_DAYS = 56
DRAFT_CACHE = PROCESSED_DIR / "draft_events.parquet"


def load_draft_events(rebuild=False) -> pd.DataFrame:
    """One row per (game, champion) that was picked or banned:
    gameid, date, league, patch, champion, picked, banned, won (if picked)."""
    if DRAFT_CACHE.exists() and not rebuild:
        return pd.read_parquet(DRAFT_CACHE)
    cols = ["gameid", "date", "league", "patch", "position", "side", "champion", "result",
            "ban1", "ban2", "ban3", "ban4", "ban5"]
    frames = []
    for f in sorted(RAW_DIR.glob(RAW_GLOB)):
        # Patch as text: read as a number, "15.20" would become 15.2.
        r = pd.read_csv(f, usecols=cols, dtype={"patch": str}, low_memory=False)
        r["date"] = pd.to_datetime(r["date"])
        picks = r[r["position"] != "team"].dropna(subset=["champion"])
        picks = picks.assign(picked=1, banned=0, won=picks["result"])[
            ["gameid", "date", "league", "patch", "champion", "picked", "banned", "won"]]
        teams = r[r["position"] == "team"].drop(columns="champion")
        bans = teams.melt(id_vars=["gameid", "date", "league", "patch"],
                          value_vars=[f"ban{i}" for i in range(1, 6)], value_name="champion")
        bans = bans.dropna(subset=["champion"]).assign(picked=0, banned=1, won=np.nan)[
            ["gameid", "date", "league", "patch", "champion", "picked", "banned", "won"]]
        frames += [picks, bans]
    ev = pd.concat(frames, ignore_index=True)
    # A champion is present once per game even if (rarely) both picked and banned.
    ev = (ev.groupby(["gameid", "champion"], as_index=False)
            .agg(date=("date", "first"), league=("league", "first"), patch=("patch", "first"),
                 picked=("picked", "max"), banned=("banned", "max"), won=("won", "max")))
    ev["patch"] = ev["patch"].astype(str)
    ev.to_parquet(DRAFT_CACHE, index=False)
    return ev


def _patch_key(p: str):
    try:
        major, minor = str(p).split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return (0, 0)


def is_tier1(league: pd.Series) -> pd.Series:
    lg = league.map(lambda x: LEAGUE_ALIAS.get(x, x))
    return lg.isin(MAJOR_LEAGUES) | lg.isin(TIER1_EVENTS) | league.isin({"LTA", "LTA N", "LTA S"})


def champion_rates(ev: pd.DataFrame, end, lookback_days=LOOKBACK_DAYS, patch_decay=PATCH_DECAY,
                   newest_weight=NEWEST_PATCH_WEIGHT):
    """Weighted pick / ban / presence rates from games in [end - lookback_days, end).

    Tier-1 games are weighted by patch recency. Games on the newest patch (the
    one an upcoming event is usually played on) count from *any* league,
    each worth `newest_weight` tier-1 games: before past Worlds only a few
    minor-league games existed on the Worlds patch, but they carried the
    meta shift. Returns (rates DataFrame, sample info)."""
    end = pd.Timestamp(end)
    recent = ev[(ev["date"] < end) & (ev["date"] >= end - pd.Timedelta(days=lookback_days))]
    newest = max(recent["patch"].unique(), key=_patch_key)
    s = recent[is_tier1(recent["league"]) | (recent["patch"] == newest)]
    games = s.drop_duplicates("gameid")[["gameid", "patch", "league"]].copy()
    patches = sorted(games["patch"].unique(), key=_patch_key)
    age = {p: len(patches) - 1 - i for i, p in enumerate(patches)}
    games["w"] = games["patch"].map(lambda p: patch_decay ** age[p])
    games.loc[games["patch"] == newest, "w"] = newest_weight
    W = games["w"].sum()
    s = s.merge(games[["gameid", "w"]], on="gameid")
    g = s.groupby("champion")
    rates = pd.DataFrame({
        "presence": g["w"].sum() / W,
        "pick": g.apply(lambda x: (x["w"] * x["picked"]).sum(), include_groups=False) / W,
        "ban": g.apply(lambda x: (x["w"] * x["banned"]).sum(), include_groups=False) / W,
        "games_present": g.size(),
    })
    wins = s[s["picked"] == 1].groupby("champion")["won"].agg(["sum", "size"])
    rates["win_rate"] = (wins["sum"] / wins["size"]).reindex(rates.index)
    rates["picks_raw"] = wins["size"].reindex(rates.index).fillna(0).astype(int)
    info = {"games": int(len(games)), "weighted_games": float(W), "newest_patch": newest,
            "newest_patch_games": int((games["patch"] == newest).sum()),
            "patches": {p: int((games["patch"] == p).sum()) for p in patches},
            "from": f"{(end - pd.Timedelta(days=lookback_days)):%Y-%m-%d}", "to": f"{end:%Y-%m-%d}"}
    return rates.sort_values("presence", ascending=False), info


def event_rates(ev: pd.DataFrame, gameids) -> pd.DataFrame:
    """Actual presence / pick / ban rates over a set of games."""
    s = ev[ev["gameid"].isin(gameids)]
    n = s["gameid"].nunique()
    g = s.groupby("champion")
    return pd.DataFrame({"presence": g.size() / n, "pick": g["picked"].sum() / n,
                         "ban": g["banned"].sum() / n}).sort_values("presence", ascending=False)


def simulate_leaders(rates: pd.DataFrame, n_games_samples, n_sims=20000, strength=META_STRENGTH, seed=0):
    """Probability each champion finishes #1 in presence / picks / bans at the
    event. n_games_samples: array of simulated event lengths (games)."""
    rng = np.random.default_rng(seed)
    champs = rates.index.to_numpy()
    out = {}
    n_games = rng.choice(np.asarray(n_games_samples), size=n_sims)
    for col in ("presence", "pick", "ban"):
        p = rates[col].clip(1e-4, 1 - 1e-4).to_numpy()
        q = rng.beta(p * strength, (1 - p) * strength, size=(n_sims, len(p)))
        counts = rng.binomial(n_games[:, None], q)
        # Ties split evenly among the tied champions.
        top = counts == counts.max(axis=1, keepdims=True)
        share = top / top.sum(axis=1, keepdims=True)
        out[col] = pd.Series(share.mean(axis=0), index=champs)
        out[f"{col}_expected"] = pd.Series((counts / n_games[:, None]).mean(axis=0), index=champs)
    return out, n_games
