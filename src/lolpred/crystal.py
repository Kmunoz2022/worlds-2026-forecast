"""Crystal Ball questions about players, games and champions (see docs/DECISIONS.md).

Player and game questions run on top of the tournament simulation: every
simulated game knows its two teams and its winner. For each team we draw one
of its own recent games with the same result (a win for the winner, a loss for
the loser) and take all five players' stats from it, so kills, assists and
team kills stay consistent within a team. A roster override (e.g. TES top laner
369) draws that player's line from his own games instead.

Rare single-game events (15+ kills, 450+ CS) use base rates from international
events instead: domestic leagues have many mismatches that inflate them.
"""
import glob

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR, RAW_GLOB

PLAYER_CACHE = PROCESSED_DIR / "player_games.parquet"
POSITIONS = ["top", "jng", "mid", "bot", "sup"]
STATS = ["kills", "deaths", "assists", "teamkills", "damagetochampions", "total cs"]
POOL_FROM = "2026-04-01"   # roughly the second half of the 2026 season
MIN_POOL = 15              # fall back to all of 2026, then 2025, below this
STRONG_OPP = 1800.0        # Elo strength of a Worlds-level opponent
INTL_EVENTS = ["WLDs", "MSI", "EWC", "FST"]
# Closest to Worlds conditions: Worlds itself, then MSI at half weight.
# EWC/FST and domestic leagues have more mismatches (more stomps, short games).
WORLDS_LIKE = {"WLDs": ("2022-01-01", 1.0), "MSI": ("2024-01-01", 0.5)}


def load_player_games(rebuild=False) -> pd.DataFrame:
    if PLAYER_CACHE.exists() and not rebuild:
        return pd.read_parquet(PLAYER_CACHE)
    cols = ["gameid", "date", "league", "teamname", "teamid", "position", "playername", "champion",
            "result", "gamelength", "patch", *STATS]
    frames = [pd.read_csv(f, usecols=cols, dtype={"patch": str}, low_memory=False)
              for f in sorted(glob.glob(str(RAW_DIR / RAW_GLOB)))]
    p = pd.concat(frames, ignore_index=True)
    p = p[p["position"] != "team"].copy()
    p["date"] = pd.to_datetime(p["date"])
    p.to_parquet(PLAYER_CACHE, index=False)
    return p


def current_rosters(cfg, pg, last_n=10):
    """{team: {position: player}} from each team's last `last_n` games, then
    config `roster_overrides` applied."""
    out = {}
    for name, t in cfg["teams"].items():
        x = pg[pg["teamid"] == t["teamid"]].sort_values("date")
        recent = x[x["gameid"].isin(x["gameid"].drop_duplicates().tail(last_n))]
        out[name] = {p: recent.loc[recent["position"] == p, "playername"].mode().iat[0] for p in POSITIONS}
        out[name].update(cfg.get("roster_overrides", {}).get(name, {}))
    return out


class TeamPools:
    """Per team and result: an array of recent team games (games x 5 positions
    x stats) plus game lengths; per overridden player, his own games."""

    def __init__(self, cfg, pg, rosters, opp_strength):
        """opp_strength: Series indexed by (gameid, teamid) -> the opponent's
        pre-game Elo strength. Only games against Worlds-level opponents are
        used, so stat lines from stomping weak domestic teams don't inflate
        a player's Worlds forecast."""
        pg = pg.join(opp_strength.rename("opp_strength"), on=["gameid", "teamid"])
        self.team_games, self.overrides = {}, {}
        for name, t in cfg["teams"].items():
            x = pg[pg["teamid"] == t["teamid"]]
            self.team_games[name] = {r: self._team_pool(x, r) for r in (0, 1)}
            for pos, player in cfg.get("roster_overrides", {}).get(name, {}).items():
                y = pg[(pg["playername"] == player) & (pg["position"] == pos)]
                self.overrides[(name, pos)] = {r: self._player_pool(y, r) for r in (0, 1)}
        self.rosters = rosters

    @staticmethod
    def _window(x, result):
        """Recent games with this result against Worlds-level opponents; if
        there are too few, the team's games against its strongest opponents."""
        x = x[x["result"] == result]
        for start in (POOL_FROM, "2026-01-01", "2025-01-01"):
            w = x[(x["date"] >= start) & (x["opp_strength"] >= STRONG_OPP)]
            if w["gameid"].nunique() >= MIN_POOL:
                return w
        recent = x[x["date"] >= "2025-07-01"]
        keep = recent.drop_duplicates("gameid").nlargest(MIN_POOL, "opp_strength")["gameid"]
        return recent[recent["gameid"].isin(keep)]

    def _team_pool(self, x, result):
        w = self._window(x, result)
        w = w[w["position"].isin(POSITIONS)]
        full = w.groupby("gameid")["position"].nunique() == 5
        w = w[w["gameid"].isin(full[full].index)]
        piv = w.pivot_table(index="gameid", columns="position", values=STATS, aggfunc="first")
        arr = np.stack([piv[s][POSITIONS].to_numpy(float) for s in STATS], axis=-1)  # games x 5 x stats
        lengths = w.groupby("gameid")["gamelength"].first().reindex(piv.index).to_numpy(float) / 60
        return {"stats": np.nan_to_num(arr), "minutes": lengths}

    def _player_pool(self, y, result):
        w = self._window(y, result)
        return np.nan_to_num(w[STATS].to_numpy(float))

    def draw(self, team, result, rng):
        """(5 x stats array, minutes) for one simulated game."""
        pool = self.team_games[team][result]
        i = rng.integers(len(pool["minutes"]))
        stats = pool["stats"][i].copy()
        for (t, pos), op in self.overrides.items():
            if t == team:
                own = op[result]
                stats[POSITIONS.index(pos)] = own[rng.integers(len(own))]
        return stats, pool["minutes"][i]


class LengthModel:
    """Game length in minutes = a + b * p_winner + pace(winner) + pace(loser)
    + noise. p_winner is the winner's pre-game win chance, so stomps by big
    favourites come out short (international games won by >85% favourites
    average ~28.6 min vs ~32.5 for close games). Pace is a team's average
    length residual in 2026, shrunk toward 0 with PACE_PRIOR games. Noise is
    resampled from residuals at international events, so the short tail
    matches what Worlds-level games actually do."""
    PACE_PRIOR = 30

    def __init__(self, m, preds, team_ids):
        d = m.merge(preds, on="gameid").dropna(subset=["p"])
        d["pw"] = np.where(d["blue_win"] == 1, d["p"], 1 - d["p"])
        d["minutes"] = d["gamelength"] / 60
        tier1 = d["league"].isin(["LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP", *INTL_EVENTS])
        fit = d[tier1 & (d["date"] >= "2025-01-01")]
        self.b, self.a = np.polyfit(fit["pw"], fit["minutes"], 1)
        d["resid"] = d["minutes"] - (self.a + self.b * d["pw"])
        intl = d[d["league"].isin(INTL_EVENTS) & (d["date"] >= "2023-01-01")]
        self.noise = (intl["resid"] - intl["resid"].mean()).to_numpy()
        recent = d[tier1 & (d["date"] >= "2026-01-01")]
        long = pd.concat([recent[["blue_teamid", "resid"]].rename(columns={"blue_teamid": "teamid"}),
                          recent[["red_teamid", "resid"]].rename(columns={"red_teamid": "teamid"})])
        g = long.groupby("teamid")["resid"].agg(["sum", "size"])
        pace = (g["sum"] / (g["size"] + self.PACE_PRIOR)).to_dict()
        self.pace = {name: pace.get(tid, 0.0) for name, tid in team_ids.items()}

    def draw(self, winner, loser, p_winner, rng):
        return (self.a + self.b * p_winner + self.pace[winner] + self.pace[loser]
                + self.noise[rng.integers(len(self.noise))])


def simulate_players(outcomes, names, pools, P, length_model, seed=0):
    """Accumulate player and game stats over simulated tournaments.
    names: team names indexed like the simulation's team indices."""
    rng = np.random.default_rng(seed)
    players = [(t, p, pools.rosters[t][p]) for t in pools.rosters for p in POSITIONS]
    pidx = {(t, p): i for i, (t, p, _) in enumerate(players)}
    K, D, A, TK, DMG = range(5)
    res = {"kda": np.zeros(len(players)), "dmg": np.zeros(len(players)), "kp": np.zeros(len(players)),
           "kills": [], "shortest_winner": {}, "under32_majority": 0, "under32_share": [],
           "mvp_role": {p: 0.0 for p in POSITIONS}, "mvp_player": {},
           "games_sum": np.zeros(len(players)), "played_sims": np.zeros(len(players)),
           "kda_sum": np.zeros(len(players)), "dmg_sum": np.zeros(len(players)), "kp_sum": np.zeros(len(players))}
    for out in outcomes:
        tot = np.zeros((len(players), 6))
        games = np.zeros(len(players))
        shortest = (1e9, None)
        under = n_games = 0
        final_line = {}
        for stage, a, b, wa, wb in out["series"]:
            ta, tb = names[a], names[b]
            for wi, li, n in ((a, b, wa), (b, a, wb)):
                winner, loser = names[wi], names[li]
                for _ in range(n):
                    sw, _ = pools.draw(winner, 1, rng)
                    sl, _ = pools.draw(loser, 0, rng)
                    minutes = length_model.draw(winner, loser, P[wi, li], rng)
                    for team, s in ((winner, sw), (loser, sl)):
                        for j, pos in enumerate(POSITIONS):
                            i = pidx[(team, pos)]
                            tot[i] += s[j]
                            games[i] += 1
                    n_games += 1
                    under += minutes < 32
                    if minutes < shortest[0]:
                        shortest = (minutes, winner)
                    if stage == "knockouts" and {ta, tb} == {names[out["champion"]], names[out["runner_up"]]}:
                        if winner == names[out["champion"]]:
                            for j, pos in enumerate(POSITIONS):
                                final_line[pos] = final_line.get(pos, np.zeros(3)) + sw[j, [K, D, A]]
                        else:
                            for j, pos in enumerate(POSITIONS):
                                final_line.setdefault(pos, np.zeros(3))
        played = games > 0
        kda = np.where(played, (tot[:, K] + tot[:, A]) / np.maximum(tot[:, D], 1), -1)
        dmg = np.where(played, tot[:, DMG] / np.maximum(games, 1), -1)
        kp = np.where(played & (tot[:, TK] > 0), (tot[:, K] + tot[:, A]) / np.maximum(tot[:, TK], 1), -1)
        for key, v in (("kda", kda), ("dmg", dmg), ("kp", kp)):
            res[key][np.argmax(v)] += 1
            res[f"{key}_sum"] += np.where(played, v, 0)
        res["games_sum"] += games
        res["played_sims"] += played
        res["kills"].append(tot[:, K].copy())
        res["shortest_winner"][shortest[1]] = res["shortest_winner"].get(shortest[1], 0) + 1
        res["under32_share"].append(under / n_games)
        res["under32_majority"] += under / n_games > 0.5
        # Finals MVP proxy: best KDA on the winning team across the final.
        if final_line:
            score = {p: (v[0] + v[2]) / max(v[1], 1) + 0.01 * (v[0] + v[2]) for p, v in final_line.items()}
            best = max(score, key=score.get)
            res["mvp_role"][best] += 1
            who = pools.rosters[names[out["champion"]]][best]
            res["mvp_player"][who] = res["mvp_player"].get(who, 0) + 1
    n = len(outcomes)
    res["players"] = players
    for key in ("kda", "dmg", "kp"):
        res[key] = res[key] / n
        # Typical tournament average for each player (over sims where he played).
        res[f"{key}_mean"] = res.pop(f"{key}_sum") / np.maximum(res["played_sims"], 1)
    res["games_mean"] = res.pop("games_sum") / n
    res["kills"] = np.array(res["kills"])
    res["shortest_winner"] = {k: v / n for k, v in res["shortest_winner"].items()}
    res["under32_majority"] /= n
    res["mvp_role"] = {k: v / n for k, v in res["mvp_role"].items()}
    res["mvp_player"] = {k: v / n for k, v in res["mvp_player"].items()}
    return res


def worlds_like_games(pg, worlds_main_ids):
    """Player rows from WORLDS_LIKE events, with a per-row event weight.
    worlds_main_ids: Worlds main-event game ids. Oracle's Elixir also files
    regional qualifiers under WLDs (e.g. LPL vs LPL), which must be left out."""
    parts = [pg[(pg["league"] == lg) & (pg["date"] >= since)].assign(w=w)
             for lg, (since, w) in WORLDS_LIKE.items()]
    x = pd.concat(parts)
    return x[(x["league"] != "WLDs") | x["gameid"].isin(worlds_main_ids)]


def rare_event_rate(pg, stat, threshold, worlds_main_ids):
    """Weighted (occurrences, games) of a player reaching `threshold` in
    `stat` in one game, at Worlds-like events."""
    x = worlds_like_games(pg, worlds_main_ids)
    per_game = x.groupby("gameid").agg(v=(stat, "max"), w=("w", "first"))
    return float((per_game["w"] * (per_game["v"] >= threshold)).sum()), float(per_game["w"].sum())


def p_at_least_one(events, games, n_games, rng, n_draw=20000):
    """P(at least one occurrence over an event of n_games), with the rate
    uncertain: Beta(events + 0.5, games - events + 0.5)."""
    rate = rng.beta(events + 0.5, games - events + 0.5, n_draw)
    n = rng.choice(n_games, n_draw)
    return float(np.mean(1 - (1 - rate) ** n))


def champion_sim(pg, rates, newest_patch, n_games, n_sims=10000, seed=0,
                 pick_strength=10.0, wr_strength=40.0, wr_prior_games=20.0, min_games=5):
    """Leaders for 'highest win rate' and 'highest average deaths per game'
    among champions with at least `min_games` games at the event.

    Per simulated event: each champion's pick rate is drawn around its recent
    rate (wide, the Worlds patch shifts the meta), its game count from that,
    its win rate around its recent win rate shrunk toward 50%, and its average
    deaths by resampling its recent games."""
    rng = np.random.default_rng(seed)
    from .draft import is_tier1  # local import: draft imports nothing from here
    recent = pg[(pg["date"] >= "2026-06-01") & (is_tier1(pg["league"]) | (pg["patch"] == newest_patch))]
    deaths_pool = {c: g["deaths"].to_numpy(float) for c, g in recent.groupby("champion")}
    champs = [c for c in rates.index if rates.loc[c, "pick"] > 0 and len(deaths_pool.get(c, [])) >= 10]
    pick = rates.loc[champs, "pick"].clip(1e-4, 0.999).to_numpy()
    n_obs = rates.loc[champs, "picks_raw"].to_numpy(float)
    wr_obs = rates.loc[champs, "win_rate"].fillna(0.5).to_numpy()
    wr_mean = (wr_obs * n_obs + 0.5 * wr_prior_games) / (n_obs + wr_prior_games)
    wr_lead = np.zeros(len(champs))
    death_lead = np.zeros(len(champs))
    eligible = np.zeros(len(champs))
    for _ in range(n_sims):
        N = rng.choice(n_games)
        q = rng.beta(pick * pick_strength, (1 - pick) * pick_strength)
        n_c = rng.binomial(N, q)
        wins = rng.binomial(n_c, rng.beta(wr_mean * wr_strength, (1 - wr_mean) * wr_strength))
        ok = n_c >= min_games
        eligible += ok
        if not ok.any():
            continue
        wr = np.where(ok, wins / np.maximum(n_c, 1), -1)
        top = wr == wr.max()
        wr_lead += top / top.sum()
        deaths = np.full(len(champs), -1.0)
        for i in np.flatnonzero(ok):
            deaths[i] = rng.choice(deaths_pool[champs[i]], n_c[i]).mean()
        death_lead[np.argmax(deaths)] += 1
    avg_deaths = {c: float(deaths_pool[c].mean()) for c in champs}
    return (pd.Series(wr_lead / n_sims, index=champs), pd.Series(death_lead / n_sims, index=champs),
            pd.Series(wr_mean, index=champs), pd.Series(avg_deaths), pd.Series(eligible / n_sims, index=champs))


def champion_pick_history(pg, champion, since="2025-01-01"):
    """(picks, games) for a champion across all leagues since `since`."""
    x = pg[pg["date"] >= since]
    return int((x["champion"] == champion).sum()), int(x["gameid"].nunique())


def unique_champions_model(pg, worlds_main_ids):
    """Fit log(unique champions) = a + b*log(games) on Fearless-era major
    international events (2025-2026). Returns (a, b, residual sd, events)."""
    x = pg[pg["league"].isin(INTL_EVENTS) & (pg["date"] >= "2025-01-01")]
    x = x[(x["league"] != "WLDs") | x["gameid"].isin(worlds_main_ids)].copy()
    x["event"] = x["league"] + " " + x["date"].dt.year.astype(str)
    ev = x.groupby("event").agg(games=("gameid", "nunique"), unique=("champion", "nunique"))
    ev = ev[ev["games"] >= 20]
    b, a = np.polyfit(np.log(ev["games"]), np.log(ev["unique"]), 1)
    resid = np.log(ev["unique"]) - (a + b * np.log(ev["games"]))
    return a, b, float(resid.std(ddof=2)), ev
