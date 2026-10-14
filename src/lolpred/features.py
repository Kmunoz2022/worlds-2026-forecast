"""Pre-game features, one row per game, from blue's perspective (blue minus red).

Everything is computed from games strictly before the one being predicted:
  - Elo parts: overall win prob, team-rating gap, league-offset gap.
  - Region record: home leagues' past cross-league win rates.
  - Form: exponentially weighted averages over each team's previous games
    (shifted by one game, so the current game is never included).
  - Context: experience, rest days, series score so far, event flags.

Early-game stats (gold/xp at 15) are left out on purpose: ~85% of LPL games
lack them, so they'd bias against LPL teams. Only stats every league has are used.
"""
import numpy as np
import pandas as pd

from .config import LEAGUE_ALIAS, MAJOR_LEAGUES

FORM_HALFLIFE = 10  # games

# Per-team per-game stats that feed the form features.
FORM_STATS = ["win", "resid", "gspd", "tower_diff", "dragon_diff", "baron_diff",
              "kd_pm", "firstblood", "minutes"]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def region_prob(wins: dict, games: dict, la: str, lb: str, prior_games=10) -> float:
    """P(league la beats league lb) from their cross-league records."""
    if la == lb:
        return 0.5
    half = prior_games / 2
    ra = (wins.get(la, 0) + half) / (games.get(la, 0) + prior_games)
    rb = (wins.get(lb, 0) + half) / (games.get(lb, 0) + prior_games)
    oa, ob = ra / (1 - ra), rb / (1 - rb)
    return oa / (oa + ob)


def region_record_prob(m: pd.DataFrame, prior_games=10, return_state=False):
    """P(blue) from each home league's past cross-league record (smoothed
    toward 50% with `prior_games` pseudo-games). Same-league games get 0.5.
    With return_state, also returns the final (wins, games) per league."""
    lb = m["blue_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).to_numpy()
    lr = m["red_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).to_numpy()
    y = m["blue_win"].to_numpy()
    wins, games = {}, {}
    out = np.full(len(m), 0.5)
    for i in range(len(m)):
        if lb[i] == lr[i]:
            continue
        out[i] = region_prob(wins, games, lb[i], lr[i], prior_games)
        wins[lb[i]] = wins.get(lb[i], 0) + y[i]
        wins[lr[i]] = wins.get(lr[i], 0) + 1 - y[i]
        games[lb[i]] = games.get(lb[i], 0) + 1
        games[lr[i]] = games.get(lr[i], 0) + 1
    return (out, (wins, games)) if return_state else out


def team_game_stats(tg: pd.DataFrame, elo_preds: pd.DataFrame) -> pd.DataFrame:
    """Per team-game values of FORM_STATS, sorted by team then date."""
    t = tg[["gameid", "teamid", "side", "date", "result", "gspd", "towers", "opp_towers",
            "dragons", "opp_dragons", "barons", "opp_barons", "kills", "deaths",
            "firstblood", "gamelength"]].copy()
    p_blue = elo_preds.set_index("gameid")["p_blue"]
    p_team = np.where(t["side"] == "Blue", t["gameid"].map(p_blue), 1 - t["gameid"].map(p_blue))
    minutes = t["gamelength"] / 60
    t["win"] = t["result"]
    t["resid"] = t["result"] - p_team  # performance vs Elo expectation
    t["tower_diff"] = t["towers"] - t["opp_towers"]
    t["dragon_diff"] = t["dragons"] - t["opp_dragons"]
    t["baron_diff"] = t["barons"] - t["opp_barons"]
    t["kd_pm"] = (t["kills"] - t["deaths"]) / minutes
    t["minutes"] = minutes
    return t.sort_values(["teamid", "date", "gameid"], kind="stable")


def team_form(tg: pd.DataFrame, elo_preds: pd.DataFrame) -> pd.DataFrame:
    """Per team-game form features using only that team's earlier games."""
    t = team_game_stats(tg, elo_preds)
    g = t.groupby("teamid", sort=False)
    out = t[["gameid", "teamid"]].copy()
    for c in FORM_STATS:
        out[f"form_{c}"] = g[c].transform(lambda s: s.shift().ewm(halflife=FORM_HALFLIFE).mean())
    out["games_before"] = g.cumcount()
    out["rest_days"] = (t["date"] - g["date"].shift()).dt.total_seconds().div(86400).clip(upper=60)
    return out


def build_features(m: pd.DataFrame, tg: pd.DataFrame, elo_preds: pd.DataFrame) -> pd.DataFrame:
    """m: matches sorted by date; elo_preds aligned row-for-row with m."""
    assert (m["gameid"].to_numpy() == elo_preds["gameid"].to_numpy()).all()
    f = pd.DataFrame({"gameid": m["gameid"].to_numpy()})

    e = elo_preds
    f["elo_logit"] = _logit(e["p_blue"].to_numpy())
    f["elo_team_diff"] = ((e["blue_strength"] - e["blue_league_off"])
                          - (e["red_strength"] - e["red_league_off"])).to_numpy() / 100
    f["elo_league_diff"] = (e["blue_league_off"] - e["red_league_off"]).to_numpy() / 100
    f["region_logit"] = _logit(region_record_prob(m))

    form = team_form(tg, elo_preds).set_index(["gameid", "teamid"])
    fb = form.reindex(pd.MultiIndex.from_arrays([m["gameid"], m["blue_teamid"]])).to_numpy()
    fr = form.reindex(pd.MultiIndex.from_arrays([m["gameid"], m["red_teamid"]])).to_numpy()
    cols = list(form.columns)
    for j, c in enumerate(cols):
        if c == "games_before":
            f["log_games_diff"] = np.log1p(fb[:, j]) - np.log1p(fr[:, j])
            f["log_games_min"] = np.log1p(np.minimum(fb[:, j], fr[:, j]))
        else:
            f[f"{c}_diff"] = fb[:, j] - fr[:, j]

    # Series score so far (pre-game) from blue's perspective.
    a_is_blue = (m["team_a"] == m["blue_teamid"]).to_numpy()
    a_w, b_w = m["a_series_wins_before"].to_numpy(), m["b_series_wins_before"].to_numpy()
    f["series_lead"] = np.where(a_is_blue, a_w - b_w, b_w - a_w)
    f["series_game_no"] = m["series_game_no"].to_numpy()

    lb = m["blue_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x))
    lr = m["red_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x))
    f["cross_league"] = m["cross_league"].astype(int).to_numpy()
    f["major_intl"] = m["major_intl"].astype(int).to_numpy()
    f["both_major"] = (lb.isin(MAJOR_LEAGUES) & lr.isin(MAJOR_LEAGUES)).astype(int).to_numpy()
    f["playoffs"] = m["playoffs"].fillna(0).astype(int).to_numpy()
    return f


FEATURES_BASIC = ["elo_logit", "elo_league_diff", "region_logit"]
FEATURES_FULL = FEATURES_BASIC + [
    "elo_team_diff",
    *[f"form_{c}_diff" for c in FORM_STATS],
    "rest_days_diff", "log_games_diff", "log_games_min",
    "series_lead", "series_game_no", "cross_league", "major_intl", "both_major", "playoffs",
]
