"""Global Elo with cross-region linking.

A team's strength = its own rating + its home league's offset.
  - Every game updates both teams' ratings (K), more when the two teams come
    from different home leagues (k_cross), since those games are rare and
    carry the only information about how regions compare.
  - League offsets move only on cross-league games (k_league), so one MSI
    upset shifts every team in that league a little. Leagues without a direct
    MSI/Worlds slot start `minor_gap` below the majors: minor leagues mostly play
    each other, so without that anchor their level floats freely (in testing a
    French 2nd-tier league drifted level with the LCK). Offsets drift back
    toward that starting level each year (league_regress). Major-vs-minor
    games move the major league's offset only by `major_minor_weight` of a
    normal step, so promotion/relegation and academy games can't swing a
    whole major region (in Oct 2026 LCS relegation games moved it +200).
  - Blue side gets a fixed bonus; ratings regress toward the mean each
    calendar year (roster turnover); new teams move faster for their first games.

Predictions are logged *before* each game is applied, so every prediction only
uses earlier games: no leakage.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import LEAGUE_ALIAS, MAJOR_LEAGUES

BASE = 1500.0


@dataclass
class EloParams:
    k: float = 24.0          # team K, same-league games
    k_cross: float = 48.0    # team K, cross-league games
    k_league: float = 8.0    # league-offset K, cross-league games only
    k_new_mult: float = 2.0  # K multiplier for a team's first `new_games` games
    new_games: int = 10
    side: float = 20.0       # blue-side bonus, rating points
    regress: float = 0.25    # share pulled back to BASE at a team's first game of each year
    minor_gap: float = 200.0     # starting offset below majors for non-MSI/Worlds leagues
    league_regress: float = 0.0  # yearly pull of league offsets back to their start
    major_minor_weight: float = 1.0  # share of a league step a major gets vs a minor team

    def to_dict(self):
        return asdict(self)


def expected(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def run_elo(m: pd.DataFrame, params: EloParams, record_state=False):
    """m: matches table sorted by date. Returns per-game predictions
    (and final state when record_state=True)."""
    p = params
    teams_b = m["blue_teamid"].to_numpy()
    teams_r = m["red_teamid"].to_numpy()
    lg_b = m["blue_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).to_numpy()
    lg_r = m["red_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).to_numpy()
    season = m["season"].to_numpy()
    y = m["blue_win"].to_numpy()
    cross = lg_b != lg_r

    rating, league_off, league_start, n_games, last_season, team_league = {}, {}, {}, {}, {}, {}
    n = len(m)
    prob = np.empty(n)
    str_b = np.empty(n)
    str_r = np.empty(n)
    off_b = np.empty(n)
    off_r = np.empty(n)

    def off(lg):
        if lg not in league_off:
            league_start[lg] = 0.0 if lg in MAJOR_LEAGUES else -p.minor_gap
            league_off[lg] = league_start[lg]
        return league_off[lg]

    def prepare(t, lg, s):
        if t not in rating:
            rating[t], n_games[t], last_season[t], team_league[t] = BASE, 0, s, lg
        if last_season[t] != s:
            rating[t] = BASE + (rating[t] - BASE) * (1 - p.regress)
            last_season[t] = s
        if team_league[t] != lg:
            # Keep total strength unchanged when a team changes home league.
            rating[t] += off(team_league[t]) - off(lg)
            team_league[t] = lg
        return rating[t] + off(lg)

    cur_season = season[0] if n else None
    for i in range(n):
        tb, tr, lb, lr, s = teams_b[i], teams_r[i], lg_b[i], lg_r[i], season[i]
        if s != cur_season:
            for lg in league_off:
                league_off[lg] = league_start[lg] + (league_off[lg] - league_start[lg]) * (1 - p.league_regress)
            cur_season = s
        sb, sr = prepare(tb, lb, s), prepare(tr, lr, s)
        e = expected(sb - sr + p.side)
        prob[i], str_b[i], str_r[i] = e, sb, sr
        off_b[i], off_r[i] = league_off[lb], league_off[lr]

        err = y[i] - e
        base_k = p.k_cross if cross[i] else p.k
        kb = base_k * (p.k_new_mult if n_games[tb] < p.new_games else 1.0)
        kr = base_k * (p.k_new_mult if n_games[tr] < p.new_games else 1.0)
        rating[tb] += kb * err
        rating[tr] -= kr * err
        if cross[i]:
            # In a major-vs-minor game (e.g. LCS relegation vs NACL), the major
            # league's offset moves only by `major_minor_weight` of the usual step.
            maj_b, maj_r = lb in MAJOR_LEAGUES, lr in MAJOR_LEAGUES
            wb = p.major_minor_weight if maj_b and not maj_r else 1.0
            wr = p.major_minor_weight if maj_r and not maj_b else 1.0
            league_off[lb] += p.k_league * wb * err
            league_off[lr] -= p.k_league * wr * err
        n_games[tb] += 1
        n_games[tr] += 1

    preds = pd.DataFrame({"gameid": m["gameid"].to_numpy(), "p_blue": prob,
                          "blue_strength": str_b, "red_strength": str_r,
                          "blue_league_off": off_b, "red_league_off": off_r})
    if not record_state:
        return preds
    state = {"rating": rating, "league_off": league_off, "team_league": team_league,
             "n_games": n_games, "last_season": last_season}
    return preds, state


def current_ratings(state, names: pd.Series) -> pd.DataFrame:
    """Final strength per team (rating + league offset), strongest first.
    Doesn't apply the next season's regression."""
    rows = [{"teamid": t, "teamname": names.get(t, t), "home_league": state["team_league"][t],
             "team_rating": r, "league_offset": state["league_off"][state["team_league"][t]],
             "games": state["n_games"][t], "last_season": state["last_season"][t]}
            for t, r in state["rating"].items()]
    df = pd.DataFrame(rows)
    df["strength"] = df["team_rating"] + df["league_offset"]
    return df.sort_values("strength", ascending=False).reset_index(drop=True)


def league_offsets(state) -> pd.Series:
    return pd.Series(state["league_off"]).sort_values(ascending=False)
