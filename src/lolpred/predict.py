"""Win probability for any matchup, using every game played so far.

Builds each team's current "snapshot" (Elo rating + league offset, form
averages including its latest game, experience) and feeds the same features
the game model was trained on. The result is averaged over both sides, since
side selection isn't known in advance, and then calibrated.
"""
import json

import joblib
import numpy as np
import pandas as pd

from .config import LEAGUE_ALIAS, MAJOR_LEAGUES, PROCESSED_DIR, ROOT
from .elo import EloParams, expected, run_elo
from .features import FORM_HALFLIFE, FORM_STATS, region_prob, region_record_prob, team_game_stats

MODELS_DIR = ROOT / "models"


class MatchPredictor:
    def __init__(self):
        m = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values(
            ["date", "gameid"], kind="stable").reset_index(drop=True)
        tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
        self.elo_params = EloParams(**json.loads((MODELS_DIR / "elo_params.json").read_text())["full"])
        elo_preds, self.elo_state = run_elo(m, self.elo_params, record_state=True)
        _, (self.reg_wins, self.reg_games) = region_record_prob(m, return_state=True)

        bundle = joblib.load(MODELS_DIR / "game_model.joblib")
        self.model, self.features = bundle["model"], bundle["features"]
        meta = json.loads((MODELS_DIR / "game_model.json").read_text())
        self.slope = meta["calibration_slope"] if meta.get("calibrated") else 1.0
        self.trained_through = meta["trained_through"]

        self.form = self._form_snapshot(tg, elo_preds)
        self.names = (tg.sort_values("date").groupby("teamid")["teamname"].last())

    @staticmethod
    def _form_snapshot(tg, elo_preds):
        """EWM over *all* of a team's games, i.e. the view before its next game."""
        t = team_game_stats(tg, elo_preds)
        g = t.groupby("teamid", sort=False)
        snap = pd.DataFrame({f"form_{c}": g[c].agg(lambda s: s.ewm(halflife=FORM_HALFLIFE).mean().iloc[-1])
                             for c in FORM_STATS})
        snap["games"] = g.size()
        snap["last_game"] = g["date"].max()
        return snap

    def team_info(self, teamid):
        st = self.elo_state
        lg = st["team_league"][teamid]
        return {"teamid": teamid, "teamname": self.names.get(teamid, teamid), "league": lg,
                "team_rating": st["rating"][teamid], "league_offset": st["league_off"][lg],
                "strength": st["rating"][teamid] + st["league_off"][lg],
                "games": int(self.form.loc[teamid, "games"]),
                "last_game": self.form.loc[teamid, "last_game"]}

    def _row(self, blue, red, playoffs=0):
        b, r = self.team_info(blue), self.team_info(red)
        fb, fr = self.form.loc[blue], self.form.loc[red]
        p_elo = expected(b["strength"] - r["strength"] + self.elo_params.side)
        p_elo = min(max(p_elo, 1e-4), 1 - 1e-4)
        lb, lr = LEAGUE_ALIAS.get(b["league"], b["league"]), LEAGUE_ALIAS.get(r["league"], r["league"])
        p_reg = min(max(region_prob(self.reg_wins, self.reg_games, lb, lr), 1e-4), 1 - 1e-4)
        row = {
            "elo_logit": np.log(p_elo / (1 - p_elo)),
            "elo_league_diff": (b["league_offset"] - r["league_offset"]) / 100,
            "region_logit": np.log(p_reg / (1 - p_reg)),
            "elo_team_diff": (b["team_rating"] - r["team_rating"]) / 100,
            "rest_days_diff": 0.0,
            "log_games_diff": np.log1p(b["games"]) - np.log1p(r["games"]),
            "log_games_min": np.log1p(min(b["games"], r["games"])),
            "series_lead": 0, "series_game_no": 1,
            "cross_league": int(lb != lr), "major_intl": 1,
            "both_major": int(lb in MAJOR_LEAGUES and lr in MAJOR_LEAGUES),
            "playoffs": playoffs,
        }
        for c in FORM_STATS:
            row[f"form_{c}_diff"] = fb[f"form_{c}"] - fr[f"form_{c}"]
        return row

    def game_prob(self, team_a, team_b, playoffs=0) -> float:
        """P(team_a wins one game vs team_b), averaged over both sides, calibrated."""
        X = pd.DataFrame([self._row(team_a, team_b, playoffs), self._row(team_b, team_a, playoffs)])
        p_blue = self.model.predict_proba(X[self.features])[:, 1]
        p = 0.5 * (p_blue[0] + (1 - p_blue[1]))
        z = np.log(p / (1 - p)) * self.slope
        return float(1 / (1 + np.exp(-z)))

    def game_prob_sided(self, blue, red, playoffs=0) -> float:
        """P(blue wins this game) with sides known (live games), calibrated."""
        X = pd.DataFrame([self._row(blue, red, playoffs)])
        p = float(self.model.predict_proba(X[self.features])[0, 1])
        z = np.log(p / (1 - p)) * self.slope
        return float(1 / (1 + np.exp(-z)))

    def prob_matrix(self, teams: list[str], playoffs=0) -> np.ndarray:
        n = len(teams)
        P = np.full((n, n), 0.5)
        for i in range(n):
            for j in range(i + 1, n):
                P[i, j] = self.game_prob(teams[i], teams[j], playoffs)
                P[j, i] = 1 - P[i, j]
        return P
