"""Live in-game win probability (see docs/DECISIONS.md).

The model (trained by scripts/live_model.py) is a logistic regression from
blue's side on: pre-game log-odds, gold / kill / CS leads, and each lead
times game time. Game time is capped at 30 minutes: training snapshots stop
at 25 minutes, so later leads are treated like 30-minute leads instead of
extrapolating further.
"""
import json

import numpy as np

from .config import ROOT

FEATURES = ["pre_logit", "gold_k", "gold_k_t", "kills", "kills_t", "cs_10", "cs_10_t"]
MAX_T = 3.0  # game time / 10, capped at 30 minutes
MODEL_PATH = ROOT / "models" / "live_model.json"


def live_features(pre_logit, minutes, gold_diff, kill_diff, cs_diff):
    """Feature matrix (n x 7). All diffs are blue minus red."""
    t = np.minimum(np.asarray(minutes, float) / 10, MAX_T)
    g, k, c = np.asarray(gold_diff, float) / 1000, np.asarray(kill_diff, float), np.asarray(cs_diff, float) / 10
    return np.column_stack([np.asarray(pre_logit, float), g, g * t, k, k * t, c, c * t])


class LiveModel:
    def __init__(self, path=MODEL_PATH):
        m = json.loads(path.read_text())
        assert m["features"] == FEATURES
        self.coef, self.intercept = np.array(m["coef"]), m["intercept"]

    def p_blue(self, p_pre_blue, minutes, gold_diff, kill_diff, cs_diff):
        p = min(max(p_pre_blue, 1e-4), 1 - 1e-4)
        x = live_features([np.log(p / (1 - p))], [minutes], [gold_diff], [kill_diff], [cs_diff])[0]
        return float(1 / (1 + np.exp(-(self.intercept + x @ self.coef))))
