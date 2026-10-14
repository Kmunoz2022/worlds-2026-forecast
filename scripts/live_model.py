"""Train and test the in-game win model used for live predictions (see docs/DECISIONS.md).

One logistic regression for any minute of the game, from blue's side:
  pre-game log-odds, gold lead, kill lead and CS lead, each also multiplied by
  game time (minutes / 10), so leads count more as the game goes on.
Trained on Oracle's Elixir snapshots at 0 (no leads yet), 10, 15, 20 and 25
minutes. XP isn't used: the live feed doesn't carry it.

Tested on 2025+ games with a model trained on 2022-2024, then refit on
everything and saved to models/live_model.json.

Usage: .venv/bin/python scripts/live_model.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.live import FEATURES, live_features  # noqa: E402
from lolpred.metrics import summarize  # noqa: E402

PRE = "p_LR full (intl x5) +cal"
SPLIT = "2025-01-01"
MINUTES = (0, 10, 15, 20, 25)


def stacked(d):
    rows = []
    for t in MINUTES:
        if t == 0:
            g = k = c = np.zeros(len(d))
            ok = np.ones(len(d), bool)
        else:
            g = d[f"golddiffat{t}"].to_numpy(float)
            k = (d[f"killsat{t}"] - d[f"opp_killsat{t}"]).to_numpy(float)
            c = d[f"csdiffat{t}"].to_numpy(float)
            ok = ~(np.isnan(g) | np.isnan(k) | np.isnan(c))
        X = live_features(d["pre_logit"].to_numpy(), np.full(len(d), t), g, k, c)
        rows.append(pd.DataFrame(X[ok], columns=FEATURES).assign(
            y=d["result"].to_numpy()[ok], minute=t, date=d["date"].to_numpy()[ok]))
    return pd.concat(rows, ignore_index=True)


def main():
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    mp = pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet")[["gameid", PRE]]
    d = tg[(tg["side"] == "Blue") & (tg["datacompleteness"] == "complete")].merge(mp, on="gameid").dropna(subset=[PRE])
    p = d[PRE].clip(1e-4, 1 - 1e-4)
    d["pre_logit"] = np.log(p / (1 - p))  # blue-side pre-game chance (out-of-sample)
    S = stacked(d)

    train, test = S["date"] < pd.Timestamp(SPLIT), S["date"] >= pd.Timestamp(SPLIT)
    model = LogisticRegression(C=1.0, max_iter=2000).fit(S.loc[train, FEATURES], S.loc[train, "y"])
    S["p"] = model.predict_proba(S[FEATURES])[:, 1]
    rows = []
    for t in MINUTES:
        x = S[test & (S["minute"] == t)]
        sure = np.maximum(x["p"], 1 - x["p"]) > 0.9
        rows.append({"minute": t, **summarize(x["y"], x["p"]), "share_over_90": float(sure.mean()),
                     "acc_over_90": float(((x["p"] >= 0.5) == (x["y"] == 1))[sure].mean()) if sure.any() else None})
    print("test 2025+ (trained 2022-2024):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    final = LogisticRegression(C=1.0, max_iter=2000).fit(S[FEATURES], S["y"])
    out = {"features": FEATURES, "coef": final.coef_[0].round(6).tolist(),
           "intercept": float(final.intercept_[0]), "test": rows, "trained_rows": int(len(S))}
    (ROOT / "models" / "live_model.json").write_text(json.dumps(out, indent=2))
    print("\ncoefficients:", dict(zip(FEATURES, out["coef"])), "intercept", round(out["intercept"], 4))


if __name__ == "__main__":
    main()
