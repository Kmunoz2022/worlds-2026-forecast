"""Evaluate an in-game win model at 10/15/20/25 minutes (see docs/DECISIONS.md).

Logistic regression on the pre-game probability (out-of-sample, from
train_models.py) plus gold / xp / cs / kill differences at the minute mark.
Trained on 2022-2024, tested on 2025+. Only games with full data (most LPL
games lack minute snapshots). Writes Data/processed/ingame_eval.json.

Usage: .venv/bin/python scripts/ingame_eval.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.metrics import summarize  # noqa: E402

PRE = "p_LR full (intl x5) +cal"
SPLIT = "2025-01-01"


def main():
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    mp = pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet")[["gameid", PRE]]
    d = tg[(tg["side"] == "Blue") & (tg["datacompleteness"] == "complete")].merge(mp, on="gameid")
    d = d.dropna(subset=[PRE])
    d["pre_logit"] = np.log(d[PRE] / (1 - d[PRE]))
    train, test = d["date"] < SPLIT, d["date"] >= SPLIT

    rows = [{"minute": 0, **summarize(d.loc[test, "result"], d.loc[test, PRE]),
             "share_over_90": 0.0, "acc_over_90": None}]
    for t in (10, 15, 20, 25):
        d[f"kd{t}"] = d[f"killsat{t}"] - d[f"opp_killsat{t}"]
        feats = [f"golddiffat{t}", f"xpdiffat{t}", f"csdiffat{t}", f"kd{t}", "pre_logit"]
        ok = d[feats].notna().all(axis=1)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(d.loc[train & ok, feats], d.loc[train & ok, "result"])
        q = model.predict_proba(d.loc[test & ok, feats])[:, 1]
        y = d.loc[test & ok, "result"].to_numpy()
        sure = np.maximum(q, 1 - q) > 0.9
        rows.append({"minute": t, **summarize(y, q), "share_over_90": float(sure.mean()),
                     "acc_over_90": float(np.mean((q[sure] >= 0.5) == (y[sure] == 1)))})
    out = {"train": "2022-2024", "test": f"{SPLIT[:4]}+", "rows": rows}
    (PROCESSED_DIR / "ingame_eval.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
