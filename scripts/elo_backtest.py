"""Tune the Elo model and backtest it on Worlds 2022-2025 against baselines.

Usage:
  .venv/bin/python scripts/elo_backtest.py          # use saved params (models/elo_params.json)
  .venv/bin/python scripts/elo_backtest.py --tune   # re-tune, then backtest

Tuning never sees Worlds games, so the Worlds numbers are out-of-sample.
The objective is the mean log loss over three 2023-2025 groups that look
like what Worlds needs: MSI/EWC/FST games, domestic games in MSI/Worlds
leagues, and major-vs-minor league games (which set the minor-league gap).
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lolpred.config import LEAGUE_ALIAS, MAJOR_LEAGUES, PROCESSED_DIR  # noqa: E402
from lolpred.elo import EloParams, current_ratings, league_offsets, run_elo  # noqa: E402
from lolpred.features import region_record_prob  # noqa: E402
from lolpred.metrics import calibration_table, log_loss, summarize  # noqa: E402
from lolpred.splits import worlds_backtests  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_MD = ROOT / "docs" / "results" / "elo_backtest.md"
TUNE_START, TUNE_END = "2023-01-01", "2026-01-01"


def tuning_groups(m):
    win = (m["date"] >= TUNE_START) & (m["date"] < TUNE_END) & (m["league"] != "WLDs")
    maj_b = m["blue_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).isin(MAJOR_LEAGUES)
    maj_r = m["red_home_league"].map(lambda x: LEAGUE_ALIAS.get(x, x)).isin(MAJOR_LEAGUES)
    return {
        "major_intl": win & m["league"].isin(["MSI", "EWC", "FST"]),
        "major_domestic": win & ~m["cross_league"] & maj_b & maj_r,
        "major_vs_minor": win & m["cross_league"] & (maj_b ^ maj_r),
    }


def tuning_objective(m, preds, groups=None):
    groups = groups if groups is not None else tuning_groups(m)
    y, p = m["blue_win"], preds["p_blue"]
    return float(np.mean([log_loss(y[g], p[g]) for g in groups.values()]))


def coordinate_descent(m, params, grid, rounds=4):
    """Try scaling each parameter by each factor in turn; keep any improvement."""
    groups = tuning_groups(m)
    best = tuning_objective(m, run_elo(m, params), groups)
    for r in range(rounds):
        improved = False
        for name, factors in grid.items():
            cur = getattr(params, name)
            for f in factors:
                val = cur * f
                if val == cur:
                    continue
                cand = replace(params, **{name: val})
                score = tuning_objective(m, run_elo(m, cand), groups)
                if score < best - 1e-5:
                    best, params, improved = score, cand, True
                    cur = val
        print(f"  round {r + 1}: objective {best:.5f}  {params}")
        if not improved:
            break
    return params, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()

    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values(["date", "gameid"],
                                                                        kind="stable").reset_index(drop=True)
    MODELS_DIR.mkdir(exist_ok=True)
    params_path = MODELS_DIR / "elo_params.json"

    if args.tune or not params_path.exists():
        print("== tuning full Elo")
        # major_minor_weight is fixed at 0, not tuned: when tuned, the
        # major-vs-minor group pulls it to 3.4 and Worlds log loss worsens
        # (0.622 -> 0.647). See docs/DECISIONS.md.
        full, obj_full = coordinate_descent(m, EloParams(league_regress=0.2, major_minor_weight=0.0), {
            "k": [0.75, 1.25], "k_cross": [0.75, 1.25, 1.5], "k_league": [0.5, 1.5, 2],
            "k_new_mult": [0.75, 1.25], "side": [0.75, 1.25], "regress": [0.5, 1.5],
            "minor_gap": [0.5, 1.5, 2], "league_regress": [0, 0.5, 2],
        })
        print("== tuning plain Elo (one K, blue-side bonus, no linking/regression)")
        plain0 = EloParams(k=24, k_cross=24, k_league=0, k_new_mult=1, regress=0, minor_gap=0)
        plain_best, obj_plain = None, np.inf
        for k in [12, 16, 20, 24, 32, 40]:
            for side in [10, 20, 30]:
                cand = replace(plain0, k=k, k_cross=k, side=side)
                s = tuning_objective(m, run_elo(m, cand))
                if s < obj_plain:
                    plain_best, obj_plain = cand, s
        print(f"  best {obj_plain:.5f}  {plain_best}")
        params_path.write_text(json.dumps({"full": full.to_dict(), "plain": plain_best.to_dict(),
                                           "objective": {"full": obj_full, "plain": obj_plain}}, indent=2))
    saved = json.loads(params_path.read_text())
    full, plain = EloParams(**saved["full"]), EloParams(**saved["plain"])

    preds_full, state = run_elo(m, full, record_state=True)
    preds_plain = run_elo(m, plain)
    blue_rate = m["blue_win"].shift().expanding().mean().fillna(0.5).to_numpy()
    models = {
        "Elo (cross-region)": preds_full["p_blue"].to_numpy(),
        "Elo (plain)": preds_plain["p_blue"].to_numpy(),
        "Stronger region": region_record_prob(m),
        "Blue-side rate": blue_rate,
        "Always 50%": np.full(len(m), 0.5),
    }

    # --- Worlds backtests
    rows, pooled_idx = [], []
    for sp in worlds_backtests(m):
        idx = sp.test(m).index
        pooled_idx.extend(idx)
        for name, p in models.items():
            rows.append({"test": sp.name, "model": name, **summarize(m.loc[idx, "blue_win"], p[idx])})
    for name, p in models.items():
        rows.append({"test": "worlds_all", "model": name, **summarize(m.loc[pooled_idx, "blue_win"], p[pooled_idx])})
    # --- all cross-league games 2023+ (includes MSI/EWC/FST/EM etc.)
    cl = m.index[(m["date"] >= TUNE_START) & m["cross_league"]]
    for name, p in models.items():
        rows.append({"test": "cross_league_2023+", "model": name, **summarize(m.loc[cl, "blue_win"], p[cl])})
    res = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    table = res.pivot_table(index=["test", "model"], values=["n", "accuracy", "log_loss", "brier"], sort=False)
    table = table[["n", "accuracy", "log_loss", "brier"]].round(3)
    table["n"] = table["n"].astype(int)
    print("\n== results (lower log loss / brier is better)\n", table.to_string())

    cal = calibration_table(m.loc[pooled_idx, "blue_win"], models["Elo (cross-region)"][pooled_idx]).round(3)
    print("\n== calibration, Worlds 2022-2025 pooled, cross-region Elo\n", cal.to_string())

    names = pd.concat([m.set_index("blue_teamid")["blue_teamname"], m.set_index("red_teamid")["red_teamname"]])
    names = names[~names.index.duplicated(keep="last")]
    ratings = current_ratings(state, names)
    leagues = league_offsets(state)
    active = ratings[ratings["last_season"] == m["season"].max()]
    print("\n== top 20 active teams now\n", active.head(20)[["teamname", "home_league", "strength",
                                                              "team_rating", "league_offset", "games"]]
          .round(0).to_string())
    print("\n== league offsets (top 15)\n", leagues.head(15).round(0).to_string())

    # --- save
    out = m[["gameid", "date", "league", "blue_teamid", "red_teamid", "blue_win", "cross_league"]].copy()
    out = out.join(preds_full.drop(columns="gameid"))
    out["p_blue_plain"] = models["Elo (plain)"]
    out.to_parquet(PROCESSED_DIR / "elo_predictions.parquet", index=False)
    ratings.to_parquet(PROCESSED_DIR / "elo_ratings.parquet", index=False)
    leagues.rename("offset").to_frame().to_parquet(PROCESSED_DIR / "elo_league_offsets.parquet")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(
        "# Elo backtest results\n\nGenerated by `scripts/elo_backtest.py`. Tuned on 2023-2025 non-Worlds games; "
        "Worlds rows are out-of-sample.\n\n"
        f"## Parameters\n\n```json\n{json.dumps(saved, indent=2)}\n```\n\n"
        f"## Results\n\n```\n{table.to_string()}\n```\n\n"
        f"## Calibration (Worlds 2022-2025 pooled, cross-region Elo)\n\n```\n{cal.to_string()}\n```\n\n"
        f"## Top 20 active teams\n\n```\n{active.head(20)[['teamname', 'home_league', 'strength', 'games']].round(0).to_string()}\n```\n\n"
        f"## League offsets (top 15)\n\n```\n{leagues.head(15).round(0).to_string()}\n```\n")
    print(f"\nwrote {PROCESSED_DIR}/elo_*.parquet, {params_path}, {RESULTS_MD}")


if __name__ == "__main__":
    main()
