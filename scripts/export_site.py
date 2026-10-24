"""Collect everything the website shows into site/data.json.

Run after simulate_worlds.py, ingame_eval.py, ledger.py grade and track_crystal.py
(scripts/refresh.py runs them all). Every number on the site
comes from this file; nothing is typed into the HTML.

Usage: .venv/bin/python scripts/export_site.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lolpred.config import LEAGUE_ALIAS, MAJOR_LEAGUES, PROCESSED_DIR  # noqa: E402
from lolpred.features import region_record_prob  # noqa: E402
from lolpred.metrics import calibration_table, summarize  # noqa: E402
from lolpred.splits import worlds_backtests  # noqa: E402

SITE_DIR = ROOT / "site"
CONFIG = ROOT / "config" / "worlds2026.json"
FINAL = "p_LR full (intl x5) +cal"
LOCKED_CRYSTAL = ROOT / "ledger" / "crystal_ball_locked.json"
LOCKED_PICKEM = ROOT / "ledger" / "pickem_locked.json"
# The pick'em actually submitted (Oct 22, Riot's 3-0 / 3-1 / 3-2 + advance format);
# pickem_locked.json is the earlier model version, kept as history.
FINAL_PICKEM = ROOT / "ledger" / "pickem_locked_final.json"
LOCKED_EXTRA = ROOT / "ledger" / "crystal_ball_locked_extra.json"


def crystal_section():
    """Locked model picks, presence question first. (Personal picks are private
    and never exported.)"""
    cb = json.loads(LOCKED_CRYSTAL.read_text())
    if LOCKED_EXTRA.exists():
        cb["questions"] = json.loads(LOCKED_EXTRA.read_text())["questions"] + cb["questions"]
    return cb


def live_section(cfg, tg):
    """What the browser needs to run live predictions itself: the live model,
    a player -> team map (the feed names players, not our teams), and each
    matchup's pre-game chance with sides known (blue vs red)."""
    from lolpred.predict import MatchPredictor
    lm = json.loads((ROOT / "models" / "live_model.json").read_text())
    pred = MatchPredictor()
    names = list(cfg["teams"])
    ids = {n: cfg["teams"][n]["teamid"] for n in names}
    p_blue = {b: {r: round(pred.game_prob_sided(ids[b], ids[r]), 4) for r in names if r != b} for b in names}
    recent = tg[(tg["season"] == tg["season"].max()) & tg["teamid"].isin(ids.values())]
    by_id = {v: k for k, v in ids.items()}
    pl = pd.read_parquet(PROCESSED_DIR / "player_games.parquet", columns=["playername", "teamid", "date"])
    pl = pl[(pl["date"] >= "2026-01-01") & pl["teamid"].isin(ids.values())].sort_values("date")
    players = {p.lower(): by_id[t] for p, t in pl.groupby("playername")["teamid"].last().items()}
    return {"coef": lm["coef"], "intercept": lm["intercept"], "features": lm["features"], "max_t": 3.0,
            "players": players, "p_blue": p_blue,
            "worlds_league_id": "98767975604431411"}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def team_section(cfg, sim, tg, elo):
    out = []
    for name, t in cfg["teams"].items():
        tid = t["teamid"]
        games = tg[(tg["teamid"] == tid) & (tg["season"] == tg["season"].max())].sort_values("date")
        hist = elo[(elo["blue_teamid"] == tid) | (elo["red_teamid"] == tid)]
        hist = hist[hist["date"] >= f"{tg['season'].max()}-01-01"]
        strength = np.where(hist["blue_teamid"] == tid, hist["blue_strength"], hist["red_strength"])
        out.append({
            "name": name, "region": t["region"], "seed": t["seed"],
            "strength": r(sim["teams"][name]["strength"], 1),
            "avg_game_win": r(sim["teams"][name]["avg_game_win"]),
            "wins": int(games["result"].sum()), "losses": int((1 - games["result"]).sum()),
            "last10": "".join("W" if x else "L" for x in games["result"].tail(10)),
            "history": [[d.strftime("%Y-%m-%d"), round(float(s), 1)]
                        for d, s in zip(hist["date"], strength)]
                       + [[sim["generated_from_data_through"], r(sim["teams"][name]["strength"], 1)]],
        })
    return out


def league_section(m, elo_offsets):
    _, (wins, games) = region_record_prob(m, return_state=True)
    rows = []
    for lg in sorted(MAJOR_LEAGUES, key=lambda x: -elo_offsets.get(x, -1e9)):
        if lg not in elo_offsets:
            continue
        rows.append({"league": lg, "offset": r(elo_offsets[lg], 1),
                     "intl_wins": int(wins.get(lg, 0)), "intl_games": int(games.get(lg, 0))})
    return rows


def model_record(m, preds):
    y = m["blue_win"].to_numpy()
    models = {"Our model": preds[FINAL].to_numpy(), "Elo only": preds["p_Elo"].to_numpy(),
              "Stronger region": region_record_prob(m), "Coin flip": np.full(len(m), 0.5)}
    by_year, pooled = [], []
    for sp in worlds_backtests(m):
        idx = sp.test(m).index.to_numpy()
        pooled.append(idx)
        for name, p in models.items():
            s = summarize(y[idx], p[idx])
            by_year.append({"year": int(sp.name[-4:]), "model": name, **{k: r(v) for k, v in s.items()}})
    idx = np.concatenate(pooled)
    overall = [{"model": name, **{k: r(v) for k, v in summarize(y[idx], p[idx]).items()}}
               for name, p in models.items()]
    cal = calibration_table(y[idx], models["Our model"][idx]).reset_index()
    calib = [{"bin": str(b), "n": int(n), "predicted": r(pr), "actual": r(a)}
             for b, n, pr, a in cal.itertuples(index=False)]
    cl = (m["date"] >= "2023-01-01") & m["cross_league"] & preds[FINAL].notna().to_numpy()
    cross = [{"model": name, **{k: r(v) for k, v in summarize(y[cl], p[cl.to_numpy()]).items()}}
             for name, p in models.items()]
    return {"by_year": by_year, "overall": overall, "calibration": calib, "cross_league": cross}


def main():
    cfg = json.loads(CONFIG.read_text())
    sim = json.loads((PROCESSED_DIR / "worlds2026_sim.json").read_text())
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet").sort_values(["date", "gameid"],
                                                                        kind="stable").reset_index(drop=True)
    tg = pd.read_parquet(PROCESSED_DIR / "team_games.parquet")
    elo = pd.read_parquet(PROCESSED_DIR / "elo_predictions.parquet")
    # Out-of-sample predictions from the last full training run, lined up with the games
    # (games added since, with --skip-train, have none; the record only uses past events).
    preds = m[["gameid"]].merge(pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet"), on="gameid", how="left")
    offsets = pd.read_parquet(PROCESSED_DIR / "elo_league_offsets.parquet")["offset"]
    offsets.index = [LEAGUE_ALIAS.get(i, i) for i in offsets.index]
    ingame = json.loads((PROCESSED_DIR / "ingame_eval.json").read_text())

    pickem = json.loads((FINAL_PICKEM if FINAL_PICKEM.exists() else LOCKED_PICKEM).read_text())
    scen_name = cfg["cblol_2nd"][0]
    scen = sim["scenarios"][scen_name]
    data = {
        "meta": {"data_through": sim["generated_from_data_through"], "exported": date.today().isoformat(),
                 "sims": sim["sims"], "kappa": sim["kappa"], "games_in_data": int(len(m))},
        "teams": team_section(cfg, sim, tg, elo),
        "game_prob": sim["game_prob"],
        "sim": [{k: (r(v) if isinstance(v, float) else v) for k, v in row.items() if k != "record_given_swiss"}
                for row in scen["teams"]],
        # Submitted picks are locked (ledger/*_locked.json); live odds above keep updating.
        "pickem": pickem["pickem"],
        "pickem_locked_at": pickem["locked_at"],
        "pickem_format": pickem.get("format", "2-2-5"),
        "play_in": {"teams": [scen_name if t == "CBLOL_2ND" else t for t in cfg["play_in"]],
                    "round1": [[scen_name if t == "CBLOL_2ND" else t for t in pair]
                               for pair in cfg.get("play_in_round1", [])]},
        "swiss_pools": {k: v for k, v in cfg["swiss_pools"].items()},
        "leagues": league_section(m, offsets.to_dict()),
        "model_record": model_record(m, preds),
        "ingame": ingame,
        "crystal_ball": crystal_section(),
        "live": live_section(cfg, tg),
        "crystal_tracker": read_json(PROCESSED_DIR / "crystal_tracker.json"),
        "ledger": read_json(PROCESSED_DIR / "ledger_graded.json"),
        # Team logos, player photos and the Data Dragon version (scripts/fetch_assets.py).
        "assets": read_json(ROOT / "config" / "team_assets.json"),
    }
    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {SITE_DIR / 'data.json'} ({(SITE_DIR / 'data.json').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
