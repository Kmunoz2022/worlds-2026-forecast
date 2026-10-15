"""Track Crystal Ball picks against Worlds 2026 as it happens (see docs/DECISIONS.md).

Two sets of picks are scored on the same measurements:
  model     ledger/crystal_ball_locked.json + ledger/crystal_ball_locked_extra.json
  personal  notes/crystal_ball_personal.json (private, git-ignored: my own picks)
Locked files are only read, never written. Personal results go only to the
private notes/crystal_tracker_personal.json, never to the website or the repo.

Every question is measured once from the Worlds main-event games in the data,
then each pick is scored by lolpred.crystal_track (shared with the automatic
feed tracker, which updates after every game) and marked: not started / on track / behind / tied / won / lost /
waiting. Questions the game data can't answer (knockout teams, placements,
Finals MVP) come from config/worlds2026_results.json.
Writes Data/processed/crystal_tracker.json (model only).

Usage: .venv/bin/python scripts/track_crystal.py [--rebuild]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.crystal_track import norm, results_facts, score_all, tally  # noqa: E402
from lolpred.crystal import load_player_games  # noqa: E402
from lolpred.draft import load_draft_events  # noqa: E402
from lolpred.splits import worlds_start  # noqa: E402

LEDGER = ROOT / "ledger"
CONFIG = ROOT / "config" / "worlds2026.json"
RESULTS = ROOT / "config" / "worlds2026_results.json"
SEASON = 2026


def load_picks():
    model = {}
    for f in ("crystal_ball_locked.json", "crystal_ball_locked_extra.json"):
        path = LEDGER / f
        if path.exists():
            for q in json.loads(path.read_text())["questions"]:
                model[q["id"]] = {"question": q["question"], "pick": q["pick"]}
    personal = None
    path = ROOT / "notes" / "crystal_ball_personal.json"
    if path.exists():
        p = json.loads(path.read_text())["picks"]
        if any(v.get("pick") for v in p.values()):
            personal = {k: v for k, v in p.items() if v.get("pick")}
    return model, personal


def measure(ids, pg, ev, games, res, cfg):
    """Everything the questions need, from the Worlds games so far."""
    w = pg[pg["gameid"].isin(ids)]
    M = {}
    d = ev[ev["gameid"].isin(ids)]
    pres = d.groupby("champion")["gameid"].nunique() / max(len(ids), 1)
    M["presence"] = [{"name": c, "value": v, "games": int(v * len(ids) + 0.5)} for c, v in pres.items()]
    ch = w.groupby("champion").agg(games=("gameid", "nunique"), wins=("result", "sum"), deaths=("deaths", "mean"))
    ch = ch[ch["games"] >= 5]
    M["champ_wr"] = [{"name": c, "value": r.wins / r.games, "games": int(r.games)} for c, r in ch.iterrows()]
    M["champ_deaths"] = [{"name": c, "value": r.deaths, "games": int(r.games)} for c, r in ch.iterrows()]
    pl = w.groupby("playername").agg(games=("gameid", "nunique"), k=("kills", "sum"), d=("deaths", "sum"),
                                     a=("assists", "sum"), tk=("teamkills", "sum"), dmg=("damagetochampions", "mean"))
    M["kda"] = [{"name": p, "value": (r.k + r.a) / max(r.d, 1), "games": int(r.games)} for p, r in pl.iterrows()]
    M["dmg"] = [{"name": p, "value": r.dmg, "games": int(r.games)} for p, r in pl.iterrows()]
    M["kp"] = [{"name": p, "value": (r.k + r.a) / max(r.tk, 1), "games": int(r.games)} for p, r in pl.iterrows()]
    M["kills"] = {norm(n): v for n, v in pl["k"].items()}
    M["teemo"] = int((w["champion"] == "Teemo").sum())
    M["unique"] = int(w["champion"].nunique())
    M["before32"] = int((games["gamelength"] < 1920).sum())
    M["after32"] = int(len(games) - M["before32"])
    for key, stat in (("max_kills", "kills"), ("max_cs", "total cs")):
        best = w.loc[w[stat].idxmax()]
        M[key] = (best["playername"], int(best[stat]))
    sg = games.loc[games["gamelength"].idxmin()]
    M["shortest"] = (sg["gamelength"] / 60, w[(w["gameid"] == sg["gameid"]) & (w["result"] == 1)]["teamname"].iat[0])
    return M | results_facts(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    res = json.loads(RESULTS.read_text())
    finished = bool(res.get("finished"))
    model_picks, personal_picks = load_picks()
    locked_at = json.loads((LEDGER / "crystal_ball_locked.json").read_text())["locked_at"]

    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    start = worlds_start(m, SEASON)
    ids = set() if start is None else set(m[(m["league"] == "WLDs") & (m["season"] == SEASON) & (m["date"] >= start)]["gameid"])
    out = {"locked_at": locked_at, "games": len(ids), "started": bool(ids), "finished": finished,
           "data_through": str(m["date"].max().date()), "source": "official data"}

    teams = list(cfg["teams"])
    regions = {t: v["region"] for t, v in cfg["teams"].items()}

    def run(picks):
        return None if picks is None else score_all(picks, M, teams, regions, finished, len(ids))

    M = {}
    if ids:
        pg = load_player_games(rebuild=args.rebuild)
        ev = load_draft_events(rebuild=args.rebuild)
        games = pg[pg["gameid"].isin(ids)].drop_duplicates("gameid")
        M = measure(ids, pg, ev, games, res, cfg)
    out["model"] = run(model_picks)
    out["tally"] = {"model": tally(out["model"])}
    (PROCESSED_DIR / "crystal_tracker.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    personal = run(personal_picks)
    if personal:
        (ROOT / "notes" / "crystal_tracker_personal.json").write_text(json.dumps(
            {"games": len(ids), "personal": personal, "tally": tally(personal)}, indent=2, ensure_ascii=False))
    print(f"games {len(ids)} | model {out['tally']['model']} | personal {tally(personal) if personal else None}")
    if ids:
        for q, v in out["model"].items():
            mine = (personal or {}).get(q)
            print(f"{q:13s} model {v['status']:10s} {v['pick']!s:30.30s} "
                  + (f"| you {mine['status']:10s} {mine['pick']!s:20.20s}" if mine else "") + f" | {v['summary']}")


if __name__ == "__main__":
    main()
