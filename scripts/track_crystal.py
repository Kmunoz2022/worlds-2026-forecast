"""Track the locked Crystal Ball picks against Worlds 2026 as it happens (see docs/DECISIONS.md).

Reads ledger/crystal_ball_locked.json (never modified), measures every
question from the Worlds main-event games in the data, and marks each pick:
  not started / on track / behind / won / lost / waiting.
Questions the game data can't answer (knockout teams, placements, Finals MVP)
come from config/worlds2026_results.json. Writes Data/processed/crystal_tracker.json.

Usage: .venv/bin/python scripts/track_crystal.py [--rebuild]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.crystal import load_player_games  # noqa: E402
from lolpred.splits import worlds_start  # noqa: E402

LOCKED = ROOT / "ledger" / "crystal_ball_locked.json"
CONFIG = ROOT / "config" / "worlds2026.json"
RESULTS = ROOT / "config" / "worlds2026_results.json"
SEASON = 2026
ROLE = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}


def board(rows, key, fmt, pick, label="name", n=3):
    """Top n rows plus the pick's own value and rank."""
    rows = sorted(rows, key=lambda r: -r[key])
    top = [{"name": r[label], "value": fmt(r[key]), "games": r.get("games")} for r in rows[:n]]
    mine = next(((i + 1, r) for i, r in enumerate(rows) if r[label] == pick), None)
    return top, (None if mine is None else {"rank": mine[0], "value": fmt(mine[1][key]), "games": mine[1].get("games")})


def status_from_rank(rank, finished):
    if rank is None:
        return "behind" if not finished else "lost"
    if finished:
        return "won" if rank == 1 else "lost"
    return "on track" if rank == 1 else "behind"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    locked = json.loads(LOCKED.read_text())
    cfg = json.loads(CONFIG.read_text())
    res = json.loads(RESULTS.read_text())
    picks = {q["id"]: q for q in locked["questions"]}
    finished = bool(res.get("finished"))

    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    start = worlds_start(m, SEASON)
    ids = set() if start is None else set(m[(m["league"] == "WLDs") & (m["season"] == SEASON) & (m["date"] >= start)]["gameid"])
    out = {"locked_at": locked["locked_at"], "games": len(ids), "started": bool(ids), "finished": finished, "questions": {}}

    def put(qid, status, summary, top=None, mine=None):
        out["questions"][qid] = {"pick": picks[qid]["pick"], "status": status, "summary": summary,
                                 "leaders": top or [], "pick_now": mine}

    if not ids:
        for qid in picks:
            put(qid, "not started", "Worlds hasn't started in the data yet.")
        (PROCESSED_DIR / "crystal_tracker.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print("Worlds 2026 hasn't started in the data; tracker reset to 'not started'.")
        return

    pg = load_player_games(rebuild=args.rebuild)
    w = pg[pg["gameid"].isin(ids)]
    games = w.drop_duplicates("gameid")

    # --- champions (min 5 games)
    ch = w.groupby("champion").agg(games=("gameid", "nunique"), wins=("result", "sum"), deaths=("deaths", "mean"))
    ch = ch[ch["games"] >= 5].assign(wr=lambda d: d["wins"] / d["games"]).reset_index()
    rows = ch.rename(columns={"champion": "name"}).to_dict("records")
    for qid, key, fmt in (("champ_wr", "wr", lambda v: f"{v:.0%}"), ("champ_deaths", "deaths", lambda v: f"{v:.2f}")):
        top, mine = board(rows, key, fmt, picks[qid]["pick"])
        note = "" if mine else f" {picks[qid]['pick']} hasn't reached 5 games yet."
        put(qid, status_from_rank(mine and mine["rank"], finished),
            (f"Leader: {top[0]['name']} {top[0]['value']}." if top else "No champion has 5 games yet.") + note, top, mine)

    # --- players
    pl = w.groupby("playername").agg(games=("gameid", "nunique"), k=("kills", "sum"), d=("deaths", "sum"),
                                     a=("assists", "sum"), tk=("teamkills", "sum"), dmg=("damagetochampions", "mean"))
    pl["kda"] = (pl["k"] + pl["a"]) / pl["d"].clip(lower=1)
    pl["kp"] = (pl["k"] + pl["a"]) / pl["tk"].clip(lower=1)
    prow = pl.reset_index().rename(columns={"playername": "name"}).to_dict("records")
    for qid, key, fmt in (("kda", "kda", lambda v: f"{v:.2f}"), ("dmg", "dmg", lambda v: f"{v:,.0f}"),
                          ("kp", "kp", lambda v: f"{v:.0%}")):
        pick_player = picks[qid]["pick"].split(" (")[0]
        top, mine = board(prow, key, fmt, pick_player)
        put(qid, status_from_rank(mine and mine["rank"], finished),
            f"Leader: {top[0]['name']} {top[0]['value']} over {top[0]['games']} games.", top, mine)

    k = pl["k"]
    peyz, guma = int(k.get("Peyz", 0)), int(k.get("Gumayusi", 0))
    lead = "Peyz" if peyz > guma else "Gumayusi" if guma > peyz else None
    st = ("won" if lead == picks["peyz_guma"]["pick"] else "lost") if finished else \
         ("on track" if lead == picks["peyz_guma"]["pick"] else "behind" if lead else "tied")
    put("peyz_guma", st, f"Peyz {peyz} kills, Gumayusi {guma}.")

    faker = 616 + int(k.get("Faker", 0))
    st = "lost" if faker >= 700 else ("won" if finished else "on track")
    put("faker_700", st, f"Faker is at {faker} career Worlds kills ({700 - faker} to go)." if faker < 700
        else f"Faker reached {faker}.")

    teemo = int((w["champion"] == "Teemo").sum())
    put("teemo", "lost" if teemo else ("won" if finished else "on track"),
        f"Teemo picked {teemo} times." if teemo else "Teemo hasn't been picked.")

    uniq = int(w["champion"].nunique())
    lo, hi = {"0-99": (0, 99), "100-104": (100, 104), "105-109": (105, 109), "110-114": (110, 114),
              "115-119": (115, 119), "120+": (120, 10**6)}[picks["unique"]["pick"]]
    st = "lost" if uniq > hi else ("won" if finished and uniq >= lo else "lost" if finished else
                                   "on track" if uniq >= lo else "waiting")
    put("unique", st, f"{uniq} different champions picked so far in {len(ids)} games.")

    before = int((games["gamelength"] < 1920).sum())
    after = len(games) - before
    lead = "Before 32:00" if before > after else "After 32:00" if after > before else None
    st = ("won" if lead == picks["len32"]["pick"] else "lost") if finished else \
         ("on track" if lead == picks["len32"]["pick"] else "behind" if lead else "tied")
    put("len32", st, f"{before} games ended before 32:00, {after} after.")

    for qid, stat, thr, unit in (("kills15", "kills", 15, "kills"), ("cs450", "total cs", 450, "CS")):
        best = w.loc[w[stat].idxmax()]
        hit = best[stat] >= thr
        yes = picks[qid]["pick"] == "Yes"
        st = ("won" if yes else "lost") if hit else (("lost" if yes else "won") if finished else
                                                      ("waiting" if yes else "on track"))
        put(qid, st, f"Best so far: {best['playername']} with {int(best[stat])} {unit}.")

    sg = games.loc[games["gamelength"].idxmin()]
    winner = w[(w["gameid"] == sg["gameid"]) & (w["result"] == 1)]["teamname"].iat[0]
    pick_team = picks["shortest"]["pick"]
    st = ("won" if winner == pick_team else "lost") if finished else ("on track" if winner == pick_team else "behind")
    put("shortest", st, f"Shortest so far: {sg['gamelength'] / 60:.1f} min, won by {winner}.")

    # --- questions that need tournament results
    ko = res.get("knockout_teams") or []
    if len(ko) == 8:
        regions = {cfg["teams"][t]["region"] for t in ko}
        put("regions_ko", "won" if str(len(regions)) == picks["regions_ko"]["pick"] else "lost",
            f"{len(regions)} regions in knockouts: {', '.join(sorted(regions))}.")
        has_lcs = any(cfg["teams"][t]["region"] == "LCS" for t in ko)
        put("lcs_ko", "won" if ("Yes" if has_lcs else "No") == picks["lcs_ko"]["pick"] else "lost",
            "An LCS team made knockouts." if has_lcs else "No LCS team made knockouts.")
    else:
        put("regions_ko", "waiting", "Decided when the Swiss stage ends.")
        put("lcs_ko", "waiting", "Decided when the Swiss stage ends.")
    pl_ = res.get("placements") or {}
    if "T1" in pl_ and "Hanwha Life Esports" in pl_:
        t1, hle = pl_["T1"], pl_["Hanwha Life Esports"]
        winner = "T1" if t1 < hle else "HLE" if hle < t1 else "Same placement"
        put("hle_t1", "won" if winner == picks["hle_t1"]["pick"] else "lost", f"T1 placed {t1}, HLE {hle}.")
    else:
        put("hle_t1", "waiting", "Decided when both teams are out (or one wins).")
    mvp = res.get("finals_mvp")
    if mvp:
        put("mvp_role", "won" if mvp["role"] == picks["mvp_role"]["pick"] else "lost",
            f"Finals MVP: {mvp['player']} ({mvp['role']}).")
    else:
        put("mvp_role", "waiting", "Decided after the final.")

    (PROCESSED_DIR / "crystal_tracker.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for qid, q in out["questions"].items():
        print(f"{qid:13s} {q['status']:11s} pick {q['pick']}: {q['summary']}")


if __name__ == "__main__":
    main()
