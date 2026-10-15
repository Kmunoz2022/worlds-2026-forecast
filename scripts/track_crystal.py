"""Track Crystal Ball picks against Worlds 2026 as it happens (see docs/DECISIONS.md).

Two sets of picks are scored on the same measurements:
  model     ledger/crystal_ball_locked.json + ledger/crystal_ball_locked_extra.json
  personal  ledger/crystal_ball_personal.json (the user's own submitted picks)
Locked files are only read, never written.

Every question is measured once from the Worlds main-event games in the data,
then each pick is marked: not started / on track / behind / tied / won / lost /
waiting. Questions the game data can't answer (knockout teams, placements,
Finals MVP) come from config/worlds2026_results.json.
Writes Data/processed/crystal_tracker.json.

Usage: .venv/bin/python scripts/track_crystal.py [--rebuild]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.crystal import load_player_games  # noqa: E402
from lolpred.draft import load_draft_events  # noqa: E402
from lolpred.splits import worlds_start  # noqa: E402

LEDGER = ROOT / "ledger"
CONFIG = ROOT / "config" / "worlds2026.json"
RESULTS = ROOT / "config" / "worlds2026_results.json"
SEASON = 2026
FAKER_CAREER, FAKER_TARGET = 616, 700
UNIQUE_BINS = {"0-99": (0, 99), "100-104": (100, 104), "105-109": (105, 109), "110-114": (110, 114),
               "115-119": (115, 119), "120+": (120, 10**6)}
# Short names people type for teams.
TEAM_ALIASES = {"hle": "Hanwha Life Esports", "hanwha": "Hanwha Life Esports", "gen": "Gen.G", "geng": "Gen.G",
                "dk": "Dplus KIA", "dplus": "Dplus KIA", "al": "Anyone's Legend", "blg": "Bilibili Gaming",
                "tes": "Top Esports", "ig": "Invictus Gaming", "g2": "G2 Esports", "mkoi": "Movistar KOI",
                "koi": "Movistar KOI", "kc": "Karmine Corp", "tl": "Team Liquid", "c9": "Cloud9",
                "tsw": "Team Secret Whales", "cfo": "CTBC Flying Oyster", "ctbc": "CTBC Flying Oyster",
                "mvk": "MVK Esports", "los": "LØS", "lgs": "LØS", "fur": "FURIA"}


def norm(s):
    return re.sub(r"[^a-z0-9ø]", "", str(s).lower())


def team_name(s, cfg):
    """Config team name for whatever the user typed ('HLE', 'Hanwha Life', ...)."""
    n = norm(s)
    for t in cfg["teams"]:
        if norm(t) == n:
            return t
    return TEAM_ALIASES.get(n, s)


def load_picks():
    model = {}
    for f in ("crystal_ball_locked.json", "crystal_ball_locked_extra.json"):
        path = LEDGER / f
        if path.exists():
            for q in json.loads(path.read_text())["questions"]:
                model[q["id"]] = {"question": q["question"], "pick": q["pick"]}
    personal = None
    path = LEDGER / "crystal_ball_personal.json"
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
    M["kills"] = pl["k"].to_dict()
    M["teemo"] = int((w["champion"] == "Teemo").sum())
    M["unique"] = int(w["champion"].nunique())
    M["before32"] = int((games["gamelength"] < 1920).sum())
    M["after32"] = int(len(games) - M["before32"])
    for key, stat in (("max_kills", "kills"), ("max_cs", "total cs")):
        best = w.loc[w[stat].idxmax()]
        M[key] = (best["playername"], int(best[stat]))
    sg = games.loc[games["gamelength"].idxmin()]
    M["shortest"] = (sg["gamelength"] / 60, w[(w["gameid"] == sg["gameid"]) & (w["result"] == 1)]["teamname"].iat[0])
    M["ko"] = res.get("knockout_teams") or []
    M["placements"] = res.get("placements") or {}
    M["mvp"] = res.get("finals_mvp")
    return M


FMT = {"presence": lambda v: f"{v:.0%}", "champ_wr": lambda v: f"{v:.0%}", "champ_deaths": lambda v: f"{v:.2f}",
       "kda": lambda v: f"{v:.2f}", "dmg": lambda v: f"{v:,.0f}", "kp": lambda v: f"{v:.0%}"}


def ranked(qid, rows, pick, finished):
    rows = sorted(rows, key=lambda r: -r["value"])
    fmt = FMT[qid]
    top = [{"name": r["name"], "value": fmt(r["value"]), "games": r["games"]} for r in rows[:3]]
    key = norm(pick.split(" (")[0])
    mine = next(((i + 1, r) for i, r in enumerate(rows) if norm(r["name"]) == key), None)
    now = None if mine is None else {"rank": mine[0], "value": fmt(mine[1]["value"]), "games": mine[1]["games"]}
    if not rows:
        return "waiting", "No one qualifies yet (champions need 5 games).", top, now
    tied_top = mine is not None and mine[1]["value"] == rows[0]["value"]
    status = ("won" if tied_top else "lost") if finished else \
             ("on track" if tied_top else "behind")
    lead = f"Leader: {top[0]['name']} {top[0]['value']} ({top[0]['games']} games)."
    if mine is None:
        lead += " Your pick isn't on the board yet." if qid in ("champ_wr", "champ_deaths") else ""
    return status, lead, top, now


def score(qid, pick, M, cfg, finished):
    """(status, summary, leaders, pick_now) for one pick."""
    yes = norm(pick) in ("yes", "y")
    if qid in FMT:
        return ranked(qid, M[qid], pick, finished)
    if qid == "peyz_guma":
        p, g = int(M["kills"].get("Peyz", 0)), int(M["kills"].get("Gumayusi", 0))
        lead = "peyz" if p > g else "gumayusi" if g > p else None
        mine = norm(pick)
        st = ("won" if lead == mine else "lost") if finished else ("tied" if lead is None else
                                                                   "on track" if lead == mine else "behind")
        return st, f"Peyz {p} kills, Gumayusi {g}.", None, None
    if qid == "faker_700":
        total = FAKER_CAREER + int(M["kills"].get("Faker", 0))
        reached = total >= FAKER_TARGET
        if reached:
            st = "won" if yes else "lost"
        else:
            st = ("lost" if yes else "won") if finished else ("on track" if not yes else "behind")
        return st, f"Faker is at {total} career Worlds kills ({max(FAKER_TARGET - total, 0)} to go).", None, None
    if qid == "teemo":
        if M["teemo"]:
            return ("won" if yes else "lost"), f"Teemo picked {M['teemo']} times.", None, None
        return (("lost" if yes else "won") if finished else ("behind" if yes else "on track")), \
            "Teemo hasn't been picked.", None, None
    if qid == "unique":
        lo, hi = UNIQUE_BINS.get(pick.replace(" ", ""), (None, None))
        u = M["unique"]
        if lo is None:
            return "waiting", f"{u} different champions so far (answer '{pick}' not recognised).", None, None
        st = "lost" if u > hi else (("won" if u >= lo else "lost") if finished else ("on track" if u >= lo else "waiting"))
        return st, f"{u} different champions picked so far.", None, None
    if qid == "len32":
        b, a = M["before32"], M["after32"]
        lead = "before" if b > a else "after" if a > b else None
        mine = "before" if norm(pick).startswith("before") else "after"
        st = ("won" if lead == mine else "lost") if finished else ("tied" if lead is None else
                                                                   "on track" if lead == mine else "behind")
        return st, f"{b} games ended before 32:00, {a} after.", None, None
    if qid in ("kills15", "cs450"):
        who, val = M["max_kills" if qid == "kills15" else "max_cs"]
        hit = val >= (15 if qid == "kills15" else 450)
        st = ("won" if yes else "lost") if hit else (("lost" if yes else "won") if finished else
                                                    ("waiting" if yes else "on track"))
        return st, f"Best so far: {who} with {val} {'kills' if qid == 'kills15' else 'CS'}.", None, None
    if qid == "shortest":
        minutes, winner = M["shortest"]
        mine = team_name(pick, cfg)
        st = ("won" if winner == mine else "lost") if finished else ("on track" if winner == mine else "behind")
        return st, f"Shortest so far: {minutes:.1f} min, won by {winner}.", None, None
    if qid in ("regions_ko", "lcs_ko"):
        if len(M["ko"]) != 8:
            return "waiting", "Decided when the Swiss stage ends.", None, None
        regions = {cfg["teams"][t]["region"] for t in M["ko"]}
        if qid == "regions_ko":
            return ("won" if str(len(regions)) == str(pick).strip() else "lost"), \
                f"{len(regions)} regions in knockouts: {', '.join(sorted(regions))}.", None, None
        has = "LCS" in regions
        return ("won" if has == yes else "lost"), ("An LCS team made knockouts." if has else
                                                   "No LCS team made knockouts."), None, None
    if qid == "hle_t1":
        p = M["placements"]
        if "T1" not in p or "Hanwha Life Esports" not in p:
            return "waiting", "Decided when both teams are out (or one wins).", None, None
        t1, hle = p["T1"], p["Hanwha Life Esports"]
        actual = "t1" if t1 < hle else "hle" if hle < t1 else "sameplacement"
        mine = norm(team_name(pick, cfg)) if norm(pick) not in ("sameplacement", "same") else "sameplacement"
        mine = {"hanwhalifeesports": "hle"}.get(mine, mine)
        return ("won" if mine == actual else "lost"), f"T1 placed {t1}, HLE {hle}.", None, None
    if qid == "mvp_role":
        if not M["mvp"]:
            return "waiting", "Decided after the final.", None, None
        return ("won" if norm(M["mvp"]["role"]) == norm(pick) else "lost"), \
            f"Finals MVP: {M['mvp']['player']} ({M['mvp']['role']}).", None, None
    return "waiting", "", None, None


def tally(qs):
    out = {}
    for q in qs.values():
        out[q["status"]] = out.get(q["status"], 0) + 1
    return out


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
    out = {"locked_at": locked_at, "games": len(ids), "started": bool(ids), "finished": finished}

    def run(picks):
        if picks is None:
            return None
        if not ids:
            return {q: {"pick": v["pick"], "status": "not started", "summary": "Worlds hasn't started in the data yet.",
                        "leaders": [], "pick_now": None} for q, v in picks.items()}
        out_q = {}
        for q, v in picks.items():
            st, summary, leaders, now = score(q, str(v["pick"]), M, cfg, finished)
            out_q[q] = {"pick": v["pick"], "status": st, "summary": summary, "leaders": leaders or [], "pick_now": now}
        return out_q

    if ids:
        pg = load_player_games(rebuild=args.rebuild)
        ev = load_draft_events(rebuild=args.rebuild)
        games = pg[pg["gameid"].isin(ids)].drop_duplicates("gameid")
        M = measure(ids, pg, ev, games, res, cfg)
    out["model"] = run(model_picks)
    out["personal"] = run(personal_picks)
    out["tally"] = {"model": tally(out["model"]), "personal": tally(out["personal"]) if out["personal"] else None}
    (PROCESSED_DIR / "crystal_tracker.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"games {len(ids)} | model {out['tally']['model']} | personal {out['tally']['personal']}")
    if ids:
        for q, v in out["model"].items():
            mine = (out["personal"] or {}).get(q)
            print(f"{q:13s} model {v['status']:10s} {v['pick']!s:30.30s} "
                  + (f"| you {mine['status']:10s} {mine['pick']!s:20.20s}" if mine else "") + f" | {v['summary']}")


if __name__ == "__main__":
    main()
