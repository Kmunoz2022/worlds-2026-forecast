"""Scoring Crystal Ball picks as Worlds 2026 goes on (see docs/DECISIONS.md).

Two sources measure the same things:
- the live feed, after every game (measure_feed, from ledger/game_stats.jsonl),
  run by the scheduled GitHub job: no CSV needed;
- Oracle's Elixir, after each data refresh (scripts/track_crystal.py). It is
  the official record and the only source for presence (needs bans) and
  damage per game (the feed only has each player's share of team damage).

Each pick gets a status: not started / on track / behind / tied / won / lost /
waiting. Standard library only.
"""
import re

FAKER_CAREER, FAKER_TARGET = 616, 700
UNIQUE_BINS = {"0-99": (0, 99), "100-104": (100, 104), "105-109": (105, 109), "110-114": (110, 114),
               "115-119": (115, 119), "120+": (120, 10**6)}
LEN_THRESHOLD_S = 32 * 60
MIN_CHAMP_GAMES = 5
# Questions the feed can't answer; they keep the last Oracle's Elixir result.
CSV_ONLY = {"presence", "dmg"}
CSV_ONLY_WHY = {"presence": "Presence counts bans, and the live feed has no bans.",
                "dmg": "The live feed only has each player's share of team damage, not totals."}
# Short names people type for teams.
TEAM_ALIASES = {"hle": "Hanwha Life Esports", "hanwha": "Hanwha Life Esports", "gen": "Gen.G", "geng": "Gen.G",
                "dk": "Dplus KIA", "dplus": "Dplus KIA", "al": "Anyone's Legend", "blg": "Bilibili Gaming",
                "tes": "Top Esports", "ig": "Invictus Gaming", "g2": "G2 Esports", "mkoi": "Movistar KOI",
                "koi": "Movistar KOI", "kc": "Karmine Corp", "tl": "Team Liquid", "c9": "Cloud9",
                "tsw": "Team Secret Whales", "cfo": "CTBC Flying Oyster", "ctbc": "CTBC Flying Oyster",
                "mvk": "MVK Esports", "los": "LØS", "lgs": "LØS", "fur": "FURIA"}
FMT = {"presence": lambda v: f"{v:.0%}", "champ_wr": lambda v: f"{v:.0%}", "champ_deaths": lambda v: f"{v:.2f}",
       "kda": lambda v: f"{v:.2f}", "dmg": lambda v: f"{v:,.0f}", "kp": lambda v: f"{v:.0%}"}


def norm(s):
    return re.sub(r"[^a-z0-9ø]", "", str(s).lower())


def team_name(s, teams):
    """Config team name for whatever was typed ('HLE', 'Hanwha Life', ...)."""
    n = norm(s)
    for t in teams:
        if norm(t) == n:
            return t
    return TEAM_ALIASES.get(n, s)


def ranked(qid, rows, pick, finished):
    rows = sorted(rows, key=lambda r: -r["value"])
    fmt = FMT[qid]
    top = [{"name": r["name"], "value": fmt(r["value"]), "games": r["games"]} for r in rows[:3]]
    key = norm(pick.split(" (")[0])
    mine = next(((i + 1, r) for i, r in enumerate(rows) if norm(r["name"]) == key), None)
    now = None if mine is None else {"rank": mine[0], "value": fmt(mine[1]["value"]), "games": mine[1]["games"]}
    if not rows:
        return "waiting", f"No one qualifies yet (champions need {MIN_CHAMP_GAMES} games).", top, now
    tied_top = mine is not None and mine[1]["value"] == rows[0]["value"]
    status = ("won" if tied_top else "lost") if finished else ("on track" if tied_top else "behind")
    lead = f"Leader: {top[0]['name']} {top[0]['value']} ({top[0]['games']} games)."
    if mine is None:
        lead += " Your pick isn't on the board yet." if qid in ("champ_wr", "champ_deaths") else ""
    return status, lead, top, now


def score(qid, pick, M, teams, team_regions, finished):
    """(status, summary, leaders, pick_now) for one pick. M: measurements;
    M["kills"] is keyed by norm(player name)."""
    yes = norm(pick) in ("yes", "y")
    if qid in FMT:
        return ranked(qid, M[qid], pick, finished)
    if qid == "peyz_guma":
        p, g = int(M["kills"].get("peyz", 0)), int(M["kills"].get("gumayusi", 0))
        lead = "peyz" if p > g else "gumayusi" if g > p else None
        mine = norm(pick)
        st = ("won" if lead == mine else "lost") if finished else ("tied" if lead is None else
                                                                   "on track" if lead == mine else "behind")
        return st, f"Peyz {p} kills, Gumayusi {g}.", None, None
    if qid == "faker_700":
        total = FAKER_CAREER + int(M["kills"].get("faker", 0))
        if total >= FAKER_TARGET:
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
        mine = team_name(pick, teams)
        st = ("won" if winner == mine else "lost") if finished else ("on track" if winner == mine else "behind")
        m, s = divmod(round(minutes * 60), 60)
        return st, f"Shortest so far: {m}:{s:02d}, won by {winner}.", None, None
    if qid in ("regions_ko", "lcs_ko"):
        if len(M["ko"]) != 8:
            return "waiting", "Decided when the Swiss stage ends.", None, None
        regions = {team_regions[t] for t in M["ko"]}
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
        mine = norm(team_name(pick, teams)) if norm(pick) not in ("sameplacement", "same") else "sameplacement"
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


def score_all(picks, M, teams, team_regions, finished, n_games):
    if not n_games:
        return {q: {"pick": v["pick"], "status": "not started", "summary": "Worlds hasn't started yet.",
                    "leaders": [], "pick_now": None} for q, v in picks.items()}
    out = {}
    for q, v in picks.items():
        st, summary, leaders, now = score(q, str(v["pick"]), M, teams, team_regions, finished)
        out[q] = {"pick": v["pick"], "status": st, "summary": summary, "leaders": leaders or [], "pick_now": now}
    return out


def results_facts(res: dict) -> dict:
    """Tournament facts from config/worlds2026_results.json."""
    return {"ko": res.get("knockout_teams") or [], "placements": res.get("placements") or {},
            "mvp": res.get("finals_mvp")}


# ---------- measurements from the live feed ----------

def measure_feed(games: list[dict], res: dict) -> dict:
    """Measurements from ledger/game_stats.jsonl rows (one per game)."""
    players = [p for g in games for p in g["players"]]
    M = {}
    champs = {}
    for g in games:
        for p in g["players"]:
            c = champs.setdefault(p["champion"], {"games": 0, "wins": 0, "deaths": 0})
            c["games"] += 1
            c["wins"] += int(p["team"] == g["winner"])
            c["deaths"] += p["deaths"]
    ok = {c: v for c, v in champs.items() if v["games"] >= MIN_CHAMP_GAMES}
    M["champ_wr"] = [{"name": c, "value": v["wins"] / v["games"], "games": v["games"]} for c, v in ok.items()]
    M["champ_deaths"] = [{"name": c, "value": v["deaths"] / v["games"], "games": v["games"]} for c, v in ok.items()]
    per = {}
    for g in games:
        team_kills = {}
        for p in g["players"]:
            team_kills[p["team"]] = team_kills.get(p["team"], 0) + p["kills"]
        for p in g["players"]:
            r = per.setdefault(p["name"], {"games": 0, "k": 0, "d": 0, "a": 0, "tk": 0})
            r["games"] += 1
            r["k"] += p["kills"]
            r["d"] += p["deaths"]
            r["a"] += p["assists"]
            r["tk"] += team_kills[p["team"]]
    M["kda"] = [{"name": n, "value": (r["k"] + r["a"]) / max(r["d"], 1), "games": r["games"]} for n, r in per.items()]
    M["kp"] = [{"name": n, "value": (r["k"] + r["a"]) / max(r["tk"], 1), "games": r["games"]} for n, r in per.items()]
    M["kills"] = {norm(n): r["k"] for n, r in per.items()}
    M["teemo"] = sum(p["champion"] == "Teemo" for p in players)
    M["unique"] = len({p["champion"] for p in players})
    M["before32"] = sum(g["duration_s"] < LEN_THRESHOLD_S for g in games)
    M["after32"] = len(games) - M["before32"]
    if players:
        best_k = max(players, key=lambda p: p["kills"])
        best_cs = max(players, key=lambda p: p["cs"])
        M["max_kills"], M["max_cs"] = (best_k["name"], best_k["kills"]), (best_cs["name"], best_cs["cs"])
        sg = min(games, key=lambda g: g["duration_s"])
        M["shortest"] = (sg["duration_s"] / 60, sg["winner"])
    return M | results_facts(res)


def feed_tracker(picks: dict, games: list[dict], res: dict, teams: list[str], team_regions: dict,
                 csv_tracker: dict | None) -> dict:
    """Tracker in the same shape as track_crystal.py's, from the feed. Presence
    and damage keep the last Oracle's Elixir result (dated)."""
    finished = bool(res.get("finished"))
    M = measure_feed(games, res)
    feed_picks = {q: v for q, v in picks.items() if q not in CSV_ONLY}
    model = score_all(feed_picks, M, teams, team_regions, finished, len(games))
    csv_model = (csv_tracker or {}).get("model") or {}
    through = (csv_tracker or {}).get("data_through")
    for q in CSV_ONLY & set(picks):
        old = csv_model.get(q)
        if old and old["status"] != "not started":
            note = f"From the official data through {through} ({csv_tracker['games']} games); updated daily."
            model[q] = dict(old, summary=f"{old['summary']} {note}")
        else:
            model[q] = {"pick": picks[q]["pick"], "status": "waiting" if games else "not started",
                        "summary": CSV_ONLY_WHY[q] + " It comes from the daily official data, which isn't in yet.",
                        "leaders": [], "pick_now": None}
    model = {q: model[q] for q in picks}  # keep the question order
    return {"games": len(games), "started": bool(games), "finished": finished, "source": "live feed",
            "model": model, "tally": {"model": tally(model)}}
