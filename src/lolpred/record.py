"""Automatic prediction log and Worlds 2026 record (see docs/DECISIONS.md).

Runs on GitHub Actions every 30 minutes (scripts/auto_ledger.py), so nothing
depends on remembering to log a match:

1. Log: every Worlds match starting within LOG_AHEAD_H hours whose two teams
   are known gets a prediction appended to ledger/predictions.jsonl. The
   numbers come from site/data.json, i.e. the model the website was showing
   at that moment. The commit on GitHub timestamps it.
2. Results: each finished match is appended to ledger/results.jsonl with the
   winner of every game, read from the LoL Esports feed.
3. Live calls: for every finished game, the live model's chance at each game
   minute is worked out from the feed (same model and pre-game odds the site
   was using) and appended to ledger/live_calls.jsonl.

build_record() then grades everything (no network) into site/record.json,
which the Pages deploy builds. All three ledger files are append-only.
This module only needs the standard library (numpy only to make a prediction).
"""
import json
import math
import re
from datetime import datetime, timedelta, timezone

from . import livefeed
from .config import ROOT

LEDGER_DIR = ROOT / "ledger"
PREDICTIONS = LEDGER_DIR / "predictions.jsonl"
RESULTS = LEDGER_DIR / "results.jsonl"
LIVE_CALLS = LEDGER_DIR / "live_calls.jsonl"
SITE_DATA = ROOT / "site" / "data.json"
CONFIG = ROOT / "config" / "worlds2026.json"

SEASON_FROM = "2026-10-01"  # Worlds 2026, Play-In onward
LOG_AHEAD_H = 36            # log a match once its teams are known and it starts within this many hours
STAGE_NAMES = {"Play-Ins": "Play-In"}
STAGE_ORDER = ["Play-In", "Swiss", "Quarterfinals", "Semifinals", "Finals"]
CHECKPOINTS = [5, 10, 15, 20, 25, 30]
CALIB_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]

# The feed's team names differ from ours (e.g. "Cloud9 Kia"); codes are the fallback.
TEAM_ALIAS = {"hle": "Hanwha Life Esports", "gen": "Gen.G", "geng": "Gen.G", "dk": "Dplus KIA",
              "al": "Anyone's Legend", "blg": "Bilibili Gaming", "tes": "Top Esports", "ig": "Invictus Gaming",
              "g2": "G2 Esports", "mkoi": "Movistar KOI", "kc": "Karmine Corp", "tl": "Team Liquid",
              "tlaw": "Team Liquid", "c9": "Cloud9", "tsw": "Team Secret Whales", "cfo": "CTBC Flying Oyster",
              "mvk": "MVK Esports", "los": "LØS", "fur": "FURIA"}


# ---------- files ----------

def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path, entries: list[dict]) -> None:
    if not entries:
        return
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- pure helpers (tested) ----------

def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("ø", "o"))


def our_team(name, code, teams) -> str | None:
    """Our team name for a feed team ("Cloud9 Kia" -> "Cloud9"), else None."""
    n, c = norm(name), norm(code or "")
    for t in teams:
        o = norm(t)
        if o == n or o in n or (n in o and len(n) > 3):
            return t
    return TEAM_ALIAS.get(c) or TEAM_ALIAS.get(n)


def stage_name(block) -> str:
    return STAGE_NAMES.get(block, block or "")


def game_edge(blue_team: dict, red_team: dict) -> float:
    """Blue's end-of-game edge: inhibitors, towers, gold. The feed doesn't say
    who won a game; whoever is ahead on these at the end did (same rule as
    the website)."""
    b, r = blue_team, red_team
    return 3 * (b["inhibitors"] - r["inhibitors"]) + (b["towers"] - r["towers"]) + (b["totalGold"] - r["totalGold"]) / 3000


def assign_winners(a_edges: list[float], a_wins: int) -> list[bool]:
    """Which games team A won, given A's edge in each game and A's official
    number of game wins: A won its a_wins best games."""
    order = sorted(range(len(a_edges)), key=lambda i: -a_edges[i])
    won = [False] * len(a_edges)
    for i in order[:a_wins]:
        won[i] = True
    return won


def live_p_blue(live: dict, p_pre_blue: float, minutes: float, gold_diff: float, kill_diff: float, cs_diff: float) -> float:
    """The live model (src/lolpred/live.py) from the coefficients in site/data.json."""
    p = min(max(p_pre_blue, 1e-4), 1 - 1e-4)
    t = min(minutes / 10, live["max_t"])
    g, c = gold_diff / 1000, cs_diff / 10
    x = [math.log(p / (1 - p)), g, g * t, kill_diff, kill_diff * t, c, c * t]
    z = live["intercept"] + sum(w * v for w, v in zip(live["coef"], x))
    return 1 / (1 + math.exp(-z))


def make_prediction(data: dict, cfg: dict, a: str, b: str, best_of: int, stage: str, match_id: str, start: str) -> dict:
    """Same fields as lolpred.ledger.make_entry, from the published site data."""
    from .series import score_distribution, series_win_prob
    p = float(data["game_prob"][a][b])
    dist = score_distribution(p, best_of)
    ps = series_win_prob(p, best_of)
    (sa, sb), _ = max(dist.items(), key=lambda kv: kv[1])
    return {"predicted_at": now_utc(), "data_through": data["meta"]["data_through"], "stage": stage,
            "team_a": a, "team_b": b, "teamid_a": cfg["teams"][a]["teamid"], "teamid_b": cfg["teams"][b]["teamid"],
            "best_of": best_of, "p_game_a": round(p, 4), "p_series_a": round(ps, 4),
            "score_dist": {f"{x}-{y}": round(v, 4) for (x, y), v in dist.items()},
            "pick": a if ps >= 0.5 else b, "likely_score": f"{sa}-{sb}",
            "match_id": match_id, "match_start": start, "source": "auto"}


def grade(predictions: list[dict], results: list[dict], open_stages: set) -> list[dict]:
    """Each played match is graded against the latest prediction for those two
    teams made before its first game started. Predictions never used become
    "superseded" (a newer one for the same pair was graded or is waiting),
    "not played" (their stage is over) or stay "pending"."""
    out = [dict(e, status="pending") for e in predictions]
    used = set()
    for r in sorted(results, key=lambda r: r["start"]):
        cutoff = r.get("first_game_start") or r["start"]
        pair = {r["team_a"], r["team_b"]}
        cands = [i for i, e in enumerate(out) if i not in used and {e["team_a"], e["team_b"]} == pair
                 and e["predicted_at"] < cutoff]
        if not cands:
            continue
        i = max(cands, key=lambda j: (out[j]["predicted_at"], j))
        used.add(i)
        e = out[i]
        flip = e["team_a"] != r["team_a"]
        aw, bw = (r["b_wins"], r["a_wins"]) if flip else (r["a_wins"], r["b_wins"])
        winner = e["team_a"] if aw > bw else e["team_b"]
        e.update(status="graded", played_at=r["start"], match_id_played=r["match_id"], a_wins=aw, b_wins=bw,
                 winner=winner, correct=winner == e["pick"], score_correct=f"{aw}-{bw}" == e["likely_score"],
                 p_actual_score=e["score_dist"].get(f"{aw}-{bw}"),
                 game_results=[int(g["winner"] == e["team_a"]) for g in r["games"]])
    for i, e in enumerate(out):
        if e["status"] != "pending":
            continue
        pair = {e["team_a"], e["team_b"]}
        if any(j != i and {f["team_a"], f["team_b"]} == pair and f["predicted_at"] >= e["predicted_at"]
               and (f["status"] == "graded" or j > i) for j, f in enumerate(out)):
            e["status"] = "superseded"
        elif e["stage"] not in open_stages:
            e["status"] = "not played"
    return out


def summary(graded: list[dict]) -> dict:
    done = [e for e in graded if e["status"] == "graded"]
    if not done:
        return {"series": 0}
    pairs = [(e["p_game_a"], y) for e in done for y in e["game_results"]]
    return {"series": len(done), "series_correct": sum(e["correct"] for e in done),
            "scores_exact": sum(e["score_correct"] for e in done), **_scores(pairs, "game")}


def _scores(pairs, prefix) -> dict:
    """Accuracy, log loss and Brier for (probability, outcome) pairs."""
    if not pairs:
        return {f"{prefix}s": 0}
    n = len(pairs)
    clip = lambda p: min(max(p, 1e-4), 1 - 1e-4)  # noqa: E731
    return {f"{prefix}s": n,
            f"{prefix}_accuracy": sum((p >= 0.5) == (y == 1) for p, y in pairs) / n,
            f"{prefix}_log_loss": -sum(y * math.log(clip(p)) + (1 - y) * math.log(1 - clip(p)) for p, y in pairs) / n,
            f"{prefix}_brier": sum((p - y) ** 2 for p, y in pairs) / n}


def live_summary(calls: list[dict]) -> dict:
    """How the live model's calls held up: by game minute and by how sure it was."""
    if not calls:
        return {"games": 0}
    by_minute = []
    for m in [0] + CHECKPOINTS:
        pairs = []
        for g in calls:
            p = g["p_pre_blue"] if m == 0 else dict(g["calls"]).get(m)
            if p is not None:
                pairs.append((p, int(g["blue_won"])))
        if pairs:
            by_minute.append({"minute": m, **_scores(pairs, "call")})
    every = [(p, int(g["blue_won"])) for g in calls for _, p in g["calls"]]
    calib = []
    for lo, hi in zip(CALIB_BINS[:-1], CALIB_BINS[1:]):
        fav = [(max(p, 1 - p), int((p >= 0.5) == (y == 1))) for p, y in every if lo <= max(p, 1 - p) < hi]
        if fav:
            calib.append({"from": lo, "to": min(hi, 1.0), "n": len(fav),
                          "said": sum(p for p, _ in fav) / len(fav), "won": sum(w for _, w in fav) / len(fav)})
    return {"games": len(calls), **_scores(every, "call"), "by_minute": by_minute, "calibration": calib,
            "recent": [{k: g[k] for k in ("match_id", "number", "blue", "red", "blue_won", "p_pre_blue", "calls")}
                       for g in calls[-6:]]}


def build_record(open_stages: set | None = None) -> dict:
    """Grade everything in the ledger (no network)."""
    preds, results, calls = read_jsonl(PREDICTIONS), read_jsonl(RESULTS), read_jsonl(LIVE_CALLS)
    if open_stages is None:
        # A stage is over once a later stage has a result.
        last = max((STAGE_ORDER.index(r["stage"]) for r in results if r["stage"] in STAGE_ORDER), default=0)
        open_stages = set(STAGE_ORDER[last:])
    g = grade(preds, results, open_stages)
    return {"updated": now_utc(), "ledger": {"summary": summary(g), "entries": g},
            "results": results, "live": live_summary(calls)}


# ---------- feed (network) ----------

def worlds_events(data: dict) -> list[dict]:
    d = livefeed._get(f"{livefeed.API}/getSchedule?hl=en-US&leagueId={data['live']['worlds_league_id']}")
    evs = (d or {}).get("data", {}).get("schedule", {}).get("events", [])
    return [e for e in evs if e.get("type") == "match" and e["startTime"] >= SEASON_FROM]


def log_upcoming(data: dict, cfg: dict, events: list[dict], existing: list[dict], now: datetime) -> list[dict]:
    teams = [t["name"] for t in data["teams"]]
    have = {(e.get("match_id"), e["data_through"]) for e in existing}
    new = []
    for e in events:
        start = livefeed.parse_ts(e["startTime"])
        if e["state"] != "unstarted" or start > now + timedelta(hours=LOG_AHEAD_H):
            continue
        a, b = (our_team(t["name"], t.get("code"), teams) for t in e["match"]["teams"])
        if not a or not b or a == b or (e["match"]["id"], data["meta"]["data_through"]) in have:
            continue
        new.append(make_prediction(data, cfg, a, b, e["match"]["strategy"]["count"], stage_name(e.get("blockName")),
                                   e["match"]["id"], e["startTime"]))
    return new


def match_result(data: dict, e: dict) -> dict | None:
    """Final score and the winner of every game of a finished match."""
    teams = [t["name"] for t in data["teams"]]
    det = livefeed.event_details(e["match"]["id"])
    if not det:
        return None
    ta, tb = det["match"]["teams"]
    A, B = our_team(ta["name"], ta.get("code"), teams), our_team(tb["name"], tb.get("code"), teams)
    if not A or not B:
        return None
    aw, bw = (ta.get("result") or {}).get("gameWins", 0), (tb.get("result") or {}).get("gameWins", 0)
    after = datetime.now(timezone.utc) - timedelta(seconds=90)
    games = []
    for g in sorted((g for g in det["match"]["games"] if g["state"] == "completed"), key=lambda g: g["number"]):
        w = livefeed.final_frames(g["id"], after)
        # The event's side list can be swapped (it was for all five games of the 2025 final);
        # the feed's own metadata, which lists the players on each side, is right.
        if w:
            a_blue = w["gameMetadata"]["blueTeamMetadata"]["esportsTeamId"] == ta["id"]
        else:
            a_blue = next((t["side"] for t in g.get("teams", []) if t["id"] == ta["id"]), "blue") == "blue"
        row = {"game_id": g["id"], "number": g["number"], "blue": A if a_blue else B, "red": B if a_blue else A,
               "a_edge": 0.0, "start": None, "end": None}
        if w:
            f = w["frames"][-1]
            edge = game_edge(f["blueTeam"], f["redTeam"])
            end = livefeed.parse_ts(f["rfc460Timestamp"])
            start = livefeed.game_start(g["id"], end - timedelta(minutes=1))
            row.update(a_edge=round(edge if a_blue else -edge, 3), start=iso(start), end=iso(end))
        games.append(row)
    for row, won in zip(games, assign_winners([r["a_edge"] for r in games], aw)):
        row["winner"] = A if won else B
    starts = [r["start"] for r in games if r["start"]]
    return {"match_id": e["match"]["id"], "start": e["startTime"], "first_game_start": min(starts) if starts else None,
            "stage": stage_name(e.get("blockName")), "best_of": det["match"]["strategy"]["count"],
            "team_a": A, "team_b": B, "a_wins": aw, "b_wins": bw, "winner": A if aw > bw else B,
            "games": games, "recorded_at": now_utc()}


def game_live_calls(data: dict, result: dict, game: dict) -> dict | None:
    """The live model's chance for blue at every game minute."""
    if not game.get("start"):
        return None
    live = data["live"]
    p_pre = live["p_blue"][game["blue"]][game["red"]]
    start, end = livefeed.parse_ts(game["start"]), livefeed.parse_ts(game["end"])
    calls, m = [], 1
    while start + timedelta(minutes=m) < end:
        w = livefeed.window(game["game_id"], start + timedelta(minutes=m))
        if w:
            s = livefeed.frame_state(w["frames"][-1])
            calls.append([m, round(live_p_blue(live, p_pre, m, s["gold_diff"], s["kill_diff"], s["cs_diff"]), 4)])
        m += 1
    return {"game_id": game["game_id"], "match_id": result["match_id"], "number": game["number"],
            "blue": game["blue"], "red": game["red"], "blue_won": game["winner"] == game["blue"],
            "p_pre_blue": p_pre, "calls": calls, "model_data_through": data["meta"]["data_through"],
            "computed_at": now_utc()}


def update(log=print) -> dict:
    """One scheduled run: log upcoming matches, record finished ones, work out live calls."""
    data = json.loads(SITE_DATA.read_text())
    cfg = json.loads(CONFIG.read_text())
    events = worlds_events(data)
    now = datetime.now(timezone.utc)

    new_preds = log_upcoming(data, cfg, events, read_jsonl(PREDICTIONS), now)
    append_jsonl(PREDICTIONS, new_preds)
    for p in new_preds:
        log(f"logged {p['team_a']} vs {p['team_b']} Bo{p['best_of']} ({p['stage']}): {p['pick']} "
            f"{max(p['p_series_a'], 1 - p['p_series_a']):.0%}, likely {p['likely_score']}")

    done = {r["match_id"] for r in read_jsonl(RESULTS)}
    new_results = []
    for e in events:
        if e["state"] == "completed" and e["match"]["id"] not in done:
            r = match_result(data, e)
            if r:
                new_results.append(r)
                log(f"result {r['team_a']} {r['a_wins']}-{r['b_wins']} {r['team_b']} ({r['stage']})")
    append_jsonl(RESULTS, new_results)

    have_calls = {c["game_id"] for c in read_jsonl(LIVE_CALLS)}
    new_calls = []
    for r in new_results:
        for g in r["games"]:
            if g["game_id"] not in have_calls:
                c = game_live_calls(data, r, g)
                if c:
                    new_calls.append(c)
                    log(f"live calls {g['blue']} vs {g['red']} game {g['number']}: {len(c['calls'])} minutes")
    append_jsonl(LIVE_CALLS, new_calls)
    return {"predictions": len(new_preds), "results": len(new_results), "live_games": len(new_calls)}
