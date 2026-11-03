"""Archive minute-by-minute game timelines from the LoL Esports feed (see docs/DECISIONS.md).

Oracle's Elixir has each game's totals plus snapshots at 10/15/20/25 minutes.
The feed has the whole game: this saves one sample per game-minute with each
team's gold, kills, towers, dragons, barons and inhibitors, and each player's
gold, CS, level, kills/deaths/assists, items, wards, damage share and kill
participation. The feed keeps games for at least a year, so past events can be
backfilled too.

Files: Data/timelines/<game id>.json.gz (git-ignored). Already-archived games
are skipped, so it's safe to run repeatedly (refresh.py runs it).

Usage:
  .venv/bin/python scripts/archive_timelines.py                   # finished Worlds 2026 games
  .venv/bin/python scripts/archive_timelines.py --since 2025-10-01 # also older Worlds games
  .venv/bin/python scripts/archive_timelines.py --league msi --since 2026-01-01
"""
import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lolpred import livefeed  # noqa: E402

OUT = ROOT / "Data" / "timelines"
LEAGUE_IDS = {"worlds": "98767975604431411", "msi": "98767991325878492", "first_stand": "113464388705111224",
              "lck": "98767991310872058", "lpl": "98767991314006698", "lec": "98767991302996019",
              "lcs": "98767991299243165", "cblol-brazil": "98767991332355509", "lcp": "113476371197627891"}
PAUSE = 0.25  # seconds between requests: the feed is unofficial, be gentle
PLAYER_KEYS = ["totalGoldEarned", "creepScore", "level", "kills", "deaths", "assists", "wardsPlaced",
               "wardsDestroyed", "championDamageShare", "killParticipation"]


def get(url):
    time.sleep(PAUSE)
    return livefeed._get(url)


def events(league, since):
    """Matches since `since`, paging back through the schedule (about 80 per page)."""
    evs, token = [], None
    for _ in range(40):
        url = f"{livefeed.API}/getSchedule?hl=en-US&leagueId={LEAGUE_IDS[league]}" + (f"&pageToken={token}" if token else "")
        sc = ((get(url) or {}).get("data") or {}).get("schedule") or {}
        page = sc.get("events", [])
        evs += page
        token = (sc.get("pages") or {}).get("older")
        if not token or not page or min(e["startTime"] for e in page) < since:
            break
    # Matches still being played count too: their finished games are archived.
    seen, out = set(), []
    for e in evs:
        mid = (e.get("match") or {}).get("id")
        if e.get("type") == "match" and e["state"] in ("completed", "inProgress") and e["startTime"] >= since and mid not in seen:
            seen.add(mid)
            out.append(e)
    return out


LITE = False  # --lite: team totals and the window's player stats only (no items/wards request)


def sample(gid, at):
    """One sample at wall-clock time `at`: team totals plus per-player stats."""
    w = livefeed.window(gid, at)
    if not w:
        return None
    f = w["frames"][-1]
    teams = {side: {k: (len(f[f"{side}Team"][k]) if k == "dragons" else f[f"{side}Team"][k])
                    for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons", "dragons")}
             for side in ("blue", "red")}
    teams["blue"]["dragon_types"] = f["blueTeam"]["dragons"]
    teams["red"]["dragon_types"] = f["redTeam"]["dragons"]
    players = {}
    if LITE:
        for side in ("blue", "red"):
            for p in f[f"{side}Team"]["participants"]:
                players[p["participantId"]] = {k: p.get(k) for k in ("totalGold", "level", "kills", "deaths", "assists",
                                                                     "creepScore", "currentHealth", "maxHealth")}
        return {"t": f["rfc460Timestamp"], "state": f["gameState"], "teams": teams, "players": players}
    d = get(f"{livefeed.FEED}/details/{gid}?startingTime={livefeed.ts(at)}")
    if d and d.get("frames"):
        for p in d["frames"][-1]["participants"]:
            players[p["participantId"]] = {k: p.get(k) for k in PLAYER_KEYS} | {"items": p.get("items", [])}
    return {"t": f["rfc460Timestamp"], "state": f["gameState"], "teams": teams, "players": players}


def archive_game(event, game):
    gid = game["id"]
    path = OUT / f"{gid}.json.gz"
    if path.exists():
        return "skip"
    end_w = livefeed.final_frames(gid, datetime.now(timezone.utc) - timedelta(seconds=90))
    if not end_w:
        return "no stats"
    meta = end_w["gameMetadata"]
    end = livefeed.parse_ts(end_w["frames"][-1]["rfc460Timestamp"])
    start = livefeed.game_start(gid, end - timedelta(minutes=1))
    samples, m = [], 0
    while start + timedelta(minutes=m) < end:
        s = sample(gid, start + timedelta(minutes=m))
        if s:
            s["minute"] = m
            samples.append(s)
        m += 1
    last = end_w["frames"][-1]
    out = {
        "game_id": gid, "match_id": event["match"]["id"], "game_number": game["number"],
        "league": event["league"]["slug"], "block": event.get("blockName"), "match_start": event["startTime"],
        "game_start": start.isoformat(), "game_end": end.isoformat(), "patch": meta.get("patchVersion"),
        "teams": {side: {"esports_team_id": meta[f"{side}TeamMetadata"]["esportsTeamId"],
                         "players": meta[f"{side}TeamMetadata"]["participantMetadata"]} for side in ("blue", "red")},
        "match_teams": [{"id": t["id"], "name": t["name"], "code": t.get("code")} for t in event["match"]["teams"]],
        "final": {side: {k: (len(last[f"{side}Team"][k]) if k == "dragons" else last[f"{side}Team"][k])
                         for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons", "dragons")}
                  for side in ("blue", "red")},
        "samples": samples,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump(out, fh)
    return f"saved {len(samples)} minutes ({(end - start).total_seconds() / 60:.0f} min game)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", nargs="+", default=["worlds"], choices=sorted(LEAGUE_IDS))
    ap.add_argument("--since", default="2026-01-01")
    ap.add_argument("--lite", action="store_true", help="skip per-player items/wards (half the requests; for backfills)")
    ap.add_argument("--out", help="save into this folder instead of Data/timelines")
    args = ap.parse_args()
    global LITE, OUT
    LITE = args.lite
    if args.out:
        OUT = Path(args.out)
    for league in args.league:
        for e in events(league, args.since):
            det = livefeed.event_details(e["match"]["id"])
            if det:
                e["match"]["teams"] = det["match"]["teams"]   # the schedule's team entries have no ids
            teams = " vs ".join(t.get("code") or t["name"] for t in e["match"]["teams"])
            for g in (det or {}).get("match", {}).get("games", []):
                if g["state"] != "completed":
                    continue
                try:
                    status = archive_game(e, g)
                except OSError as err:  # network trouble: this game is retried on the next run
                    status = f"skipped ({err.__class__.__name__}), will retry next run"
                print(f"{e['startTime'][:10]} {league} {teams} game {g['number']}: {status}", flush=True)


if __name__ == "__main__":
    main()
