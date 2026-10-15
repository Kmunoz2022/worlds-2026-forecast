"""Live in-game predictions for Worlds (see docs/DECISIONS.md).

Live mode polls the LoL Esports feed, and for every live game computes the
win chance from the pre-game prediction plus the current gold, kill and CS
leads, and the series chance from the series score. Results go to
site/live/live.json (shown by site/live.html) and every minute's call is
appended to ledger/live_log.jsonl so the live calls can be graded later.

  .venv/bin/python scripts/live.py                    # Worlds games, every 10 s, serves http://localhost:8765/live.html
  .venv/bin/python scripts/live.py --league lck lcs   # other leagues (for testing)
  .venv/bin/python scripts/live.py --replay 115565004607949429   # run a finished match through the same code

Stop with Ctrl+C.
"""
import argparse
import functools
import http.server
import json
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lolpred import livefeed  # noqa: E402
from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.live import LiveModel  # noqa: E402
from lolpred.predict import MatchPredictor  # noqa: E402
from lolpred.series import series_win_prob_live  # noqa: E402

LIVE_DIR = ROOT / "site" / "live"
LIVE_LOG = ROOT / "ledger" / "live_log.jsonl"
CONFIG = ROOT / "config" / "worlds2026.json"


class TeamResolver:
    """Map a side of a live game to our team id by its players' names
    (the feed shows 'CODE Player'); more reliable than matching team names."""

    def __init__(self):
        pg = pd.read_parquet(PROCESSED_DIR / "player_games.parquet", columns=["playername", "teamid", "teamname", "date"])
        recent = pg[pg["date"] >= "2026-01-01"].sort_values("date")
        self.player_team = recent.groupby(recent["playername"].str.lower())["teamid"].last().to_dict()
        self.names = recent.groupby("teamid")["teamname"].last().to_dict()
        cfg = json.loads(CONFIG.read_text())
        self.names.update({t["teamid"]: n for n, t in cfg["teams"].items()})

    def resolve(self, participant_meta):
        votes = {}
        for p in participant_meta:
            name = p["summonerName"].split(" ", 1)[-1].lower()
            tid = self.player_team.get(name)
            if tid:
                votes[tid] = votes.get(tid, 0) + 1
        if not votes:
            return None, None
        tid = max(votes, key=votes.get)
        return tid, self.names.get(tid, tid)


class LiveTracker:
    def __init__(self):
        self.pred = MatchPredictor()
        self.model = LiveModel()
        self.resolver = TeamResolver()
        self.starts, self.games = {}, {}
        self.logged = set()

    def game_state(self, event, game, now, log=True):
        """Everything shown for one game at wall-clock time `now`."""
        gid = game["id"]
        # The feed rejects windows ending less than 20 s ago (HTTP 400), and a
        # window is 10 s long after rounding, so ask for 40 s back.
        w = livefeed.window(gid, now - timedelta(seconds=livefeed.LIVE_DELAY_S))
        if not w or not w.get("frames"):
            return None
        meta = w["gameMetadata"]
        if gid not in self.games:
            blue_id, blue = self.resolver.resolve(meta["blueTeamMetadata"]["participantMetadata"])
            red_id, red = self.resolver.resolve(meta["redTeamMetadata"]["participantMetadata"])
            if not blue_id or not red_id:
                return None
            self.games[gid] = {"blue_id": blue_id, "blue": blue, "red_id": red_id, "red": red,
                               "p_pre_blue": self.pred.game_prob_sided(blue_id, red_id),
                               "champions": {s: [p["championId"] for p in meta[f"{s}TeamMetadata"]["participantMetadata"]]
                                             for s in ("blue", "red")},
                               "patch": meta.get("patchVersion"), "history": []}
        g = self.games[gid]
        frame = w["frames"][-1]
        if gid not in self.starts:
            self.starts[gid] = livefeed.game_start(gid, livefeed.parse_ts(frame["rfc460Timestamp"]))
        minute = (livefeed.parse_ts(frame["rfc460Timestamp"]) - self.starts[gid]).total_seconds() / 60
        st = livefeed.frame_state(frame)
        p = self.model.p_blue(g["p_pre_blue"], max(minute, 0), st["gold_diff"], st["kill_diff"], st["cs_diff"])
        if not g["history"] or minute - g["history"][-1][0] >= 0.5:
            g["history"].append([round(minute, 2), round(p, 4)])
        if log and int(minute) not in {m for m in [k[1] for k in self.logged if k[0] == gid]}:
            self.logged.add((gid, int(minute)))
            LIVE_LOG.parent.mkdir(exist_ok=True)
            with LIVE_LOG.open("a") as f:
                f.write(json.dumps({"logged_at": livefeed.ts(datetime.now(timezone.utc)), "game_id": gid,
                                    "match_id": event["match"]["id"], "league": event["league"]["slug"],
                                    "blue": g["blue"], "red": g["red"], "minute": int(minute),
                                    "p_blue": round(p, 4), "p_pre_blue": round(g["p_pre_blue"], 4),
                                    "gold_diff": st["gold_diff"], "kill_diff": st["kill_diff"],
                                    "cs_diff": st["cs_diff"], "state": st["state"]}) + "\n")

        # Series: team A = first team listed for the match.
        teams = event["match"]["teams"]
        a_wins, b_wins = (t.get("result", {}).get("gameWins", 0) for t in teams)
        a_is_blue = next(t["side"] for t in game["teams"] if t["id"] == teams[0]["id"]) == "blue"
        a_id, b_id = (g["blue_id"], g["red_id"]) if a_is_blue else (g["red_id"], g["blue_id"])
        bo = event["match"]["strategy"]["count"]
        p_a_game = self.pred.game_prob(a_id, b_id)
        p_a_now = p if a_is_blue else 1 - p
        live = st["state"] == "in_game"
        p_series = series_win_prob_live(p_a_game, bo, a_wins, b_wins, p_a_now if live else None)
        return {"game_id": gid, "game_number": game["number"], "minute": round(minute, 1), "state": st["state"],
                "blue": g["blue"], "red": g["red"], "p_blue": round(p, 4), "p_pre_blue": round(g["p_pre_blue"], 4),
                "gold_diff": st["gold_diff"], "kill_diff": st["kill_diff"], "cs_diff": st["cs_diff"],
                "objectives": {"blue": st["blue"], "red": st["red"]}, "champions": g["champions"],
                "patch": g["patch"], "history": g["history"],
                "series": {"team_a": self.resolver.names.get(a_id, teams[0]["name"]),
                           "team_b": self.resolver.names.get(b_id, teams[1]["name"]),
                           "a_wins": a_wins, "b_wins": b_wins, "best_of": bo, "p_a": round(p_series, 4)},
                "league": event["league"]["name"]}


def write_live(states):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = {"updated": livefeed.ts(datetime.now(timezone.utc)), "games": states}
    tmp = LIVE_DIR / "live.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(LIVE_DIR / "live.json")


def print_state(s):
    sr = s["series"]
    print(f"[{s['league']}] G{s['game_number']} {s['minute']:5.1f}' {s['blue']} vs {s['red']}: "
          f"blue {s['p_blue']:.0%} (pre {s['p_pre_blue']:.0%}) | gold {s['gold_diff']:+,} kills {s['kill_diff']:+} "
          f"cs {s['cs_diff']:+} | series {sr['team_a']} {sr['a_wins']}-{sr['b_wins']} {sr['team_b']}: "
          f"{sr['team_a']} {sr['p_a']:.0%}", flush=True)


def serve(port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "site"))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"live page: http://localhost:{port}/live.html")


def run_live(leagues, interval, port):
    tracker = LiveTracker()
    serve(port)
    print(f"watching {', '.join(leagues)} every {interval}s (Ctrl+C to stop)")
    while True:
        states = []
        try:
            for ev in livefeed.live_events(leagues):
                detail = livefeed.event_details(ev["match"]["id"])
                for game in (detail or {}).get("match", {}).get("games", []):
                    if game["state"] == "inProgress":
                        s = tracker.game_state(detail, game, datetime.now(timezone.utc))
                        if s:
                            states.append(s)
                            print_state(s)
        except Exception as e:  # keep watching through network hiccups
            print(f"feed error: {e}", flush=True)
        write_live(states)
        time.sleep(interval)


def run_replay(match_id, step_minutes):
    """Feed a finished match through the live code, one point per step."""
    tracker = LiveTracker()
    event = livefeed.event_details(match_id)
    t = livefeed.match_start(event["league"]["id"], match_id)
    if t is None:
        sys.exit("couldn't find the match start in its league schedule")
    wins = {tm["id"]: 0 for tm in event["match"]["teams"]}
    played = 0
    for game in event["match"]["games"]:
        if game["state"] != "completed":
            continue
        for _ in range(48):  # scan forward up to 4 hours for this game's first frames
            if livefeed.window(game["id"], t):
                break
            t += timedelta(minutes=5)
        else:
            print(f"game {game['number']}: no stats found")
            continue
        start = livefeed.game_start(game["id"], t)
        tracker.starts[game["id"]] = start
        ev = json.loads(json.dumps(event))
        for tm in ev["match"]["teams"]:
            tm["result"] = {"gameWins": wins[tm["id"]]}  # series score before this game
        m, last, misses = 0.5, None, 0
        while misses < 4:  # ride through short gaps (pauses) in the feed
            s = tracker.game_state(ev, game, start + timedelta(minutes=m), log=False)
            if s:
                last, misses = s, 0
                print_state(s)
            else:
                misses += 1
            m += step_minutes
            time.sleep(0.2)
        end = livefeed.final_frames(game["id"], start + timedelta(minutes=m + 30))
        if end and last:
            fb, fr = end["frames"][-1]["blueTeam"], end["frames"][-1]["redTeam"]
            blue_won = (fb["inhibitors"], fb["towers"]) > (fr["inhibitors"], fr["towers"])
            side = {tm["side"]: tm["id"] for tm in game["teams"]}
            wins[side["blue" if blue_won else "red"]] += 1
            winner = last["blue"] if blue_won else last["red"]
            print(f"  -> game {game['number']} won by {winner} "
                  f"(our live call at {last['minute']:.0f} min: {last['blue'] if last['p_blue'] >= 0.5 else last['red']})\n")
            write_live([last])
            played += 1
        t = start + timedelta(minutes=m + 5)
    print(f"replayed {played} games")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", nargs="+", default=["worlds"])
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--replay", help="match id of a finished match")
    ap.add_argument("--step", type=float, default=2.0, help="replay step in game minutes")
    args = ap.parse_args()
    if args.replay:
        run_replay(args.replay, args.step)
    else:
        run_live(args.league, args.interval, args.port)


if __name__ == "__main__":
    main()
