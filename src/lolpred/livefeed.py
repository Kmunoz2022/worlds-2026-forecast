"""Client for the LoL Esports live-stats feed (see docs/DECISIONS.md).

These are the endpoints lolesports.com itself uses. They're unofficial and
undocumented (community docs: vickz84259.github.io/lolesports-api-docs), need
the public key the lolesports.com site uses (see api_key), and can change
without notice. Keep polling gentle (every ~10 s per game). Second-tier events may
have stats disabled ("Stats are disabled for game ...").
"""
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import ROOT

API = "https://esports-api.lolesports.com/persisted/gw"
FEED = "https://feed.lolesports.com/livestats/v1"
LOCAL_CONFIG = ROOT / "config" / "local.json"  # git-ignored
# The feed refuses windows that end less than 20 s ago ("disallowed window with
# end time less than 20 sec old"). Windows are 10 s and start times round down
# to 10 s, so live requests look 40 s back.
LIVE_DELAY_S = 40


def api_key() -> str:
    """The lolesports.com web client's public API key. Kept out of the code:
    set LOLESPORTS_API_KEY, or put {"lolesports_api_key": "..."} in
    config/local.json. The key is listed in the community API docs."""
    key = os.environ.get("LOLESPORTS_API_KEY")
    if not key and LOCAL_CONFIG.exists():
        key = json.loads(LOCAL_CONFIG.read_text()).get("lolesports_api_key")
    if not key:
        raise RuntimeError("No LoL Esports API key: set LOLESPORTS_API_KEY or add "
                           "lolesports_api_key to config/local.json (see README).")
    return key


def _get(url, tries=3):
    """GET with the API key. Timeouts and dropped connections are retried
    (the feed occasionally stalls); 204/404 mean "nothing there"."""
    req = urllib.request.Request(url, headers={"x-api-key": api_key(), "User-Agent": "lolpred-worlds-forecast"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as e:
            if e.code in (204, 404):
                return None
            raise
        except (TimeoutError, urllib.error.URLError, ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def ts(dt: datetime) -> str:
    """Feed timestamps must be whole multiples of 10 seconds."""
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.replace(second=dt.second // 10 * 10).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def live_events(league_slugs=None):
    d = _get(f"{API}/getLive?hl=en-US")
    events = (d or {}).get("data", {}).get("schedule", {}).get("events") or []
    return [e for e in events if e.get("type") == "match"
            and (league_slugs is None or e["league"]["slug"] in league_slugs)]


def event_details(match_id):
    d = _get(f"{API}/getEventDetails?hl=en-US&id={match_id}")
    return d["data"]["event"] if d else None


def window(game_id, at: datetime):
    """10 seconds of frames starting at `at`. None if there are none for that
    moment: asked for a time after the game ended, the feed returns the
    final frames instead, so frames far from `at` are rejected."""
    w = _get(f"{FEED}/window/{game_id}?startingTime={ts(at)}")
    if not w or not w.get("frames"):
        return None
    first = parse_ts(w["frames"][0]["rfc460Timestamp"])
    return w if abs((first - at).total_seconds()) <= 40 else None


def final_frames(game_id, after: datetime):
    """The last frames of a finished game (any time after it ended works)."""
    w = _get(f"{FEED}/window/{game_id}?startingTime={ts(after)}")
    return w if w and w.get("frames") and w["frames"][-1]["gameState"] == "finished" else None


def match_start(league_id, match_id):
    """Scheduled start of a match, from the league schedule (event details
    don't include it)."""
    d = _get(f"{API}/getSchedule?hl=en-US&leagueId={league_id}")
    for e in (d or {}).get("data", {}).get("schedule", {}).get("events", []):
        if e.get("match", {}).get("id") == match_id:
            return parse_ts(e["startTime"])
    return None


def game_start(game_id, around: datetime, step=timedelta(minutes=3)):
    """Wall-clock time of the first in-game frame, found by stepping back from
    `around` (a time inside the game) and then refining to 10 seconds."""
    # The feed can go silent mid-game (a 12-minute gap in one LCS game), so
    # only 18 minutes without frames counts as "before the game started".
    t, misses = around, 0
    for _ in range(40):  # up to 2 hours back
        if misses >= 6:
            break
        if window(game_id, t - (misses + 1) * step):
            t, misses = t - (misses + 1) * step, 0
        else:
            misses += 1
    lo, hi = t - step, t  # no frames at lo, frames at hi
    while hi - lo > timedelta(seconds=10):
        mid = lo + (hi - lo) / 2
        if window(game_id, mid):
            hi = mid
        else:
            lo = mid
        time.sleep(0.2)
    for at in (lo, hi):
        w = window(game_id, at)
        if w:
            return parse_ts(w["frames"][0]["rfc460Timestamp"])
    return hi


def frame_state(frame):
    """Blue-minus-red leads and team objectives from one window frame."""
    b, r = frame["blueTeam"], frame["redTeam"]
    cs = lambda team: sum(p["creepScore"] for p in team["participants"])  # noqa: E731
    return {"gold_diff": b["totalGold"] - r["totalGold"], "kill_diff": b["totalKills"] - r["totalKills"],
            "cs_diff": cs(b) - cs(r), "state": frame["gameState"],
            "blue": {k: b[k] for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons", "dragons")},
            "red": {k: r[k] for k in ("totalGold", "totalKills", "towers", "inhibitors", "barons", "dragons")}}
