"""Crystal Ball scoring from feed game stats."""
from lolpred.crystal_track import feed_tracker, measure_feed


def game(gid, winner, minutes, players):
    return {"game_id": gid, "winner": winner, "duration_s": minutes * 60,
            "players": [{"name": n, "team": t, "champion": c, "kills": k, "deaths": d, "assists": a, "cs": cs}
                        for n, t, c, k, d, a, cs in players]}


GAMES = [
    game("1", "A", 28, [("Peyz", "A", "Ashe", 6, 1, 4, 300), ("Faker", "B", "Ahri", 2, 3, 1, 280)]),
    game("2", "B", 35, [("Peyz", "A", "Ashe", 1, 4, 2, 310), ("Faker", "B", "Teemo", 9, 0, 3, 460)]),
]
PICKS = {"peyz_guma": {"pick": "Peyz"}, "teemo": {"pick": "No"}, "len32": {"pick": "Before 32:00"},
         "cs450": {"pick": "Yes"}, "shortest": {"pick": "A"}, "kda": {"pick": "Faker (B, Mid)"},
         "presence": {"pick": "Vi"}}


def test_measure_feed_counts():
    M = measure_feed(GAMES, {})
    assert M["kills"] == {"peyz": 7, "faker": 11}
    assert M["teemo"] == 1 and M["unique"] == 3
    assert (M["before32"], M["after32"]) == (1, 1)
    assert M["max_cs"] == ("Faker", 460)
    assert M["shortest"] == (28, "A")
    kp = {r["name"]: r["value"] for r in M["kp"]}
    assert kp["Peyz"] == (6 + 4 + 1 + 2) / (6 + 1)  # (kills + assists) / team kills, as in the official stat


def test_feed_tracker_statuses():
    t = feed_tracker(PICKS, GAMES, {}, ["A", "B"], {"A": "LCK", "B": "LPL"}, None)
    m = t["model"]
    assert list(m) == list(PICKS)  # question order kept
    assert m["teemo"]["status"] == "lost"          # Teemo was picked
    assert m["cs450"]["status"] == "won"           # someone hit 450 CS
    assert m["peyz_guma"]["status"] == "on track"  # Gumayusi has 0
    assert m["len32"]["status"] == "tied"
    assert m["shortest"]["status"] == "on track"
    assert m["kda"]["status"] == "on track"
    assert m["presence"]["status"] == "waiting"    # needs bans: official data only


def test_feed_tracker_keeps_official_presence():
    csv = {"games": 5, "data_through": "2026-10-16",
           "model": {"presence": {"pick": "Vi", "status": "on track", "summary": "Leader: Vi 80%.",
                                  "leaders": [], "pick_now": None}}}
    t = feed_tracker(PICKS, GAMES, {}, ["A", "B"], {}, csv)
    assert t["model"]["presence"]["status"] == "on track"
    assert "2026-10-16" in t["model"]["presence"]["summary"]


def test_not_started_before_any_game():
    t = feed_tracker(PICKS, [], {}, ["A", "B"], {}, None)
    assert {v["status"] for v in t["model"].values()} == {"not started"}
