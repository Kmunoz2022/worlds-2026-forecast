"""Automatic ledger: team matching, game winners, grading and the live model."""
import json

import pytest

from lolpred import record
from lolpred.config import ROOT

TEAMS = ["Gen.G", "Cloud9", "Karmine Corp", "LØS", "FURIA", "MVK Esports"]


@pytest.mark.parametrize("name,code,ours", [
    ("Cloud9 Kia", "C9", "Cloud9"),
    ("Karmine Corp", "KC", "Karmine Corp"),
    ("LOS", "LOS", "LØS"),
    ("FURIA", "FUR", "FURIA"),
    ("GEN", "GEN", "Gen.G"),
    ("TBD", "TBD", None),
])
def test_our_team(name, code, ours):
    assert record.our_team(name, code, TEAMS) == ours


def test_assign_winners_gives_official_count_to_best_games():
    # A won 3 games; the three games where A ended furthest ahead are A's.
    assert record.assign_winners([-14.6, -25.0, 11.0, -21.0, 19.5], 2) == [False, False, True, False, True]
    assert record.assign_winners([1.0, -1.0], 0) == [False, False]


def test_game_edge_sign():
    ahead = {"inhibitors": 2, "towers": 10, "totalGold": 70000}
    behind = {"inhibitors": 0, "towers": 3, "totalGold": 60000}
    assert record.game_edge(ahead, behind) > 0 > record.game_edge(behind, ahead)


def test_live_p_blue_matches_the_python_live_model():
    from lolpred.live import LiveModel
    live = json.loads((ROOT / "site" / "data.json").read_text())["live"]
    m = LiveModel()
    for args in [(0.6, 12, 2500, 3, 80), (0.3, 28, -6000, -7, -150), (0.5, 40, 0, 0, 0)]:
        assert record.live_p_blue(live, *args) == pytest.approx(m.p_blue(*args), abs=1e-9)


def test_live_p_blue_rises_with_gold_lead():
    live = json.loads((ROOT / "site" / "data.json").read_text())["live"]
    ps = [record.live_p_blue(live, 0.5, 20, g, 0, 0) for g in range(-6000, 6001, 1000)]
    assert ps == sorted(ps)


def pred(a, b, at, p=0.7, stage="Play-In"):
    return {"predicted_at": at, "stage": stage, "team_a": a, "team_b": b, "p_game_a": p, "p_series_a": p,
            "pick": a if p >= 0.5 else b, "likely_score": "3-0" if p >= 0.5 else "0-3",
            "score_dist": {"3-0": 0.4, "3-1": 0.2, "3-2": 0.1, "2-3": 0.1, "1-3": 0.1, "0-3": 0.1}}


def result(a, b, aw, bw, start, winners, stage="Play-In"):
    return {"match_id": f"{a}-{b}-{start}", "start": start, "first_game_start": start, "stage": stage,
            "team_a": a, "team_b": b, "a_wins": aw, "b_wins": bw, "games": [{"winner": w} for w in winners]}


def test_grade_uses_latest_prediction_before_start_and_orients_teams():
    preds = [pred("KC", "C9", "2026-10-14T10:00:00Z", 0.6),
             pred("C9", "KC", "2026-10-15T10:00:00Z", 0.55),   # newer, teams the other way round
             pred("KC", "C9", "2026-10-15T20:00:00Z", 0.9)]    # made after the match started: never used
    res = [result("KC", "C9", 2, 3, "2026-10-15T18:00:00Z", ["C9", "KC", "C9", "KC", "C9"])]
    g = record.grade(preds, res, {"Play-In"})
    assert [e["status"] for e in g] == ["superseded", "graded", "pending"]
    e = g[1]
    assert (e["a_wins"], e["b_wins"], e["winner"]) == (3, 2, "C9")
    assert e["pick"] == "C9" and e["correct"] is True and e["game_results"] == [1, 0, 1, 0, 1]


def test_grade_never_edits_its_input():
    preds = [pred("KC", "C9", "2026-10-14T10:00:00Z")]
    before = json.dumps(preds)
    record.grade(preds, [result("KC", "C9", 3, 0, "2026-10-15T18:00:00Z", ["KC"] * 3)], {"Play-In"})
    assert json.dumps(preds) == before


def test_unplayed_pairing_from_a_finished_stage():
    g = record.grade([pred("KC", "MVK", "2026-10-14T10:00:00Z")], [], {"Swiss"})
    assert g[0]["status"] == "not played"


def test_summary_scores():
    g = record.grade([pred("KC", "C9", "2026-10-14T10:00:00Z", 0.7)],
                     [result("KC", "C9", 3, 1, "2026-10-15T18:00:00Z", ["KC", "C9", "KC", "KC"])], {"Play-In"})
    s = record.summary(g)
    assert s["series"] == 1 and s["series_correct"] == 1 and s["games"] == 4
    assert s["game_accuracy"] == pytest.approx(0.75)
    assert s["game_brier"] == pytest.approx((3 * 0.3 ** 2 + 0.7 ** 2) / 4)


def test_live_summary_calibration_and_checkpoints():
    calls = [{"match_id": "m", "number": 1, "blue": "A", "red": "B", "blue_won": True, "p_pre_blue": 0.6,
              "calls": [[5, 0.55], [10, 0.65], [15, 0.85]]},
             {"match_id": "m", "number": 2, "blue": "B", "red": "A", "blue_won": False, "p_pre_blue": 0.45,
              "calls": [[5, 0.52], [10, 0.4], [15, 0.1]]}]
    s = record.live_summary(calls)
    assert s["games"] == 2 and s["calls"] == 6
    assert s["call_accuracy"] == pytest.approx(5 / 6)
    assert {r["minute"]: r["call_accuracy"] for r in s["by_minute"]} == {0: 1.0, 5: 0.5, 10: 1.0, 15: 1.0}
    assert sum(c["n"] for c in s["calibration"]) == 6


def test_game_time_removes_pauses():
    pauses = [(600, 645), (1200, 1320)]
    assert record.game_time(500, pauses) == 500
    assert record.game_time(620, pauses) == 600       # inside the first pause: clock stopped
    assert record.game_time(1000, pauses) == 955
    assert record.game_time(2000, pauses) == 2000 - 45 - 120


def test_player_name_drops_team_tag():
    assert record.player_name("KC Canna") == "Canna"
    assert record.player_name("Faker") == "Faker"


def test_swiss_bo1_logs_the_higher_pool_on_blue():
    data = {"game_prob": {"A": {"B": 0.6}, "B": {"A": 0.4}},
            "live": {"p_blue": {"A": {"B": 0.65}, "B": {"A": 0.45}}}}
    cfg = {"swiss_pools": {"1": ["A"], "4": ["PLAY_IN_WINNER"]}}
    play_in = [{"start": f"2026-10-1{i}", "stage": "Play-In", "winner": "B"} for i in range(6)]
    pools = record.swiss_pool_of(cfg, play_in)
    assert pools == {"A": 1, "B": 4}
    assert record.game_chance(data, pools, "A", "B", 1, "Swiss")[0] == pytest.approx(0.65)
    assert record.game_chance(data, pools, "B", "A", 1, "Swiss")[0] == pytest.approx(0.35)
    # Bo3s and other stages stay side-averaged
    assert record.game_chance(data, pools, "A", "B", 3, "Swiss") == (0.6, None)
    assert record.game_chance(data, pools, "A", "B", 5, "Quarterfinals") == (0.6, None)
