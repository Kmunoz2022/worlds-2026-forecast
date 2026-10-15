"""The files the website and the public record are built from."""
import json

import pytest

from lolpred.config import ROOT

DATA = json.loads((ROOT / "site" / "data.json").read_text())
LEDGER = ROOT / "ledger"


def test_site_data_has_every_section():
    for key in ("meta", "teams", "game_prob", "sim", "pickem", "play_in", "model_record", "crystal_ball", "live"):
        assert key in DATA


def test_game_probabilities_are_complementary():
    gp = DATA["game_prob"]
    for a in gp:
        for b, p in gp[a].items():
            assert 0 < p < 1
            assert gp[b][a] == pytest.approx(1 - p, abs=1e-6)


def test_sided_probabilities_are_valid():
    for a, row in DATA["live"]["p_blue"].items():
        for b, p in row.items():
            assert 0 < p < 1


def test_live_model_shape():
    live = DATA["live"]
    assert len(live["coef"]) == len(live["features"]) == 7


def test_predictions_ledger_is_well_formed_and_in_time_order():
    lines = [json.loads(l) for l in (LEDGER / "predictions.jsonl").read_text().splitlines() if l.strip()]
    assert lines
    times = [e["predicted_at"] for e in lines]
    assert times == sorted(times)  # append-only: never back-dated
    for e in lines:
        assert e["best_of"] in (1, 3, 5)
        assert 0 < e["p_game_a"] < 1 and 0 < e["p_series_a"] < 1
        assert sum(e["score_dist"].values()) == pytest.approx(1, abs=1e-3)
        assert e["pick"] in (e["team_a"], e["team_b"])


@pytest.mark.parametrize("name", ["crystal_ball_locked.json", "crystal_ball_locked_extra.json", "pickem_locked.json"])
def test_locked_picks_exist(name):
    d = json.loads((LEDGER / name).read_text())
    # Locked before Worlds began (Play-In day 1, Oct 15 18:00 UTC).
    assert d["locked_at"] < "2026-10-15T18:00:00Z"
