"""Tournament simulator: results already played are kept, the rest is simulated."""
import numpy as np
import pytest

from lolpred.tournament import play_in, swiss

N = 16
P = np.full((N + 1, N + 1), 0.5)
REGION = [f"R{i % 4}" for i in range(N)] + ["PLAY_IN"]
POOLS = {1: [0, 1, 2, 3], 2: [4, 5, 6, 7], 3: [8, 9, 10, 11], 4: [12, 13, 14, 15]}


def test_play_in_keeps_played_results():
    # The real 2026 Play-In (0 KC, 1 C9, 2 MVK, 3 FURIA), all six matches played.
    played = [(0, 1, 3, 2), (2, 3, 3, 0), (0, 2, 3, 0), (3, 1, 3, 1), (2, 3, 3, 1), (0, 2, 3, 0)]
    rng = np.random.default_rng(0)
    assert {play_in(P, [0, 1, 2, 3], rng, [(0, 1), (3, 2)], played=played) for _ in range(50)} == {0}


def test_play_in_rejects_a_result_that_does_not_fit_the_bracket():
    with pytest.raises(ValueError):
        play_in(P, [0, 1, 2, 3], np.random.default_rng(0), [(0, 1), (2, 3)], played=[(0, 2, 3, 0)])


def test_swiss_uses_drawn_round_one_and_known_results():
    round1 = [(0, 12), (1, 13), (2, 14), (3, 15), (4, 8), (5, 9), (6, 10), (7, 11)]
    results = {frozenset(pr): (pr[0], pr[1], 1, 0) for pr in round1}  # every pool-1/2 team won
    for seed in range(20):
        tally = []
        rec = swiss(P, POOLS, REGION, np.random.default_rng(seed), tally, round1, results)
        assert [tuple(t[:2]) for t in tally[:8]] == round1
        assert all(rec[t][0] >= 1 for t in range(8)) and all(rec[t][1] >= 1 for t in range(8, 16))
        assert all(w == 3 or l == 3 for w, l in rec.values())


def test_swiss_round_two_keeps_known_pairs():
    round1 = [(0, 12), (1, 13), (2, 14), (3, 15), (4, 8), (5, 9), (6, 10), (7, 11)]
    results = {frozenset(pr): (pr[0], pr[1], 1, 0) for pr in round1}
    known = {1: [(0, 7)]}
    tally = []
    swiss(P, POOLS, REGION, np.random.default_rng(3), tally, round1, results, known)
    assert (0, 7) in [tuple(t[:2]) for t in tally[8:16]]
