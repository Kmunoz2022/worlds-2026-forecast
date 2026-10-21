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


def test_form_posterior_moves_toward_results():
    from lolpred.tournament import FORM_SD, form_posterior
    L = np.zeros((3, 3))
    mu, sd = form_posterior(L, [])
    assert np.allclose(mu, 0) and np.allclose(sd, FORM_SD)
    mu, sd = form_posterior(L, [(0, 1), (0, 2)])   # team 0 won two even games
    assert mu[0] > 0 > mu[1] and sd[0] < FORM_SD


def test_tournament_form_spreads_the_odds():
    # 15 direct teams (1-15), four Play-In teams (0, 16, 17, 18), placeholder slot 19.
    # Team 0 is a clear favourite: with form switched on it wins the Play-In less often.
    from lolpred.tournament import simulate
    n = 19
    P = np.full((n, n), 0.5)
    P[0, :], P[:, 0] = 0.8, 0.2
    P[0, 0] = 0.5
    pools = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8], 3: [9, 10, 11, 12], 4: [13, 14, 15, 19]}
    region = [f"R{i % 4}" for i in range(n)] + ["PLAY_IN"]
    args = dict(play_in_teams=[0, 16, 17, 18], pools=pools, pi_slot=19, region=region, n_sims=3000, seed=1)
    base = simulate(P, **args)[0]["play_in_win"]
    form = simulate(P, **args, form_sd=0.8)[0]["play_in_win"]
    assert form < base


def test_swiss_bo1_higher_pool_plays_blue():
    # Blue always wins, red never: in round-1 Bo1s the higher-pool team (pools 1 and 2) must win.
    round1 = [(0, 12), (1, 13), (2, 14), (3, 15), (4, 8), (5, 9), (6, 10), (7, 11)]
    PB = np.full((N + 1, N + 1), 0.999)
    for seed in range(5):
        tally = []
        swiss(P, POOLS, REGION, np.random.default_rng(seed), tally, round1, PB=PB)
        assert all(t[2] == 1 for t in tally[:8])
