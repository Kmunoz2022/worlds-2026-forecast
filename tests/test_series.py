"""Series maths: valid scores, probabilities that add up, consistent live updates."""
import pytest

from lolpred.series import score_distribution, series_win_prob, series_win_prob_live


@pytest.mark.parametrize("best_of,scores", [
    (1, {(1, 0), (0, 1)}),
    (3, {(2, 0), (2, 1), (1, 2), (0, 2)}),
    (5, {(3, 0), (3, 1), (3, 2), (2, 3), (1, 3), (0, 3)}),
])
def test_only_valid_final_scores(best_of, scores):
    dist = score_distribution(0.63, best_of)
    assert set(dist) == scores  # e.g. a Bo5 can never end 4-1 or 2-1
    assert sum(dist.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.1, 0.35, 0.5, 0.8, 0.97])
def test_bo1_keeps_the_game_probability(p):
    assert series_win_prob(p, 1) == pytest.approx(p)


def test_even_matchup_is_symmetric():
    dist = score_distribution(0.5, 5)
    for (a, b), v in dist.items():
        assert dist[(b, a)] == pytest.approx(v)


def test_favourite_gains_in_longer_series():
    assert series_win_prob(0.6, 1) < series_win_prob(0.6, 3) < series_win_prob(0.6, 5)


def test_series_win_increases_with_game_probability():
    ps = [series_win_prob(p / 20, 5) for p in range(1, 20)]
    assert ps == sorted(ps)


def test_live_update_before_any_game_matches_pregame():
    for p in (0.3, 0.55, 0.9):
        assert series_win_prob_live(p, 5, 0, 0) == pytest.approx(series_win_prob(p, 5))


def test_live_update_settled_series():
    assert series_win_prob_live(0.2, 5, 3, 1) == 1.0
    assert series_win_prob_live(0.9, 3, 0, 2) == 0.0


def test_losing_a_game_lowers_series_chance():
    p = 0.63
    assert series_win_prob_live(p, 5, 0, 1) < series_win_prob(p, 5) < series_win_prob_live(p, 5, 1, 0)


def test_live_game_chance_averages_to_pregame():
    # Before game 1, weighting by the game-1 chance must give back the plain series chance.
    p = 0.6
    assert series_win_prob_live(p, 5, 0, 0, p_current=p) == pytest.approx(series_win_prob(p, 5))
