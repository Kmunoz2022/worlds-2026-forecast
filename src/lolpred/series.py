"""Series outcomes (Bo1/Bo3/Bo5) from a single-game win probability.

Games in a series are NOT independent: sweeps happen far more often than
coin-flip maths says (2023+ data: Bo3 2-0 in 65% of series vs 55.5% predicted
by independent games; Bo5 3-0 in 39% vs 30%). One matchup's true edge varies
from series to series (draft prep, form on the day, meta fit).

Model: beta-binomial. For each series, team A's game win prob q is drawn from
Beta(p*KAPPA, (1-p)*KAPPA), then every game in that series uses q. This keeps
the single-game probability at p (so Bo1 odds are unchanged) while making
lopsided scores more likely. KAPPA fitted on ~11k Bo3/Bo5 series, 2023+.
"""
from math import comb, exp, lgamma

import numpy as np

KAPPA = 6.0


def _lbeta(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def score_distribution(p: float, best_of: int, kappa: float = KAPPA) -> dict[tuple[int, int], float]:
    """{(a_wins, b_wins): probability} for every possible final score."""
    w = (best_of + 1) // 2
    al, be = p * kappa, (1 - p) * kappa
    out = {}
    for lose in range(w):
        n_ways = comb(w - 1 + lose, lose)  # the winner takes the last game
        out[(w, lose)] = n_ways * exp(_lbeta(al + w, be + lose) - _lbeta(al, be))
        out[(lose, w)] = n_ways * exp(_lbeta(al + lose, be + w) - _lbeta(al, be))
    return out


def series_win_prob(p: float, best_of: int, kappa: float = KAPPA) -> float:
    w = (best_of + 1) // 2
    return sum(v for (a, b), v in score_distribution(p, best_of, kappa).items() if a == w)


def sample_series(p: float, best_of: int, rng: np.random.Generator, kappa: float = KAPPA) -> tuple[int, int]:
    """Play one series; returns (a_wins, b_wins)."""
    w = (best_of + 1) // 2
    q = rng.beta(p * kappa, (1 - p) * kappa)
    a = b = 0
    while a < w and b < w:
        if rng.random() < q:
            a += 1
        else:
            b += 1
    return a, b


def series_win_prob_live(p: float, best_of: int, a_wins: int, b_wins: int, p_current: float | None = None,
                         kappa: float = KAPPA) -> float:
    """P(team A wins the series) given the games already played and, if a game
    is in progress, A's live chance of winning it.

    Same beta-binomial model as above, updated with the finished games: each
    win makes the next one a bit more likely for the same team (the series
    reveals which team is better on the day)."""
    w = (best_of + 1) // 2
    al, be = p * kappa, (1 - p) * kappa

    def rest(a, b):
        if a >= w:
            return 1.0
        if b >= w:
            return 0.0
        q = (al + a) / (al + be + a + b)
        return q * rest(a + 1, b) + (1 - q) * rest(a, b + 1)

    if p_current is None:
        return rest(a_wins, b_wins)
    return p_current * rest(a_wins + 1, b_wins) + (1 - p_current) * rest(a_wins, b_wins + 1)
