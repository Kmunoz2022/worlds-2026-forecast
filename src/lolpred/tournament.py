"""Monte Carlo simulation of Worlds 2026: Play-In -> Swiss -> Knockouts.

Teams are integer indices into a game-probability matrix P, where P[i, j] is
team i's single-game win probability vs team j. Series are drawn with the
beta-binomial series model (series.sample_series). Team strengths stay fixed
for the whole tournament (no in-event rating updates).

Format (web search, 2026-09-30):
  Play-In: 4 teams, double elimination, all Bo5, 1 team advances.
  Swiss: 16 teams, 5 rounds, 3 wins qualify / 3 losses eliminated. Bo1, except
    Bo3 when a team can qualify or be eliminated (a 2-win or 2-loss record).
    Round 1: pool 1 vs pool 4, pool 2 vs pool 3, no same-region matches.
    Later rounds: random pairing within the same record, avoiding rematches.
  Knockouts: 8 teams, single elimination, Bo5.

Approximations (official draw rules not confirmed):
  - Play-In round 1 pairings are drawn at random unless given in the config.
  - Knockout draw: the two 3-0 teams go to opposite halves and face 3-2
    teams; the 3-1 teams fill the rest; same-region quarterfinals are
    avoided when possible.
"""
from collections import Counter, defaultdict

import numpy as np

from .series import sample_series

RECORDS = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]

# Tournament form: in each simulated tournament every team gets one strength shift,
# N(0, FORM_SD) on the log-odds scale, kept for the whole event. A team that is off
# at Worlds tends to stay off; winning early in a simulated run then also means that
# team is likely in form, so its later odds rise. Sized on 2022-25 internationals
# (MSI, Worlds, EWC, First Stand): residuals against pre-event predictions persist
# with SD ~0.4, and predicting each later game with this learning was best at
# 0.5-0.6 (log loss 0.628 -> 0.617). See docs/DECISIONS.md.
FORM_SD = 0.5


def form_posterior(L, games, sd=FORM_SD):
    """Per-team (mean, sd) of the form shift given real games the model hasn't
    seen yet (one Newton step from zero, i.e. a Laplace approximation).
    L: log-odds matrix; games: [(winner, loser), ...] as team indices."""
    n = L.shape[0]
    resid, info = np.zeros(n), np.zeros(n)
    for w, l in games:
        p = 1 / (1 + np.exp(-L[w, l]))
        resid[w] += 1 - p
        resid[l] -= 1 - p
        info[w] += p * (1 - p)
        info[l] += p * (1 - p)
    prec = 1 / sd ** 2 + info
    return resid / prec, np.sqrt(1 / prec)


def play(P, a, b, best_of, rng, tally=None):
    """Returns (winner, loser). Appends (a, b, a_wins, b_wins) to `tally`."""
    wa, wb = sample_series(P[a, b], best_of, rng)
    if tally is not None:
        tally.append((a, b, wa, wb))
    return (a, b) if wa > wb else (b, a)


def play_in(P, teams, rng, round1=None, tally=None, played=None):
    """Double elimination, all Bo5: two openers, winners' match, elimination
    match, losers' final, then a grand final between the winners' bracket
    winner and the losers' bracket winner. round1: [(a, b), (c, d)] if known,
    else drawn at random. played: results so far, in that order, as
    (a, b, a_wins, b_wins); those matches keep their real result."""
    played = list(played or [])
    if round1 is None:
        t = list(rng.permutation(teams))
        round1 = [(t[0], t[3]), (t[1], t[2])]

    def match(a, b):
        if played:
            pa, pb, wa, wb = played.pop(0)
            if {pa, pb} != {a, b}:
                raise ValueError(f"Play-In result {pa} vs {pb} doesn't fit the bracket (expected {a} vs {b})")
            if tally is not None:
                tally.append((pa, pb, wa, wb))
            return (pa, pb) if wa > wb else (pb, pa)
        return play(P, a, b, 5, rng, tally)

    w1, l1 = match(*round1[0])
    w2, l2 = match(*round1[1])
    ub_w, ub_l = match(w1, w2)
    lb_w, _ = match(l1, l2)
    lbf_w, _ = match(ub_l, lb_w)
    gf_w, _ = match(ub_w, lbf_w)
    return gf_w


def _pair_avoiding(group, bad, rng, tries=200):
    """Random perfect pairing of `group` avoiding pairs where bad(a, b)."""
    group = list(group)
    for _ in range(tries):
        g = list(rng.permutation(group))
        pairs = [(g[i], g[i + 1]) for i in range(0, len(g), 2)]
        if not any(bad(a, b) for a, b in pairs):
            return pairs
    return pairs  # no clean pairing found; accept the last draw


def _cross_pairs(A, B, bad, rng, tries=200):
    """Pair every team in A with one in B, avoiding bad pairs."""
    for _ in range(tries):
        perm = list(rng.permutation(B))
        pairs = list(zip(A, perm))
        if not any(bad(a, b) for a, b in pairs):
            return pairs
    return pairs


def swiss(P, pools, region, rng, tally=None, round1=None, results=None, known_pairs=None):
    """pools: {1: [idx...], 2: ..., 3: ..., 4: ...}. Returns {team: (wins, losses)}.
    round1: the drawn round-1 pairs, if known (else drawn from the pools).
    results: {frozenset((a, b)): (a, b, a_wins, b_wins)} for series already
    played; Swiss never repeats a pairing, so the pair identifies the series.
    known_pairs: {round index: [(a, b), ...]} pairings already set for later
    rounds (from results); the other teams in that round are paired at random."""
    results, known_pairs = results or {}, known_pairs or {}
    wins, losses = defaultdict(int), defaultdict(int)
    played = set()
    same_region = lambda a, b: region[a] == region[b]  # noqa: E731
    rematch = lambda a, b: frozenset((a, b)) in played  # noqa: E731

    pairs = list(round1) if round1 else (_cross_pairs(pools[1], pools[4], same_region, rng)
                                         + _cross_pairs(pools[2], pools[3], same_region, rng))
    for rnd in range(5):
        for a, b in pairs:
            known = results.get(frozenset((a, b)))
            if known:
                ka, kb, wa, wb = known
                if tally is not None:
                    tally.append(known)
                w, l = (ka, kb) if wa > wb else (kb, ka)
            else:
                deciding = wins[a] == 2 or losses[a] == 2
                w, l = play(P, a, b, 3 if deciding else 1, rng, tally)
            wins[w] += 1
            losses[l] += 1
            played.add(frozenset((a, b)))
        active = [t for pool in pools.values() for t in pool if wins[t] < 3 and losses[t] < 3]
        if not active:
            break
        fixed = [(a, b) for a, b in known_pairs.get(rnd + 1, []) if a in active and b in active]
        taken = {t for pr in fixed for t in pr}
        groups = defaultdict(list)
        for t in active:
            if t not in taken:
                groups[(wins[t], losses[t])].append(t)
        pairs = fixed + [p for g in groups.values() for p in _pair_avoiding(g, rematch, rng)]
    return {t: (wins[t], losses[t]) for pool in pools.values() for t in pool}


def knockouts(P, records, region, rng, tally=None):
    """records: {team: (3, losses)} for the 8 qualifiers.
    Returns (quarterfinal losers, semifinal losers, runner-up, champion)."""
    by = defaultdict(list)
    for t, (_, l) in records.items():
        by[l].append(t)
    for _ in range(100):
        s0, s1, s2 = (list(rng.permutation(by[k])) for k in (0, 1, 2))
        # Half 1: 3-0 vs 3-2, 3-1 vs 3-1. Half 2: 3-0 vs 3-2, 3-1 vs 3-2.
        qf = [(s0[0], s2[0]), (s1[0], s1[1]), (s0[1], s2[1]), (s1[2], s2[2])]
        if not any(region[a] == region[b] for a, b in qf):
            break
    qf_w = [play(P, a, b, 5, rng, tally)[0] for a, b in qf]
    sf = [(qf_w[0], qf_w[1]), (qf_w[2], qf_w[3])]
    sf_w = [play(P, a, b, 5, rng, tally)[0] for a, b in sf]
    champ, runner_up = play(P, sf_w[0], sf_w[1], 5, rng, tally)
    qf_l = [t for pair in qf for t in pair if t not in qf_w]
    sf_l = [t for t in qf_w if t not in sf_w]
    return qf_l, sf_l, runner_up, champ


def simulate(P, play_in_teams, pools, pi_slot, region, n_sims=20000, seed=0, play_in_round1=None,
             game_totals=None, outcomes=None, play_in_played=None, swiss_round1=None, swiss_results=None,
             swiss_known_pairs=None, form_sd=0.0, form_games=None):
    """pools[4] must contain `pi_slot`, a placeholder index replaced each sim by
    the Play-In winner. Returns per-team counts and Swiss-record counts.
    If `game_totals` is a list, appends {stage: games played} for each sim.
    If `outcomes` is a list, appends each sim's full result: every series
    (stage, a, b, a_wins, b_wins), Swiss records, and knockout placements."""
    rng = np.random.default_rng(seed)
    counts = defaultdict(Counter)
    P0 = np.clip(np.asarray(P, float), 1e-6, 1 - 1e-6)
    L = np.log(P0 / (1 - P0))
    mu, sd = form_posterior(L, form_games or [], form_sd) if form_sd > 0 else (np.zeros(len(L)), np.zeros(len(L)))
    for _ in range(n_sims):
        if form_sd > 0:   # this tournament's form for every team, then the game odds it implies
            u = mu + sd * rng.standard_normal(len(L))
            P = 1 / (1 + np.exp(-(L + u[:, None] - u[None, :])))
        t_pi, t_sw, t_ko = [], [], []
        pi_winner = play_in(P, play_in_teams, rng, play_in_round1, t_pi, play_in_played)
        counts[pi_winner]["play_in_win"] += 1
        sim_pools = {k: [pi_winner if t == pi_slot else t for t in v] for k, v in pools.items()}
        r1 = [tuple(pi_winner if t == pi_slot else t for t in pr) for pr in swiss_round1] if swiss_round1 else None
        rec = swiss(P, sim_pools, region, rng, t_sw, r1, swiss_results, swiss_known_pairs)
        for t, (w, l) in rec.items():
            counts[t][f"{w}-{l}"] += 1
            counts[t]["swiss"] += 1
            if w == 3:
                counts[t]["qualify"] += 1
        quals = {t: r for t, r in rec.items() if r[0] == 3}
        qf_l, sf_l, ru, champ = knockouts(P, quals, region, rng, t_ko)
        if game_totals is not None:
            game_totals.append({k: sum(s[2] + s[3] for s in v)
                                for k, v in (("play_in", t_pi), ("swiss", t_sw), ("knockouts", t_ko))})
        if outcomes is not None:
            outcomes.append({
                "series": [("play_in", *s) for s in t_pi] + [("swiss", *s) for s in t_sw]
                          + [("knockouts", *s) for s in t_ko],
                "play_in_teams": list(play_in_teams), "play_in_winner": pi_winner,
                "records": rec, "qf_losers": qf_l, "sf_losers": sf_l,
                "runner_up": ru, "champion": champ})
        for t in qf_l + sf_l + [ru, champ]:
            counts[t]["qf"] += 1
        for t in sf_l + [ru, champ]:
            counts[t]["sf"] += 1
        counts[ru]["final"] += 1
        counts[champ]["final"] += 1
        counts[champ]["champion"] += 1
    return counts


def pickem(probs, n_30=2, n_03=2, n_adv=5):
    """Greedy Swiss pick'em: the teams most likely to go 3-0 and 0-3, then the
    most likely to advance among the rest (assumed Riot-style 2/2/5 picks).
    probs: {team_name: {"3-0": p, "0-3": p, "qualify": p}}."""
    left = dict(probs)
    t30 = sorted(left, key=lambda t: -left[t]["3-0"])[:n_30]
    for t in t30:
        left.pop(t)
    t03 = sorted(left, key=lambda t: -left[t]["0-3"])[:n_03]
    for t in t03:
        left.pop(t)
    adv = sorted(left, key=lambda t: -left[t]["qualify"])[:n_adv]
    return {"3-0": t30, "0-3": t03, "advance": adv}


def setup_from_config(cfg, cblol_2nd):
    """Turn config/worlds2026.json into simulator inputs for one choice of the
    CBLOL #2 slot. Team indices follow the order of cfg['teams']; index
    len(teams) is the Play-In winner placeholder."""
    names = list(cfg["teams"])
    idx = {n: i for i, n in enumerate(names)}
    pi_slot = len(names)
    resolve = lambda t: idx[cblol_2nd] if t == "CBLOL_2ND" else idx[t]  # noqa: E731
    return {
        "names": names, "idx": idx, "pi_slot": pi_slot,
        "region": [cfg["teams"][n]["region"] for n in names] + ["PLAY_IN"],
        "play_in": [resolve(t) for t in cfg["play_in"]],
        "pools": {int(k): [pi_slot if t == "PLAY_IN_WINNER" else idx[t] for t in v]
                  for k, v in cfg["swiss_pools"].items()},
        "round1": [tuple(resolve(t) for t in pair) for pair in cfg.get("play_in_round1", [])] or None,
        "swiss_round1": [tuple(pi_slot if t == "PLAY_IN_WINNER" else idx[t] for t in pair)
                         for pair in cfg.get("swiss_round1", [])] or None,
    }
