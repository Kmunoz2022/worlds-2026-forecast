"""Answer the Worlds 2026 Crystal Ball questions (see docs/DECISIONS.md).

Runs the tournament simulation with full per-series logs, then layers player
stats, game lengths and champion picks on top. Writes
Data/processed/crystal_ball.json for the website.

Usage: .venv/bin/python scripts/crystal_ball.py [--sims 5000] [--rebuild]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.crystal import (INTL_EVENTS, POSITIONS, LengthModel, TeamPools, champion_pick_history,  # noqa: E402
                             champion_sim, current_rosters, load_player_games, p_at_least_one,
                             rare_event_rate, simulate_players, unique_champions_model,
                             worlds_like_games)
from lolpred.draft import champion_rates, load_draft_events  # noqa: E402
from lolpred.predict import MatchPredictor  # noqa: E402
from lolpred.splits import worlds_start  # noqa: E402
from lolpred.tournament import setup_from_config, simulate  # noqa: E402

CONFIG = ROOT / "config" / "worlds2026.json"
ROLE = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
# Worlds Finals MVPs 2017-2024 by role, from public records (not in our data):
# 2017 Ruler (bot), 2018 Ning (jng), 2019 Tian (jng), 2020 Canyon (jng),
# 2021 Scout (mid), 2022 Zeka (mid), 2023 Faker (mid), 2024 Faker (mid).
MVP_HISTORY = {"mid": 4, "jng": 3, "bot": 1, "top": 0, "sup": 0}
FAKER_CAREER, FAKER_TARGET = 616, 700

def options(series, labeller=str, top=8):
    s = series.sort_values(ascending=False).head(top)
    return [{"label": labeller(k), "p": round(float(v), 4)} for k, v in s.items() if v > 0]


def question(qid, text, opts, method, pick=None, extra=None):
    pick = pick or (opts[0]["label"] if opts else None)
    p = next((o["p"] for o in opts if o["label"] == pick), None)
    return {"id": qid, "question": text, "pick": pick, "pick_p": p, "options": opts,
            "method": method, **(extra or {})}


def explain(Q, ps, plabel, outcomes, st, cfg, names, kills, pid, need, n_games, rates, wr_mean, avg_deaths,
            eligible, lengths, shares, mu, uev, draws, teemo_picks, teemo_games, top_mvp, hist):
    """Plain-language 'why this pick' for every question, built from the numbers."""
    n = len(outcomes)
    idx = st["idx"]
    play_in = {names[i] for i in st["play_in"]}
    reach = {t: np.mean([o["play_in_winner"] == idx[t] for o in outcomes]) if t in play_in else 1.0 for t in names}
    qual = {t: np.mean([o["records"].get(idx[t], (0, 0))[0] == 3 for o in outcomes]) for t in names}
    semis = {t: np.mean([idx[t] in o["sf_losers"] or idx[t] in (o["champion"], o["runner_up"]) for o in outcomes])
             for t in names}
    team_games = {t: float(np.mean([sum(s[3] + s[4] for s in o["series"] if idx[t] in (s[1], s[2])) for o in outcomes]))
                  for t in names}
    pl = ps["players"]

    def who(i):
        return pl[i][2]

    def top_idx(key, k=3):
        return list(np.argsort(-ps[key])[:k])

    W = {}
    c = Q[0]["options"]
    lead = c[0]["label"]
    W["champ_wr"] = (
        f"The win-rate leader is almost always a champion that plays only 5-8 games and barely loses, so luck "
        f"decides most of it and no champion is above {c[0]['p']:.0%}. {lead} has one of the best win rates in "
        f"recent top-tier games ({wr_mean[lead]:.0%} after adjusting for small samples) and is picked often enough "
        f"to reach 5 games in {eligible[lead]:.0%} of simulated Worlds. "
        + " ".join(f"{o['label']} ({wr_mean[o['label']]:.0%}, reaches 5 games {eligible[o['label']]:.0%})."
                   for o in c[1:3]))
    c = Q[1]["options"]
    lead = c[0]["label"]
    W["champ_deaths"] = (
        f"{lead} dies {avg_deaths[lead]:.1f} times per game in recent top-tier and newest-patch games, the most of any "
        f"champion that's picked often enough to reach 5 games (it does in {eligible[lead]:.0%} of simulations). "
        f"Divers and engage supports die most because they start fights. Next: "
        + ", ".join(f"{o['label']} {avg_deaths[o['label']]:.1f} deaths" for o in c[1:3]) + ".")
    for key, stat, fmt in (("kda", "KDA", "{:.1f}"), ("dmg", "damage per game", "{:,.0f}"),
                           ("kp", "kill participation", "{:.0%}")):
        ti = top_idx(key)
        i = ti[0]
        team = pl[i][0]
        stage = (f"{team} reaches the Swiss stage {reach[team]:.0%} of the time" if team in play_in
                 else f"{team} advances from Swiss {qual[team]:.0%} of the time")
        text = (f"{who(i)} has a typical Worlds {stat} of {fmt.format(ps[key + '_mean'][i])} in the simulation, "
                f"over about {ps['games_mean'][i]:.0f} games ({stage}). ")
        if key == "kda":
            text += ("KDA leaders come from winning teams, since losses wreck KDA, and mids and bot laners die least. "
                     "A deep run helps, but a short hot streak can also top the list.")
        elif key == "dmg":
            text += ("Bot laners and mids deal the most damage, and longer games mean more damage, so a carry on a team "
                     "that plays long, close series is the best bet.")
        else:
            text += ("That's the highest typical number of any player: supports join most of their team's kills, and "
                     "in FURIA's games against strong opponents he's in almost all of them. The Play-In is all best-of-"
                     "fives, so FURIA plays about as many games as a Swiss team. Teams that rarely get kills tend to "
                     "produce the leader: Atlen (CTBC) led in 2022, Harp (DetonatioN) in 2023. The risk is that his "
                     "numbers come from CBLOL; against Worlds teams it could be lower, which is why the steadier "
                     "pick from a stronger team is next on the list.")
        text += " Next: " + ", ".join(
            f"{who(j)} ({pl[j][0]}, {fmt.format(ps[key + '_mean'][j])} over ~{ps['games_mean'][j]:.0f} games)"
            for j in ti[1:]) + "."
        W[key] = text
    peyz, guma = kills[:, pid["Peyz"]], kills[:, pid["Gumayusi"]]
    W["peyz_guma"] = (
        f"T1 and HLE are expected to play a similar number of games ({team_games['T1']:.0f} vs "
        f"{team_games['Hanwha Life Esports']:.0f}), so kills per game decides it. Peyz averages more kills in T1's "
        f"games than Gumayusi does in HLE's, which adds up to about {peyz.mean():.0f} vs {guma.mean():.0f} expected. "
        f"Gumayusi wins it mainly if HLE goes much deeper than T1.")
    faker = kills[:, pid["Faker"]]
    W["faker_700"] = (
        f"Faker needs {need} kills at this Worlds. He'd get there only with a near-record number of games and a "
        f"high kill rate: T1 plays about {team_games['T1']:.0f} games on average, giving {faker.mean():.0f} kills "
        f"expected, and even in T1's best 10% of runs he gets about {np.percentile(faker, 90):.0f}.")
    W["teemo"] = (
        f"Teemo has {teemo_picks} picks in {teemo_games:,} pro games since 2025 across every league, none in "
        f"recent top-tier play, and one at a Worlds since 2022 (a 2022 Play-In game). Fearless draft makes "
        f"teams dig deeper into the champion pool, but not that deep.")
    W["unique"] = (
        f"With Fearless draft a champion can't be reused within a series, so recent international events picked "
        f"around 100 different champions in 70-100 games (Worlds 2025: 95 in 84 games; MSI 2026: 101 in 71). "
        f"Worlds 2026 should run about {n_games.mean():.0f} games, so the most likely count is about "
        f"{np.median(draws):.0f}. 0-99 is the single most likely answer only because it's the widest range; "
        f"100 or more is more likely overall ({np.mean(draws >= 100):.0%}).")
    W["len32"] = (
        f"Recent Worlds-level events land close to half and half: Worlds 2024 and 2025 had 48-49% of games end "
        f"before 32:00, while MSI 2026 had 51% and Worlds 2022-23 had 60%+. Weighted toward recent years the "
        f"estimate is {mu:.0%} before, so 'before' is a lean, not a lock.")
    W["kills15"] = (
        "Fifteen kills in one game needs a long, bloody stomp. It happened in only 1 of about 430 Worlds games "
        "since 2022, but 5 times at MSI since 2024. Over about 100 games that adds up to roughly a coin flip.")
    W["cs450"] = (
        "450 CS needs a long game and a farming carry. Every Worlds since 2022 had 2-6 such games, about 4% of "
        f"games, so over ~{n_games.mean():.0f} games it's nearly certain.")
    mvp = ", ".join(f"{p} {v:.0%}" for p, v in top_mvp[:3])
    W["mvp_role"] = (
        f"Mid players won 4 of the 8 Finals MVPs from 2017 to 2024 and junglers 3. Gen.G is the title favourite, "
        f"and the simulation's most likely individual MVPs are {mvp}. Ruler would make it 'Bot', but across all "
        f"the likely winners mid is still the best single answer.")
    reg = {}
    for o in outcomes:
        for r in {st["region"][t] for t, rr in o["records"].items() if rr[0] == 3}:
            reg[r] = reg.get(r, 0) + 1
    reg = {r: v / n for r, v in sorted(reg.items(), key=lambda kv: -kv[1])}
    W["regions_ko"] = (
        "LCK and LPL are nearly certain to have a team in knockouts. The count comes down to how many of LEC, "
        "LCS, CBLOL and LCP also get one: " + ", ".join(f"{r} {v:.0%}" for r, v in reg.items()) + ".")
    W["lcs_ko"] = (
        f"Three LCS teams get a shot: Team Liquid advances {qual['Team Liquid']:.0%} of the time, LYON "
        f"{qual['LYON']:.0%}, and Cloud9 {qual['Cloud9']:.0%} (it must win the Play-In first). At least one of "
        f"them makes it in about {Q[[q['id'] for q in Q].index('lcs_ko')]['options'][0]['p']:.0%} of simulations.")
    W["hle_t1"] = (
        f"This is close to a coin flip. T1 and HLE are rated almost the same and both are heavy favourites to "
        f"advance (T1 {qual['T1']:.0%}, HLE {qual['Hanwha Life Esports']:.0%}); T1 reaches semis "
        f"{semis['T1']:.0%} vs HLE {semis['Hanwha Life Esports']:.0%}. About 1 time in 8 they go out in the "
        f"same round.")
    pace = sorted(lengths.pace.items(), key=lambda kv: kv[1])
    sq = Q[[q["id"] for q in Q].index("shortest")]["options"]
    W["shortest"] = (
        "The shortest game is usually a heavy favourite stomping a much weaker team, so the best bets are strong "
        f"teams that play several mismatches. {sq[0]['label']} plays {abs(lengths.pace[sq[0]['label']]):.1f} minutes "
        f"{'faster' if lengths.pace[sq[0]['label']] < 0 else 'slower'} than expected this year and is a big favourite "
        f"in most of its Swiss games; {sq[1]['label']} is the biggest favourite of all "
        f"({lengths.pace[sq[1]['label']]:+.1f} min pace). Fastest teams in 2026: "
        + ", ".join(f"{t} {abs(v):.1f} min faster" for t, v in pace[:3] if v < 0) + ".")
    for q in Q:
        q["why"] = W.get(q["id"], "")


def placement(out, t):
    """Lower is better. Knockout stages by round; Swiss exits by wins; Play-In exits last."""
    if out["champion"] == t:
        return 1
    if out["runner_up"] == t:
        return 2
    if t in out["sf_losers"]:
        return 3
    if t in out["qf_losers"]:
        return 5
    if t in out["records"]:
        return 9 + (2 - out["records"][t][0])
    return 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--rebuild", action="store_true", help="re-read raw CSVs for player and draft data")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    st = setup_from_config(cfg, cfg["cblol_2nd"][0])
    names = st["names"]
    pred = MatchPredictor()
    P = pred.prob_matrix([cfg["teams"][n]["teamid"] for n in names])
    outcomes, totals = [], []
    simulate(P, st["play_in"], st["pools"], st["pi_slot"], st["region"], n_sims=args.sims, seed=1,
             play_in_round1=st["round1"], game_totals=totals, outcomes=outcomes)
    n_games = np.array([sum(t.values()) for t in totals])
    rng = np.random.default_rng(2)
    print(f"{args.sims} tournaments, {n_games.mean():.0f} games on average")

    pg = load_player_games(rebuild=args.rebuild)
    ev = load_draft_events(rebuild=args.rebuild)
    rosters = current_rosters(cfg, pg)
    elo = pd.read_parquet(PROCESSED_DIR / "elo_predictions.parquet")
    opp_strength = pd.concat([
        elo.set_index(["gameid", "blue_teamid"])["red_strength"].rename_axis(["gameid", "teamid"]),
        elo.set_index(["gameid", "red_teamid"])["blue_strength"].rename_axis(["gameid", "teamid"])])
    pools = TeamPools(cfg, pg, rosters, opp_strength)
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    mp = pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet")[["gameid", "p_LR full (intl x5) +cal"]]
    lengths = LengthModel(m, mp.rename(columns={"p_LR full (intl x5) +cal": "p"}),
                          {n: cfg["teams"][n]["teamid"] for n in names})
    ps = simulate_players(outcomes, names, pools, P, lengths, seed=3)
    players = ps["players"]
    plabel = {i: f"{p} ({t}, {ROLE[pos]})" for i, (t, pos, p) in enumerate(players)}
    pid = {p: i for i, (_, _, p) in enumerate(players)}

    end = ev["date"].max().normalize() + pd.Timedelta(days=1)
    rates, info = champion_rates(ev, end)
    wr_lead, death_lead, wr_mean, avg_deaths, eligible = champion_sim(pg, rates, info["newest_patch"], n_games)

    Q = []
    Q.append(question("champ_wr", "Which champion will have the highest Win Rate (min 5 games played)?",
                      options(wr_lead), "Simulated champion games and wins: pick rates from recent top-tier drafts "
                      "(newest patch weighted 10x), win rates shrunk toward 50%. Usually won by a champion with "
                      "only 5-8 games, so every option is a long shot.",
                      extra={"detail": {c: {"recent_wr": round(float(wr_mean[c]), 3)} for c in wr_lead.nlargest(8).index}}))
    Q.append(question("champ_deaths", "Which champion will have the highest average Deaths per Game (min 5 games played)?",
                      options(death_lead), "Same simulated champion games; deaths per game resampled from each "
                      "champion's recent games.",
                      extra={"detail": {c: {"avg_deaths": round(float(avg_deaths[c]), 2)} for c in death_lead.nlargest(8).index}}))
    for qid, key, text in (("kda", "kda", "Which pro will finish with the highest average KDA?"),
                           ("dmg", "dmg", "Which pro will finish with the highest average Damage per Game?"),
                           ("kp", "kp", "Which pro will finish with the highest average Kill Participation?")):
        Q.append(question(qid, text, options(pd.Series(ps[key]), plabel.get),
                          "Each simulated game draws the team's own recent game with the same result (win or loss) "
                          "against a Worlds-level opponent and adds its players' stats. Teams that go deep play more "
                          "games, which matters a lot. No minimum games assumed."))

    kills = ps["kills"]
    peyz, guma = kills[:, pid["Peyz"]], kills[:, pid["Gumayusi"]]
    Q.append(question("peyz_guma", "Peyz vs Gumayusi: Who will finish with more total Kills?",
                      [{"label": "Peyz", "p": round(float(np.mean(peyz > guma) + 0.5 * np.mean(peyz == guma)), 4)},
                       {"label": "Gumayusi", "p": round(float(np.mean(guma > peyz) + 0.5 * np.mean(peyz == guma)), 4)}],
                      f"Total kills over the simulated tournament. Expected: Peyz {peyz.mean():.0f}, Gumayusi {guma.mean():.0f}. "
                      "T1 and HLE go about equally deep; Peyz averages more kills per game in 2026."))
    Q[-1]["pick"] = max(Q[-1]["options"], key=lambda o: o["p"])["label"]
    Q[-1]["pick_p"] = max(o["p"] for o in Q[-1]["options"])

    faker = kills[:, pid["Faker"]]
    need = FAKER_TARGET - FAKER_CAREER
    p_faker = float(np.mean(faker >= need))
    Q.append(question("faker_700", "Will Faker reach 700 career Worlds Kills this year? (616 entering)",
                      [{"label": "No", "p": round(1 - p_faker, 4)}, {"label": "Yes", "p": round(p_faker, 4)}],
                      f"Needs {need} kills this Worlds. Simulated: {faker.mean():.0f} on average, 90th percentile "
                      f"{np.percentile(faker, 90):.0f}. He'd need a deep run with many long series and a high kill rate."))

    picks, games = champion_pick_history(pg, "Teemo")
    picks_t1 = int(((pg["champion"] == "Teemo") & pg["league"].isin(INTL_EVENTS)).sum())
    p_teemo = p_at_least_one(picks, games, n_games, rng)
    Q.append(question("teemo", "Will Teemo be picked at least once?",
                      [{"label": "No", "p": round(1 - p_teemo, 4)}, {"label": "Yes", "p": round(p_teemo, 4)}],
                      f"Teemo was picked {picks} times in {games:,} pro games since 2025 (all leagues), and "
                      f"{picks_t1} times at Worlds/MSI/EWC/FST since 2022 (a 2022 Play-In game). It isn't in recent top-tier drafts."))

    main_ids = set()
    for y in sorted(m["season"].unique()):
        start = worlds_start(m, y)
        if start is not None:
            main_ids |= set(m[(m["league"] == "WLDs") & (m["season"] == y) & (m["date"] >= start)]["gameid"])
    a, b, sd, uev = unique_champions_model(pg, main_ids)
    draws = np.exp(a + b * np.log(rng.choice(n_games, 20000)) + rng.normal(0, sd, 20000))
    # The official Crystal Ball answer ranges.
    bins = [(0, 100, "0-99"), (100, 105, "100-104"), (105, 110, "105-109"), (110, 115, "110-114"),
            (115, 120, "115-119"), (120, 999, "120+")]
    Q.append(question("unique", "How many unique champions will be picked?",
                      [{"label": lab, "p": round(float(np.mean((draws >= lo) & (draws < hi))), 4)} for lo, hi, lab in bins],
                      f"Fit on Fearless-draft-era international events (games vs unique champions): "
                      + ", ".join(f"{e} {r['games']} games/{r['unique']}" for e, r in uev.iterrows())
                      + f". Median forecast {np.median(draws):.0f} over ~{n_games.mean():.0f} games.",
                      extra={"median": round(float(np.median(draws)), 1)}))
    Q[-1]["pick"] = max(Q[-1]["options"], key=lambda o: o["p"])["label"]
    Q[-1]["pick_p"] = max(o["p"] for o in Q[-1]["options"])

    # 32-minute split: event-level share before 32:00 varies a lot by meta, so
    # draw the share from the spread seen across recent international events.
    gl = worlds_like_games(pg, main_ids).drop_duplicates("gameid").copy()
    gl["event"] = gl["league"] + " " + gl["date"].dt.year.astype(str)
    shares = gl.groupby("event").agg(n=("gameid", "size"), w=("w", "first"), year=("date", lambda d: d.dt.year.iat[0]),
                                     share=("gamelength", lambda s: float((s < 1920).mean())))
    # Game length follows the meta, so older events count less (half per year).
    shares["w"] = shares["w"] * 0.5 ** (shares["year"].max() - shares["year"])
    mu = float(np.average(shares["share"], weights=shares["n"] * shares["w"]))
    var = float(np.average((shares["share"] - mu) ** 2, weights=shares["w"]))
    k = max(mu * (1 - mu) / max(var, 1e-4) - 1, 2)
    sh = rng.beta(mu * k, (1 - mu) * k, 20000)
    before = rng.binomial(rng.choice(n_games, 20000), sh) / rng.choice(n_games, 20000)
    p_before = float(np.mean(before > 0.5))
    Q.append(question("len32", "Will more games finish before or after the 32 minute mark?",
                      [{"label": "Before 32:00", "p": round(p_before, 4)}, {"label": "After 32:00", "p": round(1 - p_before, 4)}],
                      "Share of games under 32:00 at Worlds since 2022 and MSI since 2024 (half weight): "
                      + ", ".join(f"{e} {s:.0%}" for e, s in shares["share"].items())
                      + f". Weighted average {mu:.0%}, with each year back counting half as much (game length follows the meta; "
                      "the last two Worlds both ran slightly long). Player-stat simulation gives {ps['under32_majority']:.0%} for 'before' as a cross-check."))
    Q[-1]["pick"] = max(Q[-1]["options"], key=lambda o: o["p"])["label"]
    Q[-1]["pick_p"] = max(o["p"] for o in Q[-1]["options"])

    for qid, stat, thr, text in (("kills15", "kills", 15, "Will we see any pro get at least 15 kills in a single game?"),
                                 ("cs450", "total cs", 450, "Will we see any pro reach a CS of at least 450 in a single game?")):
        e, g = rare_event_rate(pg, stat, thr, main_ids)
        p = p_at_least_one(e, g, n_games, rng)
        Q.append(question(qid, text, [{"label": "Yes", "p": round(p, 4)}, {"label": "No", "p": round(1 - p, 4)}],
                          f"Happened in {e:g} of {g:g} weighted games at Worlds since 2022 and MSI since 2024 (MSI at "
                          f"half weight), {e / g:.1%} per game, over ~{n_games.mean():.0f} games. EWC and domestic "
                          f"leagues have more mismatches, so they're left out."))
        Q[-1]["pick"] = max(Q[-1]["options"], key=lambda o: o["p"])["label"]
        Q[-1]["pick_p"] = max(o["p"] for o in Q[-1]["options"])

    hist = {r: v / sum(MVP_HISTORY.values()) for r, v in MVP_HISTORY.items()}
    blend = {r: 0.5 * ps["mvp_role"][r] + 0.5 * hist[r] for r in POSITIONS}
    top_mvp = sorted(ps["mvp_player"].items(), key=lambda kv: -kv[1])[:5]
    Q.append(question("mvp_role", "What will be the role of the Finals MVP winner?",
                      sorted([{"label": ROLE[r], "p": round(v, 4)} for r, v in blend.items()], key=lambda o: -o["p"]),
                      "Half from history (Finals MVPs 2017-2024: mid 4, jungle 3, bot 1), half from the simulation "
                      "(best KDA on the winning team in the simulated final). Most likely individual MVPs in the "
                      "simulation: " + ", ".join(f"{p} {v:.0%}" for p, v in top_mvp) + "."))

    regions = [len({st["region"][t] for t, r in o["records"].items() if r[0] == 3}) for o in outcomes]
    rc = pd.Series(regions).value_counts(normalize=True).sort_index()
    Q.append(question("regions_ko", "How many different regions will be represented in Knockouts?",
                      sorted([{"label": str(k), "p": round(float(v), 4)} for k, v in rc.items()], key=lambda o: -o["p"]),
                      "Distinct regions among the 8 Swiss qualifiers in each simulated tournament."))
    lcs = [st["idx"][n] for n in names if cfg["teams"][n]["region"] == "LCS"]
    p_lcs = float(np.mean([any(o["records"].get(t, (0, 0))[0] == 3 for t in lcs) for o in outcomes]))
    Q.append(question("lcs_ko", "Will a team from LCS reach Knockouts?",
                      sorted([{"label": "Yes", "p": round(p_lcs, 4)}, {"label": "No", "p": round(1 - p_lcs, 4)}], key=lambda o: -o["p"]),
                      "Any of Team Liquid, LYON or Cloud9 (via the Play-In) going 3-x in Swiss."))
    h, t = st["idx"]["Hanwha Life Esports"], st["idx"]["T1"]
    ph = np.array([placement(o, h) for o in outcomes])
    pt = np.array([placement(o, t) for o in outcomes])
    Q.append(question("hle_t1", "Which team will place higher: HLE or T1?",
                      [{"label": "T1", "p": round(float(np.mean(pt < ph)), 4)},
                       {"label": "HLE", "p": round(float(np.mean(ph < pt)), 4)},
                       {"label": "Same placement", "p": round(float(np.mean(ph == pt)), 4)}],
                      "Placement by stage reached (Swiss exits ranked by wins). 'Same placement' is both out in the "
                      "same round, e.g. both losing quarterfinals; check how Riot scores ties."))
    Q[-1]["pick"] = max(Q[-1]["options"][:2], key=lambda o: o["p"])["label"]
    Q[-1]["pick_p"] = next(o["p"] for o in Q[-1]["options"] if o["label"] == Q[-1]["pick"])
    Q.append(question("shortest", "Which team will win the shortest game (duration)?",
                      options(pd.Series(ps["shortest_winner"])),
                      f"Each simulated game's length comes from how lopsided it is (games won by big favourites "
                      f"are shorter: {lengths.a:.1f} {lengths.b:+.1f} x the winner's pre-game chance, in minutes), plus "
                      "each team's own pace in 2026 and random variation copied from real international games. "
                      "Teams that are heavy favourites in several games, and teams that play many games, get the "
                      "most chances. Fastest teams this year (minutes vs expected): "
                      + ", ".join(f"{t} {v:+.1f}" for t, v in sorted(lengths.pace.items(), key=lambda kv: kv[1])[:4]) + "."))

    explain(Q, ps, plabel, outcomes, st, cfg, names, kills, pid, need, n_games, rates, wr_mean, avg_deaths,
            eligible, lengths, shares, mu, uev, draws, picks, games, top_mvp, hist)

    out = {"sims": args.sims, "games_mean": float(n_games.mean()), "sample": info,
           "rosters": rosters, "roster_overrides": cfg.get("roster_overrides", {}), "questions": Q}
    (PROCESSED_DIR / "crystal_ball.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for q in Q:
        print(f"\n{q['question']}\n  PICK: {q['pick']} ({q['pick_p']:.0%})   "
              + " | ".join(f"{o['label']} {o['p']:.0%}" for o in q["options"][:5]))


if __name__ == "__main__":
    main()
