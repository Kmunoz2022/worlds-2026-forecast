"""Simulate Worlds 2026 and produce pick'em recommendations.

Reads config/worlds2026.json (teams, seeds, Play-In, Swiss pools), predicts
every matchup with the calibrated game model, and runs the Monte Carlo
tournament once per candidate for the unresolved CBLOL slot.

Usage: .venv/bin/python scripts/simulate_worlds.py [--sims 20000]
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
from lolpred.predict import MatchPredictor  # noqa: E402
from lolpred.series import KAPPA, score_distribution, series_win_prob  # noqa: E402
from lolpred.record import RESULTS, read_jsonl  # noqa: E402
from lolpred.tournament import RECORDS, pickem, setup_from_config, simulate  # noqa: E402

CONFIG = ROOT / "config" / "worlds2026.json"
OUT_JSON = PROCESSED_DIR / "worlds2026_sim.json"
RESULTS_MD = ROOT / "docs" / "results" / "worlds2026_sim.md"


def series_model_check():
    """Predicted vs actual sweep rates on 2023+ Bo3/Bo5 series (out-of-sample game probs)."""
    m = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    p = pd.read_parquet(PROCESSED_DIR / "model_predictions.parquet")[["gameid", "p_LR full (intl x5) +cal"]]
    m = m.merge(p.rename(columns={"p_LR full (intl x5) +cal": "p"}), on="gameid")
    g1 = m[(m["date"] >= "2023-01-01") & m["p"].notna() & m["best_of"].isin([3, 5]) & (m["series_game_no"] == 1)]
    s = pd.read_parquet(PROCESSED_DIR / "series.parquet").set_index("series_id")[["a_wins", "b_wins"]]
    g1 = g1.join(s, on="series_id")
    pa = np.where(g1["blue_teamid"] == g1["team_a"], g1["p"], 1 - g1["p"])
    rows = []
    for bo in (3, 5):
        sel = (g1["best_of"] == bo).to_numpy()
        w = (bo + 1) // 2
        actual = ((g1.loc[sel, "a_wins"] == 0) | (g1.loc[sel, "b_wins"] == 0)).mean()
        for name, kappa in (("independent games", 1e9), (f"beta-binomial (kappa={KAPPA:g})", KAPPA)):
            pred = np.mean([score_distribution(x, bo, kappa)[(w, 0)] + score_distribution(x, bo, kappa)[(0, w)]
                            for x in pa[sel]])
            rows.append({"format": f"Bo{bo}", "series": int(sel.sum()), "model": name,
                         "predicted sweep %": round(100 * pred, 1), "actual sweep %": round(100 * actual, 1)})
    return pd.DataFrame(rows)


def played_series():
    """Worlds series already played, oldest first: ledger/results.jsonl, which the
    scheduled job fills from the official feed after every match (the CSV can be
    days behind)."""
    return sorted(read_jsonl(RESULTS), key=lambda r: r["start"])


def swiss_state(results, idx):
    """Played Swiss series by pair, and each round's pairings (a team's n-th
    Swiss series is in round n)."""
    by_pair, rounds, n = {}, {}, {}
    for r in results:
        if r["stage"] != "Swiss":
            continue
        a, b = idx[r["team_a"]], idx[r["team_b"]]
        by_pair[frozenset((a, b))] = (a, b, r["a_wins"], r["b_wins"])
        rounds.setdefault(n.get(a, 0), []).append((a, b))
        n[a], n[b] = n.get(a, 0) + 1, n.get(b, 0) + 1
    return by_pair, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    teams = cfg["teams"]
    names = list(teams)
    idx = {n: i for i, n in enumerate(names)}
    region = [teams[n]["region"] for n in names] + ["PLAY_IN"]
    pi_slot = len(names)

    pred = MatchPredictor()
    print(f"model data through {pred.trained_through}")
    P = pred.prob_matrix([teams[n]["teamid"] for n in names])

    info = pd.DataFrame([{**pred.team_info(teams[n]["teamid"]), "team": n, "region": teams[n]["region"]}
                         for n in names]).set_index("team")
    # Neutral single-game win prob vs the average of the other qualified teams.
    info["avg_game_win%"] = [100 * np.mean([P[i, j] for j in range(len(names)) if j != i]) for i in range(len(names))]
    info = info.sort_values("strength", ascending=False)
    pd.set_option("display.width", 200)
    print("\n== team strength\n", info[["region", "strength", "team_rating", "league_offset", "games",
                                        "avg_game_win%"]].round(1).to_string())

    results = played_series()
    played = [(r["team_a"], r["team_b"], r["a_wins"], r["b_wins"]) for r in results if r["stage"] == "Play-In"]
    sw_results, sw_rounds = swiss_state(results, idx)
    if results:
        print("\n== Results so far (kept as played)")
        for r in results:
            print(f"   {r['stage']}: {r['team_a']} {r['a_wins']}-{r['b_wins']} {r['team_b']}")

    scenarios = {}
    for cand in cfg["cblol_2nd"]:
        others = [c for c in cfg["cblol_2nd"] if c != cand]
        st = setup_from_config(cfg, cand)
        play_in, pools, round1 = st["play_in"], st["pools"], st["round1"]
        totals = []
        counts = simulate(P, play_in, pools, pi_slot, region, n_sims=args.sims, seed=0,
                          play_in_round1=round1, game_totals=totals,
                          play_in_played=[(idx[a], idx[b], wa, wb) for a, b, wa, wb in played],
                          swiss_round1=st["swiss_round1"], swiss_results=sw_results, swiss_known_pairs=sw_rounds)
        rows = []
        for n in names:
            if n in others:
                continue
            c = counts[idx[n]]
            swiss_n = max(c["swiss"], 1)
            row = {"team": n, "region": teams[n]["region"],
                   "play_in_win": c["play_in_win"] / args.sims if idx[n] in play_in else None,
                   "reach_swiss": c["swiss"] / args.sims,
                   **{r: c[r] / args.sims for r in RECORDS},
                   "qualify": c["qualify"] / args.sims, "semis": c["sf"] / args.sims,
                   "final": c["final"] / args.sims, "champion": c["champion"] / args.sims,
                   "record_given_swiss": {r: c[r] / swiss_n for r in RECORDS}}
            rows.append(row)
        df = pd.DataFrame(rows).set_index("team").sort_values("champion", ascending=False)
        picks = pickem({n: {"3-0": r["3-0"], "0-3": r["0-3"], "qualify": r["qualify"]}
                        for n, r in df.iterrows() if r["reach_swiss"] > 0.99})
        scenarios[cand] = {"table": df, "pickem": picks,
                           "games": {k: [t[k] for t in totals] for k in ("play_in", "swiss", "knockouts")}}
        show = df.drop(columns=["record_given_swiss"]).copy()
        pct = show.columns.drop("region")
        show[pct] = (show[pct].astype(float) * 100).round(1)
        print(f"\n== scenario: CBLOL 2nd seed = {cand}  ({args.sims} sims, % of sims)\n{show.to_string()}")
        print(f"\npick'em: 3-0 {picks['3-0']} | 0-3 {picks['0-3']} | advance {picks['advance']}")

    # Example series score distributions (neutral side).
    examples = [("Gen.G", "Anyone's Legend", 5), ("Hanwha Life Esports", "Bilibili Gaming", 5),
                ("T1", "G2 Esports", 3), ("Gen.G", "Team Liquid", 1)]
    ex_rows = []
    for a, b, bo in examples:
        p = P[idx[a], idx[b]]
        dist = score_distribution(p, bo)
        ex_rows.append({"matchup": f"{a} vs {b} (Bo{bo})", "game %": round(100 * p, 1),
                        "series %": round(100 * series_win_prob(p, bo), 1),
                        **{f"{x}-{y}": round(100 * v, 1) for (x, y), v in sorted(dist.items(), key=lambda kv: (-kv[0][0], kv[0][1]))}})
    ex = pd.DataFrame(ex_rows).fillna("").set_index("matchup")
    print("\n== example series score distributions (%)\n", ex.to_string())

    check = series_model_check()
    print("\n== series model check (2023+)\n", check.to_string(index=False))

    # --- save for the website
    out = {"generated_from_data_through": pred.trained_through, "sims": args.sims, "kappa": KAPPA,
           "teams": {n: {"region": teams[n]["region"], "seed": teams[n]["seed"],
                         "strength": float(info.loc[n, "strength"]),
                         "avg_game_win": float(info.loc[n, "avg_game_win%"]) / 100} for n in names},
           "game_prob": {a: {b: float(P[idx[a], idx[b]]) for b in names if b != a} for a in names},
           "scenarios": {k: {"pickem": v["pickem"], "games_per_sim": v["games"],
                             "teams": json.loads(v["table"].reset_index().to_json(orient="records"))}
                         for k, v in scenarios.items()}}
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    md = ["# Worlds 2026 simulation\n",
          f"Generated by `scripts/simulate_worlds.py` ({args.sims} simulations, data through "
          f"{pred.trained_through}). Seeds/pools from `config/worlds2026.json` (partly unconfirmed).\n",
          "## Team strength\n", "```", info[["region", "strength", "games", "avg_game_win%"]].round(1).to_string(), "```\n"]
    for k, v in scenarios.items():
        show = v["table"].drop(columns=["record_given_swiss"]).copy()
        pct = show.columns.drop("region")
        show[pct] = (show[pct].astype(float) * 100).round(1)
        md += [f"## Scenario: CBLOL 2nd seed = {k} (% of simulations)\n", "```", show.to_string(), "```\n",
               f"Pick'em: **3-0** {', '.join(v['pickem']['3-0'])} · **0-3** {', '.join(v['pickem']['0-3'])} · "
               f"**advance** {', '.join(v['pickem']['advance'])}\n"]
    md += ["## Example series score distributions (%)\n", "```", ex.to_string(), "```\n",
           "## Series model check (2023+ Bo3/Bo5)\n", "```", check.to_string(index=False), "```\n"]
    RESULTS_MD.write_text("\n".join(md))
    print(f"\nwrote {OUT_JSON}, {RESULTS_MD}")


if __name__ == "__main__":
    main()
