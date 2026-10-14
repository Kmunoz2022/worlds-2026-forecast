"""Log predictions before matches and grade them after (see docs/DECISIONS.md).

Usage:
  .venv/bin/python scripts/ledger.py auto                 # Play-In: predict every pairing still possible
  .venv/bin/python scripts/ledger.py add "Gen.G" "T1" --bo 3 --stage "Swiss R1"
  .venv/bin/python scripts/ledger.py grade                # grade against Data/processed, write the record

Predictions are only ever appended. Run `auto`/`add` before the matches are
played; `grade` matches each played Worlds series to the latest prediction
made before it started.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from lolpred import ledger  # noqa: E402
from lolpred.config import PROCESSED_DIR  # noqa: E402

CONFIG = ROOT / "config" / "worlds2026.json"
SEASON = 2026


def load_tables():
    return (pd.read_parquet(PROCESSED_DIR / "matches.parquet"), pd.read_parquet(PROCESSED_DIR / "series.parquet"))


def graded_now(cfg):
    m, s = load_tables()
    return ledger.grade(ledger.load(), m, s, SEASON)


def cmd_auto(cfg):
    """Play-In is double elimination: a team is out after 2 series losses and
    the event ends when one team is left. Predict every pairing of teams
    still alive that has no open (ungraded) prediction."""
    from lolpred.predict import MatchPredictor
    teams = [cfg["cblol_2nd"][0] if t == "CBLOL_2ND" else t for t in cfg["play_in"]]
    g = graded_now(cfg)
    losses = {t: 0 for t in teams}
    for e in g:
        if e["status"] == "graded" and e["stage"] == "Play-In":
            losses[e["team_b"] if e["winner"] == e["team_a"] else e["team_a"]] += 1
    alive = [t for t in teams if losses[t] < 2]
    if len(alive) <= 1:
        print(f"Play-In finished: {alive[0] if alive else '?'} advanced. Nothing to predict.")
        return
    open_pairs = {frozenset((e["team_a"], e["team_b"])) for e in g if e["status"] == "pending"}
    pred = MatchPredictor()
    new = [ledger.make_entry(pred, cfg, a, b, 5, "Play-In")
           for a, b in itertools.combinations(alive, 2) if frozenset((a, b)) not in open_pairs]
    ledger.append(new)
    for e in new:
        print(f"logged {e['team_a']} vs {e['team_b']} Bo5: {e['team_a']} {e['p_series_a']:.0%} -> {e['pick']} {e['likely_score']}")
    print(f"{len(new)} new predictions; alive: {', '.join(alive)}")


def cmd_add(cfg, a, b, bo, stage):
    from lolpred.predict import MatchPredictor
    for t in (a, b):
        if t not in cfg["teams"]:
            sys.exit(f"unknown team {t!r}; use names from config/worlds2026.json")
    e = ledger.make_entry(MatchPredictor(), cfg, a, b, bo, stage)
    ledger.append([e])
    print(f"logged {a} vs {b} Bo{bo} ({stage}): {a} {e['p_series_a']:.0%} -> {e['pick']} {e['likely_score']}")


def cmd_grade(cfg):
    g = graded_now(cfg)
    summ = ledger.summary(g)
    (PROCESSED_DIR / "ledger_graded.json").write_text(json.dumps({"summary": summ, "entries": g}, indent=2, ensure_ascii=False))
    done = [e for e in g if e["status"] == "graded"]
    print(f"{len(g)} predictions, {len(done)} graded. Summary: {summ}")
    for e in done:
        print(f"  {e['played_at']} {e['stage']}: {e['team_a']} {e['a_wins']}-{e['b_wins']} {e['team_b']} | "
              f"pick {e['pick']} {e['likely_score']} ({'right' if e['correct'] else 'wrong'})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("auto")
    add = sub.add_parser("add")
    add.add_argument("team_a")
    add.add_argument("team_b")
    add.add_argument("--bo", type=int, required=True, choices=[1, 3, 5])
    add.add_argument("--stage", required=True)
    sub.add_parser("grade")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.cmd == "auto":
        cmd_auto(cfg)
    elif args.cmd == "add":
        cmd_add(cfg, args.team_a, args.team_b, args.bo, args.stage)
    else:
        cmd_grade(cfg)


if __name__ == "__main__":
    main()
