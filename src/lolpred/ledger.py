"""Prediction ledger: predictions are written down *before* a match and graded
after it, so the 2026 record can't be fitted to results (see docs/DECISIONS.md).

ledger/predictions.jsonl is append-only: one JSON object per predicted series.
An entry is never edited. If the same two teams meet again (e.g. a Play-In
rematch in the grand final), a new entry is added before that match, and each
played series is graded against the most recent unused prediction made
before it started.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import ROOT
from .series import score_distribution, series_win_prob

LEDGER = ROOT / "ledger" / "predictions.jsonl"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def append(entries: list[dict]) -> None:
    LEDGER.parent.mkdir(exist_ok=True)
    with LEDGER.open("a") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def make_entry(predictor, cfg, a: str, b: str, best_of: int, stage: str) -> dict:
    ta, tb = cfg["teams"][a]["teamid"], cfg["teams"][b]["teamid"]
    p = predictor.game_prob(ta, tb)
    dist = score_distribution(p, best_of)
    ps = series_win_prob(p, best_of)
    (sa, sb), _ = max(dist.items(), key=lambda kv: kv[1])
    return {"predicted_at": now_utc(), "data_through": predictor.trained_through, "stage": stage,
            "team_a": a, "team_b": b, "teamid_a": ta, "teamid_b": tb, "best_of": best_of,
            "p_game_a": round(p, 4), "p_series_a": round(ps, 4),
            "score_dist": {f"{x}-{y}": round(v, 4) for (x, y), v in dist.items()},
            "pick": a if ps >= 0.5 else b, "likely_score": f"{sa}-{sb}"}


def grade(entries: list[dict], matches: pd.DataFrame, series: pd.DataFrame, season: int) -> list[dict]:
    """Attach the actual result to each entry whose series has been played.
    Returns a new list (entries themselves are not modified)."""
    s = series[(series["league"] == "WLDs") & (series["season"] == season)].sort_values("start")
    games = matches.groupby("series_id")
    out = [dict(e, status="pending") for e in entries]
    used = set()
    for _, row in s.iterrows():
        pair = {row["team_a"], row["team_b"]}
        cands = [i for i, e in enumerate(out) if i not in used and {e["teamid_a"], e["teamid_b"]} == pair
                 and pd.Timestamp(e["predicted_at"]).tz_localize(None) < row["start"]]
        if not cands:
            continue
        i = max(cands, key=lambda j: out[j]["predicted_at"])
        used.add(i)
        e = out[i]
        a_is_series_a = e["teamid_a"] == row["team_a"]
        aw = int(row["a_wins"] if a_is_series_a else row["b_wins"])
        bw = int(row["b_wins"] if a_is_series_a else row["a_wins"])
        g = games.get_group(row["series_id"]).sort_values("date")
        a_won = np.where(g["blue_teamid"] == e["teamid_a"], g["blue_win"], 1 - g["blue_win"]).astype(int)
        winner = e["team_a"] if aw > bw else e["team_b"]
        e.update(status="graded", played_at=row["start"].strftime("%Y-%m-%d %H:%M"), a_wins=aw, b_wins=bw,
                 winner=winner, correct=bool(winner == e["pick"]),
                 score_correct=bool(f"{aw}-{bw}" == e["likely_score"]),
                 p_actual_score=e["score_dist"].get(f"{aw}-{bw}"), game_results=a_won.tolist())
    return out


def summary(graded: list[dict]) -> dict:
    done = [e for e in graded if e["status"] == "graded"]
    if not done:
        return {"series": 0}
    y = np.concatenate([e["game_results"] for e in done]).astype(float)
    p = np.concatenate([[e["p_game_a"]] * len(e["game_results"]) for e in done])
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    return {"series": len(done), "series_correct": sum(e["correct"] for e in done),
            "scores_exact": sum(e["score_correct"] for e in done),
            "games": int(len(y)), "game_accuracy": float(np.mean((p >= 0.5) == (y == 1))),
            "game_log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "game_brier": float(np.mean((p - y) ** 2))}
