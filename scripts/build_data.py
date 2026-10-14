"""Build Data/processed/{team_games,matches,series}.parquet and print a data report.

Usage: .venv/bin/python scripts/build_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from lolpred.config import PROCESSED_DIR  # noqa: E402
from lolpred.data import build_all  # noqa: E402
from lolpred.splits import holdout_2026_worlds, walk_forward, worlds_backtests  # noqa: E402


def main():
    tg, m, s, log = build_all()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tg.to_parquet(PROCESSED_DIR / "team_games.parquet", index=False)
    m.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
    s.to_parquet(PROCESSED_DIR / "series.parquet", index=False)

    print("== cleaning")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print(f"\n== tables\n  team_games {tg.shape}  matches {m.shape}  series {s.shape}")
    print(f"  dates {m['date'].min():%Y-%m-%d} -> {m['date'].max():%Y-%m-%d}")
    print(f"  teams {tg['teamid'].nunique()}  leagues {m['league'].nunique()}")

    print("\n== games per season (all / cross-league / major intl)")
    print(m.groupby("season").agg(all=("gameid", "size"), cross=("cross_league", "sum"),
                                  major_intl=("major_intl", "sum")).to_string())

    print(f"\n== blue side win rate: {m['blue_win'].mean():.3f}")

    print("\n== series formats (inferred best_of)")
    print(s["best_of"].value_counts(dropna=False).rename(index={float('nan'): 'ambiguous'}).to_string())
    print("\n  score distribution by format")
    print(s.dropna(subset=["best_of"]).groupby("best_of")["score"].value_counts(normalize=True)
          .round(3).to_string())
    print(f"\n  ambiguous series by league (top 8):\n"
          f"{s[s['best_of'].isna()]['league'].value_counts().head(8).to_string()}")

    print("\n== EVENT_ONLY teams (no domestic data), by event")
    eo = tg[tg["home_league"].str.startswith("EVENT_ONLY")]
    print(eo.groupby("home_league")["teamid"].nunique().to_string() if len(eo) else "  none")

    print("\n== splits")
    for sp in worlds_backtests(m):
        print(f"  {sp.name}: train {len(sp.train(m)):>6} games | test {len(sp.test(m)):>4} "
              f"(from {sp.test_start:%Y-%m-%d})")
    wf = walk_forward(m)
    print(f"  walk_forward: {len(wf)} quarterly folds, {wf[0].name} .. {wf[-1].name}")
    h = holdout_2026_worlds(m)
    print(f"  holdout 2026 Worlds: {len(h.test(m)) if h else 0} games so far"
          + (f" (from {h.test_start:%Y-%m-%d})" if h else ""))

    print("\n== 2026 WLDs-labelled games that are NOT cross-league (likely qualifiers)")
    q = m[(m["league"] == "WLDs") & (m["season"] == 2026) & ~m["cross_league"]]
    print(q[["date", "blue_teamname", "red_teamname", "blue_home_league"]].to_string(index=False)
          if len(q) else "  none")
    print(f"\nwrote {PROCESSED_DIR}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
