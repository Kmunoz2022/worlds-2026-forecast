"""One command after downloading a new Oracle's Elixir CSV (see docs/DECISIONS.md).

Rebuilds the data, ratings and model, re-simulates the rest of Worlds,
grades the prediction ledger, updates the Crystal Ball tracker, archives finished games' minute-by-
minute timelines, and exports the website data.
Locked picks (ledger/*_locked.json) are never touched.

Usage: .venv/bin/python scripts/refresh.py [--skip-train]
Then republish site/index.html + site/data.json.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args):
    t = time.time()
    print(f"\n>>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(args)}")
    print(f"    done in {time.time() - t:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="reuse the saved game model (saves ~4 min)")
    args = ap.parse_args()

    empty = [p.name for p in sorted((ROOT / "Data").glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
             if p.stat().st_size == 0]
    if empty:
        sys.exit(f"empty CSV files, re-download them first: {', '.join(empty)}")

    run("scripts/build_data.py")
    run("scripts/elo_backtest.py")
    if not args.skip_train:
        run("scripts/train_models.py")
    run("scripts/simulate_worlds.py")
    run("scripts/ingame_eval.py")
    run("scripts/live_model.py")
    # Predictions are logged by the scheduled GitHub job (scripts/auto_ledger.py), not here,
    # so the ledger has one writer. This grade from Oracle's Elixir is a cross-check.
    run("scripts/ledger.py", "grade")
    run("scripts/track_crystal.py", "--rebuild")
    run("scripts/archive_timelines.py")
    run("scripts/export_site.py")
    run("scripts/auto_ledger.py", "record")  # local copy of site/record.json for previews
    print("\nAll done. Republish site/index.html with site/data.json to update the website.")


if __name__ == "__main__":
    main()
