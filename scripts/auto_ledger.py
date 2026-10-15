"""Automatic prediction log, results and live-call grading (see src/lolpred/record.py).

Usage:
  python scripts/auto_ledger.py update            # log upcoming matches, record finished ones (network)
  python scripts/auto_ledger.py record --out site/record.json   # grade everything (no network)

GitHub Actions runs `update` every 30 minutes (.github/workflows/ledger.yml) and
commits the ledger; the Pages deploy runs `record`. Needs LOLESPORTS_API_KEY
(or config/local.json) for `update`.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lolpred import record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "record"])
    ap.add_argument("--out", default=str(ROOT / "site" / "record.json"))
    ap.add_argument("--ledger-dir", help="read/write the ledger files here instead (for testing)")
    args = ap.parse_args()
    if args.ledger_dir:
        d = Path(args.ledger_dir)
        record.PREDICTIONS, record.RESULTS, record.LIVE_CALLS = (d / "predictions.jsonl", d / "results.jsonl",
                                                                  d / "live_calls.jsonl")
    if args.cmd == "update":
        print(json.dumps(record.update()))
    else:
        rec = record.build_record()
        Path(args.out).write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
        s, lv = rec["ledger"]["summary"], rec["live"]
        print(f"record: {s.get('series', 0)} series graded, {lv['games']} games of live calls -> {args.out}")


if __name__ == "__main__":
    main()
