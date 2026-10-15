"""CI check: ledger files are append-only and locked picks never change.

Compares the working tree with an earlier commit (default: the previous one).
Every ledger/*.jsonl must start with exactly its earlier content, and every
ledger/*_locked*.json must be byte-for-byte unchanged.

Usage: python scripts/check_ledger.py [BASE_COMMIT]
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def at(commit, path):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    return r.stdout if r.returncode == 0 else None  # None: the file didn't exist then


def main():
    base = sys.argv[1] if len(sys.argv) > 1 and set(sys.argv[1]) != {"0"} else "HEAD~1"
    if subprocess.run(["git", "rev-parse", "--verify", "-q", base], cwd=ROOT, capture_output=True).returncode:
        print(f"no commit {base} to compare with; skipping")
        return
    bad = []
    for f in sorted((ROOT / "ledger").glob("*")):
        rel = f.relative_to(ROOT).as_posix()
        old, new = at(base, rel), f.read_bytes()
        if old is None:
            continue
        if f.suffix == ".jsonl" and not new.startswith(old):
            bad.append(f"{rel}: earlier lines were changed or removed (append-only)")
        elif "_locked" in f.name and new != old:
            bad.append(f"{rel}: locked picks changed")
    if bad:
        sys.exit("\n".join(bad))
    print(f"ledger OK against {base}")


if __name__ == "__main__":
    main()
