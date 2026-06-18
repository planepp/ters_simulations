#!/usr/bin/env python3
import sys
from pathlib import Path

def is_leaf(d):
    return not any(p.is_dir() for p in d.iterdir())

def is_finished(d):
    out = d / "aims.out"
    if not out.exists():
        return False
    try:
        text = out.read_text(errors="ignore")
        return ("Have a nice day" in text) or ("Invalid ovlp_type" in text)
    except Exception:
        return False

if len(sys.argv) < 2:
    print("Usage: collect_dirs.py [mode]", file=sys.stderr)
    sys.exit(1)

mode = sys.argv[1]

dirs = []
for d in sorted(Path(".").glob(f"{mode}*/*")):
    if not d.is_dir():
        continue
    if not is_leaf(d):
        continue
    if is_finished(d):
        continue
    dirs.append(str(d))

for d in dirs:
    print(d)
