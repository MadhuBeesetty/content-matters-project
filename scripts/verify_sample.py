#!/usr/bin/env python3
"""
Smoke test: verify the data and pipeline wiring without a GPU.

Checks:
  1. data/synthetic splits load and have the expected schema
  2. labels are well-formed and construction rules hold (negation <=> False)
  3. ollama CLI is available (needed for the evaluation scripts)
"""

import json
import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"

failures = 0


def check(name: str, ok: bool, detail: str = ""):
    global failures
    print(f"{'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures += 1


print("SMOKE TEST")
print("=" * 60)

# 1. Splits load
splits = {}
for split in ["train", "test", "dev"]:
    path = DATA_DIR / f"{split}.json"
    if path.exists():
        with open(path) as f:
            splits[split] = json.load(f)

check("splits exist", len(splits) == 3,
      f"found: {list(splits)} (run scripts/generate_synthetic_data.py if missing)")
if len(splits) < 3:
    sys.exit(1)

# 2. Schema
required = {"id", "context", "theorem", "type", "label"}
bad = [e["id"] for e in splits["train"][:100] if not required <= set(e)]
check("schema (first 100 train)", not bad, f"bad: {bad[:3]}")

# 3. Label construction rules
neg_markers = ["no ", "not ", "does not", "false", "some elements"]
misconstructed = []
for split, data in splits.items():
    for e in data:
        has_neg = any(m in e["theorem"] for m in neg_markers)
        if e["label"] == "False" and not has_neg:
            misconstructed.append(e["id"])
        if e["label"] in ("True", "Uncertain") and has_neg:
            misconstructed.append(e["id"])
        if "{" in e["context"] or "{" in e["theorem"]:
            misconstructed.append(e["id"])
check("label construction rules", not misconstructed,
      f"{len(misconstructed)} violations, e.g. {misconstructed[:3]}")

labels = [e["label"] for e in splits["test"]]
check("labels are valid values",
      set(labels) <= {"True", "False", "Uncertain"},
      str({l: labels.count(l) for l in set(labels)}))

# 4. Ollama available
check("ollama CLI available", shutil.which("ollama") is not None)

print()
if failures:
    print(f"✗ {failures} check(s) failed")
    sys.exit(1)
print("✓ All checks passed — pipeline wiring is sound.")
