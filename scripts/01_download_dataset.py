#!/usr/bin/env python3
"""
Download the real ProverQA dataset into data/proverqa/.

Sources (from the paper's own data statement and the Logic-LM lineage):
  - ProverQA repo:  https://github.com/kkk-an/ProverQA
  - Logic-LM mirror: https://github.com/lgw863/Logic-LM (data/proverqa/)
  - FOLIO (OOD eval): https://github.com/Yale-LILY/FOLIO

This script NEVER fabricates data. If downloads fail (e.g. offline), it
prints manual instructions and exits non-zero. For a synthetic stand-in
used in pipeline development, run scripts/generate_synthetic_data.py.
"""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict

DATA_DIR = Path(__file__).parent.parent / "data" / "proverqa"

URL_SETS = [
    {
        "name": "Logic-LM mirror (raw GitHub)",
        "urls": {
            "train": "https://raw.githubusercontent.com/lgw863/Logic-LM/main/data/proverqa/train.json",
            "dev": "https://raw.githubusercontent.com/lgw863/Logic-LM/main/data/proverqa/dev.json",
            "test": "https://raw.githubusercontent.com/lgw863/Logic-LM/main/data/proverqa/test.json",
        },
    },
]


def try_download(urls: Dict[str, str]) -> bool:
    ok = True
    for split, url in urls.items():
        out = DATA_DIR / f"{split}.json"
        if out.exists():
            print(f"✓ {split}.json already exists, skipping")
            continue
        try:
            print(f"⏳ {split}: {url}")
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  ✓ {len(data)} examples")
        except Exception as e:
            print(f"  ✗ {type(e).__name__}: {e}")
            ok = False
    return ok


def main() -> int:
    print("=" * 80)
    print("PROVERQA DOWNLOADER (real dataset only — never fabricates)")
    print("=" * 80)
    print()

    for source in URL_SETS:
        print(f"Trying: {source['name']}")
        if try_download(source["urls"]):
            print()
            print(f"✓ Done. Data in: {DATA_DIR}")
            return 0
        print()

    print("✗ Automatic download failed (offline or URLs moved).")
    print()
    print("Manual options:")
    print("  1. git clone https://github.com/kkk-an/ProverQA   (paper's data statement)")
    print("  2. git clone https://github.com/lgw863/Logic-LM && cp -r Logic-LM/data/proverqa data/")
    print("  3. FOLIO (OOD eval): https://github.com/Yale-LILY/FOLIO")
    print()
    print("For pipeline development without the real data:")
    print("  python3 scripts/generate_synthetic_data.py   # writes data/synthetic/")
    return 1


if __name__ == "__main__":
    sys.exit(main())
