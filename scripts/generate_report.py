#!/usr/bin/env python3
"""
Write docs/RESULTS.md from whatever result files exist in data/results/.

Only real, label-scored results are included. If no result files exist yet,
the report says so instead of inventing numbers.
"""

import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
REPORT_FILE = Path(__file__).parent.parent / "docs" / "RESULTS.md"


def load_results() -> dict:
    results = {}
    baseline = RESULTS_DIR / "ollama_phi3.5_baseline.json"
    if baseline.exists():
        results["baseline"] = json.load(open(baseline))
    for t in ["control", "econst", "cdiag"]:
        f = RESULTS_DIR / f"ollama_treatment_{t}.json"
        if f.exists():
            results[f"treatment_{t}"] = json.load(open(f))
    return results


def main():
    results = load_results()
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Results",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Dataset: `data/synthetic` (synthetic stand-in with correct-by-construction",
        "labels — NOT ProverQA; see docs/DATA.md). Metric: answer-level accuracy",
        "against gold labels. Random guessing scores ~33%.",
        "",
    ]

    if not results:
        lines += ["No result files found in `data/results/` yet.",
                  "Run `python3 scripts/ollama_run_all.py` first.", ""]
    else:
        lines += ["| Experiment | Accuracy | Correct/Total | Unparsed | Notes |",
                  "|---|---|---|---|---|"]
        if "baseline" in results:
            b = results["baseline"]
            scored = b["correct"] + b["wrong"] + b.get("unparsed", 0)
            err = f", {b['errors']} call errors" if b.get("errors") else ""
            lines.append(
                f"| Baseline ({b['model']}) | {b['accuracy']*100:.1f}% | "
                f"{b['correct']}/{scored} scored{err} | {b.get('unparsed', 0)} | no treatment |")
        for t in ["control", "econst", "cdiag"]:
            key = f"treatment_{t}"
            if key in results:
                r = results[key]
                lines.append(
                    f"| Treatment: {t} | {r['accuracy']*100:.1f}% | "
                    f"{r['correct']}/{r['total']} | {r.get('unparsed', 0)} | "
                    f"prompt-level only, no training |")
        lines += [
            "",
            "**Caveats**: small samples, synthetic data, one local model. These",
            "numbers are pipeline smoke tests, not reproductions of the paper's",
            "results (which require ProverQA + Prover9 + real RL training).",
            "",
        ]

    REPORT_FILE.write_text("\n".join(lines))
    print(f"✓ Wrote {REPORT_FILE}")


if __name__ == "__main__":
    main()
