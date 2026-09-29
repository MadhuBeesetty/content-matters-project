# Results — Curated Snapshot

Submission-ready copy of all final results. The working store is
`data/results/` (the pipeline reads/writes there); this folder is a curated
snapshot for sharing, review, and the artifact submission.

Refresh after any new run: `bash scripts/sync_results.sh`

## Layout

```
results/
├── RESULTS.md           # all final tables (auto-generated from raw records)
├── ARTIFACTS.md         # claim -> data -> reproduction command map
├── provenance.log       # complete run history (hourly_check.log snapshot)
├── raw/                 # per-problem JSONL records (every generation,
│                        #   verdict, timing — the audit trail)
└── summaries/           # canonical per-model/condition summary JSONs
```

## The two magistral fragments

`raw/full_eval_magistral_24b*.jsonl` are PARTIAL (74 + 15 records) — the model
was dropped as hardware-infeasible (measured 2.7-5 tok/s; see provenance.log
2026-09-28 18:27 entry). Included for transparency, excluded from all tables.

## Correction notice

The Sep-24 checkpoint reward aggregates (data/checkpoints/) are artifacts of a
broken serving stack and are deliberately NOT in this folder. See the
CORRECTION notice in RESULTS.md.
