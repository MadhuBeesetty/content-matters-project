# Artifact Appendix — Reproducibility Map

Every number in the paper is produced locally by the scripts below from raw
per-problem records (`data/results/*.jsonl`). No number is transcribed by hand.

## Claims → Evidence → Reproduction

| Claim | Data (raw records) | Analysis | Reproduce |
|---|---|---|---|
| Strong models are framing-insensitive at ceiling (deepseek null) | `full_eval_deepseek-r1_14b.jsonl` (N=1,500×4) | `analyze_full_eval.py deepseek-r1_14b` | `python3 scripts/ollama_eval_parallel.py deepseek-r1:14b 1500 synthetic` |
| Scale substitutes for reasoning (qwen-32B ≈ deepseek-14B) | `full_eval_qwen2.5_32b.jsonl` | `analyze_full_eval.py qwen2.5_32b` | `... qwen2.5:32b 500 synthetic` |
| Dose-response: verbose feedback hurts mid-capability reasoner (phi4) | `full_eval_phi4-reasoning_14b.jsonl` (N=500×4) | `analyze_full_eval.py phi4-reasoning_14b` (control>cdiag chi²=18.35) | `... phi4-reasoning:14b 500 synthetic` |
| Weak-model format-compliance channel (nemo unparsed 17-25%) | `full_eval_mistral-nemo_12b.jsonl` | `analyze_full_eval.py mistral-nemo_12b` | `... mistral-nemo:12b 500 synthetic` |
| econst helps the weakest model (phi3.5) | `full_eval_phi3.5.jsonl` | `analyze_full_eval.py phi3.5` | `... phi3.5 500 synthetic` |
| Below-ceiling separation, hard set (deepseek, qwen) | `full_eval_*_hard.jsonl` | `analyze_full_eval.py <tag>_hard` | `... <model> 500 synthetic_hard` |
| Constrained-decoding probe (nemo intact under econst; retraction evidence) | `gbnf_collapse_nemo.jsonl` | inline summary in script output | `python3 scripts/gbnf_collapse_test.py 250` |

## Datasets (correct-by-construction, seeded)

| Set | Script | Config |
|---|---|---|
| `data/synthetic/` (easy, 2-hop) | `generate_synthetic_data.py` | seed 42, 4,966/1,500/100 |
| `data/synthetic_hard/` (3-5 hop + distractors, shuffled) | `generate_synthetic_data_hard.py` | seed 123, 3,000/1,000/200 |
| `data/synthetic_xhard/` (6-8 hop, 5-7 distractors) | same script, parameters | seed 456 |

## Infrastructure (the reproducible-eval engineering)

- `ollama_eval_parallel.py` — async checkpointed runner: per-problem JSONL,
  resume-safe (error rows auto-retry), dataset-aware tags, env-tunable
  (`OLLAMA_EVAL_PARALLEL`, `OLLAMA_EVAL_TIMEOUT`, `OLLAMA_EVAL_NUM_PREDICT`)
- `queue_*.sh` — chained watcher pattern for multi-stage overnight pipelines
- `generate_report.py` — regenerates `docs/RESULTS.md` from raw records only
- `verify_sample.py` — dataset schema + label-construction smoke test
- `logs/hourly_check.log` — the full provenance trail of every run

## Known limitations (stated in the paper)

- Ollama 0.32.14 + Apple M4 Max; 24B+ Mistral-arch models are infeasible on
  this stack (measured 2.7-5 tok/s) — see magistral fragment + log note
- The Sep-24 checkpoint reward aggregates are artifacts (broken serving stack);
  only JSONL eval records are trustworthy (see RESULTS.md correction notice)
- Training-time experiments require Qwen2.5-7B weights + mlx-lm (offline-blocked
  during this work); see TRAINING_SETUP.md for the unblock path

## Quickstart for reviewers (<10 min)

```bash
python3 scripts/verify_sample.py                                  # data sanity
python3 scripts/ollama_eval_parallel.py phi3.5 50 synthetic       # fast demo run
python3 scripts/analyze_full_eval.py phi3.5                       # stats from records
python3 scripts/generate_report.py                                # rebuild RESULTS.md
```
