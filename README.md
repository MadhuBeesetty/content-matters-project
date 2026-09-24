# Content Matters — Extension Project

An effort to **extend** the anonymous NeurIPS 2026 submission *"Content Matters
More Than Placement: Privileged Supervision in LLM Post-Training"* using local,
offline models (Ollama) and Apple Silicon training (MLX).

The paper's finding: in RL post-training with privileged teacher feedback, the
**content** of the feedback drives the gains (correct demonstrations and
difficulty-aware slotting reach ~85% on ProverQA; generic constraints collapse
the model to 8.7%), while **where** the feedback is placed is statistically
irrelevant (0.67-point spread across slots).

This repo is **not** a reproduction of the paper — the authors release their
code only upon acceptance, so no reference implementation exists. It is an
independent extension effort. The paper PDF lives in `paper/` (gitignored —
it is someone else's anonymous submission; we do not redistribute it).

## Honest status

| Component | Status |
|---|---|
| Synthetic logic dataset (`data/synthetic/`) | ✅ Works. 4,966/1,500/100 splits, **correct-by-construction labels**. NOT ProverQA — see `docs/DATA.md` |
| Real ProverQA | ❌ Not yet obtained (offline). `scripts/01_download_dataset.py` when online |
| Answer-level baseline via Ollama | ✅ Real inference, scored against gold labels |
| Prompt-level treatment comparison | ✅ Real inference (prompting only — **no training**) |
| Prover9 proof-level verification | ⚠️ Interface rewritten correctly (`src/prover9_interface.py`); needs Prover9 installed |
| RL training (GRPO + teacher gate) | ❌ Not yet. Planned on MLX/LoRA — see roadmap |
| `scripts/legacy/` | ⚠️ Quarantined first-draft scripts — non-functional, kept for reference only |

## Quick start

```bash
bash scripts/00_setup_environment.sh     # venv + deps (MLX for training track)
python3 scripts/verify_sample.py         # smoke test: data + wiring
python3 scripts/ollama_run_all.py        # baseline + treatments + docs/RESULTS.md
```

## Layout

```
├── scripts/            # working pipeline (see docs/RUNNING.md)
│   └── legacy/         # quarantined non-functional drafts — do not run
├── src/                # library modules (dataset, prover9, evaluation; training is WIP)
├── data/
│   ├── synthetic/      # generated stand-in data (correct labels, not ProverQA)
│   ├── proverqa/       # real dataset lands here when downloaded
│   └── results/        # experiment outputs (JSON)
├── docs/               # DATA.md, RUNNING.md, RESULTS.md
├── paper/              # the submission PDF (gitignored)
└── requirements.txt
```

## Roadmap (the extension, i.e. the publishable part)

1. **E-CONST collapse: syntactic or semantic?** The paper's generic-constraint
   condition collapses because models "stopped writing programs a parser
   accepts" (75% malformed). Test with grammar-constrained decoding (GBNF via
   llama.cpp/Ollama): if constrained decoding restores format but not accuracy,
   the failure is in the reasoning, not the syntax. Pure inference — no training.
2. **Cross-scale replication + empirical test of their Theorem 1.** The paper
   uses one model (Qwen2.5-7B). Replicate the placement null and the teacher
   gate across 1.5B–8B students (MLX/LoRA) with strong local teachers
   (QwQ-32B, phi4-reasoning), and check whether measured variance reduction
   tracks the predicted 1/(1−ρ²).
3. Real ProverQA + Prover9-gated rewards, replacing the synthetic stand-in.

## Attribution

- Paper under extension: Anonymous, *Content Matters More Than Placement:
  Privileged Supervision in LLM Post-Training*, NeurIPS 2026 submission (OpenReview).
- ProverQA: Qi et al., 2025 (arXiv:2502.03659), `github.com/kkk-an/ProverQA`.
- FOLIO: Han et al., 2024, `github.com/Yale-LILY/FOLIO`.

All numbers reported in `docs/RESULTS.md` are generated locally by
`scripts/generate_report.py` from `data/results/` — none are transcribed from
the paper.
