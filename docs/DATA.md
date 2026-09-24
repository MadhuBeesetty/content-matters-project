# Data

## What is in this repo

`data/synthetic/` — a **synthetic** first-order-logic reasoning set generated
by `scripts/generate_synthetic_data.py` (seed 42):

| Split | Examples |
|---|---|
| train | 4,966 |
| test | 1,500 |
| dev | 100 |

Each example: `id, context, theorem, type, label` where `label ∈ {True, False, Uncertain}`.

**Labels are correct by construction.** The generator picks the label first,
then builds premises/conclusion to match:

- `True` — premises form a valid entailment chain; the conclusion follows.
- `False` — premises entail the conclusion; the stated theorem is its negation.
- `Uncertain` — the chain is broken with an unrelated entity, so neither the
  conclusion nor its negation follows.

## What this is NOT

This is not **ProverQA**, the dataset used in the paper (Qi et al., 2025,
arXiv:2502.03659). Real ProverQA differs in ways that matter:

- longer, harder contexts (many premises, distractors; easy/medium/hard
  difficulty gradient — 500/500/500 in the test split)
- schema is `context / question / options (A. True, B. False, C. Uncertain) /
  answer`, not `theorem`
- designed so that formal verification (Prover9) is the ground truth

**Do not report numbers from `data/synthetic/` as ProverQA results.** They are
pipeline smoke tests. The synthetic set uses the same split sizes only so the
pipeline is shaped correctly when the real data drops in.

## Getting the real ProverQA

When online:

```bash
python3 scripts/01_download_dataset.py        # tries known mirrors
# or manually:
git clone https://github.com/kkk-an/ProverQA  # source named in the paper
```

Real data lands in `data/proverqa/` and is then preferred automatically by
`src/dataset.py`. FOLIO (out-of-distribution eval): `github.com/Yale-LILY/FOLIO`.

## Prover9

Proof-level verification (the paper's reward signal) requires Prover9. See
`scripts/02_setup_prover9.sh` and `src/prover9_interface.py`.
