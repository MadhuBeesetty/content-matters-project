# Results

Generated: 2026-09-23 16:33

Dataset: `data/synthetic` (synthetic stand-in with correct-by-construction
labels — NOT ProverQA; see docs/DATA.md). Metric: answer-level accuracy
against gold labels. Random guessing scores ~33%.

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (phi3.5) | 71.4% | 35/49 scored, 1 call errors | 0 | no treatment |

**Caveats**: small samples, synthetic data, one local model. These
numbers are pipeline smoke tests, not reproductions of the paper's
results (which require ProverQA + Prover9 + real RL training).
