# Results

Generated: 2026-09-28 19:48

Dataset: `data/synthetic` (synthetic stand-in with correct-by-construction
labels — NOT ProverQA; see docs/DATA.md). Metric: answer-level accuracy
against gold labels, evaluated at inference time (prompt-level treatment
framing only — NOT the GRPO-trained checkpoints; see data/checkpoints/
and check_progress_all.py for training-side reward tracking, a separate,
non-comparable metric). Random guessing scores ~33%.

**CORRECTION 2026-09-27**: the Sep-24 nemo checkpoint rewards
(control 0.684 / econst 0.020 — the apparent 'E-CONST collapse') were
measured with a broken serving stack (serialized 1-slot server, failed
turbo run's empty generations contaminating checkpoints, 128-token
truncation). Under correct serving the same prompts yield 0.897/0.900 —
no collapse. Treat checkpoint reward aggregates as artifacts; only the
eval JSONL records (tables below) are trustworthy.

## deepseek-r1_14b

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (deepseek-r1:14b) | 95.5% | 1433/1500 scored | 0 | no treatment |
| Treatment: control | 96.3% | 1445/1500 | 0 | prompt-level only, no training |
| Treatment: econst | 96.4% | 1446/1500 | 0 | prompt-level only, no training |
| Treatment: cdiag | 95.9% | 1438/1500 | 0 | prompt-level only, no training |

## deepseek-r1_14b_hard

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (deepseek-r1:14b) | 93.4% | 466/499 scored, 1 call errors | 0 | no treatment |
| Treatment: control | 95.0% | 475/500 | 0 | prompt-level only, no training |
| Treatment: econst | 96.0% | 478/500 | 0 | prompt-level only, no training |
| Treatment: cdiag | 93.2% | 464/500 | 0 | prompt-level only, no training |

## mistral-nemo_12b

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (mistral-nemo:12b) | 61.2% | 306/500 scored | 43 | no treatment |
| Treatment: control | 50.0% | 250/500 | 123 | prompt-level only, no training |
| Treatment: econst | 53.4% | 267/500 | 100 | prompt-level only, no training |
| Treatment: cdiag | 59.0% | 295/500 | 76 | prompt-level only, no training |

## phi3.5

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (phi3.5) | 76.0% | 380/500 scored | 2 | no treatment |
| Treatment: control | 74.6% | 373/500 | 1 | prompt-level only, no training |
| Treatment: econst | 79.4% | 397/500 | 3 | prompt-level only, no training |
| Treatment: cdiag | 76.8% | 384/500 | 2 | prompt-level only, no training |

## phi4-reasoning_14b

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (phi4-reasoning:14b) | 87.9% | 435/495 scored, 5 call errors | 0 | no treatment |
| Treatment: control | 93.5% | 463/500 | 0 | prompt-level only, no training |
| Treatment: econst | 90.3% | 446/500 | 0 | prompt-level only, no training |
| Treatment: cdiag | 87.7% | 428/500 | 0 | prompt-level only, no training |

## qwen2.5_32b

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (qwen2.5:32b) | 96.2% | 481/500 scored | 0 | no treatment |
| Treatment: control | 95.8% | 479/500 | 0 | prompt-level only, no training |
| Treatment: econst | 94.0% | 470/500 | 0 | prompt-level only, no training |
| Treatment: cdiag | 95.4% | 477/500 | 0 | prompt-level only, no training |

## qwen2.5_32b_hard

| Experiment | Accuracy | Correct/Total | Unparsed | Notes |
|---|---|---|---|---|
| Baseline (qwen2.5:32b) | 93.4% | 467/500 scored | 0 | no treatment |
| Treatment: control | 94.2% | 471/500 | 0 | prompt-level only, no training |
| Treatment: econst | 92.0% | 460/500 | 0 | prompt-level only, no training |
| Treatment: cdiag | 90.4% | 452/500 | 2 | prompt-level only, no training |

**Caveats**: small samples (N per experiment noted above), synthetic
data, local quantized models via Ollama. These numbers are pipeline
smoke tests / cross-model probes, not reproductions of the paper's
results (which require ProverQA + Prover9 + real RL training).
