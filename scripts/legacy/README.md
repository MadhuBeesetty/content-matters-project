# scripts/legacy — quarantined, do not run

These scripts came from the project's first scaffold and are **non-functional
or misleading**. They are kept for reference only and are not part of the
pipeline. Nothing here should be cited, run, or extended.

| Script | Defect |
|---|---|
| `10_reproduce_baseline.py` | "Baseline" is a hardcoded heuristic with noise (the source comment says it "simulates that the model can solve ~40%"). Produces fabricated numbers. |
| `30_grpo_experiment.py` | Claims to be the main GRPO experiment but performs no training: no `backward()` anywhere, `optimizer.step()` runs on empty gradients, the `placement` argument is accepted but ignored, and any non-empty generation is scored "correct". |
| `40_evaluate_results.py` | Evaluates outputs of the above. |
| `50_extend_models.py` | Extension scaffold built on the same vacuous metrics. |
| `baseline_real_model.py` | Loads a real HF model but scores any response longer than 10 characters as correct. |

The working pipeline is one directory up: `generate_synthetic_data.py` →
`ollama_baseline.py` → `ollama_treatment_eval.py` → `generate_report.py`.
