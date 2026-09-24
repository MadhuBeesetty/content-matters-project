# Running the pipeline

Everything runs locally on macOS with Ollama models. No GPU cluster, no CUDA.

## Setup (once)

```bash
bash scripts/00_setup_environment.sh    # creates .venv, installs requirements
source .venv/bin/activate
```

Ollama must be installed and serving models (`ollama list` should show at
least one small model, e.g. `phi3.5`).

## The pipeline

```bash
python3 scripts/ollama_run_all.py
```

runs, in order:

| Step | Script | What it does | Time |
|---|---|---|---|
| 0 | `generate_synthetic_data.py` | writes `data/synthetic/` if missing (only if missing) | seconds |
| 1 | `ollama_baseline.py 50` | phi3.5 answers 50 test problems; scored vs gold labels | ~4 min |
| 2 | `ollama_treatment_eval.py 50` | control / econst / cdiag prompt framings, same scoring | ~12 min |
| 3 | `generate_report.py` | writes `docs/RESULTS.md` from `data/results/` | seconds |

Run any step individually with different sizes/models, e.g.:

```bash
python3 scripts/ollama_baseline.py 200 phi3.5
python3 scripts/ollama_treatment_eval.py 100 mistral-nemo
```

## Smoke test (no model calls)

```bash
python3 scripts/verify_sample.py
```

Validates data files, schema, and label-construction rules.

## What these numbers mean (and don't)

- **Metric**: answer-level accuracy — the model must end with `ANSWER: True|
  False|Uncertain`; compared to the gold label. Unparsable responses are
  counted separately (a rough analog of the paper's "malformed" bucket).
- **Random floor**: ~33% (3-way labels).
- These are smoke tests on **synthetic** data with small local models. They
  are not comparable to the paper's ProverQA numbers and must not be reported
  as such.

## Legacy scripts

`scripts/legacy/` contains the project's first-draft scripts. They are
non-functional (simulated training, heuristic scoring) and quarantined for
reference only — see `scripts/legacy/README.md`.
