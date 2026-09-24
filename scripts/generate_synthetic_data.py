#!/usr/bin/env python3
"""
Generate a SYNTHETIC first-order-logic reasoning dataset.

This is NOT ProverQA. It is a small, self-contained synthetic benchmark for
pipeline development and smoke tests while the real ProverQA dataset is
unavailable (see docs/DATA.md for how to obtain the real data).

Labels are CORRECT BY CONSTRUCTION: the label is chosen first, then the
premises and conclusion are generated to match it:

  - True:      premises form a valid entailment chain; conclusion follows.
  - False:     premises entail the conclusion; the stated theorem is its
               negation (contradicted by the premises).
  - Uncertain: the chain is broken with an unrelated entity, so neither the
               conclusion nor its negation follows from the premises.

Every problem uses exactly the number of distinct entities its template
requires, and each predicate placeholder is substituted once per problem.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Pattern table.
# Each pattern provides:
#   ctx_valid(A, B, C, D, P)    -> premises that entail the conclusion
#   ctx_broken(A, B, C, D, P)   -> premises with a broken chain (no entailment)
#   thm(A, B, C, P)             -> the entailed conclusion
#   thm_neg(A, B, C, P)         -> its negation (contradicted by valid premises)
# D is an unrelated fourth entity used only to break chains for "Uncertain".
# ---------------------------------------------------------------------------

PATTERNS = [
    {
        "type": "syllogistic",
        "ctx_valid": lambda A, B, C, D, P: f"All {A} are {B}. All {B} are {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"All {A} are {B}. All {D} are {C}.",
        "thm": lambda A, B, C, P: f"Therefore, all {A} are {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, no {A} are {C}.",
    },
    {
        "type": "syllogistic",
        "ctx_valid": lambda A, B, C, D, P: f"{A} implies {B}. {B} implies {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"{A} implies {B}. {D} implies {C}.",
        "thm": lambda A, B, C, P: f"Therefore, {A} implies {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {A} does not imply {C}.",
    },
    {
        "type": "modus_ponens",
        "ctx_valid": lambda A, B, C, D, P: f"If {A} is true then {B} is true. {A} is true.",
        "ctx_broken": lambda A, B, C, D, P: f"If {A} is true then {B} is true. {D} is true.",
        "thm": lambda A, B, C, P: f"Therefore, {B} is true.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {B} is false.",
    },
    {
        "type": "transitive",
        "ctx_valid": lambda A, B, C, D, P: f"{A} is greater than {B}. {B} is greater than {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"{A} is greater than {B}. {D} is greater than {C}.",
        "thm": lambda A, B, C, P: f"Therefore, {A} is greater than {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {A} is not greater than {C}.",
    },
    {
        "type": "genealogical",
        "ctx_valid": lambda A, B, C, D, P: f"{A} is parent of {B}. {B} is parent of {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"{A} is parent of {B}. {D} is parent of {C}.",
        "thm": lambda A, B, C, P: f"Therefore, {A} is grandparent of {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {A} is not grandparent of {C}.",
    },
    {
        "type": "set_theory",
        "ctx_valid": lambda A, B, C, D, P: f"{A} is subset of {B}. {B} is subset of {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"{A} is subset of {B}. {D} is subset of {C}.",
        "thm": lambda A, B, C, P: f"Therefore, {A} is subset of {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {A} is not a subset of {C}.",
    },
    {
        "type": "set_containment",
        "ctx_valid": lambda A, B, C, D, P: f"All elements in {A} are in {B}. All elements in {B} are in {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"All elements in {A} are in {B}. All elements in {D} are in {C}.",
        "thm": lambda A, B, C, P: f"Therefore, all elements in {A} are in {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, some elements in {A} are not in {C}.",
    },
    {
        "type": "property_inheritance",
        "ctx_valid": lambda A, B, C, D, P: f"All {A} satisfy property {P}. All {B} are {A}.",
        "ctx_broken": lambda A, B, C, D, P: f"All {A} satisfy property {P}. All {B} are {D}.",
        "thm": lambda A, B, C, P: f"Therefore, all {B} satisfy property {P}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, no {B} satisfy property {P}.",
    },
    {
        "type": "containment",
        "ctx_valid": lambda A, B, C, D, P: f"{A} is in {B}. {B} is in {C}.",
        "ctx_broken": lambda A, B, C, D, P: f"{A} is in {B}. {D} is in {C}.",
        "thm": lambda A, B, C, P: f"Therefore, {A} is in {C}.",
        "thm_neg": lambda A, B, C, P: f"Therefore, {A} is not in {C}.",
    },
]

ENTITIES = [
    "objects", "things", "creatures", "items", "entities",
    "people", "humans", "mammals", "animals", "beings",
    "concepts", "ideas", "principles", "facts", "statements",
    "sets", "groups", "classes", "categories", "collections",
]

PREDICATES = [
    "red", "blue", "large", "small", "fast", "slow",
    "happy", "sad", "intelligent", "strong", "weak",
    "visible", "hidden", "complete", "partial", "total",
]

LABELS = ["True", "False", "Uncertain"]


def generate_problem(problem_id: int, rng: random.Random) -> Dict:
    """Generate one problem whose label is correct by construction."""

    pattern = rng.choice(PATTERNS)
    label = rng.choice(LABELS)

    # Always sample exactly the entities the template needs: A, B, C plus an
    # unrelated D used only to break the chain for "Uncertain".
    A, B, C, D = rng.sample(ENTITIES, 4)
    P = rng.choice(PREDICATES)  # one predicate per problem, used consistently

    context = pattern["ctx_valid"](A, B, C, D, P) if label != "Uncertain" \
        else pattern["ctx_broken"](A, B, C, D, P)
    theorem = pattern["thm"](A, B, C, P) if label != "False" \
        else pattern["thm_neg"](A, B, C, P)

    return {
        "id": f"synthetic_{problem_id:04d}",
        "context": context,
        "theorem": theorem,
        "type": pattern["type"],
        "label": label,
    }


def generate_dataset(train_size: int = 4966, test_size: int = 1500,
                     dev_size: int = 100, seed: int = 42) -> None:
    """Generate train/test/dev splits with correct-by-construction labels."""

    rng = random.Random(seed)

    print("=" * 80)
    print("GENERATING SYNTHETIC LOGIC DATASET (correct-by-construction labels)")
    print("=" * 80)
    print()
    print("NOTE: This is NOT ProverQA. It is a synthetic stand-in for pipeline")
    print("development. See docs/DATA.md for how to obtain the real dataset.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = train_size + test_size + dev_size
    problems = [generate_problem(i, rng) for i in range(total)]

    splits = {
        "train": problems[:train_size],
        "test": problems[train_size:train_size + test_size],
        "dev": problems[train_size + test_size:train_size + test_size + dev_size],
    }

    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ {name:5s}: {len(data):,} problems -> {path}")

    # Sanity report
    print()
    from collections import Counter
    all_probs = problems
    print("Labels:", dict(Counter(p["label"] for p in all_probs)))
    print("Types:", dict(Counter(p["type"] for p in all_probs)))
    bad = [p for p in all_probs if "{" in p["context"] or "{" in p["theorem"]]
    print(f"Unsubstituted placeholders: {len(bad)} (must be 0)")
    print()
    print("✓ Done.")


if __name__ == "__main__":
    generate_dataset()
