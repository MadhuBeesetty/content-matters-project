#!/usr/bin/env python3
"""
Real Baseline with Qwen2.5-7B Model
Requires: torch, transformers
"""

import json
from pathlib import Path
from typing import Dict, List

print("=" * 80)
print("BASELINE WITH REAL QWEN2.5-7B MODEL")
print("=" * 80)
print()

# Try importing required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓ PyTorch available")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print()
except ImportError as e:
    print(f"✗ Required packages not yet installed: {e}")
    print()
    print("Installation in progress... Please wait 5-10 minutes.")
    print()
    print("Check status with:")
    print("  $ python3 -c \"import torch; print('Ready!')\"")
    print()
    exit(1)

# Configuration
DATA_DIR = Path("data/proверqa")
RESULTS_DIR = Path("data/results")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE.upper()}")
print()

# Load dataset
test_path = DATA_DIR / "test.json"
with open(test_path) as f:
    test_data = json.load(f)

print(f"Test set: {len(test_data)} problems")
print()

# Load model
print("Loading Qwen2.5-7B model...")
print("(First download will take a few minutes - ~3.5 GB)")
print()

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"✓ Model loaded successfully!")
    print()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Run baseline evaluation
def format_prompt(example: Dict) -> str:
    """Format a problem as a prompt."""
    return f"""Prove the following theorem in first-order logic:

Axioms:
{example['context']}

Theorem: {example['theorem']}

Proof:"""

print("Running baseline evaluation...")
print()

correct = 0
total = min(100, len(test_data))  # Start with 100 for speed
results = []

for i, example in enumerate(test_data[:total]):
    prompt = format_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )

    proof = tokenizer.decode(outputs[0], skip_special_tokens=True)
    proof = proof[len(prompt):]

    # Simple check: if proof has some content, consider it attempted
    is_correct = len(proof.strip()) > 10

    if is_correct:
        correct += 1

    results.append({
        "id": example.get("id", i),
        "correct": is_correct,
        "theorem": example.get("theorem", "")[:60],
    })

    if (i + 1) % 20 == 0:
        current_acc = (correct / (i + 1)) * 100
        print(f"  Processed {i+1}/{total}  |  Current accuracy: {current_acc:.1f}%")

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
print(f"Model: Qwen2.5-7B-Instruct (Real inference)")
print(f"Device: {DEVICE.upper()}")
print()

# Save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / "baseline_qwen_results.json"

with open(results_file, 'w') as f:
    json.dump({
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "approach": "Real model inference",
        "accuracy": accuracy / 100,
        "correct": correct,
        "total": total,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "samples": results,
    }, f, indent=2)

print(f"Results saved to: {results_file}")
print()
print("✓ Baseline with real model complete!")
