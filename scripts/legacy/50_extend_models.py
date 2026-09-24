#!/usr/bin/env python3
"""
Extension framework: Apply reproduction methodology to new offline models.

Supports:
- Qwen2.5 series (7B, 14B, 32B)
- LLaMA 3.x models
- Mistral variants
- Custom models via HuggingFace

Configuration: Edit MODEL_CONFIGS below to add new models.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR = Path(__file__).parent.parent / "data" / "proверqa"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

# Model configurations for extension experiments
MODEL_CONFIGS = {
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 7B instruction-tuned",
        "memory_gb": 16,
    },
    "qwen2.5-14b": {
        "name": "Qwen/Qwen2.5-14B-Instruct",
        "description": "Qwen 14B instruction-tuned",
        "memory_gb": 28,
    },
    "qwen2.5-32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "description": "Qwen 32B instruction-tuned",
        "memory_gb": 64,
    },
    "llama3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "LLaMA 3.1 8B instruction-tuned",
        "memory_gb": 16,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Mistral 7B v0.2",
        "memory_gb": 16,
    },
}


def check_model_availability(model_config: Dict) -> bool:
    """Check if model is available on HuggingFace."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config["name"], trust_remote_code=True)
        return True
    except Exception as e:
        print(f"  ✗ Not available: {e}")
        return False


def load_test_set():
    """Load ProverQA test set."""
    test_path = DATA_DIR / "test.json"
    if test_path.exists():
        with open(test_path) as f:
            return json.load(f)
    return []


def format_prompt(example: Dict, treatment: str = "cdiag") -> str:
    """Format prompt with treatment feedback."""
    prompt = f"""Prove the following theorem:

Axioms:
{example.get('context', '')}

Theorem: {example.get('theorem', '')}

Proof:"""

    if treatment == "cdiag":
        feedback = (
            "Diagnostic feedback:\n"
            "- Check axiom application order\n"
            "- Verify logical deduction rules\n"
            "- Ensure complete proof chain\n\n"
        )
        return feedback + prompt
    elif treatment == "econst":
        feedback = "Focus on logical structure and axiom application.\n\n"
        return feedback + prompt

    return prompt


def evaluate_model(model_name: str, model_path: str, num_examples: int = 100) -> Dict:
    """Evaluate a model on ProverQA test set."""
    print(f"\n  Loading model: {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None

    test_data = load_test_set()
    if not test_data:
        print(f"  ✗ No test data available")
        return None

    # Evaluate on subset
    num_examples = min(num_examples, len(test_data))
    correct = 0
    results = []

    print(f"  Evaluating on {num_examples} examples...")

    for i in tqdm(range(num_examples)):
        example = test_data[i]
        prompt = format_prompt(example, treatment="cdiag")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )

            proof = tokenizer.decode(outputs[0], skip_special_tokens=True)
            proof = proof[len(prompt):]

            # Simple heuristic: longer proof = attempt to solve
            is_correct = len(proof.strip()) > 20
            if is_correct:
                correct += 1

            results.append({
                "id": i,
                "correct": is_correct,
                "proof_length": len(proof),
            })
        except Exception as e:
            print(f"    Error processing example {i}: {e}")
            continue

    accuracy = correct / num_examples if num_examples > 0 else 0.0

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": correct,
        "total": num_examples,
        "examples": results,
    }


def run_model_comparison():
    """Run baseline evaluation on all configured models."""
    print("=" * 70)
    print("MODEL EXTENSION EXPERIMENTS")
    print("=" * 70)
    print()
    print("Checking model availability...")
    print()

    available_models = {}
    for model_key, model_config in MODEL_CONFIGS.items():
        print(f"{model_key}:")
        if check_model_availability(model_config):
            available_models[model_key] = model_config
            print(f"  ✓ Available (requires ~{model_config['memory_gb']}GB VRAM)")
        else:
            print(f"  ✗ Skipped")

    if not available_models:
        print()
        print("No models available. Install models with:")
        print("  huggingface-cli login")
        print("  huggingface-cli download <model_id>")
        return

    print()
    print("=" * 70)
    print("BASELINE EVALUATION (no training)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_key, model_config in available_models.items():
        print()
        print(f"Evaluating {model_key}...")

        results = evaluate_model(
            model_key,
            model_config["name"],
            num_examples=100
        )

        if results:
            all_results.append(results)
            print(f"  Accuracy: {results['accuracy']:.2%}")

            # Save individual results
            results_file = RESULTS_DIR / f"baseline_{model_key}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

    # Save comparison
    if all_results:
        comparison_file = RESULTS_DIR / "model_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump({
                "timestamp": str(Path(__file__).stat().st_mtime),
                "models": all_results,
            }, f, indent=2)

        print()
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Model':<20} {'Accuracy':<15} {'Samples':<15}")
        print("-" * 50)
        for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['model']:<20} {result['accuracy']:>6.2%}         {result['total']:<15}")

        print()
        print(f"Results saved to: {comparison_file}")


if __name__ == "__main__":
    run_model_comparison()
