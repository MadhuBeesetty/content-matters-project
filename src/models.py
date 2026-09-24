"""Model wrappers and utilities."""

from typing import Optional, Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelWrapper:
    """Wrapper for consistent model interface."""

    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize model wrapper.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **kwargs,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):]

    def get_logits(self, prompt: str) -> torch.Tensor:
        """Get model logits for prompt.

        Args:
            prompt: Input prompt

        Returns:
            Logits tensor
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits

    def set_mode(self, training: bool) -> None:
        """Set model training/eval mode.

        Args:
            training: Whether to set training mode
        """
        if training:
            self.model.train()
        else:
            self.model.eval()


class PromptTemplate:
    """Prompt template for consistent formatting."""

    def __init__(self, template: str):
        """Initialize template.

        Args:
            template: Template string with {field} placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """Format template with variables.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        return self.template.format(**kwargs)

    @staticmethod
    def prover_qa_prompt(context: str, theorem: str, feedback: Optional[str] = None) -> str:
        """Format ProverQA problem as prompt.

        Args:
            context: Axioms/context
            theorem: Theorem to prove
            feedback: Optional feedback

        Returns:
            Formatted prompt
        """
        prompt = f"""Prove the following theorem:

Axioms:
{context}

Theorem: {theorem}

Proof:"""

        if feedback:
            prompt = f"{feedback}\n\n{prompt}"

        return prompt
