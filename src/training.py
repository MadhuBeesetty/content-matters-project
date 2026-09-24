"""Training utilities — WORK IN PROGRESS, not used by any script yet.

Honest status:
  - `GRPOOptimizer.compute_reward` is REINFORCE with a group-mean baseline.
    It is NOT full GRPO: it lacks the importance-sampling ratio, PPO-style
    clipping, and the KL penalty to a frozen reference policy. A real GRPO
    loop for this project will be built on mlx-lm (see README roadmap).
  - `TeacherQualityGate` mirrors the paper's pre-gradient gate idea but uses
    an absolute log-prob threshold; the paper gates on the top-2 token margin
    under a frozen reference policy.
  - `LoraAdapter` is a thin peft wrapper and is fine as-is.
"""

from typing import Dict, Optional
import torch


class GRPOOptimizer:
    """Group-mean-baseline policy gradient (simplified; see module docstring)."""

    def __init__(self, model, learning_rate: float = 1e-5):
        """Initialize GRPO optimizer.

        Args:
            model: Transformer model to train
            learning_rate: Learning rate
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def compute_reward(self, logits: torch.Tensor, target_ids: torch.Tensor, group_rewards: torch.Tensor) -> torch.Tensor:
        """Compute GRPO loss from group rewards.

        Args:
            logits: Model output logits
            target_ids: Target token IDs
            group_rewards: Scalar rewards for examples

        Returns:
            Loss tensor
        """
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Extract probabilities for target tokens
        token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Compute relative rewards within group
        mean_reward = group_rewards.mean()
        relative_rewards = group_rewards - mean_reward

        # GRPO loss: minimize -log_prob * relative_reward
        loss = -(token_log_probs * relative_rewards.unsqueeze(-1)).mean()

        return loss


class LoraAdapter:
    """LoRA fine-tuning adapter for parameter-efficient training."""

    @staticmethod
    def apply_lora(model, r: int = 8, lora_alpha: int = 16):
        """Apply LoRA to model.

        Args:
            model: Model to adapt
            r: LoRA rank
            lora_alpha: LoRA scaling

        Returns:
            Adapted model
        """
        try:
            from peft import get_peft_model, LoraConfig

            config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            return get_peft_model(model, config)
        except ImportError:
            print("Warning: peft not installed, skipping LoRA")
            return model


class TeacherQualityGate:
    """Pre-gradient teacher quality screening."""

    def __init__(self, log_prob_threshold: float = -2.0):
        """Initialize quality gate.

        Args:
            log_prob_threshold: Minimum log-probability for acceptance
        """
        self.threshold = log_prob_threshold
        self.rejected_count = 0
        self.accepted_count = 0

    def filter_batch(self, log_probs: torch.Tensor, examples: list) -> tuple:
        """Filter low-quality examples before gradient computation.

        Args:
            log_probs: Log probabilities from teacher
            examples: Corresponding examples

        Returns:
            (filtered_indices, filtered_examples)
        """
        valid_indices = (log_probs >= self.threshold).nonzero(as_tuple=True)[0]

        self.accepted_count += len(valid_indices)
        self.rejected_count += len(log_probs) - len(valid_indices)

        filtered_examples = [examples[i] for i in valid_indices]

        return valid_indices, filtered_examples

    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics."""
        return {
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "total": self.accepted_count + self.rejected_count,
        }
