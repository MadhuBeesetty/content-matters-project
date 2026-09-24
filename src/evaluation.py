"""Evaluation metrics and utilities."""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ProofEvaluator:
    """Evaluate proof quality and correctness."""

    @staticmethod
    def compute_metrics(
        predictions: List[bool],
        ground_truth: List[bool],
    ) -> Dict[str, float]:
        """Compute standard classification metrics.

        Args:
            predictions: Predicted correctness
            ground_truth: Ground truth correctness

        Returns:
            Dict of metrics
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        return {
            "accuracy": accuracy_score(ground_truth, predictions),
            "precision": precision_score(ground_truth, predictions, zero_division=0),
            "recall": recall_score(ground_truth, predictions, zero_division=0),
            "f1": f1_score(ground_truth, predictions, zero_division=0),
        }

    @staticmethod
    def analyze_placement_independence(
        results_by_placement: Dict[str, List[bool]],
        ground_truth: List[bool],
    ) -> Dict[str, any]:
        """Analyze whether placement affects accuracy.

        Args:
            results_by_placement: Accuracy by placement (first, middle, last)
            ground_truth: Ground truth labels

        Returns:
            Placement independence analysis
        """
        accuracies = {}
        for placement, predictions in results_by_placement.items():
            acc = accuracy_score(ground_truth, predictions)
            accuracies[placement] = acc

        mean_acc = np.mean(list(accuracies.values()))
        std_acc = np.std(list(accuracies.values()))
        max_diff = max(accuracies.values()) - min(accuracies.values())

        return {
            "accuracies_by_placement": accuracies,
            "mean_accuracy": mean_acc,
            "std_dev": std_acc,
            "max_difference": max_diff,
            "is_placement_independent": max_diff < 0.05,  # threshold
        }

    @staticmethod
    def compare_treatments(
        treatment_results: Dict[str, Dict],
    ) -> Tuple[List[str], List[float]]:
        """Compare performance across treatments.

        Args:
            treatment_results: Results for each treatment

        Returns:
            (treatment_names, accuracies)
        """
        treatments = sorted(treatment_results.keys())
        accuracies = [treatment_results[t]["accuracy"] for t in treatments]

        return treatments, accuracies


class MetricsAggregator:
    """Aggregate metrics across batches and epochs."""

    def __init__(self):
        """Initialize aggregator."""
        self.metrics_history = []

    def add_batch_metrics(self, epoch: int, batch: int, metrics: Dict) -> None:
        """Record metrics for a batch.

        Args:
            epoch: Epoch number
            batch: Batch number
            metrics: Metrics dict
        """
        self.metrics_history.append({
            "epoch": epoch,
            "batch": batch,
            **metrics,
        })

    def get_epoch_summary(self, epoch: int) -> Dict:
        """Get aggregated metrics for an epoch.

        Args:
            epoch: Epoch number

        Returns:
            Aggregated metrics
        """
        epoch_metrics = [m for m in self.metrics_history if m["epoch"] == epoch]

        if not epoch_metrics:
            return {}

        result = {"epoch": epoch}
        for key in epoch_metrics[0].keys():
            if key not in ["epoch", "batch"]:
                values = [m[key] for m in epoch_metrics]
                result[f"{key}_mean"] = np.mean(values)
                result[f"{key}_std"] = np.std(values)

        return result
