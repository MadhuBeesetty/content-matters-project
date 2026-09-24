"""ProverQA dataset utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ProverQADataset:
    """ProverQA dataset from Logic-LM."""

    def __init__(self, split: str = "train", data_dir: Optional[Path] = None):
        """Load ProverQA split.

        Args:
            split: One of "train", "dev", "test"
            data_dir: Path to data directory
        """
        if data_dir is None:
            root = Path(__file__).parent.parent / "data"
            # Prefer the real ProverQA dataset when present; otherwise fall
            # back to the synthetic stand-in (see docs/DATA.md).
            data_dir = root / "proverqa" if (root / "proverqa" / f"{split}.json").exists() \
                else root / "synthetic"

        split_path = data_dir / f"{split}.json"

        with open(split_path) as f:
            self.data = json.load(f)

        self.split = split
        self.data_dir = data_dir

    def __len__(self) -> int:
        """Number of examples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        return self.data[idx]

    def get_batch(self, indices: List[int]) -> List[Dict]:
        """Get batch of examples."""
        return [self.data[i] for i in indices]

    def iterate_batches(self, batch_size: int):
        """Iterate over batches."""
        for i in range(0, len(self.data), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(self.data))))
            yield self.get_batch(batch_indices)

    @property
    def example_fields(self) -> List[str]:
        """Available fields in examples."""
        if len(self.data) > 0:
            return list(self.data[0].keys())
        return []
