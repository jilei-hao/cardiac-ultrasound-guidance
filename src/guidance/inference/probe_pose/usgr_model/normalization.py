"""Data normalization utilities.

Vendored verbatim from USGR ``usgr/layers/normalization.py`` (no functional changes).
"""

import torch
from typing import Iterable


class Normalizer:
    """
    Normalize spatial data to zero mean and unit standard deviation.
    Stores per-component statistics including min/max for bounding box computation.
    """

    def __init__(
        self,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        min: torch.Tensor = None,
        max: torch.Tensor = None,
    ):
        """
        Initialize with optional pre-computed statistics.

        Args:
            mean: Tensor with per-component means
            std: Tensor with per-component standard deviations
            min: Tensor with per-component minimum values
            max: Tensor with per-component maximum values
        """
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max

    def compute_statistics(self, tensors: Iterable[torch.Tensor], min_std: float = 1e-6):
        """
        Compute normalization statistics from an iterable of tensors.

        Args:
            tensors: Iterable of tensors to compute statistics from
            min_std: Minimum std value to avoid division by zero
        """
        data = []
        for t in tensors:
            data.append(t)

        data = torch.stack(data, dim=0)

        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values

        # Avoid division by zero for constant components
        self.std = torch.clamp(self.std, min=min_std)

        print(f"Computed statistics from {len(data)} samples")

        return self

    def symmetric_extent(self) -> torch.Tensor:
        """
        Compute symmetric extent around mean that encompasses all data.

        Returns:
            Tensor with per-component extent values such that
            [mean - extent, mean + extent] contains all observed data.
        """
        if self.mean is None or self.min is None or self.max is None:
            raise ValueError("Statistics not computed. Call compute_statistics() first.")
        return torch.maximum(self.mean - self.min, self.max - self.mean)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to zero mean and unit std.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Statistics not computed. Call compute_statistics() first.")

        mean = self.mean.to(data.device)
        std = self.std.to(data.device)

        # Handle batched data
        if data.dim() > self.mean.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return (data - mean) / std

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Statistics not computed. Call compute_statistics() first.")

        mean = self.mean.to(data.device)
        std = self.std.to(data.device)

        # Handle batched data
        if data.dim() > self.mean.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return data * std + mean

    def save(self, path: str):
        """Save statistics to a file."""
        torch.save({
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
        }, path)
        print(f"Saved normalization statistics to {path}")

    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Load statistics from a file."""
        data = torch.load(path)
        instance = cls(
            mean=data['mean'],
            std=data['std'],
            min=data.get('min'),  # backwards compatible
            max=data.get('max'),
        )
        print(f"Loaded normalization statistics from {path}")
        return instance
