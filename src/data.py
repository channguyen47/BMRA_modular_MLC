from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultilabelDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        feature_missingness: torch.Tensor,
        label_availability: torch.Tensor,
    ) -> None:
        self.tensors = features, labels, feature_missingness, label_availability

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)


def split_indices(
    size: int, train: float, val: float, test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("Split fractions must sum to 1")
    indices = np.random.default_rng(seed).permutation(size)
    train_end = int(size * train)
    val_end = train_end + int(size * val)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def fit_standardizer(
    frame: pd.DataFrame, feature_columns: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    means = np.nanmean(features, axis=0)
    scales = np.nanstd(features, axis=0)
    if np.isnan(means).any():
        raise ValueError("Every feature needs at least one observed training value")
    scales[(scales == 0) | np.isnan(scales)] = 1.0
    return means, scales


def make_dataset(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    means: np.ndarray,
    scales: np.ndarray,
) -> MultilabelDataset:
    features = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    labels = frame.loc[:, label_columns].to_numpy(dtype=np.float32)

    feature_missingness = np.isnan(features)
    label_availability = ~np.isnan(labels)
    observed_labels = labels[label_availability]
    if not np.isin(observed_labels, [0.0, 1.0]).all():
        raise ValueError("Observed labels must be binary (0 or 1)")

    features = (features - means) / scales
    features[feature_missingness] = 0.0
    labels[~label_availability] = 0.0

    return MultilabelDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
        torch.from_numpy(feature_missingness.astype(np.float32)),
        torch.from_numpy(label_availability.astype(np.float32)),
    )
