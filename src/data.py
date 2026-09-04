from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _grouped_strata(
    frame: pd.DataFrame, label_columns: Sequence[str], group_column: str
) -> tuple[np.ndarray, np.ndarray]:
    if group_column not in frame:
        raise ValueError(f"Group column not found: {group_column}")
    if frame[group_column].isna().any():
        raise ValueError(f"Group column contains missing values: {group_column}")

    labels = frame.loc[:, label_columns]
    observed = labels.notna().to_numpy(dtype=np.uint8)
    values = labels.fillna(0).to_numpy()
    if not np.isin(values[observed.astype(bool)], [0, 1]).all():
        raise ValueError("Observed labels must be binary (0 or 1)")

    group_codes, groups = pd.factorize(frame[group_column], sort=False)
    positive = labels.eq(1).to_numpy(dtype=np.uint8)
    group_positive = np.zeros((len(groups), positive.shape[1]), dtype=np.uint8)
    group_observed = np.zeros((len(groups), observed.shape[1]), dtype=np.uint8)
    np.maximum.at(group_positive, group_codes, positive)
    np.maximum.at(group_observed, group_codes, observed)
    variable_observed = group_observed[:, np.ptp(group_observed, axis=0) > 0]
    return group_codes, np.concatenate([group_positive, variable_observed], axis=1)


def _iterative_partitions(
    strata: np.ndarray, proportions: Sequence[float], seed: int
) -> list[np.ndarray]:
    proportions = np.asarray(proportions, dtype=float)
    if not np.isclose(proportions.sum(), 1.0) or (proportions <= 0).any():
        raise ValueError("Partition proportions must be positive and sum to 1")

    rng = np.random.default_rng(seed)
    assignments = np.full(len(strata), -1, dtype=int)
    desired_sizes = proportions * len(strata)
    desired_labels = proportions[:, None] * strata.sum(axis=0)

    while True:
        unassigned = assignments < 0
        remaining_counts = strata[unassigned].sum(axis=0)
        positive_labels = np.flatnonzero(remaining_counts > 0)
        if not len(positive_labels):
            break
        rarest_count = remaining_counts[positive_labels].min()
        rarest_labels = positive_labels[
            remaining_counts[positive_labels] == rarest_count
        ]
        label = int(rng.choice(rarest_labels))
        candidates = np.flatnonzero(unassigned & strata[:, label].astype(bool))
        rng.shuffle(candidates)

        for index in candidates:
            label_need = desired_labels[:, label]
            eligible = np.flatnonzero(label_need == label_need.max())
            size_need = desired_sizes[eligible]
            eligible = eligible[size_need == size_need.max()]
            partition = int(rng.choice(eligible))
            assignments[index] = partition
            desired_sizes[partition] -= 1
            desired_labels[partition] -= strata[index]

    for index in rng.permutation(np.flatnonzero(assignments < 0)):
        eligible = np.flatnonzero(desired_sizes == desired_sizes.max())
        partition = int(rng.choice(eligible))
        assignments[index] = partition
        desired_sizes[partition] -= 1

    return [np.flatnonzero(assignments == index) for index in range(len(proportions))]


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
    size: int,
    train: float,
    val: float,
    test: float,
    seed: int,
    *,
    frame: pd.DataFrame | None = None,
    label_columns: Sequence[str] | None = None,
    strategy: str = "random",
    group_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train + val + test, 1.0):
        raise ValueError("Split fractions must sum to 1")
    if min(train, val, test) <= 0:
        raise ValueError("Split fractions must be positive")
    if strategy == "grouped_multilabel":
        if frame is None or label_columns is None or group_column is None:
            raise ValueError(
                "Grouped multilabel splitting requires frame, label_columns, and group_column"
            )
        if len(frame) != size:
            raise ValueError("size must match the number of frame rows")
        group_codes, strata = _grouped_strata(frame, label_columns, group_column)
        group_count = int(group_codes.max()) + 1
        if group_count < 3:
            raise ValueError("Grouped splitting requires at least three groups")

        development_fraction = train + val
        development_groups, test_groups = _iterative_partitions(
            strata, [development_fraction, test], seed
        )
        train_local, val_local = _iterative_partitions(
            strata[development_groups],
            [train / development_fraction, val / development_fraction],
            seed + 1,
        )
        train_groups = development_groups[train_local]
        val_groups = development_groups[val_local]

        return tuple(
            np.flatnonzero(np.isin(group_codes, selected_groups))
            for selected_groups in (train_groups, val_groups, test_groups)
        )
    if strategy != "random":
        raise ValueError(f"Unsupported split strategy: {strategy}")
    indices = np.random.default_rng(seed).permutation(size)
    train_end = int(size * train)
    val_end = train_end + int(size * val)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def grouped_multilabel_folds(
    frame: pd.DataFrame,
    label_columns: Sequence[str],
    group_column: str,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds")

    group_codes, strata = _grouped_strata(frame, label_columns, group_column)
    group_count = int(group_codes.max()) + 1
    if group_count < folds:
        raise ValueError("Number of groups must be at least the number of folds")
    validation_groups = _iterative_partitions(
        strata, np.repeat(1 / folds, folds), seed
    )
    all_groups = np.arange(group_count)
    return [
        (
            np.flatnonzero(np.isin(group_codes, np.setdiff1d(all_groups, held_out))),
            np.flatnonzero(np.isin(group_codes, held_out)),
        )
        for held_out in validation_groups
    ]


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
