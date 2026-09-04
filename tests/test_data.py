import numpy as np
import pandas as pd

from src.data import grouped_multilabel_folds, split_indices


def test_grouped_multilabel_split_is_deterministic_and_leak_free():
    patient_ids = np.repeat(np.arange(60), 2)
    frame = pd.DataFrame(
        {
            "ID": patient_ids,
            "a": (patient_ids % 2 == 0).astype(int),
            "b": (patient_ids % 5 == 0).astype(int),
            "c": (patient_ids % 7 == 0).astype(int),
        }
    )
    kwargs = {
        "frame": frame,
        "label_columns": ["a", "b", "c"],
        "strategy": "grouped_multilabel",
        "group_column": "ID",
    }

    first = split_indices(len(frame), 0.7, 0.15, 0.15, 42, **kwargs)
    second = split_indices(len(frame), 0.7, 0.15, 0.15, 42, **kwargs)

    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert set(np.concatenate(first)) == set(range(len(frame)))

    patient_sets = [set(frame.iloc[indices]["ID"]) for indices in first]
    assert patient_sets[0].isdisjoint(patient_sets[1])
    assert patient_sets[0].isdisjoint(patient_sets[2])
    assert patient_sets[1].isdisjoint(patient_sets[2])

    overall = frame[["a", "b", "c"]].mean()
    for indices in first:
        difference = (frame.iloc[indices][["a", "b", "c"]].mean() - overall).abs()
        assert (difference < 0.12).all()


def test_random_split_remains_available_for_legacy_artifacts():
    first = split_indices(20, 0.7, 0.15, 0.15, 42)
    second = split_indices(20, 0.7, 0.15, 0.15, 42)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_grouped_multilabel_folds_keep_patients_together():
    patient_ids = np.repeat(np.arange(30), 2)
    frame = pd.DataFrame(
        {
            "ID": patient_ids,
            "a": (patient_ids % 2 == 0).astype(int),
            "b": (patient_ids % 3 == 0).astype(int),
        }
    )
    folds = grouped_multilabel_folds(frame, ["a", "b"], "ID", folds=3, seed=42)

    validation_rows = []
    for train_indices, validation_indices in folds:
        train_patients = set(frame.iloc[train_indices]["ID"])
        validation_patients = set(frame.iloc[validation_indices]["ID"])
        assert train_patients.isdisjoint(validation_patients)
        validation_rows.extend(validation_indices)
    assert sorted(validation_rows) == list(range(len(frame)))
