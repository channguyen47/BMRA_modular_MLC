import torch

from src.training import MaskedBCELoss, resolve_final_epochs, select_thresholds


def test_missing_labels_do_not_contribute_to_loss():
    criterion = MaskedBCELoss()
    targets = torch.tensor([[1.0, float("nan")]])
    availability = torch.tensor([[1.0, 0.0]])
    first = criterion(torch.tensor([[0.5, -100.0]]), targets, availability)
    second = criterion(torch.tensor([[0.5, 100.0]]), targets, availability)
    assert torch.equal(first, second)


def test_threshold_selection_ignores_missing_labels():
    probabilities = torch.tensor([[0.4, 0.99], [0.6, 0.01]])
    targets = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    availability = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    global_threshold, label_thresholds, scores = select_thresholds(
        probabilities, targets, availability, [0.3, 0.5, 0.7], 1
    )

    assert global_threshold == 0.5
    assert torch.equal(label_thresholds, torch.tensor([0.5, 0.5]))
    assert scores[0.5] == 1.0


def test_final_epochs_can_use_cv_median_or_fixed_value():
    assert resolve_final_epochs("median_best_epoch", [20, 21, 17]) == 20
    assert resolve_final_epochs(40, [20, 21, 17]) == 40
