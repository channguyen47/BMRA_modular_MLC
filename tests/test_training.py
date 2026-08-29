import torch

from src.training import MaskedBCELoss


def test_missing_labels_do_not_contribute_to_loss():
    criterion = MaskedBCELoss()
    targets = torch.tensor([[1.0, float("nan")]])
    availability = torch.tensor([[1.0, 0.0]])
    first = criterion(torch.tensor([[0.5, -100.0]]), targets, availability)
    second = criterion(torch.tensor([[0.5, 100.0]]), targets, availability)
    assert torch.equal(first, second)
