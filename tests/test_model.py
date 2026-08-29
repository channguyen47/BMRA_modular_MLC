import torch

from src.model import MultilabelMLP


def test_model_forward_dimensions():
    model = MultilabelMLP(5, 3, [8], use_missingness_mask=True)
    output = model(torch.zeros(4, 5), torch.zeros(4, 5))
    assert output.shape == (4, 3)
