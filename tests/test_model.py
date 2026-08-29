import torch

from src.model import build_model


def test_model_forward_dimensions():
    framework = {
        "model": {
            "hidden_dims": [8],
            "activation": "relu",
            "dropout": 0.0,
            "use_missingness_mask": True,
        }
    }
    project = {"features": [f"x{i}" for i in range(5)], "labels": ["a", "b", "c"]}
    model = build_model(framework, project)
    output = model(torch.zeros(4, 5), torch.zeros(4, 5))
    assert output.shape == (4, 3)
