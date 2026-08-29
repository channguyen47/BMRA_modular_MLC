from pathlib import Path

from torch import nn

from src.training import save_artifacts


def test_artifact_directory_creation(tmp_path: Path):
    config_snapshot = b"value: 1\n"
    output = save_artifacts(
        "abc123",
        nn.Linear(2, 1),
        config_snapshot,
        {"test_loss": 0.5},
        [{"epoch": 1, "train_loss": 0.7}],
        tmp_path / "artifacts",
    )
    assert {path.name for path in output.iterdir()} == {
        "model.pt",
        "framework.yaml",
        "metrics.json",
        "training_history.csv",
    }
    assert (output / "framework.yaml").read_bytes() == config_snapshot
