from pathlib import Path

from torch import nn

from src.training import save_artifacts


def test_artifact_directory_creation(tmp_path: Path):
    framework_snapshot = b"value: 1\n"
    project_snapshot = b"features: [x]\nlabels: [y]\n"
    output = save_artifacts(
        "abc123",
        "student_project",
        nn.Linear(2, 1),
        framework_snapshot,
        project_snapshot,
        {"test_loss": 0.5},
        [{"epoch": 1, "train_loss": 0.7}],
        tmp_path / "artifacts",
    )
    assert {path.name for path in output.iterdir()} == {
        "model.pt",
        "framework.yaml",
        "student_project.yaml",
        "metrics.json",
        "training_history.csv",
    }
    assert output == tmp_path / "artifacts/student_project/abc123"
    assert (output / "framework.yaml").read_bytes() == framework_snapshot
    assert (output / "student_project.yaml").read_bytes() == project_snapshot
