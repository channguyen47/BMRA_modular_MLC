from copy import deepcopy
from pathlib import Path

import yaml

from src.hashing import calculate_run_id


def load_config():
    with open("configs/framework.yaml", encoding="utf-8") as file:
        framework = yaml.safe_load(file)
    project_path = Path(framework["project"]["config_path"])
    with project_path.open(encoding="utf-8") as file:
        project = yaml.safe_load(file)
    framework["project"].update(project)
    return framework


def test_hashing_is_deterministic():
    config = load_config()
    reordered = deepcopy(config)
    reordered["experiment"]["name"] = "not_hashed"
    reordered["hash_params"].reverse()
    assert calculate_run_id(config) == calculate_run_id(reordered)


def test_hash_parameter_changes_run_id():
    config = load_config()
    changed = deepcopy(config)
    changed["project"]["features"][0] = "changed feature"
    assert calculate_run_id(config) != calculate_run_id(changed)
