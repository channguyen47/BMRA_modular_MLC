from copy import deepcopy

import yaml

from src.hashing import calculate_run_id


def test_hashing_is_deterministic():
    with open("configs/framework.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    reordered = deepcopy(config)
    reordered["experiment"]["name"] = "not_hashed"
    reordered["hash_params"].reverse()
    assert calculate_run_id(config) == calculate_run_id(reordered)


def test_hash_parameter_changes_run_id():
    with open("configs/framework.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    changed = deepcopy(config)
    changed["model"]["dropout"] = 0.3
    assert calculate_run_id(config) != calculate_run_id(changed)
