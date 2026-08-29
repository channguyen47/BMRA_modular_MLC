import hashlib
import json
from typing import Any


def _get(config: dict[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing hash parameter: {dotted_path}")
        value = value[key]
    return value


def canonical_hash_input(config: dict[str, Any]) -> str:
    values = {path: _get(config, path) for path in config["hash_params"]}
    return json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def calculate_run_id(config: dict[str, Any], length: int = 12) -> str:
    digest = hashlib.sha256(canonical_hash_input(config).encode("utf-8")).hexdigest()
    return digest[:length]
