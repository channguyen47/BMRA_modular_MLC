import csv
import json
from collections.abc import Iterable
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, availability: torch.Tensor
    ) -> torch.Tensor:
        availability = availability.to(logits.dtype)
        targets = torch.where(availability.bool(), targets, torch.zeros_like(targets))
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        return (losses * availability).sum() / availability.sum().clamp_min(1.0)


def build_loss(loss_config: dict[str, Any]) -> MaskedBCELoss:
    loss_type = loss_config["type"]
    if loss_type == "masked_bce":
        return MaskedBCELoss()
    if loss_type == "masked_weighted_bce":
        if loss_config.get("pos_weight") is None:
            raise ValueError("masked_weighted_bce requires loss.pos_weight")
        return MaskedBCELoss(torch.as_tensor(loss_config["pos_weight"], dtype=torch.float32))
    raise ValueError(f"Unsupported loss type: {loss_type}")


def build_optimizer(
    model: nn.Module, training_config: dict[str, Any]
) -> torch.optim.Optimizer:
    if training_config["optimizer"].lower() != "adam":
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
    return torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )


def multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    availability: torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, float]:
    observed = availability.bool()
    predictions = torch.sigmoid(logits) >= torch.as_tensor(
        threshold, dtype=logits.dtype, device=logits.device
    )
    targets = targets.bool()
    total = observed.sum().item()
    if total == 0:
        return {"accuracy": 0.0, "micro_f1": 0.0}
    correct = ((predictions == targets) & observed).sum().item()
    tp = (predictions & targets & observed).sum().item()
    fp = (predictions & ~targets & observed).sum().item()
    fn = (~predictions & targets & observed).sum().item()
    denominator = 2 * tp + fp + fn
    return {
        "accuracy": correct / total,
        "micro_f1": (2 * tp / denominator) if denominator else 0.0,
    }


def _epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_observed = 0.0
    all_logits, all_targets, all_availability = [], [], []

    with torch.set_grad_enabled(optimizer is not None):
        for features, targets, feature_mask, label_mask in loader:
            features, targets = features.to(device), targets.to(device)
            feature_mask, label_mask = feature_mask.to(device), label_mask.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(features, feature_mask)
            loss = criterion(logits, targets, label_mask)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            observed = label_mask.sum().item()
            total_loss += loss.item() * observed
            total_observed += observed
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_availability.append(label_mask.detach().cpu())

    if not all_logits:
        raise ValueError("DataLoader is empty")
    metrics = multilabel_metrics(
        torch.cat(all_logits),
        torch.cat(all_targets),
        torch.cat(all_availability),
        threshold,
    )
    metrics["loss"] = total_loss / max(total_observed, 1.0)
    return metrics


def train_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    return _epoch(model, loader, criterion, device, optimizer)


def evaluate_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, float]:
    return _epoch(model, loader, criterion, device, optimizer=None, threshold=threshold)


def predict_loader(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_targets, all_availability = [], [], []
    with torch.no_grad():
        for features, targets, feature_mask, label_mask in loader:
            all_logits.append(model(features.to(device), feature_mask.to(device)).cpu())
            all_targets.append(targets)
            all_availability.append(label_mask)
    if not all_logits:
        raise ValueError("DataLoader is empty")
    return (
        torch.cat(all_logits),
        torch.cat(all_targets),
        torch.cat(all_availability),
    )


def _probability_micro_f1(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    availability: torch.Tensor,
    threshold: float,
) -> float:
    observed = availability.bool()
    predictions = probabilities >= threshold
    targets = targets.bool()
    tp = (predictions & targets & observed).sum().item()
    fp = (predictions & ~targets & observed).sum().item()
    fn = (~predictions & targets & observed).sum().item()
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def select_thresholds(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    availability: torch.Tensor,
    candidates: list[float],
    minimum_positive_count: int,
) -> tuple[float, torch.Tensor, dict[float, float]]:
    if not candidates:
        raise ValueError("At least one threshold candidate is required")
    scores = {
        float(threshold): _probability_micro_f1(
            probabilities, targets, availability, float(threshold)
        )
        for threshold in candidates
    }
    global_threshold = max(
        scores, key=lambda threshold: (scores[threshold], -abs(threshold - 0.5))
    )
    label_thresholds = torch.full(
        (targets.shape[1],), global_threshold, dtype=probabilities.dtype
    )
    for index in range(targets.shape[1]):
        observed = availability[:, index].bool()
        observed_targets = targets[observed, index]
        positives = int(observed_targets.sum().item())
        negatives = len(observed_targets) - positives
        if positives < minimum_positive_count or negatives == 0:
            continue
        label_thresholds[index] = max(
            scores,
            key=lambda threshold: (
                _probability_micro_f1(
                    probabilities[:, index],
                    targets[:, index],
                    availability[:, index],
                    threshold,
                ),
                -abs(threshold - global_threshold),
            ),
        )
    return global_threshold, label_thresholds, scores


def resolve_final_epochs(setting: str | int, fold_best_epochs: list[int]) -> int:
    if isinstance(setting, int) and not isinstance(setting, bool) and setting > 0:
        return setting
    if setting == "median_best_epoch" and fold_best_epochs:
        return max(1, int(round(median(fold_best_epochs))))
    raise ValueError(
        "cross_validation.final_epochs must be a positive integer or median_best_epoch"
    )


def save_artifacts(
    run_id: str,
    project_basename: str,
    model: nn.Module,
    framework_snapshot: bytes,
    project_snapshot: bytes,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    artifacts_root: str | Path = "artifacts",
) -> Path:
    output_dir = Path(artifacts_root) / project_basename / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    (output_dir / "framework.yaml").write_bytes(framework_snapshot)
    (output_dir / f"{project_basename}.yaml").write_bytes(project_snapshot)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        if not history:
            raise ValueError("Training history cannot be empty")
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return output_dir
