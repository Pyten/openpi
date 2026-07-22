"""Train a small real-data value prototype on grouped rollout splits."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    auc = float("nan")
    if positives and negatives:
        order = np.argsort(probabilities, kind="stable")
        ranks = np.empty(len(order), dtype=np.float64)
        sorted_probabilities = probabilities[order]
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and sorted_probabilities[end] == sorted_probabilities[start]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    brier = float(np.mean((probabilities - labels) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        upper_closed = index == bins - 1
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if upper_closed else probabilities < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return {"auroc": auc, "brier": brier, "ece": ece}


def summarize_outputs(
    labels: np.ndarray,
    probabilities: np.ndarray,
    episode_ids: list[str],
    train_prior: float,
) -> dict[str, Any]:
    """Report frame and episode metrics from per-frame predictions."""
    frame_metrics = binary_metrics(labels, probabilities)
    episodes: dict[str, list[tuple[float, float]]] = {}
    for episode_id, label, probability in zip(
        episode_ids, labels, probabilities, strict=True
    ):
        episodes.setdefault(episode_id, []).append((float(label), float(probability)))
    episode_labels = np.asarray([values[0][0] for values in episodes.values()])
    episode_probabilities = np.asarray(
        [np.mean([item[1] for item in values]) for values in episodes.values()]
    )
    return {
        "frames": int(len(labels)),
        "episodes": int(len(episodes)),
        "frame": frame_metrics,
        "episode": binary_metrics(episode_labels, episode_probabilities),
        "constant_prior": binary_metrics(
            episode_labels, np.full_like(episode_labels, train_prior, dtype=np.float64)
        ),
    }


def multitask_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    episode_ids: list[str],
    task_ids: np.ndarray,
    task_priors: dict[int, float],
) -> dict[str, Any]:
    """Report pooled, macro, and per-task metrics without hiding weak tasks."""
    pooled_prior = float(np.mean([task_priors[int(task)] for task in task_ids]))
    per_task = {}
    for task_id in sorted(set(task_ids.tolist())):
        mask = task_ids == task_id
        per_task[str(task_id)] = summarize_outputs(
            labels[mask],
            probabilities[mask],
            [episode_id for episode_id, keep in zip(episode_ids, mask, strict=True) if keep],
            task_priors[task_id],
        )
    macro = {
        level: {
            metric: float(np.nanmean([report[level][metric] for report in per_task.values()]))
            for metric in ("auroc", "brier", "ece")
        }
        for level in ("frame", "episode")
    }
    return {
        "pooled": summarize_outputs(labels, probabilities, episode_ids, pooled_prior),
        "macro": macro,
        "per_task": per_task,
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one positive temperature on validation logits only."""
    logit_tensor = torch.as_tensor(logits, dtype=torch.float64)
    label_tensor = torch.as_tensor(labels, dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            logit_tensor / temperature, label_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20.0).detach())


def progress_heuristic(
    samples: list[tuple[str, dict[str, Any], int]],
    bins: int = 10,
    smoothing: float = 1.0,
) -> dict[int, dict[int, float]]:
    """Estimate success probability from train-only progress bins."""
    estimates = {}
    task_ids = sorted({int(episode["task_id"]) for _, episode, _ in samples})
    for task_id in task_ids:
        sums = np.zeros(bins, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.float64)
        seen_episodes: set[tuple[str, int]] = set()
        for episode_id, episode, timestep in samples:
            if int(episode["task_id"]) != task_id:
                continue
            # Each episode contributes once per progress bin, avoiding frame weighting.
            progress = timestep / max(len(episode["actions"]) - 1, 1)
            index = min(int(progress * bins), bins - 1)
            if (episode_id, index) in seen_episodes:
                continue
            seen_episodes.add((episode_id, index))
            sums[index] += float(episode["success"])
            counts[index] += 1.0
        prior = sums.sum() / max(counts.sum(), 1.0)
        estimates[task_id] = {}
        for index in range(bins):
            denominator = counts[index] + smoothing
            estimates[task_id][index] = float(
                (sums[index] + smoothing * prior) / denominator
                if denominator
                else prior
            )
    return estimates


def heuristic_outputs(
    samples: list[tuple[str, dict[str, Any], int]],
    estimates: dict[int, dict[int, float]],
    bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    labels, probabilities, task_ids, episode_ids = [], [], [], []
    for episode_id, episode, timestep in samples:
        progress = timestep / max(len(episode["actions"]) - 1, 1)
        index = min(int(progress * bins), bins - 1)
        task_id = int(episode["task_id"])
        labels.append(float(episode["success"]))
        probabilities.append(estimates[task_id][index])
        task_ids.append(task_id)
        episode_ids.append(episode_id)
    return (
        np.asarray(labels),
        np.asarray(probabilities),
        np.asarray(task_ids, dtype=np.int64),
        episode_ids,
    )


def load_split_references(
    metadata_dir: Path, split: str, task_ids: list[int]
) -> list[dict[str, Any]]:
    payload = json.loads((metadata_dir / f"{split}.json").read_text())
    selected = set(task_ids)
    return [record for record in payload["episodes"] if record["task_id"] in selected]


class RolloutFrames(Dataset):
    def __init__(
        self,
        data_dir: Path,
        references: list[dict[str, Any]],
        frames_per_episode: int,
        image_size: int,
    ) -> None:
        shards: dict[str, list[dict[str, Any]]] = {}
        self.samples = []
        self.image_size = image_size
        for reference in references:
            source_file = reference["source_file"]
            if source_file not in shards:
                with (data_dir / source_file).open("rb") as handle:
                    shards[source_file] = pickle.load(handle)
            episode = shards[source_file][reference["task_index"]]
            timesteps = np.unique(
                np.linspace(0, len(episode["actions"]) - 1, frames_per_episode, dtype=np.int64)
            )
            for timestep in timesteps:
                self.samples.append((reference["episode_id"], episode, int(timestep)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_id, episode, timestep = self.samples[index]
        image = np.concatenate([episode["images"][timestep], episode["wrist_imgs"][timestep]], axis=-1)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        image_tensor = F.interpolate(
            image_tensor[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )[0]
        return {
            "episode_id": episode_id,
            "image": image_tensor,
            "state": torch.from_numpy(np.asarray(episode["states"][timestep], dtype=np.float32)),
            "progress": torch.tensor(timestep / max(len(episode["actions"]) - 1, 1), dtype=torch.float32),
            "task_id": torch.tensor(int(episode["task_id"]), dtype=torch.long),
            "label": torch.tensor(float(episode["success"]), dtype=torch.float32),
        }


class ValuePrototype(nn.Module):
    def __init__(self, state_dim: int, num_tasks: int, task_dim: int = 16) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.task_embedding = nn.Embedding(num_tasks, task_dim)
        self.head = nn.Sequential(
            nn.Linear(128 + state_dim + 1 + task_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        progress: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [self.vision(image), state, progress[:, None], self.task_embedding(task_id)],
            dim=-1,
        )
        return self.head(features).squeeze(-1)


def predict_outputs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels, logits, progresses, task_ids, episode_ids = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_logits = model(
                batch["image"].to(device),
                batch["state"].to(device),
                batch["progress"].to(device),
                batch["task_id"].to(device),
            )
            logits.extend(batch_logits.cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            progresses.extend(batch["progress"].numpy().tolist())
            task_ids.extend(batch["task_id"].numpy().tolist())
            episode_ids.extend(batch["episode_id"])
    return (
        np.asarray(labels),
        np.asarray(logits),
        np.asarray(progresses),
        np.asarray(task_ids, dtype=np.int64),
        episode_ids,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_prior: float,
    temperature: float = 1.0,
) -> dict[str, Any]:
    labels, logits, _, _, episode_ids = predict_outputs(model, loader, device)
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
    return summarize_outputs(labels, probabilities, episode_ids, train_prior)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument(
        "--task-ids",
        help="Comma-separated task IDs for joint training; overrides --task-id",
    )
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    task_ids = (
        sorted({int(value.strip()) for value in args.task_ids.split(",")})
        if args.task_ids
        else [args.task_id]
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: RolloutFrames(
            args.data_dir,
            load_split_references(args.metadata_dir, split, task_ids),
            args.frames_per_episode,
            args.image_size,
        )
        for split in ("train", "val", "test")
    }
    train_sample_tasks = [int(episode["task_id"]) for _, episode, _ in datasets["train"].samples]
    task_sample_counts = {task: train_sample_tasks.count(task) for task in task_ids}
    sample_weights = [1.0 / task_sample_counts[task] for task in train_sample_tasks]
    train_sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, sampler=train_sampler, num_workers=0
        ),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0),
    }
    state_dim = datasets["train"][0]["state"].numel()
    model = ValuePrototype(state_dim, num_tasks=max(task_ids) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_priors = {}
    for task_id in task_ids:
        labels_for_task = {
            episode_id: float(episode["success"])
            for episode_id, episode, _ in datasets["train"].samples
            if int(episode["task_id"]) == task_id
        }
        train_priors[task_id] = sum(labels_for_task.values()) / len(labels_for_task)
    train_prior = float(np.mean(list(train_priors.values())))
    criterion = nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch["image"].to(device),
                batch["state"].to(device),
                batch["progress"].to(device),
                batch["task_id"].to(device),
            )
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_labels, val_logits, _, val_task_ids, val_episode_ids = predict_outputs(
            model, loaders["val"], device
        )
        validation = multitask_metrics(
            val_labels,
            1.0 / (1.0 + np.exp(-val_logits)),
            val_episode_ids,
            val_task_ids,
            train_priors,
        )
        score = validation["macro"]["episode"]["auroc"] - 0.1 * validation["macro"]["episode"]["brier"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation})
        print(json.dumps(history[-1], allow_nan=True), flush=True)
        if score > best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "state_dim": state_dim, "task_ids": task_ids, "args": vars(args)},
                args.output_dir / "best.pt",
            )

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_labels, val_logits, _, val_task_ids, val_episode_ids = predict_outputs(
        model, loaders["val"], device
    )
    test_labels, test_logits, _, test_task_ids, test_episode_ids = predict_outputs(
        model, loaders["test"], device
    )
    # Calibrate using validation logits only; test labels are never used to fit T.
    temperature = fit_temperature(val_logits, val_labels)
    raw_validation = multitask_metrics(
        val_labels,
        1.0 / (1.0 + np.exp(-val_logits)),
        val_episode_ids,
        val_task_ids,
        train_priors,
    )
    raw_test = multitask_metrics(
        test_labels,
        1.0 / (1.0 + np.exp(-test_logits)),
        test_episode_ids,
        test_task_ids,
        train_priors,
    )
    calibrated_test = multitask_metrics(
        test_labels,
        1.0 / (1.0 + np.exp(-test_logits / temperature)),
        test_episode_ids,
        test_task_ids,
        train_priors,
    )
    heuristic_estimates = progress_heuristic(datasets["train"].samples)
    heuristic_labels, heuristic_probabilities, heuristic_task_ids, heuristic_ids = heuristic_outputs(
        datasets["test"].samples, heuristic_estimates
    )
    task_only_probabilities = np.asarray([train_priors[int(task)] for task in test_task_ids])
    report = {
        "task_ids": task_ids,
        "seed": args.seed,
        "device": str(device),
        "train_prior": train_prior,
        "train_priors": train_priors,
        "sampling": "task_balanced",
        "conditioning": "task_embedding",
        "split_grouping": ["task_id", "init_state_index"],
        "calibration": {
            "temperature": temperature,
            "fit_split": "val",
            "raw_test": raw_test,
            "temperature_scaled_test": calibrated_test,
            "task_id_only_test": multitask_metrics(
                test_labels,
                task_only_probabilities,
                test_episode_ids,
                test_task_ids,
                train_priors,
            ),
            "progress_task_id_test": multitask_metrics(
                heuristic_labels,
                heuristic_probabilities,
                heuristic_ids,
                heuristic_task_ids,
                train_priors,
            ),
            "progress_bin_estimates": heuristic_estimates,
        },
        "validation": raw_validation,
        "test": raw_test,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["test"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
