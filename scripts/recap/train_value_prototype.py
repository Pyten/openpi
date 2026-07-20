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
from torch.utils.data import DataLoader, Dataset


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


def load_split_references(metadata_dir: Path, split: str, task_id: int) -> list[dict[str, Any]]:
    payload = json.loads((metadata_dir / f"{split}.json").read_text())
    return [record for record in payload["episodes"] if record["task_id"] == task_id]


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
            "label": torch.tensor(float(episode["success"]), dtype=torch.float32),
        }


class ValuePrototype(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(6, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + state_dim + 1, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.vision(image), state, progress[:, None]], dim=-1)
        return self.head(features).squeeze(-1)


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, train_prior: float
) -> dict[str, Any]:
    model.eval()
    labels, probabilities, episode_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device), batch["state"].to(device), batch["progress"].to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())
            episode_ids.extend(batch["episode_id"])
    frame_metrics = binary_metrics(np.asarray(labels), np.asarray(probabilities))
    episodes: dict[str, list[tuple[float, float]]] = {}
    for episode_id, label, probability in zip(episode_ids, labels, probabilities, strict=True):
        episodes.setdefault(episode_id, []).append((label, probability))
    episode_labels = np.asarray([values[0][0] for values in episodes.values()])
    episode_probabilities = np.asarray([np.mean([item[1] for item in values]) for values in episodes.values()])
    return {
        "frames": len(labels),
        "episodes": len(episodes),
        "frame": frame_metrics,
        "episode": binary_metrics(episode_labels, episode_probabilities),
        "constant_prior": binary_metrics(
            episode_labels, np.full_like(episode_labels, train_prior, dtype=np.float64)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=5)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {
        split: RolloutFrames(
            args.data_dir,
            load_split_references(args.metadata_dir, split, args.task_id),
            args.frames_per_episode,
            args.image_size,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=0)
        for split, dataset in datasets.items()
    }
    state_dim = datasets["train"][0]["state"].numel()
    model = ValuePrototype(state_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_episode_labels = {
        episode_id: float(episode["success"])
        for episode_id, episode, _ in datasets["train"].samples
    }
    positive = sum(train_episode_labels.values())
    train_prior = positive / len(train_episode_labels)
    criterion = nn.BCEWithLogitsLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].to(device), batch["state"].to(device), batch["progress"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, loaders["val"], device, train_prior)
        score = validation["episode"]["auroc"] - 0.1 * validation["episode"]["brier"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation})
        print(json.dumps(history[-1], allow_nan=True), flush=True)
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "state_dim": state_dim, "args": vars(args)}, args.output_dir / "best.pt")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    report = {
        "task_id": args.task_id,
        "seed": args.seed,
        "device": str(device),
        "train_prior": train_prior,
        "split_grouping": ["task_id", "init_state_index"],
        "validation": evaluate(model, loaders["val"], device, train_prior),
        "test": evaluate(model, loaders["test"], device, train_prior),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report["test"], indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
