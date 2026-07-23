"""Run value inference on rollout shards and write compact advantage sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_value_prototype import ValuePrototype


def signed_rewards(episode: dict[str, Any]) -> np.ndarray:
    """Use a +1/-1 terminal outcome so rewards match signed value predictions."""
    rewards = np.asarray(episode["rewards"], dtype=np.float32).copy()
    if rewards.size:
        rewards[-1] = 1.0 if episode["success"] else -1.0
    return rewards


def monte_carlo_returns(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        returns[index] = running
    return returns


def nstep_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    n_step: int = 50,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(rewards)
    returns = np.zeros(length, dtype=np.float32)
    advantages = np.zeros(length, dtype=np.float32)
    for timestep in range(length):
        value = 0.0
        discount = 1.0
        terminal = False
        for offset in range(min(n_step, length - timestep)):
            index = timestep + offset
            value += discount * float(rewards[index])
            discount *= gamma
            if bool(dones[index]):
                terminal = True
                break
        bootstrap_index = timestep + n_step
        if not terminal and bootstrap_index < length:
            value += discount * float(values[bootstrap_index])
        returns[timestep] = value
        advantages[timestep] = value - float(values[timestep])
    return advantages, returns


def task_threshold(values: list[np.ndarray], positive_fraction: float) -> float:
    if not values:
        raise ValueError("cannot fit an advantage threshold without train values")
    flattened = np.concatenate(values)
    return float(np.percentile(flattened, (1.0 - positive_fraction) * 100.0))


def _predict_episode(
    model: ValuePrototype,
    episode: dict[str, Any],
    device: torch.device,
    image_size: int,
    batch_size: int,
    temperature: float,
) -> np.ndarray:
    probabilities = []
    length = len(episode["actions"])
    task_id = int(episode["task_id"])
    model.eval()
    with torch.no_grad():
        for start in range(0, length, batch_size):
            end = min(start + batch_size, length)
            images = np.concatenate(
                [episode["images"][start:end], episode["wrist_imgs"][start:end]], axis=-1
            )
            image_tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float().div_(255.0)
            image_tensor = F.interpolate(
                image_tensor,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).to(device)
            states = torch.from_numpy(
                np.asarray(episode["states"][start:end], dtype=np.float32)
            ).to(device)
            progress = torch.arange(start, end, dtype=torch.float32, device=device)
            progress /= max(length - 1, 1)
            tasks = torch.full((end - start,), task_id, dtype=torch.long, device=device)
            logits = model(image_tensor, states, progress, tasks)
            probabilities.append(torch.sigmoid(logits / temperature).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)


def _split_episode_ids(metadata_dir: Path) -> dict[str, set[str]]:
    result = {}
    for split in ("train", "val", "test"):
        payload = json.loads((metadata_dir / f"{split}.json").read_text())
        result[split] = {record["episode_id"] for record in payload["episodes"]}
    return result


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize(sidecars: list[dict[str, Any]], split_ids: dict[str, set[str]]) -> dict[str, Any]:
    summary = {}
    for split, identifiers in split_ids.items():
        selected = [item for item in sidecars if item["episode_id"] in identifiers]
        frames = sum(item["length"] for item in selected)
        summary[split] = {
            "episodes": len(selected),
            "frames": frames,
            "success_rate": float(np.mean([item["success"] for item in selected])) if selected else float("nan"),
            "return_only_positive_rate": float(
                sum(item["return_only_labels"].sum() for item in selected) / max(frames, 1)
            ),
            "mc_positive_rate": float(
                sum(item["mc_labels"].sum() for item in selected) / max(frames, 1)
            ),
            "n50_positive_rate": float(
                sum(item["n50_labels"].sum() for item in selected) / max(frames, 1)
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-step", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--positive-fraction", type=float, default=0.40)
    args = parser.parse_args()
    if not 0.0 < args.positive_fraction < 1.0:
        parser.error("--positive-fraction must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task_ids = [int(task) for task in checkpoint["task_ids"]]
    model = ValuePrototype(
        int(checkpoint["state_dim"]), num_tasks=max(task_ids) + 1
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = json.loads(args.metrics.read_text())
    temperature = float(metrics["calibration"]["temperature"])
    split_ids = _split_episode_ids(args.metadata_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_reports = {}
    total_episodes = total_frames = 0
    for source_path in sorted(args.data_dir.glob("task_*.pkl")):
        with source_path.open("rb") as stream:
            episodes = pickle.load(stream)
        if not episodes:
            continue
        task_id = int(episodes[0]["task_id"])
        if task_id not in task_ids:
            continue
        sidecars = []
        train_mc, train_n50 = [], []
        for episode in episodes:
            ep_idx = int(episode["ep_idx"])
            episode_id = f"task_{task_id:02d}/ep_{ep_idx:03d}"
            probabilities = _predict_episode(
                model,
                episode,
                device,
                args.image_size,
                args.batch_size,
                temperature,
            )
            values = 2.0 * probabilities - 1.0
            rewards = signed_rewards(episode)
            mc_returns = monte_carlo_returns(rewards, args.gamma)
            mc_advantages = mc_returns - values
            n50_adv, n50_returns = nstep_advantages(
                rewards,
                np.asarray(episode["dones"], dtype=bool),
                values,
                args.n_step,
                args.gamma,
            )
            item = {
                "episode_id": episode_id,
                "task_id": task_id,
                "ep_idx": ep_idx,
                "length": len(rewards),
                "success": bool(episode["success"]),
                "value_probabilities": probabilities,
                "value_predictions": values.astype(np.float32),
                "mc_returns": mc_returns,
                "mc_advantages": mc_advantages.astype(np.float32),
                "n50_returns": n50_returns,
                "n50_advantages": n50_adv.astype(np.float32),
            }
            sidecars.append(item)
            if episode_id in split_ids["train"]:
                train_mc.append(item["mc_advantages"])
                train_n50.append(item["n50_advantages"])

        mc_threshold = task_threshold(train_mc, args.positive_fraction)
        n50_threshold = task_threshold(train_n50, args.positive_fraction)
        for item in sidecars:
            item["return_only_labels"] = np.full(
                item["length"], int(item["success"]), dtype=np.int8
            )
            item["mc_labels"] = (item["mc_advantages"] > mc_threshold).astype(np.int8)
            item["n50_labels"] = (item["n50_advantages"] > n50_threshold).astype(np.int8)

        output_path = args.output_dir / f"task_{task_id:02d}_value_labels.pkl"
        with output_path.open("wb") as stream:
            pickle.dump(sidecars, stream)
        task_reports[str(task_id)] = {
            "source_file": source_path.name,
            "sidecar_file": output_path.name,
            "mc_threshold": mc_threshold,
            "n50_threshold": n50_threshold,
            "splits": _summarize(sidecars, split_ids),
        }
        total_episodes += len(sidecars)
        total_frames += sum(item["length"] for item in sidecars)
        print(json.dumps({"task_id": task_id, **task_reports[str(task_id)]}), flush=True)

    manifest = {
        "schema_version": 1,
        "source_data_dir": str(args.data_dir.resolve()),
        "metadata_dir": str(args.metadata_dir.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "temperature": temperature,
        "value_scale": "signed_success_probability_2p_minus_1",
        "threshold_fit_split": "train",
        "threshold_scope": "per_task",
        "positive_fraction": args.positive_fraction,
        "n_step": args.n_step,
        "gamma": args.gamma,
        "episodes": total_episodes,
        "frames": total_frames,
        "tasks": task_reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    print(json.dumps({"episodes": total_episodes, "frames": total_frames}, indent=2))


if __name__ == "__main__":
    main()
