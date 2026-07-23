"""Lazy train-split loader joining rollout payloads with compact label sidecars."""

from __future__ import annotations

import json
import pathlib
import pickle
from typing import Any, Callable

import numpy as np


ACTION_HORIZON = 50
LABEL_KEYS = {
    "return_only": "return_only_labels",
    "mc": "mc_labels",
    "n50": "n50_labels",
}


class SidecarRolloutDataLoader:
    def __init__(
        self,
        rollout_dir: pathlib.Path,
        sidecar_dir: pathlib.Path,
        metadata_dir: pathlib.Path,
        label_variant: str,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if label_variant not in LABEL_KEYS:
            raise ValueError(f"unknown label variant {label_variant!r}")
        train_payload = json.loads((metadata_dir / "train.json").read_text())
        train_ids = {record["episode_id"] for record in train_payload["episodes"]}
        label_key = LABEL_KEYS[label_variant]

        self.samples_by_task: dict[int, list[tuple[dict[str, Any], np.ndarray, int]]] = {}
        self.episodes = []
        for source_path in sorted(rollout_dir.glob("task_*.pkl")):
            with source_path.open("rb") as stream:
                episodes = pickle.load(stream)
            if not episodes:
                continue
            task_id = int(episodes[0]["task_id"])
            sidecar_path = sidecar_dir / f"task_{task_id:02d}_value_labels.pkl"
            with sidecar_path.open("rb") as stream:
                sidecars = pickle.load(stream)
            sidecars_by_id = {item["episode_id"]: item for item in sidecars}

            task_samples = self.samples_by_task.setdefault(task_id, [])
            for episode in episodes:
                episode_id = f"task_{task_id:02d}/ep_{int(episode['ep_idx']):03d}"
                if episode_id not in train_ids:
                    continue
                sidecar = sidecars_by_id[episode_id]
                labels = np.asarray(sidecar[label_key], dtype=np.int32)
                if labels.shape != (int(episode["length"]),):
                    raise ValueError(
                        f"{episode_id}: {label_key} shape {labels.shape} does not match length"
                    )
                self.episodes.append(episode)
                task_samples.extend((episode, labels, timestep) for timestep in range(len(labels)))

        if not self.samples_by_task:
            raise ValueError("no train samples loaded")
        self.task_ids = sorted(self.samples_by_task)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.label_variant = label_variant

    @property
    def num_samples(self) -> int:
        return sum(len(samples) for samples in self.samples_by_task.values())

    @property
    def positive_rate(self) -> float:
        positives = sum(
            int(labels[timestep])
            for samples in self.samples_by_task.values()
            for _, labels, timestep in samples
        )
        return positives / self.num_samples

    def sample_batch(
        self, tokenize: Callable[[list[str]], tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, ...]:
        selected = []
        # Choose tasks uniformly, then timesteps uniformly within each task.
        for task_id in self.rng.choice(self.task_ids, self.batch_size, replace=True):
            samples = self.samples_by_task[int(task_id)]
            selected.append(samples[int(self.rng.integers(0, len(samples)))])

        images, wrists, states, actions, labels, prompts = [], [], [], [], [], []
        for episode, episode_labels, timestep in selected:
            episode_actions = episode["actions"]
            end = min(timestep + ACTION_HORIZON, len(episode_actions))
            chunk = np.asarray(episode_actions[timestep:end], dtype=np.float32)
            if len(chunk) < ACTION_HORIZON:
                chunk = np.concatenate(
                    [chunk, np.repeat(chunk[-1:], ACTION_HORIZON - len(chunk), axis=0)]
                )
            images.append(episode["images"][timestep].astype(np.float32) / 127.5 - 1.0)
            wrists.append(episode["wrist_imgs"][timestep].astype(np.float32) / 127.5 - 1.0)
            states.append(np.asarray(episode["states"][timestep], dtype=np.float32))
            actions.append(chunk)
            labels.append(int(episode_labels[timestep]))
            prompts.append(episode["prompt"])

        tokens, token_mask = tokenize(prompts)
        return (
            np.stack(images),
            np.stack(wrists),
            np.stack(states),
            np.stack(actions),
            np.asarray(labels, dtype=np.int32),
            tokens,
            token_mask,
        )
