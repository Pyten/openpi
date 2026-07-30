"""Closed-loop evaluation of counterfactual ranker-selected first action chunks."""
from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import time
import types

import numpy as np
import torch

if "numba" not in sys.modules:
    numba_mock = types.ModuleType("numba")
    numba_mock.jit = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["numba"] = numba_mock

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy as ws_client
from openpi.recap import evaluation_protocol as protocol
from train_initial_action_ranker import Ranker, features


# Do not let expensive first inference be mistaken for a dead connection.
websocket_connect = ws_client.websockets.sync.client.connect
def connect_without_ping(*args, **kwargs):
    kwargs.setdefault("ping_interval", None)
    return websocket_connect(*args, **kwargs)
ws_client.websockets.sync.client.connect = connect_without_ping


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(1.0 - quat[3] ** 2)
    return np.zeros(3) if math.isclose(den, 0.0) else (quat[:3] * 2.0 * math.acos(quat[3])) / den


def prep_obs(obs):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    return image, wrist, state


def held_out_selection(data_path: pathlib.Path, checkpoints: list[pathlib.Path], cv_fold: int, num_folds: int):
    data = dict(np.load(data_path))
    groups = sorted({(int(t), int(s)) for t, s in zip(data["task_ids"], data["init_state_indices"], strict=True)})
    rng = np.random.default_rng(20260720)
    rng.shuffle(groups)
    test_groups = {group for i, group in enumerate(groups) if i % num_folds == cv_fold}
    device = "cuda"
    models = []
    for checkpoint in checkpoints:
        model = Ranker().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        model.eval()
        models.append(model)
    candidate_rows = {}
    for i, key in enumerate(zip(data["task_ids"], data["init_state_indices"], data["candidate_ids"], strict=True)):
        task, state, candidate = map(int, key)
        if (task, state) in test_groups:
            candidate_rows.setdefault((task, state, candidate), []).append(i)
    selected = {}
    with torch.no_grad():
        for task, state in sorted(test_groups):
            rows = []
            for candidate in range(24):
                indices = candidate_rows[(task, state, candidate)]
                # Candidate action is identical across two continuation rows.
                index = indices[0]
                score = float(np.mean([model(*features(data, np.asarray([index]), device)).item() for model in models]))
                rows.append((score, candidate, data["action_chunks"][index]))
            selected[(task, state)] = max(rows, key=lambda row: row[0])[1:]
    return selected


def rollout(job):
    task_id, task_name, prompt, bddl, init_states, action, candidate, continuation, host, port, seed = job
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    client = ws_client.WebsocketClientPolicy(host, port)
    obs = env.reset()
    obs = env.set_init_state(init_states[seed])
    for _ in range(protocol.NUM_WAIT_STEPS):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    plan = collections.deque(action)
    steps = 0
    reward = 0.0
    info = {}
    while steps < protocol.MAX_STEPS:
        if not plan:
            image, wrist, state = prep_obs(obs)
            result = client.infer({
                "observation/image": image, "observation/wrist_image": wrist,
                "observation/state": state, "prompt": prompt,
                "_openpi_rng_seed": protocol.action_rng_seed(100000 + continuation, 0, steps),
            })
            plan.extend(result["actions"][: protocol.REPLAN_STEPS])
        obs, reward, done, info = env.step(plan.popleft().tolist())
        steps += 1
        if done:
            break
    env.close()
    return {"task_id": task_id, "task_name": task_name, "init_state_index": seed, "candidate_id": candidate,
            "continuation_id": continuation, "success": bool(info.get("success", reward > 0.5)), "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--ranker", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--continuations", type=int, default=3)
    parser.add_argument("--continuation-start", type=int, default=0)
    parser.add_argument("--cv-fold", type=int, default=0)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--arms", nargs="+", choices=("selected", "policy0", "random"), default=("selected", "policy0"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    selected = held_out_selection(args.data, args.ranker, args.cv_fold, args.num_folds)
    suite = benchmark.get_benchmark_dict()[protocol.SUITE_NAME]()
    task = suite.get_task(9)
    init_path = pathlib.Path(get_libero_path("init_states")) / suite.tasks[9].problem_folder / suite.tasks[9].init_states_file
    init_states = torch.load(init_path, weights_only=False)
    raw = dict(np.load(args.data))
    actions = {}
    for i, key in enumerate(zip(raw["task_ids"], raw["init_state_indices"], raw["candidate_ids"], strict=True)):
        actions.setdefault(tuple(map(int, key)), raw["action_chunks"][i])
    bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    jobs = []
    for state, (chosen_candidate, chosen_action) in selected.items():
        task_id, state_index = state
        random_candidate = int(np.random.default_rng(20260730 + state_index).integers(24))
        options = {
            "selected": (chosen_candidate, chosen_action),
            "random": (random_candidate, actions[(task_id, state_index, random_candidate)]),
            "policy0": (0, actions[(task_id, state_index, 0)]),
        }
        for arm in args.arms:
            candidate, action = options[arm]
            for continuation in range(args.continuation_start, args.continuation_start + args.continuations):
                jobs.append((arm, task_id, state_index, candidate, action, continuation))
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard]
    def invoke(job):
        arm, task_id, state_index, candidate, action, continuation = job
        record = rollout((task_id, task.name, task.language, bddl, init_states, action, candidate, continuation, args.host, args.port, state_index))
        record["arm"] = arm
        return record
    results = []
    for job in jobs:
        record = invoke(job); results.append(record); print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle: json.dump(results, handle, indent=2)
    print(json.dumps({"shard": args.shard, "episodes": len(results), "success_rate": np.mean([r["success"] for r in results])}, indent=2))


if __name__ == "__main__":
    main()
