"""
Phase 1C: 多进程 Rollout 采集
策略: serve_policy.py (WebSocket server) + 多进程 worker 并行采集

用法:
  # 先在另一个终端启动策略 server:
  # .venv/bin/python scripts/serve_policy.py --env libero \
  #   --policy.config pi0_libero \
  #   --policy.dir checkpoints/pi0_libero/recap_sft_baseline/29999 \
  #   --port 8000

  # 然后启动采集:
  # MUJOCO_GL=osmesa python collect_rollouts.py
"""

import collections
import json
import math
import multiprocessing as mp
import os
import pathlib
import pickle
import sys
import time
import traceback
import types
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

# LIBERO imports robosuite's optional numba path, which is incompatible with
# the NumPy version used by openpi. Rollout collection does not need JIT here.
if "numba" not in sys.modules:
    numba_mock = types.ModuleType("numba")
    numba_mock.jit = lambda *args, **kwargs: (lambda fn: fn)
    sys.modules["numba"] = numba_mock

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools, websocket_client_policy as _ws_client
from openpi.recap import evaluation_protocol as protocol

# First JAX inference can block longer than WebSocket's default keepalive window.
_websocket_connect = _ws_client.websockets.sync.client.connect
def _connect_without_ping(*args, **kwargs):
    kwargs.setdefault("ping_interval", None)
    return _websocket_connect(*args, **kwargs)
_ws_client.websockets.sync.client.connect = _connect_without_ping

# ── constants ────────────────────────────────────────────────────────────────
POLICY_HOST = os.environ.get("RECAP_POLICY_HOST", "127.0.0.1")
POLICY_PORT = int(os.environ.get("RECAP_POLICY_PORT", "8000"))
POLICY_ID   = os.environ.get("RECAP_POLICY_ID", "unknown")
SUITE_NAME   = protocol.SUITE_NAME
NUM_EPISODES = int(os.environ.get("RECAP_NUM_EPISODES", "50"))
EPISODE_START = int(os.environ.get("RECAP_EPISODE_START", "0"))
PILOT_TASK_IDS = {int(x) for x in os.environ.get("RECAP_PILOT_TASK_IDS", "9").split(",")}
PILOT_INIT_STATES = int(os.environ.get("RECAP_PILOT_INIT_STATES", "10"))
INIT_STATE_INDICES = [int(x) for x in os.environ.get("RECAP_INIT_STATE_INDICES", "").split(",") if x.strip()]
PILOT_CANDIDATES = int(os.environ.get("RECAP_PILOT_CANDIDATES", "24"))
PILOT_CONTINUATIONS = int(os.environ.get("RECAP_PILOT_CONTINUATIONS", "2"))
MAX_STEPS    = protocol.MAX_STEPS
NUM_WAIT     = protocol.NUM_WAIT_STEPS
RESIZE       = protocol.RESIZE
REPLAN_STEPS = protocol.REPLAN_STEPS
NUM_WORKERS  = 10          # one process per LIBERO-Spatial task
COLLECTION_SEED = int(os.environ.get(
    "RECAP_COLLECTION_SEED", os.environ.get("RECAP_SEED_BASE", "0")
))
WRITE_MERGED = os.environ.get("RECAP_WRITE_MERGED", "0") == "1"
OUTPUT_DIR   = pathlib.Path(os.environ.get(
    "RECAP_OUTPUT_DIR",
    "/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts",
))
LOG_DIR      = pathlib.Path(os.environ.get(
    "RECAP_LOG_DIR",
    "/mnt/vepfs/pyten/Programs/code/pi0.6/logs",
))
DUMMY_ACTION = [0.0] * 6 + [-1.0]


# ── helpers ──────────────────────────────────────────────────────────────────
def _quat2axisangle(quat):
    if quat[3] > 1.0:  quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _prep_obs(obs, resize):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img   = image_tools.convert_to_uint8(image_tools.resize_with_pad(img,   resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate((
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ))
    return img, wrist, state


# ── per-task worker ───────────────────────────────────────────────────────────
def worker_fn(task_id: int, task_name: str, task_desc: str, task_bddl: str,
              init_states, num_episodes: int, output_path: pathlib.Path,
              log_path: pathlib.Path, seed: int, policy_id: str):
    """Each worker runs one LIBERO task, collects num_episodes rollouts."""

    os.environ["MUJOCO_GL"] = "osmesa"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", buffering=1)

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][task{task_id}] {msg}", flush=True)
        log_f.write(f"[{ts}][task{task_id}] {msg}\n")

    log(f"Starting: {task_name}")
    log(f"  desc: {task_desc}")
    log(f"  episodes: {num_episodes}")

    # connect to policy server with retry
    client = None
    for attempt in range(30):
        try:
            client = _ws_client.WebsocketClientPolicy(POLICY_HOST, POLICY_PORT)
            log("Connected to policy server")
            break
        except Exception as e:
            if attempt % 5 == 0:
                log(f"  waiting for policy server... ({e})")
            time.sleep(2)
    if client is None:
        log("ERROR: could not connect to policy server after 60s")
        return

    # init env
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl,
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(seed)
    except Exception as e:
        log(f"ERROR: env init failed: {e}")
        traceback.print_exc(file=log_f)
        return

    episodes = []
    successes = 0

    for local_ep_idx in range(num_episodes):
        ep_idx = EPISODE_START + local_ep_idx
        ep_start = time.time()
        try:
            env.reset()
            # use fixed initial state for reproducibility
            state_slot = (ep_idx // (PILOT_CANDIDATES * PILOT_CONTINUATIONS)) % PILOT_INIT_STATES
            state_idx = INIT_STATE_INDICES[state_slot] if INIT_STATE_INDICES else state_slot
            candidate_id = ep_idx % PILOT_CANDIDATES
            continuation_id = (ep_idx // PILOT_CANDIDATES) % PILOT_CONTINUATIONS
            obs = env.set_init_state(init_states[state_idx])
        except Exception as e:
            log(f"  ep{ep_idx}: reset failed: {e}")
            continue

        ep_obs_imgs   = []
        ep_wrist_imgs = []
        ep_states     = []
        ep_actions    = []
        ep_rewards    = []
        ep_dones      = []

        action_plan = collections.deque()
        t = 0
        done = False
        reward = 0.0
        info = {}

        while t < MAX_STEPS + NUM_WAIT:
            try:
                if t < NUM_WAIT:
                    obs, reward, done, info = env.step(DUMMY_ACTION)
                    t += 1
                    continue

                img, wrist, state = _prep_obs(obs, RESIZE)

                if not action_plan:
                    action_step = t - NUM_WAIT
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist,
                        "observation/state": state,
                        "prompt": task_desc,
                        # Only the first chunk changes by candidate; later policy noise is shared.
                        "_openpi_rng_seed": protocol.action_rng_seed(
                            seed + (candidate_id * 100 if action_step == 0 else continuation_id),
                            0, action_step,
                        ),
                    }
                    try:
                        result = client.infer(element)
                        chunk = result["actions"]
                        action_plan.extend(chunk[:REPLAN_STEPS])
                    except Exception as e:
                        log(f"  ep{ep_idx} t={t}: infer failed: {e}")
                        break

                action = action_plan.popleft()

                ep_obs_imgs.append(img)
                ep_wrist_imgs.append(wrist)
                ep_states.append(state.copy())
                ep_actions.append(np.array(action, dtype=np.float32))

                obs, reward, done, info = env.step(action.tolist())
                ep_rewards.append(float(reward))
                ep_dones.append(bool(done))

                if done:
                    break

                t += 1

            except Exception as e:
                log(f"  ep{ep_idx} t={t}: step error: {e}")
                break

        success = bool(info.get("success", reward > 0.5))
        successes += int(success)

        if len(ep_actions) > 0:
            episodes.append({
                "task_id":    task_id,
                "task_name":  task_name,
                "prompt":     task_desc,
                "ep_idx":     ep_idx,
                "success":    success,
                "length":     len(ep_actions),
                "images":     np.stack(ep_obs_imgs[: REPLAN_STEPS + 1]),
                "wrist_imgs": np.stack(ep_wrist_imgs[: REPLAN_STEPS + 1]),
                "states":     np.stack(ep_states[: REPLAN_STEPS + 1]),
                "actions":    np.stack(ep_actions[:REPLAN_STEPS]),
                "collection_seed": seed,
                "init_state_index": state_idx,
                "candidate_id": candidate_id,
                "continuation_id": continuation_id,
                "policy_id": policy_id,
                "action_rng_mode": "protocol_v1_explicit",
                "protocol": {
                    "suite": SUITE_NAME,
                    "wait_steps": NUM_WAIT,
                    "max_steps": MAX_STEPS,
                    "replan_steps": REPLAN_STEPS,
                    "resize": RESIZE,
                },
            })

        ep_time = time.time() - ep_start
        log(f"  ep{ep_idx}: success={success} len={len(ep_actions)} t={ep_time:.1f}s  [{successes}/{ep_idx+1}]")

    env.close()
    log_f.close()

    # save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(episodes, f)

    print(f"[task{task_id}] Done: {successes}/{num_episodes} success, saved {len(episodes)} eps → {output_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    bm_dict = benchmark.get_benchmark_dict()
    suite   = bm_dict[SUITE_NAME]()
    n_tasks = suite.n_tasks
    print(f"Suite: {SUITE_NAME}, tasks: {n_tasks}")

    jobs = []
    for task_id in range(n_tasks):
        if task_id not in PILOT_TASK_IDS:
            continue
        task       = suite.get_task(task_id)
        task_bddl  = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        # PyTorch 2.6 changed torch.load default to weights_only=True which
        # breaks loading numpy arrays. Patch to load with weights_only=False.
        import torch, os as _os
        _benchmark_mod = suite.__class__.__module__.split('.')[0]
        init_states_path = _os.path.join(
            get_libero_path("init_states"),
            suite.tasks[task_id].problem_folder,
            suite.tasks[task_id].init_states_file,
        )
        init_states = torch.load(init_states_path, weights_only=False)
        out_path   = OUTPUT_DIR / f"task_{task_id:02d}_{task.name}.pkl"
        log_path   = LOG_DIR / f"rollout_task_{task_id:02d}.log"

        jobs.append(dict(
            task_id=task_id,
            task_name=task.name,
            task_desc=task.language,
            task_bddl=task_bddl,
            init_states=init_states,
            num_episodes=NUM_EPISODES,
            output_path=out_path,
            log_path=log_path,
            seed=COLLECTION_SEED,
            policy_id=POLICY_ID,
        ))

    print(f"Launching {min(NUM_WORKERS, n_tasks)} parallel workers for {n_tasks} tasks × {NUM_EPISODES} eps each")
    t0 = time.time()

    procs = []
    for j in jobs:
        p = mp.Process(target=worker_fn, kwargs=j, daemon=True)
        p.start()
        procs.append((p, j["task_id"]))
        print(f"  Started worker for task {j['task_id']}: {j['task_name']}")

    for p, tid in procs:
        p.join()
        print(f"  Worker task {tid} finished (exit={p.exitcode})")

    elapsed = time.time() - t0
    print(f"\nAll workers done in {elapsed/3600:.2f}h")

    # Summarize task shards without retaining the full dataset in memory.
    all_eps = [] if WRITE_MERGED else None
    total = 0
    n_succ = 0
    for j in jobs:
        fpath = j["output_path"]
        if fpath.exists():
            with open(fpath, "rb") as f:
                eps = pickle.load(f)
            total += len(eps)
            n_succ += sum(1 for episode in eps if episode["success"])
            if all_eps is not None:
                all_eps.extend(eps)
            print(f"  task{j['task_id']}: {len(eps)} episodes loaded")
        else:
            print(f"  task{j['task_id']}: NO OUTPUT FILE")

    print(f"\nCollected {total} episodes, success rate: {n_succ}/{total} = {n_succ/max(total,1)*100:.1f}%")
    if all_eps is not None:
        merged_path = OUTPUT_DIR / "all_episodes.pkl"
        with open(merged_path, "wb") as f:
            pickle.dump(all_eps, f)
        print(f"Merged compatibility file saved to {merged_path}")

    manifest = {
        "policy_id": POLICY_ID,
        "policy_host": POLICY_HOST,
        "policy_port": POLICY_PORT,
        "suite": SUITE_NAME,
        "episodes_per_task": NUM_EPISODES,
        "num_tasks": n_tasks,
        "num_episodes": total,
        "num_successes": n_succ,
        "success_rate": n_succ / max(total, 1),
        "collection_seed": COLLECTION_SEED,
        "action_rng_mode": "protocol_v1_explicit",
        "storage_mode": "task_shards_with_merged_copy" if WRITE_MERGED else "task_shards",
        "protocol": {
            "wait_steps": NUM_WAIT,
            "max_steps": MAX_STEPS,
            "replan_steps": REPLAN_STEPS,
            "resize": RESIZE,
        },
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
