"""
Phase 1E: Evaluate RECAP policy vs SFT baseline on LIBERO spatial tasks.

RECAP inference uses CFG (beta=2.0):
  v_guided = v_uncond + beta * (v_cond - v_uncond)
  where cond = adv_label=1, uncond = adv_label=0

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  MUJOCO_GL=osmesa XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    .venv/bin/python scripts/recap/eval_recap.py \
      --recap-ckpt checkpoints/pi0_libero/recap_policy/10000 \
      --sft-ckpt   checkpoints/pi0_libero/recap_sft_baseline/29999 \
      --episodes-per-task 20
"""

import argparse
import math
import os
import pathlib
import sys
import time
import traceback

import einops
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from openpi.models import pi0 as _pi0
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils

HIDDEN_DIM    = 2048
CFG_BETA      = 2.0
ACTION_HORIZON = 50
ACTION_DIM     = 32
REPLAN_STEPS   = 5
MAX_STEPS      = 220
SUITE_NAME     = "libero_spatial"
RESIZE         = 224


# ── env helpers (same as collect_rollouts.py) ─────────────────────────────────
def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _prep_obs(raw_obs, resize):
    from openpi_client import image_tools
    img   = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img   = image_tools.convert_to_uint8(image_tools.resize_with_pad(img,   resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate((
        raw_obs["robot0_eef_pos"],
        _quat2axisangle(raw_obs["robot0_eef_quat"]),
        raw_obs["robot0_gripper_qpos"],
    ))
    return img, wrist, state


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(ckpt_dir: str, train_config):
    model_config = train_config.model
    ckpt_params_dir = pathlib.Path(ckpt_dir) / "params"
    if not ckpt_params_dir.exists():
        ckpt_params_dir = pathlib.Path(ckpt_dir) / "model_params"
    if not ckpt_params_dir.exists():
        ckpt_params_dir = pathlib.Path(ckpt_dir)

    raw_params = _model.restore_params(str(ckpt_params_dir), restore_type=np.ndarray)
    model = model_config.create(jax.random.PRNGKey(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(raw_params)
    return nnx.merge(graphdef, state)


def load_adv_embed(adv_embed_dir: str):
    import orbax.checkpoint as ocp
    adv_embed = nnx.Embed(num_embeddings=2, features=HIDDEN_DIM, rngs=nnx.Rngs(params=jax.random.PRNGKey(42)))
    ckptr = ocp.StandardCheckpointer()
    raw = ckptr.restore(str(pathlib.Path(adv_embed_dir).resolve()))
    graphdef, state = nnx.split(adv_embed)
    state.replace_by_pure_dict(raw)
    return nnx.merge(graphdef, state)


# ── tokenizer helper ──────────────────────────────────────────────────────────
def make_tokenize_fn(train_config):
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len)

    def tokenize(prompt: str):
        tokens, mask = tokenizer.tokenize(prompt)
        return (np.array(tokens)[None], np.array(mask, dtype=bool)[None])

    return tokenize


# ── CFG inference (RECAP) ─────────────────────────────────────────────────────
def sample_actions_cfg(model, adv_embed, observation, rng, *, num_steps=10, beta=CFG_BETA):
    """
    CFG inference: run conditional (adv=1) and unconditional (adv=0) prefix passes,
    build separate KV caches, then combine velocities during denoising.
    """
    # preprocess observation once
    obs = _model.preprocess_observation(None, observation, train=False)
    B = obs.state.shape[0]

    def _build_kv_cache(adv_label_val: int):
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        adv_tok = adv_embed(jnp.full((B,), adv_label_val, dtype=jnp.int32))[:, None, :]
        prefix_tokens_aug  = jnp.concatenate([adv_tok, prefix_tokens], axis=1)
        prefix_mask_aug    = jnp.concatenate([jnp.ones((B, 1), dtype=bool), prefix_mask], axis=1)
        prefix_ar_mask_aug = jnp.concatenate([jnp.array([False]), prefix_ar_mask])
        prefix_attn_mask = make_attn_mask(prefix_mask_aug, prefix_ar_mask_aug)
        positions = jnp.cumsum(prefix_mask_aug, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens_aug, None], mask=prefix_attn_mask, positions=positions
        )
        return kv_cache, prefix_mask_aug

    kv_cond,   prefix_mask_cond   = _build_kv_cache(1)
    kv_uncond, prefix_mask_uncond = _build_kv_cache(0)

    noise = jax.random.normal(rng, (B, model.action_horizon, model.action_dim))
    dt = -1.0 / num_steps

    def step(carry):
        x_t, time_val = carry
        t_arr = jnp.broadcast_to(time_val, B)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, t_arr)

        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

        def _run_suffix(kv_cache, prefix_mask_aug):
            prefix_attn = einops.repeat(prefix_mask_aug, "b p -> b s p", s=suffix_tokens.shape[1])
            full_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            positions = (jnp.sum(prefix_mask_aug, axis=-1)[:, None]
                         + jnp.cumsum(suffix_mask, axis=-1) - 1)
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return model.action_out_proj(suffix_out[:, -model.action_horizon:])

        v_cond   = _run_suffix(kv_cond,   prefix_mask_cond)
        v_uncond = _run_suffix(kv_uncond, prefix_mask_uncond)
        v_guided = v_uncond + beta * (v_cond - v_uncond)
        return x_t + dt * v_guided, time_val + dt

    def cond_fn(carry):
        _, time_val = carry
        return time_val >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond_fn, step, (noise, 1.0))
    return x_0


# ── plain SFT inference ───────────────────────────────────────────────────────
def sample_actions_sft(model, observation, rng, *, num_steps=10):
    return model.sample_actions(rng, observation, num_steps=num_steps)


# ── eval one task ─────────────────────────────────────────────────────────────
def eval_task(task_id, task_name, task_desc, task_bddl, init_states,
              infer_fn, tokenize, norm_stats, num_episodes, seed, label):
    """Run num_episodes on one task, return (successes, total)."""
    from libero.libero.envs import OffScreenRenderEnv

    state_mean = norm_stats["state"].mean.astype(np.float32)
    state_std  = norm_stats["state"].std.astype(np.float32)
    action_mean = norm_stats["actions"].mean.astype(np.float32)
    action_std  = norm_stats["actions"].std.astype(np.float32)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl,
        camera_heights=256, camera_widths=256,
    )
    env.seed(seed)

    successes = 0
    for ep_idx in range(num_episodes):
        env.reset()
        init_state = init_states[ep_idx % len(init_states)]
        env.set_init_state(init_state)

        # warm-up steps
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        tokens, token_mask = tokenize(task_desc)
        actions_chunk = None
        chunk_step = REPLAN_STEPS  # trigger re-plan on first real step

        done = False
        for step_i in range(MAX_STEPS):
            if chunk_step >= REPLAN_STEPS or actions_chunk is None:
                img, wrist, state = _prep_obs(obs, RESIZE)

                # z-score normalize state (8d), then pad to 32d
                state_norm = (state - state_mean) / (state_std + 1e-6)
                state_pad = np.concatenate(
                    [state_norm, np.zeros(ACTION_DIM - len(state_norm))]
                ) if len(state_norm) < ACTION_DIM else state_norm

                observation = _model.Observation(
                    images={
                        "base_0_rgb":        jnp.array(img)[None].astype(jnp.float32) / 127.5 - 1.0,
                        "left_wrist_0_rgb":  jnp.array(wrist)[None].astype(jnp.float32) / 127.5 - 1.0,
                        "right_wrist_0_rgb": jnp.zeros((1, RESIZE, RESIZE, 3), dtype=jnp.float32),
                    },
                    image_masks={
                        "base_0_rgb":        jnp.ones(1, dtype=bool),
                        "left_wrist_0_rgb":  jnp.ones(1, dtype=bool),
                        "right_wrist_0_rgb": jnp.zeros(1, dtype=bool),
                    },
                    state=jnp.array(state_pad)[None].astype(jnp.float32),
                    tokenized_prompt=jnp.array(tokens),
                    tokenized_prompt_mask=jnp.array(token_mask),
                )

                rng = jax.random.PRNGKey(seed * 10000 + ep_idx * 1000 + step_i)
                actions_raw = np.array(infer_fn(observation, rng))  # (1, 50, 32)
                # unnormalize first 7 action dims
                actions_7d = actions_raw[0, :, :7]
                actions_chunk = actions_7d * (action_std + 1e-6) + action_mean
                chunk_step = 0

            action = actions_chunk[chunk_step]
            obs, reward, done, info = env.step(action.tolist())
            chunk_step += 1

            if done:
                break

        success = bool(info.get("success", reward > 0.5))
        successes += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"  [{label}] task{task_id} ep{ep_idx+1:02d}/{num_episodes} {status} "
              f"(total {successes}/{ep_idx+1})", flush=True)

    env.close()
    return successes, num_episodes


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recap-ckpt", default="checkpoints/pi0_libero/recap_policy/10000")
    parser.add_argument("--sft-ckpt",   default="checkpoints/pi0_libero/recap_sft_baseline/29999")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--cfg-beta", type=float, default=CFG_BETA)
    parser.add_argument("--eval-sft",  action="store_true", default=True)
    parser.add_argument("--eval-recap", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}", flush=True)

    # disable augmax compile for faster eval startup
    jax.config.update("jax_compilation_cache_dir",
                      str(pathlib.Path("~/.cache/jax").expanduser()))

    # ── load LIBERO task suite ────────────────────────────────────────────────
    from libero.libero import benchmark as _bench
    suite = _bench.get_benchmark_dict()[SUITE_NAME]()
    num_tasks = suite.n_tasks
    print(f"Tasks: {num_tasks}", flush=True)

    train_config = _config.get_config("pi0_libero")
    tokenize = make_tokenize_fn(train_config)

    # ── load norm_stats from SFT checkpoint ──────────────────────────────────
    from openpi.training import checkpoints as _checkpoints
    norm_stats = _checkpoints.load_norm_stats(
        pathlib.Path(args.sft_ckpt) / "assets/physical-intelligence", "libero"
    )
    print(f"Norm stats loaded: state mean {norm_stats['state'].mean[:3]}", flush=True)

    # ── preload task metadata (bddl paths + init_states) ─────────────────────
    from libero.libero import get_libero_path
    import os as _os

    task_infos = []
    for task_id in range(num_tasks):
        task = suite.get_task(task_id)
        bddl_path = str(pathlib.Path(get_libero_path("bddl_files"))
                        / task.problem_folder / task.bddl_file)
        init_states_path = _os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            task.init_states_file,
        )
        init_states = torch.load(init_states_path, weights_only=False)
        task_infos.append({
            "task_id": task_id,
            "task_name": task.name,
            "task_desc": task.language,
            "task_bddl": bddl_path,
            "init_states": init_states,
        })

    results = {}

    # ── evaluate SFT baseline ─────────────────────────────────────────────────
    if args.eval_sft:
        print("\n=== Evaluating SFT baseline ===", flush=True)
        sft_model = load_model(args.sft_ckpt, train_config)
        sft_model.eval()

        sft_infer_fn = nnx_utils.module_jit(sft_model.sample_actions)

        sft_results = []
        for info in task_infos:
            task_id = info["task_id"]
            print(f"\n[SFT] Task {task_id}: {info['task_name']}", flush=True)
            t0 = time.time()
            succ, total = eval_task(
                task_id, info["task_name"], info["task_desc"],
                info["task_bddl"], info["init_states"],
                lambda obs, rng: sft_infer_fn(rng, obs),
                tokenize, norm_stats, args.episodes_per_task, args.seed, "SFT",
            )
            elapsed = time.time() - t0
            sft_results.append(succ / total)
            print(f"  Task {task_id} SFT: {succ}/{total} = {succ/total*100:.1f}% ({elapsed:.0f}s)", flush=True)

        sft_overall = sum(sft_results) / len(sft_results)
        results["SFT"] = sft_results
        print(f"\nSFT overall: {sft_overall*100:.1f}%", flush=True)
        del sft_model

    # ── evaluate RECAP ────────────────────────────────────────────────────────
    if args.eval_recap:
        print("\n=== Evaluating RECAP policy ===", flush=True)
        recap_model = load_model(args.recap_ckpt, train_config)
        recap_model.eval()

        adv_embed_dir = str(pathlib.Path(args.recap_ckpt) / "adv_embed")
        adv_embed = load_adv_embed(adv_embed_dir)

        recap_infer_fn = nnx.jit(
            lambda obs, rng: sample_actions_cfg(recap_model, adv_embed, obs, rng, beta=args.cfg_beta)
        )

        recap_results = []
        for info in task_infos:
            task_id = info["task_id"]
            print(f"\n[RECAP] Task {task_id}: {info['task_name']}", flush=True)
            t0 = time.time()
            succ, total = eval_task(
                task_id, info["task_name"], info["task_desc"],
                info["task_bddl"], info["init_states"],
                lambda obs, rng: recap_infer_fn(obs, rng),
                tokenize, norm_stats, args.episodes_per_task, args.seed, "RECAP",
            )
            elapsed = time.time() - t0
            recap_results.append(succ / total)
            print(f"  Task {task_id} RECAP: {succ}/{total} = {succ/total*100:.1f}% ({elapsed:.0f}s)", flush=True)

        recap_overall = sum(recap_results) / len(recap_results)
        results["RECAP"] = recap_results
        print(f"\nRECAP overall: {recap_overall*100:.1f}%", flush=True)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"{'Task':<5} {'Name':<35} {'SFT':>6} {'RECAP':>6}", flush=True)
    print("-" * 60, flush=True)
    for info in task_infos:
        task_id = info["task_id"]
        name = info["task_name"][:33]
        sft_r   = f"{results['SFT'][task_id]*100:.0f}%"   if "SFT"   in results else "  -"
        recap_r = f"{results['RECAP'][task_id]*100:.0f}%" if "RECAP" in results else "  -"
        print(f"{task_id:<5} {name:<35} {sft_r:>6} {recap_r:>6}", flush=True)
    print("-" * 60, flush=True)
    sft_avg   = f"{sum(results['SFT'])/len(results['SFT'])*100:.1f}%"   if "SFT"   in results else "  -"
    recap_avg = f"{sum(results['RECAP'])/len(results['RECAP'])*100:.1f}%" if "RECAP" in results else "  -"
    print(f"{'AVG':<5} {'':<35} {sft_avg:>6} {recap_avg:>6}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
