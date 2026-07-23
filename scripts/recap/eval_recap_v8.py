"""
Evaluate RECAP v8 policy on LIBERO spatial.

v8 inference:
  v_sft    = frozen SFT backbone (no adv_token)
  v_guided = v_sft + beta * (delta(label=1) - delta(label=0))
  delta_net: small MLP, input=[state(8), adv_embed(64), t(1)]

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  MUJOCO_GL=osmesa XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    .venv/bin/python scripts/recap/eval_recap_v8.py \
      --recap-ckpt checkpoints/pi0_libero/recap_policy_v8/30000 \
      --sft-ckpt   checkpoints/pi0_libero/recap_sft_baseline/29999 \
      --episodes-per-task 20
"""

import sys as _sys
import types as _types
if 'numba' not in _sys.modules:
    _numba_mock = _types.ModuleType('numba')
    _numba_mock.jit = lambda *a, **kw: (lambda f: f)
    _sys.modules['numba'] = _numba_mock

import argparse
import math
import os
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils
from openpi.recap import evaluation_protocol as protocol
import einops

ADV_DIM        = 64
ACTION_HORIZON = 50
ACTION_DIM     = 32   # model internal dim
CFG_BETA       = 2.0
REPLAN_STEPS   = protocol.REPLAN_STEPS
MAX_STEPS      = protocol.MAX_STEPS
SUITE_NAME     = protocol.SUITE_NAME
RESIZE         = protocol.RESIZE


# ── ActionDeltaNet (must match train_recap_v8.py) ─────────────────────────────
class ActionDeltaNet(nnx.Module):
    def __init__(self, state_dim, adv_dim, hidden, action_horizon, action_dim, rngs):
        in_dim = state_dim + adv_dim + 1
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.fc3 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, action_horizon * action_dim, rngs=rngs)
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def __call__(self, state, adv_vec, t):
        t_feat = t[:, None]
        x = jnp.concatenate([state, adv_vec, t_feat], axis=-1)
        x = jax.nn.silu(self.fc1(x))
        x = jax.nn.silu(self.fc2(x))
        x = jax.nn.silu(self.fc3(x))
        return self.out(x).reshape(-1, self.action_horizon, self.action_dim)


# ── env helpers ───────────────────────────────────────────────────────────────
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
def load_sft_model(ckpt_dir, train_config):
    model = train_config.model.create(jax.random.PRNGKey(0))
    ckpt_path = pathlib.Path(ckpt_dir)
    params_dir = ckpt_path / "params"
    if params_dir.exists():
        raw_params = _model.restore_params(str(params_dir), restore_type=np.ndarray)
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(raw_params)
        return nnx.merge(graphdef, state)
    import orbax.checkpoint as ocp
    model_state = nnx.state(model)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(str((ckpt_path / "model_params").resolve()), target=model_state)
    nnx.update(model, restored)
    return model


def load_v8_modules(recap_ckpt_dir):
    import orbax.checkpoint as ocp
    ckpt_path = pathlib.Path(recap_ckpt_dir)

    adv_embed = nnx.Embed(num_embeddings=2, features=ADV_DIM,
                          rngs=nnx.Rngs(params=jax.random.PRNGKey(1)))
    delta_net = ActionDeltaNet(
        state_dim=8, adv_dim=ADV_DIM, hidden=512,
        action_horizon=50, action_dim=7,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(2)),
    )

    ckptr = ocp.StandardCheckpointer()
    adv_state   = nnx.state(adv_embed)
    delta_state = nnx.state(delta_net)

    restored_adv   = ckptr.restore(str((ckpt_path / "adv_embed").resolve()),   target=adv_state)
    restored_delta = ckptr.restore(str((ckpt_path / "delta_net").resolve()), target=delta_state)
    nnx.update(adv_embed, restored_adv)
    nnx.update(delta_net, restored_delta)
    return adv_embed, delta_net


# ── tokenizer ─────────────────────────────────────────────────────────────────
def make_tokenize_fn(train_config):
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len)
    def tokenize(prompt):
        tokens, mask = tokenizer.tokenize(prompt)
        return np.array(tokens)[None], np.array(mask, dtype=bool)[None]
    return tokenize


# ── v8 CFG inference ──────────────────────────────────────────────────────────
def make_v8_infer_fn(model, adv_embed, delta_net, *, num_steps=10, beta=CFG_BETA):
    graphdef,   model_state = nnx.split(model)
    adv_gdef,   adv_state   = nnx.split(adv_embed)
    delta_gdef, delta_state = nnx.split(delta_net)

    @jax.jit
    def build_kv_cache(m_state, obs):
        m = nnx.merge(graphdef, m_state)
        prefix_tokens, prefix_mask, prefix_ar_mask = m.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = m.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return kv_cache, prefix_mask

    @jax.jit
    def run_suffix(m_state, obs, x_t, time_val, kv_cache, prefix_mask):
        m = nnx.merge(graphdef, m_state)
        B = obs.state.shape[0]
        t_arr = jnp.broadcast_to(time_val, B)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = m.embed_suffix(obs, x_t, t_arr)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
        positions = (jnp.sum(prefix_mask, axis=-1)[:, None]
                     + jnp.cumsum(suffix_mask, axis=-1) - 1)
        (_, suffix_out), _ = m.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return m.action_out_proj(suffix_out[:, -m.action_horizon:])

    @jax.jit
    def run_delta(adv_st, delta_st, state_raw, adv_label_val, time_val):
        adv   = nnx.merge(adv_gdef,   adv_st)
        delta = nnx.merge(delta_gdef, delta_st)
        B = state_raw.shape[0]
        adv_vec = adv(jnp.full((B,), adv_label_val, dtype=jnp.int32))
        t_arr   = jnp.broadcast_to(time_val, B)
        return delta(state_raw, adv_vec, t_arr)  # [B, 50, 7]

    def infer(observation, rng):
        B = observation.state.shape[0]
        # state_raw: first 8 dims (before padding)
        state_raw = observation.state[:, :8]

        kv_cache, prefix_mask = build_kv_cache(model_state, observation)

        noise   = jax.random.normal(rng, (B, model.action_horizon, model.action_dim))
        dt      = -1.0 / num_steps
        x_t     = noise
        time_val = jnp.float32(1.0)

        for _ in range(num_steps):
            v_sft = run_suffix(model_state, observation, x_t, time_val, kv_cache, prefix_mask)

            # delta only uses first 7 dims; pad zeros for remaining action dims
            d_cond   = run_delta(adv_state, delta_state, state_raw, jnp.int32(1), time_val)
            d_uncond = run_delta(adv_state, delta_state, state_raw, jnp.int32(0), time_val)
            delta_7  = beta * (d_cond - d_uncond)  # [B, 50, 7]
            delta_pad = jnp.concatenate([
                delta_7,
                jnp.zeros((B, 50, model.action_dim - 7))
            ], axis=-1)

            v_guided = v_sft + delta_pad
            x_t = x_t + dt * v_guided
            time_val = time_val + dt

        return x_t

    return infer


# ── eval one task ─────────────────────────────────────────────────────────────
def eval_task(task_id, task_name, task_desc, task_bddl, init_states,
              infer_fn, tokenize, norm_stats, num_episodes, seed, label):
    from libero.libero.envs import OffScreenRenderEnv

    state_mean  = norm_stats["state"].mean.astype(np.float32)
    state_std   = norm_stats["state"].std.astype(np.float32)
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

        for _ in range(protocol.NUM_WAIT_STEPS):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        tokens, token_mask = tokenize(task_desc)
        actions_chunk = None
        chunk_step = REPLAN_STEPS

        for step_i in range(MAX_STEPS):
            if chunk_step >= REPLAN_STEPS or actions_chunk is None:
                img, wrist, state = _prep_obs(obs, RESIZE)

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

                rng = jax.random.PRNGKey(protocol.action_rng_seed(seed, ep_idx, step_i))
                actions_raw = np.array(infer_fn(observation, rng))  # [1, 50, 32]
                actions_7d  = actions_raw[0, :, :7]
                actions_chunk = actions_7d * (action_std + 1e-6) + action_mean
                _abs_mask = np.array([True, True, True, True, True, True, False], dtype=bool)
                actions_chunk[:, :7] += np.where(_abs_mask, state[:7], 0.0)
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
    parser.add_argument("--recap-ckpt", default="checkpoints/pi0_libero/recap_policy_v8/30000")
    parser.add_argument("--sft-ckpt",   default="checkpoints/pi0_libero/recap_sft_baseline/29999")
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--cfg-beta",   type=float, default=CFG_BETA)
    parser.add_argument("--eval-sft",   action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-recap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    print(
        "Protocol: "
        f"suite={SUITE_NAME} episodes/task={args.episodes_per_task} seed={args.seed} "
        f"wait={protocol.NUM_WAIT_STEPS} max_steps={MAX_STEPS} "
        f"replan={REPLAN_STEPS} flow_steps={protocol.FLOW_STEPS}",
        flush=True,
    )
    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}", flush=True)
    jax.config.update("jax_compilation_cache_dir",
                      str(pathlib.Path("~/.cache/jax").expanduser()))

    from libero.libero import benchmark as _bench
    suite = _bench.get_benchmark_dict()[SUITE_NAME]()
    print(f"Tasks: {suite.n_tasks}", flush=True)

    train_config = _config.get_config("pi0_libero")
    tokenize = make_tokenize_fn(train_config)

    from openpi.training import checkpoints as _checkpoints
    norm_stats = _checkpoints.load_norm_stats(
        pathlib.Path(args.sft_ckpt) / "assets/physical-intelligence", "libero"
    )

    from libero.libero import get_libero_path
    import os as _os

    task_infos = []
    for task_id in range(suite.n_tasks):
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

    if args.eval_sft:
        print("\n=== SFT baseline ===", flush=True)
        sft_model = load_sft_model(args.sft_ckpt, train_config)
        sft_model.eval()
        sft_infer_fn = nnx_utils.module_jit(sft_model.sample_actions)

        sft_results = []
        for info in task_infos:
            print(f"\n[SFT] Task {info['task_id']}: {info['task_name']}", flush=True)
            succ, total = eval_task(
                info["task_id"], info["task_name"], info["task_desc"],
                info["task_bddl"], info["init_states"],
                lambda obs, rng: sft_infer_fn(rng, obs),
                tokenize, norm_stats, args.episodes_per_task, args.seed, "SFT",
            )
            sft_results.append(succ / total)
            print(f"  Task {info['task_id']} SFT: {succ}/{total} = {succ/total*100:.1f}%", flush=True)

        results["SFT"] = sft_results
        print(f"\nSFT overall: {sum(sft_results)/len(sft_results)*100:.1f}%", flush=True)
        del sft_model

    if args.eval_recap:
        print("\n=== RECAP v8 ===", flush=True)
        model = load_sft_model(args.sft_ckpt, train_config)
        model.eval()
        adv_embed, delta_net = load_v8_modules(args.recap_ckpt)

        recap_infer_fn = make_v8_infer_fn(
            model, adv_embed, delta_net, num_steps=10, beta=args.cfg_beta
        )

        recap_results = []
        for info in task_infos:
            print(f"\n[RECAP-v8] Task {info['task_id']}: {info['task_name']}", flush=True)
            succ, total = eval_task(
                info["task_id"], info["task_name"], info["task_desc"],
                info["task_bddl"], info["init_states"],
                lambda obs, rng: recap_infer_fn(obs, rng),
                tokenize, norm_stats, args.episodes_per_task, args.seed, "RECAP-v8",
            )
            recap_results.append(succ / total)
            print(f"  Task {info['task_id']} RECAP-v8: {succ}/{total} = {succ/total*100:.1f}%", flush=True)

        results["RECAP-v8"] = recap_results
        print(f"\nRECAP-v8 overall: {sum(recap_results)/len(recap_results)*100:.1f}%", flush=True)

    print("\n" + "=" * 65, flush=True)
    print(f"{'Task':<5} {'Name':<35} {'SFT':>6} {'RECAP-v8':>9}", flush=True)
    print("-" * 65, flush=True)
    for info in task_infos:
        tid  = info["task_id"]
        name = info["task_name"][:33]
        sr   = f"{results['SFT'][tid]*100:.0f}%"       if "SFT"      in results else "  -"
        rr   = f"{results['RECAP-v8'][tid]*100:.0f}%"  if "RECAP-v8" in results else "  -"
        print(f"{tid:<5} {name:<35} {sr:>6} {rr:>9}", flush=True)
    print("-" * 65, flush=True)
    sa = f"{sum(results['SFT'])/len(results['SFT'])*100:.1f}%"           if "SFT"      in results else "  -"
    ra = f"{sum(results['RECAP-v8'])/len(results['RECAP-v8'])*100:.1f}%" if "RECAP-v8" in results else "  -"
    print(f"{'AVG':<5} {'':<35} {sa:>6} {ra:>9}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
