"""
RECAP v8: Action-Space CFG with completely frozen backbone.

Problem with v1-v7: inserting adv_token into prefix/suffix corrupts backbone
outputs because the backbone was never trained with extra tokens in the sequence.
Joint training (v7) also fails because SFT anchor and RECAP losses conflict,
causing all actions to shift to z-up regardless of label.

v8 solution: don't touch the backbone at all.
- Backbone frozen (SFT weights, never updated)
- Runs ONE forward pass to get v_sft (the SFT velocity prediction)
- ActionDeltaNet: small MLP that predicts a residual delta in action space
  Input: [obs_state(8), adv_embed(64), t(1)] -> delta[action_horizon * action_dim]
- Training loss: ||stop_gradient(v_sft) + delta(label) - u_t||^2
  => delta learns to correct v_sft toward the target velocity for each label
- CFG inference:
  v_guided = v_sft + beta * (delta(label=1) - delta(label=0))
  backbone runs once (not twice), delta_net runs twice

Architecture:
  adv_embed: 2 x ADV_DIM (64)
  delta_net: Linear(8+64+1, 512) -> SiLU -> Linear(512,512) -> SiLU -> Linear(512, H*7)
  H=50, 7=action_dim

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  nohup .venv/bin/python -u scripts/recap/train_recap_v8.py \\
    --sft-ckpt checkpoints/pi0_libero/recap_sft_baseline/29999 \\
    --labeled-data data/rollouts/labeled_episodes_v5.pkl \\
    --exp-name recap_policy_v8 \\
    --num-steps 30000 --batch-size 64 --lr 3e-4 \\
    > logs/recap_train_v8.log 2>&1 &
"""

import argparse
import os
import pathlib
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import sharding

ADV_DIM        = 64     # embedding dim (small! not 2048)
ACTION_HORIZON = 50
ACTION_DIM     = 7      # raw action dim (not 32; flow matching in 7-dim space)
SAVE_INTERVAL  = 1000
LOG_INTERVAL   = 50


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ── ActionDeltaNet ─────────────────────────────────────────────────────────────
class ActionDeltaNet(nnx.Module):
    """Small MLP: (state, adv_embed, t) -> action_delta [B, H, 7]"""

    def __init__(self, state_dim: int, adv_dim: int, hidden: int, action_horizon: int,
                 action_dim: int, rngs: nnx.Rngs):
        in_dim = state_dim + adv_dim + 1  # +1 for time
        self.fc1 = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.fc3 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, action_horizon * action_dim, rngs=rngs)
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def __call__(self, state, adv_vec, t):
        # state: [B, state_dim], adv_vec: [B, adv_dim], t: [B]
        t_feat = t[:, None]  # [B, 1]
        x = jnp.concatenate([state, adv_vec, t_feat], axis=-1)
        x = jax.nn.silu(self.fc1(x))
        x = jax.nn.silu(self.fc2(x))
        x = jax.nn.silu(self.fc3(x))
        x = self.out(x)
        return x.reshape(-1, self.action_horizon, self.action_dim)


# ── Dataset ───────────────────────────────────────────────────────────────────
class RolloutDataLoader:
    def __init__(self, labeled_pkl: pathlib.Path, batch_size: int, seed: int = 0):
        with open(labeled_pkl, "rb") as f:
            episodes = pickle.load(f)

        self.flat_samples = []
        for ep in episodes:
            T = ep["length"]
            actions = ep["actions"]
            for t in range(T):
                end = min(t + ACTION_HORIZON, T)
                chunk = actions[t:end]
                if len(chunk) < ACTION_HORIZON:
                    pad = np.tile(chunk[-1:], (ACTION_HORIZON - len(chunk), 1))
                    chunk = np.concatenate([chunk, pad], axis=0)
                self.flat_samples.append({
                    "image":       ep["images"][t].astype(np.float32) / 127.5 - 1.0,
                    "wrist_image": ep["wrist_imgs"][t].astype(np.float32) / 127.5 - 1.0,
                    "state":       ep["states"][t].astype(np.float32),
                    "actions":     chunk.astype(np.float32),
                    "advantage_label": np.int32(ep["advantage_labels"][t]),
                    "prompt":      ep["prompt"],
                })

        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        log(f"Dataset: {len(self.flat_samples)} samples, "
            f"success={sum(1 for s in self.flat_samples if s['advantage_label']==1)}, "
            f"fail={sum(1 for s in self.flat_samples if s['advantage_label']==0)}")

    def sample_batch(self, tokenize):
        idx = self.rng.choice(len(self.flat_samples), self.batch_size, replace=True)
        batch = [self.flat_samples[i] for i in idx]
        images      = np.stack([b["image"] for b in batch])
        wrist_imgs  = np.stack([b["wrist_image"] for b in batch])
        states      = np.stack([b["state"] for b in batch])
        actions     = np.stack([b["actions"] for b in batch])
        adv_labels  = np.stack([b["advantage_label"] for b in batch])
        prompts     = [b["prompt"] for b in batch]
        tokens, tok_mask = tokenize(prompts)
        return images, wrist_imgs, states, actions, adv_labels, tokens, tok_mask


# ── normalization helpers ──────────────────────────────────────────────────────
def get_norm_stats(train_config):
    import json
    assets_dir = pathlib.Path(train_config.assets_dirs)
    norm_file  = assets_dir / "physical-intelligence" / "libero" / "norm_stats.json"
    with open(norm_file) as f:
        d = json.load(f)
    ns   = d["norm_stats"]["actions"]
    mean = np.array(ns["mean"])
    std  = np.array(ns["std"])
    return mean.astype(np.float32), std.astype(np.float32)


# ── loss function ──────────────────────────────────────────────────────────────
def compute_loss(model, adv_embed, delta_net, rng,
                 images, wrist_images, states_raw, actions_raw, adv_labels,
                 tokens, token_mask, model_action_dim: int, train: bool):
    """
    states_raw: [B, 8] (7-dim EEF + 1 gripper, from env)
    actions_raw: [B, H, 7] (7-dim delta EEF actions)
    model expects state/actions padded to model_action_dim (32)
    """
    from openpi.models import model as _model_mod
    from openpi.models.pi0 import make_attn_mask
    import einops

    B = images.shape[0]
    rng_pre, rng_n, rng_t = jax.random.split(rng, 3)

    # Pad state and actions to model_action_dim
    pad_s = model_action_dim - states_raw.shape[-1]
    states = jnp.concatenate([states_raw, jnp.zeros((B, pad_s))], axis=-1) if pad_s > 0 else states_raw
    pad_a = model_action_dim - actions_raw.shape[-1]
    actions = jnp.concatenate([actions_raw, jnp.zeros((B, actions_raw.shape[1], pad_a))], axis=-1) if pad_a > 0 else actions_raw

    # Build observation
    obs = _model_mod.Observation(
        images={
            "base_0_rgb":        images,
            "left_wrist_0_rgb":  wrist_images,
            "right_wrist_0_rgb": jnp.zeros_like(images),
        },
        image_masks={
            "base_0_rgb":        jnp.ones(B, dtype=bool),
            "left_wrist_0_rgb":  jnp.ones(B, dtype=bool),
            "right_wrist_0_rgb": jnp.zeros(B, dtype=bool),
        },
        state=states,
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
    )
    obs = _model_mod.preprocess_observation(rng_pre, obs, train=False)

    # Flow matching noise/time (match v7: beta distribution)
    noise = jax.random.normal(rng_n, actions.shape)
    time  = jax.random.beta(rng_t, 1.5, 1, (B,)) * 0.999 + 0.001
    te    = time[:, None, None]
    x_t   = te * noise + (1 - te) * actions
    u_t   = noise - actions  # ground-truth velocity (model_action_dim-dim)

    # SFT backbone forward (frozen, no gradient through backbone)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)

    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_mask = jnp.concatenate([prefix_attn_for_suffix, suffix_attn_mask], axis=-1)

    positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
    positions_suffix = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions_prefix
    )
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_mask,
        positions=positions_suffix,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    # v_sft: [B, H, model_action_dim], stop_gradient so backbone never updates
    v_sft = jax.lax.stop_gradient(
        model.action_out_proj(suffix_out[:, -model.action_horizon:])
    )

    # ActionDeltaNet: operates on 7-dim actions only
    adv_vecs = adv_embed(adv_labels)          # [B, ADV_DIM]
    delta_7  = delta_net(states_raw, adv_vecs, time)  # [B, H, 7]
    # Pad delta to model_action_dim for comparison
    delta = jnp.concatenate([delta_7, jnp.zeros((B, delta_7.shape[1], pad_a))], axis=-1) if pad_a > 0 else delta_7

    v_pred = v_sft + delta
    loss = jnp.mean(jnp.square(v_pred - u_t))
    return loss


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",      required=True)
    parser.add_argument("--labeled-data",  required=True)
    parser.add_argument("--exp-name",      default="recap_policy_v8")
    parser.add_argument("--num-steps",     type=int, default=30000)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--ckpt-base-dir", default="checkpoints")
    parser.add_argument("--fsdp-devices",  type=int, default=1)
    parser.add_argument("--state-dim",     type=int, default=8)
    parser.add_argument("--hidden-dim",    type=int, default=512)
    args = parser.parse_args()

    log(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    log(f"v8: FROZEN backbone + ActionDeltaNet in action space. lr={args.lr}")

    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    train_config = _config.get_config("pi0_libero")
    model_config = train_config.model

    import orbax.checkpoint as ocp

    # Load frozen SFT backbone
    log("Loading SFT checkpoint (frozen)...")
    ckpt_params_dir = pathlib.Path(args.sft_ckpt) / "params"
    if not ckpt_params_dir.exists():
        ckpt_params_dir = pathlib.Path(args.sft_ckpt)
    raw_params = _model.restore_params(str(ckpt_params_dir), restore_type=np.ndarray)
    model = model_config.create(jax.random.PRNGKey(0))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(raw_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    log("SFT backbone loaded (frozen).")

    # Init trainable modules: adv_embed + delta_net
    adv_embed = nnx.Embed(num_embeddings=2, features=ADV_DIM, rngs=nnx.Rngs(params=jax.random.PRNGKey(1)))
    delta_net = ActionDeltaNet(
        state_dim=args.state_dim,
        adv_dim=ADV_DIM,
        hidden=args.hidden_dim,
        action_horizon=ACTION_HORIZON,
        action_dim=ACTION_DIM,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(2)),
    )
    log(f"ActionDeltaNet: state_dim={args.state_dim}, adv_dim={ADV_DIM}, hidden={args.hidden_dim}")

    model_action_dim = model_config.action_dim  # 32

    # Optimizer: only adv_embed + delta_net
    tx = optax.adamw(args.lr, weight_decay=1e-4)
    adv_params   = nnx.state(adv_embed)
    delta_params = nnx.state(delta_net)

    # Combine into single param tree for optimizer
    combined_params = {"adv": adv_params, "delta": delta_params}
    opt_state = tx.init(combined_params)

    model_graphdef = nnx.graphdef(model)
    adv_graphdef   = nnx.graphdef(adv_embed)
    delta_graphdef = nnx.graphdef(delta_net)

    data_loader = RolloutDataLoader(pathlib.Path(args.labeled_data), args.batch_size)

    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len)

    def tokenize(prompts):
        results = [tokenizer.tokenize(p) for p in prompts]
        return (np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]).astype(bool))

    # Model params: frozen, put on GPU but no gradient
    if args.fsdp_devices > 1:
        mesh = sharding.make_mesh(args.fsdp_devices)
        model_params_shape    = jax.eval_shape(lambda: nnx.state(model))
        model_params_sharding = sharding.fsdp_sharding(model_params_shape, mesh, log=True)
        model_params = jax.device_put(nnx.state(model), model_params_sharding)
        replicated   = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_shd     = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    else:
        model_params = nnx.state(model)
        mesh = None
        replicated = None
        data_shd   = None

    def train_step(model_params, combined_params, opt_state,
                   images, wrist_images, states, actions, adv_labels,
                   tokens, token_mask, rng):
        m       = nnx.merge(model_graphdef, model_params)
        adv     = nnx.merge(adv_graphdef,   combined_params["adv"])
        delta   = nnx.merge(delta_graphdef, combined_params["delta"])

        def loss_fn(adv_e, delta_n):
            return compute_loss(
                m, adv_e, delta_n, rng,
                images, wrist_images, states, actions, adv_labels,
                tokens, token_mask, model_action_dim, train=True,
            )

        diff_adv   = nnx.DiffState(0, nnx.Param)
        diff_delta = nnx.DiffState(1, nnx.Param)
        loss, (adv_grads, delta_grads) = nnx.value_and_grad(
            loss_fn, argnums=(diff_adv, diff_delta)
        )(adv, delta)

        adv_params_new   = nnx.state(adv)
        delta_params_new = nnx.state(delta)
        new_combined_grads  = {"adv": adv_grads, "delta": delta_grads}
        updates, new_opt_state = tx.update(new_combined_grads, opt_state, combined_params)
        new_combined_params = optax.apply_updates(combined_params, updates)
        return new_combined_params, new_opt_state, loss

    if mesh is not None:
        train_step_jit = jax.jit(
            train_step,
            in_shardings=(
                model_params_sharding, replicated, replicated,
                data_shd, data_shd, data_shd, data_shd, data_shd,
                data_shd, data_shd, replicated,
            ),
            out_shardings=(replicated, replicated, replicated),
        )
    else:
        train_step_jit = jax.jit(train_step)

    ckpt_dir = pathlib.Path(args.ckpt_base_dir).resolve() / "pi0_libero" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(8)
    t0  = time.time()
    total_steps = args.num_steps
    log(f"Starting RECAP v8 training, steps=1..{total_steps} ...")

    for step in range(1, total_steps + 1):
        rng, step_rng = jax.random.split(rng)

        images, wrist_imgs, sts, actions, adv_labels, tokens, token_mask = \
            data_loader.sample_batch(tokenize)

        combined_params, opt_state, loss = train_step_jit(
            model_params, combined_params, opt_state,
            jnp.array(images), jnp.array(wrist_imgs), jnp.array(sts),
            jnp.array(actions), jnp.array(adv_labels),
            jnp.array(tokens), jnp.array(token_mask), step_rng,
        )

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            rate    = LOG_INTERVAL / elapsed
            remain  = (total_steps - step) / rate
            # compute delta magnitude
            adv_w = np.array(jax.device_get(combined_params["adv"].embedding.value))
            e0_n  = np.linalg.norm(adv_w[0])
            e1_n  = np.linalg.norm(adv_w[1])
            cos   = np.dot(adv_w[0], adv_w[1]) / (e0_n * e1_n + 1e-8)
            log(f"Step {step:5d}/{total_steps} loss={float(loss):.4f} "
                f"adv_e0={e0_n:.3f} adv_e1={e1_n:.3f} cos={cos:.3f} "
                f"rate={rate:.2f}s/s eta={remain/3600:.2f}h")
            t0 = time.time()

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            import shutil
            step_dir = ckpt_dir / str(step)
            step_dir.mkdir(parents=True, exist_ok=True)
            ckptr = ocp.StandardCheckpointer()
            host_adv   = jax.device_get(combined_params["adv"])
            host_delta = jax.device_get(combined_params["delta"])
            for sub, h in [("adv_embed", host_adv), ("delta_net", host_delta)]:
                d = step_dir / sub
                if d.exists():
                    shutil.rmtree(d)
                ckptr.save(d, h)
            ckptr.wait_until_finished()
            log(f"Checkpoint saved: {step_dir}")

    log("RECAP v8 training complete.")


if __name__ == "__main__":
    main()
