"""
ARFM: Adaptive Reward-weighted Flow Matching fine-tuning for pi0.6.

Method (arxiv 2509.04063):
  - No architectural changes. Backbone fully trainable (full fine-tuning from SFT init).
  - Per-sample loss weight: w_i(α) = exp(α·R*_i) / Σ_j exp(α·R*_j)
    where R*_i is the normalized advantage (episode-level: success=+1, fail=0 → normalized).
  - α is solved per-batch via bisection: balances gradient variance vs RL signal.
  - Simple version first: α fixed (α=1.0), then adaptive bisection.
  - Loss: Σ_i w_i(α) · ||v_t_i - u_t_i||²

Key insight: reward signal is injected only through LOSS WEIGHTS, not through
any architectural conditioning. Backbone inference is unchanged — no token injection,
no CFG at test time (unless we want to run two forward passes with different weights).

Inference: standard SFT forward pass (no change). The fine-tuned policy has
learned to imitate success behavior more strongly than failure behavior.
For CFG-style: could run with α→+∞ (success-only) vs α→-∞ (failure-only) and subtract,
but basic ARFM just evaluates the fine-tuned policy directly.

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  nohup .venv/bin/python -u scripts/recap/train_arfm.py \\
    --sft-ckpt checkpoints/pi0_libero/recap_sft_baseline/29999 \\
    --labeled-data data/rollouts/labeled_episodes_v5.pkl \\
    --exp-name arfm_v1 \\
    --num-steps 20000 --batch-size 64 --lr 2e-5 --alpha 2.0 \\
    > logs/train_arfm.log 2>&1 &
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
from openpi.models.pi0 import make_attn_mask
from openpi.training import config as _config
from openpi.training import sharding

import einops

SAVE_INTERVAL = 1000
LOG_INTERVAL  = 50
ACTION_HORIZON = 50


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class RolloutDataLoader:
    def __init__(self, labeled_pkl: pathlib.Path, batch_size: int, seed: int = 0):
        with open(labeled_pkl, "rb") as f:
            episodes = pickle.load(f)

        self.flat_samples = []
        for ep in episodes:
            T = ep["length"]
            actions = ep["actions"]
            label = 1.0 if ep["success"] else 0.0
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
                    "reward":      np.float32(label),
                    "prompt":      ep["prompt"],
                })

        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        n_succ = sum(1 for s in self.flat_samples if s["reward"] > 0.5)
        n_fail = len(self.flat_samples) - n_succ
        log(f"Dataset: {len(self.flat_samples)} samples, success={n_succ}, fail={n_fail}")

    def sample_batch(self, tokenize):
        idx = self.rng.choice(len(self.flat_samples), self.batch_size, replace=True)
        batch = [self.flat_samples[i] for i in idx]
        images      = np.stack([b["image"] for b in batch])
        wrist_imgs  = np.stack([b["wrist_image"] for b in batch])
        states      = np.stack([b["state"] for b in batch])
        actions     = np.stack([b["actions"] for b in batch])
        rewards     = np.stack([b["reward"] for b in batch])
        prompts     = [b["prompt"] for b in batch]
        tokens, tok_mask = tokenize(prompts)
        return images, wrist_imgs, states, actions, rewards, tokens, tok_mask


# ── ARFM weight computation ────────────────────────────────────────────────────
def arfm_weights(rewards: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Softmax-style weights: w_i = exp(α·R_i) / Σ exp(α·R_j)"""
    # Normalize rewards to zero mean unit std first
    r_mean = jnp.mean(rewards)
    r_std  = jnp.std(rewards) + 1e-8
    r_norm = (rewards - r_mean) / r_std
    log_w  = alpha * r_norm
    log_w  = log_w - jax.scipy.special.logsumexp(log_w)  # log-normalize
    return jnp.exp(log_w) * len(rewards)  # scale so mean weight ≈ 1


# ── Loss function ─────────────────────────────────────────────────────────────
def compute_loss(model, rng, images, wrist_images, states_raw, actions_raw,
                 rewards, tokens, token_mask, model_action_dim: int,
                 alpha: float, train: bool):
    B = images.shape[0]
    rng_pre, rng_n, rng_t = jax.random.split(rng, 3)

    # Pad state and actions to model_action_dim
    pad_s = model_action_dim - states_raw.shape[-1]
    states = (jnp.concatenate([states_raw, jnp.zeros((B, pad_s))], axis=-1)
              if pad_s > 0 else states_raw)
    pad_a = model_action_dim - actions_raw.shape[-1]
    actions = (jnp.concatenate([actions_raw, jnp.zeros((B, actions_raw.shape[1], pad_a))], axis=-1)
               if pad_a > 0 else actions_raw)

    obs = _model.Observation(
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
    obs = _model.preprocess_observation(rng_pre, obs, train=train)

    # Flow matching
    noise = jax.random.normal(rng_n, actions.shape)
    t_val = jax.random.beta(rng_t, 1.5, 1, (B,)) * 0.999 + 0.001
    t_exp = t_val[:, None, None]
    x_t   = t_exp * noise + (1 - t_exp) * actions
    u_t   = noise - actions

    # Forward pass
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, t_val)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask  = make_attn_mask(input_mask, ar_mask)
    positions  = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])

    # Per-sample MSE loss: [B]
    per_sample_loss = jnp.mean(jnp.square(v_t - u_t), axis=(1, 2))

    # ARFM weights
    weights = arfm_weights(rewards, alpha)  # [B], mean ≈ 1

    weighted_loss = jnp.mean(weights * per_sample_loss)
    unweighted_loss = jnp.mean(per_sample_loss)
    return weighted_loss, unweighted_loss


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",      required=True)
    parser.add_argument("--labeled-data",  required=True)
    parser.add_argument("--exp-name",      default="arfm_v1")
    parser.add_argument("--num-steps",     type=int,   default=20000)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--alpha",         type=float, default=2.0,
                        help="ARFM temperature: higher = more advantage-weighted")
    parser.add_argument("--ckpt-base-dir", default="checkpoints")
    parser.add_argument("--fsdp-devices",  type=int,   default=4)
    parser.add_argument("--save-interval", type=int,   default=200,
                        help="Save checkpoint every N steps (default 200 for Phase 1)")
    parser.add_argument("--seed",          type=int,   default=0,
                        help="Training seed for data sampling and flow-matching noise")
    args = parser.parse_args()

    log(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    log(f"ARFM: lr={args.lr}, alpha={args.alpha}, steps={args.num_steps}, "
        f"batch={args.batch_size}, seed={args.seed}")

    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding       = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_config = _config.get_config("pi0_libero")
    model_config = train_config.model

    import orbax.checkpoint as ocp

    # Load SFT checkpoint
    log("Loading SFT checkpoint...")
    ckpt_params_dir = pathlib.Path(args.sft_ckpt) / "params"
    if not ckpt_params_dir.exists():
        ckpt_params_dir = pathlib.Path(args.sft_ckpt)
    raw_params = _model.restore_params(str(ckpt_params_dir), restore_type=np.ndarray)
    model = model_config.create(jax.random.PRNGKey(args.seed))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(raw_params)
    model = nnx.merge(graphdef, state)
    log("SFT checkpoint loaded.")

    model_action_dim = model.action_dim

    # Optimizer (full fine-tuning, cosine decay)
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=min(500, args.num_steps // 4), decay_steps=args.num_steps,
    )
    tx = optax.adamw(lr_sched, weight_decay=1e-4)

    model_params = nnx.state(model)
    model_params_shape    = jax.eval_shape(lambda: model_params)
    model_params_sharding = sharding.fsdp_sharding(model_params_shape, mesh, log=True)
    model_params = jax.device_put(model_params, model_params_sharding)

    opt_state = tx.init(model_params)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    log("Optimizer initialized.")

    data_loader = RolloutDataLoader(
        pathlib.Path(args.labeled_data), args.batch_size, seed=args.seed
    )

    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len)

    def tokenize(prompts):
        results = [tokenizer.tokenize(p) for p in prompts]
        return (np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]).astype(bool))

    model_graphdef = nnx.graphdef(model)
    alpha = args.alpha

    def train_step(model_params, opt_state,
                   images, wrist_images, states, actions, rewards,
                   tokens, token_mask, rng):
        m = nnx.merge(model_graphdef, model_params)
        m.train()

        def loss_fn(model):
            wl, uwl = compute_loss(
                model, rng, images, wrist_images, states, actions, rewards,
                tokens, token_mask, model_action_dim, alpha, train=True,
            )
            return wl, uwl

        diff_state = nnx.DiffState(0, nnx.Param)
        (weighted_loss, unweighted_loss), grads = nnx.value_and_grad(
            loss_fn, argnums=diff_state, has_aux=True
        )(m)

        new_params = nnx.state(m)
        updates, new_opt_state = tx.update(grads, opt_state, model_params)
        new_model_params = optax.apply_updates(model_params, updates)
        return new_model_params, new_opt_state, weighted_loss, unweighted_loss

    train_step_jit = jax.jit(
        train_step,
        in_shardings=(
            model_params_sharding, replicated_sharding,
            data_sharding, data_sharding, data_sharding, data_sharding, data_sharding,
            data_sharding, data_sharding,
            replicated_sharding,
        ),
        out_shardings=(
            model_params_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding,
        ),
        donate_argnums=(0, 1),  # donate model_params + opt_state for in-place update
    )

    ckpt_dir = pathlib.Path(args.ckpt_base_dir).resolve() / "pi0_libero" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the historical seed-0 trajectory while making retrains reproducible.
    rng = jax.random.PRNGKey(9 + args.seed)
    t0  = time.time()
    total_steps = args.num_steps
    save_interval = args.save_interval
    log(f"Starting ARFM training, steps=1..{total_steps} ...")

    for step in range(1, total_steps + 1):
        rng, step_rng = jax.random.split(rng)

        images, wrist_imgs, states, actions, rewards, tokens, token_mask = \
            data_loader.sample_batch(tokenize)

        def _dev(arr, shrd):
            return jax.device_put(jnp.array(arr), shrd)

        with sharding.set_mesh(mesh):
            model_params, opt_state, w_loss, uw_loss = train_step_jit(
                model_params, opt_state,
                _dev(images,     data_sharding),
                _dev(wrist_imgs, data_sharding),
                _dev(states,     data_sharding),
                _dev(actions,    data_sharding),
                _dev(rewards,    data_sharding),
                _dev(tokens,     data_sharding),
                _dev(token_mask, data_sharding),
                step_rng,
            )

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            rate    = LOG_INTERVAL / elapsed
            remain  = (total_steps - step) / rate
            log(f"Step {step:5d}/{total_steps} "
                f"w_loss={float(w_loss):.4f} uw_loss={float(uw_loss):.4f} "
                f"rate={rate:.2f}s/s eta={remain/3600:.2f}h")
            t0 = time.time()

        if step % save_interval == 0 or step == total_steps:
            import shutil
            step_dir = ckpt_dir / str(step)
            step_dir.mkdir(parents=True, exist_ok=True)
            ckptr = ocp.StandardCheckpointer()
            host_params = jax.device_get(model_params)
            params_dir  = step_dir / "model_params"
            if params_dir.exists():
                shutil.rmtree(params_dir)
            ckptr.save(params_dir, host_params)
            ckptr.wait_until_finished()
            log(f"Checkpoint saved: {step_dir}")

    log("ARFM training complete.")


if __name__ == "__main__":
    main()
