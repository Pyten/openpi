"""
Phase 1D v7: RECAP — joint training + SFT anchor loss

Key improvements over v3:
1. SFT anchor loss: runs forward pass WITHOUT adv_token, penalizes deviation
   from SFT actions. Prevents backbone degradation while still learning CFG.
2. Episode-level advantage labeling (not timestep-level N-step return).

Architecture: adv_token in prefix (same as v3), backbone also trains (same as v3).
Loss = flow_loss(with_adv) + sft_lambda * flow_loss(without_adv)

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  MUJOCO_GL=osmesa nohup .venv/bin/python -u scripts/recap/train_recap_v7.py \
    --sft-ckpt checkpoints/pi0_libero/recap_sft_baseline/29999 \
    --labeled-data data/rollouts/labeled_episodes_v5.pkl \
    --exp-name recap_policy_v7 \
    --num-steps 20000 --batch-size 64 \
    --lr 2e-6 --adv-lr 1e-3 --sft-lambda 1.0 \
    --ortho-lambda 0.1 --norm-lambda 0.001 --fsdp-devices 4 \
    > logs/recap_train_v7.log 2>&1 &
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

HIDDEN_DIM       = 2048
ADV_DROPOUT_RATE = 0.30
ACTION_HORIZON   = 50
SAVE_INTERVAL    = 1000
LOG_INTERVAL     = 50


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} {msg}", flush=True)


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
        n_pos = sum(1 for s in self.flat_samples if s["advantage_label"] == 1)
        log(f"Data: {len(self.flat_samples)} timesteps, {n_pos/len(self.flat_samples)*100:.1f}% positive")

    def sample_batch(self, tokenize_fn):
        idx = self.rng.integers(0, len(self.flat_samples), size=self.batch_size)
        batch = [self.flat_samples[i] for i in idx]
        images     = np.stack([s["image"] for s in batch])
        wrist_imgs = np.stack([s["wrist_image"] for s in batch])
        states     = np.stack([s["state"] for s in batch])
        actions    = np.stack([s["actions"] for s in batch])
        adv_labels = np.array([s["advantage_label"] for s in batch], dtype=np.int32)
        tokens, token_mask = tokenize_fn([s["prompt"] for s in batch])
        return images, wrist_imgs, states, actions, adv_labels, tokens, token_mask


def compute_loss(
    model, adv_embed, rng, ortho_lambda, norm_lambda, sft_lambda,
    images, wrist_images, states, actions, adv_labels,
    tokenized_prompt, tokenized_prompt_mask, train=True,
):
    """Flow matching loss with adv_token in prefix + SFT anchor loss."""
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)

    model_action_dim = 32
    if states.shape[-1] < model_action_dim:
        states = jnp.concatenate(
            [states, jnp.zeros((*states.shape[:-1], model_action_dim - states.shape[-1]), dtype=states.dtype)],
            axis=-1)
    if actions.shape[-1] < model_action_dim:
        actions = jnp.concatenate(
            [actions, jnp.zeros((*actions.shape[:-1], model_action_dim - actions.shape[-1]), dtype=actions.dtype)],
            axis=-1)

    obs = _model.Observation(
        images={
            "base_0_rgb":        images,
            "left_wrist_0_rgb":  wrist_images,
            "right_wrist_0_rgb": jnp.zeros_like(images),
        },
        image_masks={
            "base_0_rgb":        jnp.ones(images.shape[0], dtype=bool),
            "left_wrist_0_rgb":  jnp.ones(images.shape[0], dtype=bool),
            "right_wrist_0_rgb": jnp.zeros(images.shape[0], dtype=bool),
        },
        state=states,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    obs = _model.preprocess_observation(preprocess_rng, obs, train=False)

    B = actions.shape[0]
    batch_shape = actions.shape[:1]

    noise    = jax.random.normal(noise_rng, actions.shape)
    time     = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_exp = time[..., None, None]
    x_t      = time_exp * noise + (1 - time_exp) * actions
    u_t      = noise - actions

    # ── SFT anchor loss: pure prefix, no adv_token ────────────────────────
    prefix_tokens_sft, prefix_mask_sft, prefix_ar_mask_sft = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)

    input_mask_sft = jnp.concatenate([prefix_mask_sft, suffix_mask], axis=1)
    ar_mask_sft    = jnp.concatenate([prefix_ar_mask_sft, suffix_ar_mask], axis=0)
    attn_mask_sft  = make_attn_mask(input_mask_sft, ar_mask_sft)
    positions_sft  = jnp.cumsum(input_mask_sft, axis=1) - 1

    (_, suffix_out_sft), _ = model.PaliGemma.llm(
        [prefix_tokens_sft, suffix_tokens],
        mask=attn_mask_sft,
        positions=positions_sft,
        adarms_cond=[None, adarms_cond],
    )
    v_t_sft = model.action_out_proj(suffix_out_sft[:, -model.action_horizon:])
    sft_loss = jnp.mean(jnp.square(v_t_sft - u_t))

    # ── RECAP loss: prefix with adv_token ─────────────────────────────────
    if train:
        keep = jax.random.bernoulli(dropout_rng, p=1.0 - ADV_DROPOUT_RATE, shape=(B,)).astype(jnp.int32)
        effective_labels = adv_labels * keep
    else:
        effective_labels = adv_labels

    adv_tokens = adv_embed(effective_labels)[:, None, :]
    prefix_tokens_adv  = jnp.concatenate([adv_tokens, prefix_tokens_sft], axis=1)
    prefix_mask_adv    = jnp.concatenate([jnp.ones((B, 1), dtype=bool), prefix_mask_sft], axis=1)
    prefix_ar_mask_adv = jnp.concatenate([jnp.array([False]), prefix_ar_mask_sft])

    input_mask_adv = jnp.concatenate([prefix_mask_adv, suffix_mask], axis=1)
    ar_mask_adv    = jnp.concatenate([prefix_ar_mask_adv, suffix_ar_mask], axis=0)
    attn_mask_adv  = make_attn_mask(input_mask_adv, ar_mask_adv)
    positions_adv  = jnp.cumsum(input_mask_adv, axis=1) - 1

    (_, suffix_out_adv), _ = model.PaliGemma.llm(
        [prefix_tokens_adv, suffix_tokens],
        mask=attn_mask_adv,
        positions=positions_adv,
        adarms_cond=[None, adarms_cond],
    )
    v_t_adv = model.action_out_proj(suffix_out_adv[:, -model.action_horizon:])
    flow_loss = jnp.mean(jnp.square(v_t_adv - u_t))

    # ── regularization ────────────────────────────────────────────────────
    W = adv_embed.embedding.value
    e0 = W[0] / (jnp.linalg.norm(W[0]) + 1e-8)
    e1 = W[1] / (jnp.linalg.norm(W[1]) + 1e-8)
    ortho_loss = jnp.square(jnp.dot(e0, e1))
    norm_loss  = jnp.mean(jnp.square(W))

    total = flow_loss + sft_lambda * sft_loss + ortho_lambda * ortho_loss + norm_lambda * norm_loss
    return total, flow_loss, sft_loss, ortho_loss, norm_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",      default=None)
    parser.add_argument("--resume-ckpt",   default=None)
    parser.add_argument("--labeled-data",  required=True)
    parser.add_argument("--exp-name",      default="recap_policy_v7")
    parser.add_argument("--num-steps",     type=int, default=20000)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=2e-6)
    parser.add_argument("--adv-lr",        type=float, default=1e-3)
    parser.add_argument("--sft-lambda",    type=float, default=1.0)
    parser.add_argument("--ortho-lambda",  type=float, default=0.1)
    parser.add_argument("--norm-lambda",   type=float, default=0.001)
    parser.add_argument("--ckpt-base-dir", default="checkpoints")
    parser.add_argument("--fsdp-devices",  type=int, default=4)
    args = parser.parse_args()
    if args.resume_ckpt is None and args.sft_ckpt is None:
        parser.error("Either --sft-ckpt or --resume-ckpt is required")

    log(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    log(f"Config: steps={args.num_steps}, batch={args.batch_size}, "
        f"lr={args.lr}, adv_lr={args.adv_lr}, sft_lambda={args.sft_lambda}, "
        f"ortho_lambda={args.ortho_lambda}  [JOINT TRAINING + SFT ANCHOR]")

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {args.batch_size} must be divisible by {jax.device_count()}")

    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding       = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    log(f"Mesh: {mesh.shape}")

    train_config = _config.get_config("pi0_libero")
    model_config = train_config.model

    import orbax.checkpoint as ocp

    if args.resume_ckpt is not None:
        resume_path = pathlib.Path(args.resume_ckpt)
        try:
            start_step = int(resume_path.name)
        except ValueError:
            start_step = 0
        log(f"Resuming from: {resume_path} (start_step={start_step})")
        def _init_model(rng):
            m = model_config.create(rng)
            m_state = nnx.state(m)
            ckptr_r = ocp.StandardCheckpointer()
            restored = ckptr_r.restore(str((resume_path / "model_params").resolve()), target=m_state)
            nnx.update(m, restored)
            return m
        def _init_adv(rng):
            a = nnx.Embed(num_embeddings=2, features=HIDDEN_DIM, rngs=nnx.Rngs(params=rng))
            a_state = nnx.state(a)
            ckptr_r = ocp.StandardCheckpointer()
            restored = ckptr_r.restore(str((resume_path / "adv_embed").resolve()), target=a_state)
            nnx.update(a, restored)
            return a
    else:
        start_step = 0
        log("Loading SFT checkpoint...")
        ckpt_params_dir = pathlib.Path(args.sft_ckpt) / "params"
        if not ckpt_params_dir.exists():
            ckpt_params_dir = pathlib.Path(args.sft_ckpt)
        raw_params = _model.restore_params(str(ckpt_params_dir), restore_type=np.ndarray)
        def _init_model(rng):
            m = model_config.create(rng)
            graphdef, state = nnx.split(m)
            state.replace_by_pure_dict(raw_params)
            return nnx.merge(graphdef, state)
        def _init_adv(rng):
            return nnx.Embed(num_embeddings=2, features=HIDDEN_DIM, rngs=nnx.Rngs(params=rng))

    model = _init_model(jax.random.PRNGKey(0))
    adv_embed = _init_adv(jax.random.PRNGKey(42))
    log("Model and adv_embed loaded.")

    # Separate optimizers: low lr for backbone, high lr for adv_embed
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=500, decay_steps=args.num_steps,
    )
    model_tx = optax.adamw(lr_sched, weight_decay=1e-4)
    adv_tx   = optax.adamw(args.adv_lr, weight_decay=0.0)

    model_params = nnx.state(model)
    adv_params   = nnx.state(adv_embed)

    model_params_shape    = jax.eval_shape(lambda: model_params)
    model_params_sharding = sharding.fsdp_sharding(model_params_shape, mesh, log=True)

    model_params   = jax.device_put(model_params,   model_params_sharding)
    adv_params     = jax.device_put(adv_params,     replicated_sharding)

    model_opt_state = model_tx.init(model_params)
    adv_opt_state   = adv_tx.init(adv_params)
    model_opt_state = jax.device_put(model_opt_state, replicated_sharding)
    adv_opt_state   = jax.device_put(adv_opt_state,   replicated_sharding)
    log("Optimizers initialized. JOINT backbone + adv_embed training.")

    data_loader = RolloutDataLoader(pathlib.Path(args.labeled_data), args.batch_size)

    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer = PaligemmaTokenizer(max_len=train_config.model.max_token_len)

    def tokenize(prompts):
        results = [tokenizer.tokenize(p) for p in prompts]
        return (np.stack([r[0] for r in results]),
                np.stack([r[1] for r in results]).astype(bool))

    model_graphdef = nnx.graphdef(model)
    adv_graphdef   = nnx.graphdef(adv_embed)
    ortho_lambda   = args.ortho_lambda
    norm_lambda    = args.norm_lambda
    sft_lambda     = args.sft_lambda

    def train_step(model_params, adv_params, model_opt_state, adv_opt_state,
                   images, wrist_images, states, actions, adv_labels,
                   tokens, token_mask, rng):

        m = nnx.merge(model_graphdef, model_params)
        a = nnx.merge(adv_graphdef, adv_params)
        m.train()

        def loss_fn(model, adv_embed):
            total, flow, sft, ortho, norm = compute_loss(
                model, adv_embed, rng, ortho_lambda, norm_lambda, sft_lambda,
                images, wrist_images, states, actions, adv_labels,
                tokens, token_mask, train=True,
            )
            return total, (flow, sft, ortho, norm)

        diff_state_model = nnx.DiffState(0, nnx.Param)
        diff_state_adv   = nnx.DiffState(1, nnx.Param)
        (total_loss, (flow_loss, sft_loss, ortho_loss, norm_loss)), (model_grads, adv_grads) = nnx.value_and_grad(
            loss_fn, argnums=(diff_state_model, diff_state_adv), has_aux=True
        )(m, a)

        m_updates, new_model_opt_state = model_tx.update(model_grads, model_opt_state, model_params)
        a_updates, new_adv_opt_state   = adv_tx.update(adv_grads, adv_opt_state, adv_params)

        new_model_params = optax.apply_updates(model_params, m_updates)
        new_adv_params   = optax.apply_updates(adv_params, a_updates)

        return (new_model_params, new_adv_params, new_model_opt_state, new_adv_opt_state,
                total_loss, flow_loss, sft_loss, ortho_loss, norm_loss)

    train_step_jit = jax.jit(
        train_step,
        in_shardings=(
            model_params_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding,
            data_sharding, data_sharding, data_sharding, data_sharding, data_sharding,
            data_sharding, data_sharding,
            replicated_sharding,
        ),
        out_shardings=(
            model_params_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding,
        ),
        donate_argnums=(0, 1, 2, 3),
    )

    ckpt_dir = pathlib.Path(args.ckpt_base_dir).resolve() / "pi0_libero" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(7)
    t0  = time.time()
    total_steps = start_step + args.num_steps
    log(f"Starting RECAP v7 training from step {start_step+1} to {total_steps}...")

    for step in range(start_step + 1, total_steps + 1):
        rng, step_rng = jax.random.split(rng)

        images, wrist_imgs, states, actions, adv_labels, tokens, token_mask = data_loader.sample_batch(tokenize)

        def _to_device(arr, shrd):
            return jax.device_put(jnp.array(arr), shrd)

        images_d     = _to_device(images,     data_sharding)
        wrist_d      = _to_device(wrist_imgs, data_sharding)
        states_d     = _to_device(states,     data_sharding)
        actions_d    = _to_device(actions,    data_sharding)
        adv_labels_d = _to_device(adv_labels, data_sharding)
        tokens_d     = _to_device(tokens,     data_sharding)
        tmask_d      = _to_device(token_mask, data_sharding)

        with sharding.set_mesh(mesh):
            (model_params, adv_params, model_opt_state, adv_opt_state,
             total_loss, flow_loss, sft_loss, ortho_loss, norm_loss) = train_step_jit(
                model_params, adv_params, model_opt_state, adv_opt_state,
                images_d, wrist_d, states_d, actions_d, adv_labels_d,
                tokens_d, tmask_d, step_rng,
            )

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            rate    = LOG_INTERVAL / elapsed
            remain  = (total_steps - step) / rate
            W = np.array(jax.device_get(adv_params.embedding.value))
            e0_n = np.linalg.norm(W[0])
            e1_n = np.linalg.norm(W[1])
            cos  = np.dot(W[0], W[1]) / (e0_n * e1_n + 1e-8)
            log(f"Step {step:5d}/{total_steps} "
                f"loss={float(total_loss):.4f} flow={float(flow_loss):.4f} "
                f"sft={float(sft_loss):.4f} ortho={float(ortho_loss):.4f} "
                f"e0={e0_n:.4f} e1={e1_n:.4f} cos={cos:.4f} "
                f"rate={rate:.2f}s/s eta={remain/3600:.2f}h")
            t0 = time.time()

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            import shutil
            step_dir = ckpt_dir / str(step)
            step_dir.mkdir(parents=True, exist_ok=True)
            ckptr = ocp.StandardCheckpointer()
            host_model_params = jax.device_get(model_params)
            host_adv_params   = jax.device_get(adv_params)
            model_params_dir = step_dir / "model_params"
            adv_embed_dir    = step_dir / "adv_embed"
            if model_params_dir.exists():
                shutil.rmtree(model_params_dir)
            if adv_embed_dir.exists():
                shutil.rmtree(adv_embed_dir)
            ckptr.save(model_params_dir, host_model_params)
            ckptr.wait_until_finished()
            ckptr.save(adv_embed_dir, host_adv_params)
            ckptr.wait_until_finished()
            log(f"Checkpoint saved: {step_dir}")

    log("RECAP v7 training complete.")


if __name__ == "__main__":
    main()
