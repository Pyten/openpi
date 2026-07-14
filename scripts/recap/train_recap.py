"""
Phase 1D v3: RECAP advantage-conditioned policy training (with FSDP)

Key fix over v2:
- adv_embed uses separate high lr (default 1e-3) vs model lr (default 5e-6)
- Added orthogonal regularization loss to force embed[0] != embed[1]

Usage:
  cd /mnt/vepfs/pyten/Programs/code/pi0.6
  MUJOCO_GL=osmesa nohup .venv/bin/python scripts/recap/train_recap.py \
    --sft-ckpt checkpoints/pi0_libero/recap_sft_baseline/29999 \
    --labeled-data data/rollouts/labeled_episodes.pkl \
    --exp-name recap_policy_v3 \
    --num-steps 20000 \
    --batch-size 64 \
    --lr 5e-6 \
    --adv-lr 1e-3 \
    --ortho-lambda 0.1 \
    --fsdp-devices 4 \
    > logs/recap_train_v3.log 2>&1 &
"""

import argparse
import logging
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

from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.recap.conditioning import ConditioningState
from openpi.recap.conditioning import apply_condition_dropout
from openpi.training import config as _config
from openpi.training import sharding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HIDDEN_DIM       = 2048
ADV_DROPOUT_RATE = 0.30
ACTION_HORIZON   = 50
SAVE_INTERVAL    = 1000
LOG_INTERVAL     = 50


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
        logger.info(f"Data: {len(self.flat_samples)} timesteps, {n_pos/len(self.flat_samples)*100:.1f}% positive")

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


def compute_loss_with_advantage(
    model, adv_embed, rng, ortho_lambda,
    images, wrist_images, states, actions, adv_labels,
    tokenized_prompt, tokenized_prompt_mask, train=True,
):
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

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)

    if train:
        keep = jax.random.bernoulli(dropout_rng, p=1.0 - ADV_DROPOUT_RATE, shape=(B,))
        effective_labels = apply_condition_dropout(adv_labels, keep)
    else:
        effective_labels = adv_labels

    adv_tokens = adv_embed(effective_labels)[:, None, :]
    prefix_tokens  = jnp.concatenate([adv_tokens, prefix_tokens], axis=1)
    prefix_mask    = jnp.concatenate([jnp.ones((B, 1), dtype=bool), prefix_mask], axis=1)
    prefix_ar_mask = jnp.concatenate([jnp.array([False]), prefix_ar_mask])

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask    = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask  = make_attn_mask(input_mask, ar_mask)
    positions  = jnp.cumsum(input_mask, axis=1) - 1

    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    flow_loss = jnp.mean(jnp.square(v_t - u_t))

    # Keep negative, positive, and null embeddings distinct.
    W = adv_embed.embedding.value
    normalized = W / (jnp.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    gram = normalized @ normalized.T
    off_diagonal = gram - jnp.eye(gram.shape[0], dtype=gram.dtype)
    ortho_loss = jnp.sum(jnp.square(off_diagonal)) / (gram.shape[0] * (gram.shape[0] - 1))

    return flow_loss + ortho_lambda * ortho_loss, flow_loss, ortho_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt",      default=None,
                        help="SFT checkpoint dir (required if not --resume-ckpt)")
    parser.add_argument("--resume-ckpt",   default=None,
                        help="RECAP checkpoint dir to resume from")
    parser.add_argument("--labeled-data",  required=True)
    parser.add_argument("--exp-name",      default="recap_policy_v3")
    parser.add_argument("--num-steps",     type=int, default=20000)
    parser.add_argument("--batch-size",    type=int, default=64)
    parser.add_argument("--lr",            type=float, default=5e-6,
                        help="Main model learning rate")
    parser.add_argument("--adv-lr",        type=float, default=1e-3,
                        help="Advantage embedding learning rate (much larger than model lr)")
    parser.add_argument("--ortho-lambda",  type=float, default=0.1,
                        help="Weight for orthogonal regularization on adv_embed")
    parser.add_argument("--ckpt-base-dir", default="checkpoints")
    parser.add_argument("--fsdp-devices",  type=int, default=4)
    args = parser.parse_args()
    if args.resume_ckpt is None and args.sft_ckpt is None:
        parser.error("Either --sft-ckpt or --resume-ckpt is required")

    logger.info(f"JAX devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    logger.info(f"Config: steps={args.num_steps}, batch={args.batch_size}, "
                f"lr={args.lr}, adv_lr={args.adv_lr}, ortho_lambda={args.ortho_lambda}")

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch_size {args.batch_size} must be divisible by {jax.device_count()}")

    jax.config.update("jax_compilation_cache_dir", str(pathlib.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding       = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    logger.info(f"Mesh: {mesh.shape}")

    train_config = _config.get_config("pi0_libero")
    model_config = train_config.model

    import orbax.checkpoint as ocp

    if args.resume_ckpt is not None:
        resume_path = pathlib.Path(args.resume_ckpt)
        try:
            start_step = int(resume_path.name)
        except ValueError:
            start_step = 0
        logger.info(f"Resuming from RECAP checkpoint: {resume_path} (start_step={start_step})")
        def _init_model(rng):
            m = model_config.create(rng)
            m_state = nnx.state(m)
            ckptr = ocp.StandardCheckpointer()
            restored = ckptr.restore(str((resume_path / "model_params").resolve()), target=m_state)
            nnx.update(m, restored)
            return m
    else:
        start_step = 0
        logger.info("Loading SFT checkpoint...")
        ckpt_params_dir = pathlib.Path(args.sft_ckpt) / "params"
        if not ckpt_params_dir.exists():
            ckpt_params_dir = pathlib.Path(args.sft_ckpt)
        raw_params = _model.restore_params(str(ckpt_params_dir), restore_type=np.ndarray)
        def _init_model(rng):
            m = model_config.create(rng)
            graphdef, state = nnx.split(m)
            state.replace_by_pure_dict(raw_params)
            return nnx.merge(graphdef, state)

    model = _init_model(jax.random.PRNGKey(0))
    logger.info("Model loaded.")

    # Always fresh-init adv_embed from scratch (no resume — previous run barely moved it anyway)
    adv_embed = nnx.Embed(
        num_embeddings=len(ConditioningState),
        features=HIDDEN_DIM,
        rngs=nnx.Rngs(params=jax.random.PRNGKey(42)),
    )
    logger.info("adv_embed initialized fresh (orthogonal init via random seed).")

    # Separate optimizers: high lr for adv_embed, cosine decay for model
    warmup_steps = min(200, max(1, args.num_steps // 10))
    lr_sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=max(args.num_steps, warmup_steps + 1),
    )
    model_tx = optax.adamw(lr_sched, weight_decay=1e-4)
    adv_tx   = optax.adamw(args.adv_lr, weight_decay=0.0)  # flat lr, no wd for tiny embed

    freeze_filter    = model_config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))

    model_params = nnx.state(model)
    adv_params   = nnx.state(adv_embed)

    model_params_shape = jax.eval_shape(lambda: model_params)
    model_params_sharding = sharding.fsdp_sharding(model_params_shape, mesh, log=True)

    model_params = jax.device_put(model_params, model_params_sharding)
    adv_params   = jax.device_put(adv_params,   replicated_sharding)
    logger.info("Params sharded. Initializing optimizer states...")

    trainable_mp  = model_params.filter(trainable_filter)
    opt_state     = model_tx.init(trainable_mp)
    adv_opt_state = adv_tx.init(adv_params)

    opt_state_shape    = jax.eval_shape(lambda: opt_state)
    opt_state_sharding = sharding.fsdp_sharding(opt_state_shape, mesh)
    opt_state          = jax.device_put(opt_state,     opt_state_sharding)
    adv_opt_state      = jax.device_put(adv_opt_state, replicated_sharding)

    state_sharding = {
        "model_params":  model_params_sharding,
        "opt_state":     opt_state_sharding,
        "adv_params":    replicated_sharding,
        "adv_opt_state": replicated_sharding,
    }
    logger.info("Train state sharded across GPUs.")

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

    def train_step(model_params, opt_state, adv_params, adv_opt_state,
                   images, wrist_images, states, actions, adv_labels,
                   tokens, token_mask, rng):

        m = nnx.merge(model_graphdef, model_params)
        a = nnx.merge(adv_graphdef, adv_params)
        m.train()

        def loss_fn(model, adv_embed):
            total, flow, ortho = compute_loss_with_advantage(
                model, adv_embed, rng, ortho_lambda,
                images, wrist_images, states, actions, adv_labels,
                tokens, token_mask, train=True,
            )
            return total, (flow, ortho)

        diff_state_model = nnx.DiffState(0, trainable_filter)
        diff_state_adv   = nnx.DiffState(1, nnx.Param)
        (total_loss, (flow_loss, ortho_loss)), (model_grads, adv_grads) = nnx.value_and_grad(
            loss_fn, argnums=(diff_state_model, diff_state_adv), has_aux=True
        )(m, a)

        trainable_mp = model_params.filter(trainable_filter)
        mp_updates, new_opt_state = model_tx.update(model_grads, opt_state, trainable_mp)
        new_trainable_mp = optax.apply_updates(trainable_mp, mp_updates)
        nnx.update(m, new_trainable_mp)
        new_model_params = nnx.state(m)

        ap_updates, new_adv_opt_state = adv_tx.update(adv_grads, adv_opt_state, adv_params)
        new_adv_params = optax.apply_updates(adv_params, ap_updates)

        return new_model_params, new_opt_state, new_adv_params, new_adv_opt_state, total_loss, flow_loss, ortho_loss

    train_step_jit = jax.jit(
        train_step,
        in_shardings=(
            state_sharding["model_params"], state_sharding["opt_state"],
            replicated_sharding, replicated_sharding,
            data_sharding, data_sharding, data_sharding, data_sharding, data_sharding,
            data_sharding, data_sharding,
            replicated_sharding,
        ),
        out_shardings=(
            state_sharding["model_params"], state_sharding["opt_state"],
            replicated_sharding, replicated_sharding,
            replicated_sharding, replicated_sharding, replicated_sharding,
        ),
        donate_argnums=(0, 1, 2, 3),
    )

    ckpt_dir = pathlib.Path(args.ckpt_base_dir).resolve() / "pi0_libero" / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(7)
    t0  = time.time()
    total_steps = start_step + args.num_steps
    logger.info(f"Starting RECAP training from step {start_step+1} to {total_steps}...")

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
            (model_params, opt_state, adv_params, adv_opt_state,
             total_loss, flow_loss, ortho_loss) = train_step_jit(
                model_params, opt_state, adv_params, adv_opt_state,
                images_d, wrist_d, states_d, actions_d, adv_labels_d,
                tokens_d, tmask_d, step_rng,
            )

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            rate    = LOG_INTERVAL / elapsed
            remain  = (total_steps - step) / rate
            logger.info(f"Step {step:5d}/{total_steps} "
                        f"loss={float(total_loss):.4f} flow={float(flow_loss):.4f} "
                        f"ortho={float(ortho_loss):.4f} "
                        f"rate={rate:.2f}s/s remaining={remain/3600:.2f}h")
            t0 = time.time()

        if step % SAVE_INTERVAL == 0 or step == total_steps:
            import orbax.checkpoint as ocp
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
            logger.info(f"Checkpoint saved: {step_dir}")

    logger.info("RECAP training complete.")


if __name__ == "__main__":
    main()
