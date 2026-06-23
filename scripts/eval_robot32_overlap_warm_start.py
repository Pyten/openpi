"""
Evaluate a trained policy checkpoint on a subset of a LeRobot-format dataset.

This script is designed for the custom `robot_32` config in `src/openpi/training/config.py`,
and focuses on "single-sample" offline sanity checks to avoid common pitfalls:

- Verify shapes match: (action_horizon, action_dim)
- Verify output action space is ABSOLUTE (vs. DELTA) by comparing errors in both spaces
- Verify determinism when providing a fixed diffusion noise tensor (PI0/PI05)
- Optionally load normalization statistics (norm_stats.json) from dataset root

Example:
  python scripts/eval_robot32_overlap.py \
    --dataset-root "/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/our_data/lerobot_1&2" \
    --ckpt "/iag_ad_vepfs_volc/iag_ad_vepfs_volc/wangkeqiu/res/robot_32/grab_delta_finetune/19999" \
    --num-episodes 10 --frames-per-episode 1 --split test --eps 0.05
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Any

import numpy as np

from warm_start_from_dtw import DEFAULT_MEAN, DEFAULT_STD, load_stats, warm_start_chunk_from_stats
from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import policy_config as policy_config_lib
from openpi.shared import normalize as normalize_lib
from openpi.training import config as train_config_lib
from openpi.training import data_loader as data_loader_lib


def _as_int(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar int-like value, got shape={arr.shape}, dtype={arr.dtype}")
    return int(arr.reshape(()))


def _as_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray) and x.dtype.kind in ("U", "S", "O") and x.size == 1:
        return str(x.reshape(()).item())
    # LeRobot PromptFromLeRobotTask returns a python str, but keep a fallback.
    return str(x)


def _cosine_mean(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    # pred, gt: (H, D)
    num = np.sum(pred * gt, axis=-1)
    den = (np.linalg.norm(pred, axis=-1) * np.linalg.norm(gt, axis=-1) + eps)
    return float(np.mean(num / den))


def _overlap_at_eps(pred: np.ndarray, gt: np.ndarray, eps: float) -> float:
    return float(np.mean(np.abs(pred - gt) < eps))


def _delta_mask_robot32() -> np.ndarray:
    # Same as LeRobot32DataConfig / LeRobotGrab32DataConfig:
    # head2 delta, left arm7 delta, left hand6 absolute, right arm7 delta, right hand6 absolute, waist2 delta, leg2 delta
    return np.asarray(transforms.make_bool_mask(2, 7, -6, 7, -6, 2, 2), dtype=bool)


def _to_delta(actions_abs: np.ndarray, state_abs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Mirror transforms.DeltaActions for comparison.
    actions = np.array(actions_abs, copy=True)
    dims = mask.shape[0]
    base = np.where(mask, state_abs[:dims], 0.0)
    actions[..., :dims] -= base[None, :]
    return actions


def _describe_array(name: str, x: np.ndarray) -> str:
    x = np.asarray(x)
    return (
        f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={np.min(x):.4g} max={np.max(x):.4g} mean={np.mean(x):.4g} std={np.std(x):.4g}"
    )


def _load_splits(dataset_root: pathlib.Path) -> dict[str, list[int]]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    splits = info.get("splits", {})
    out: dict[str, list[int]] = {}
    for k, v in splits.items():
        # LeRobot commonly stores splits as a string "start:end" (e.g., "0:30").
        if isinstance(v, str):
            if ":" in v:
                a, b = v.split(":", 1)
                start, end = int(a), int(b)
                out[k] = list(range(start, end))
            else:
                # Fallback: allow comma-separated list like "0,1,2"
                out[k] = [int(x) for x in v.split(",") if x.strip()]
            continue
        # Some datasets may store explicit lists.
        if isinstance(v, (list, tuple)):
            out[k] = [int(x) for x in v]
            continue
        raise ValueError(f"Unsupported split format for {k!r}: {type(v)!r} value={v!r}")
    return out


def _load_action_names(dataset_root: pathlib.Path) -> list[str] | None:
    """Load action dimension names from meta/info.json, if available."""
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    action = feats.get("action", {})
    names = action.get("names")
    # LeRobot stores names like: [["head_pitch", ... , "knee_pitch"]]
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        if all(isinstance(x, str) for x in names[0]):
            return list(names[0])
    return None


def _load_episode_lengths(dataset_root: pathlib.Path) -> dict[int, int]:
    """Read meta/episodes.jsonl -> {episode_index: length}."""
    ep_path = dataset_root / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    with ep_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lengths[int(obj["episode_index"])] = int(obj["length"])
    if not lengths:
        raise ValueError(f"No episodes found in {ep_path}")
    return lengths


def _compute_episode_starts(episode_lengths: dict[int, int]) -> dict[int, int]:
    """Compute global start index for each episode, assuming episodes are concatenated by ascending episode_index."""
    starts: dict[int, int] = {}
    offset = 0
    for ep in sorted(episode_lengths.keys()):
        starts[ep] = offset
        offset += int(episode_lengths[ep])
    return starts


def _choose_norm_stats(dataset_root: pathlib.Path, mode: str) -> dict[str, normalize_lib.NormStats] | None:
    mode = mode.lower()
    if mode == "none":
        return None
    if mode == "dataset":
        # `normalize.load(dir)` expects `dir/norm_stats.json`.
        return normalize_lib.load(dataset_root)
    raise ValueError(f"Unsupported --norm-stats mode: {mode} (expected: none|dataset)")


def _deepcopy_for_infer(x: Any) -> Any:
    """Deep-copy a nested dict/list/tuple structure, copying array-like leaves.

    Why: `policy.infer()` applies transforms that may mutate arrays in-place (e.g. DeltaActions / Normalize).
    If we keep references to the same underlying torch tensor / numpy array (or a shared numpy view),
    our "ground truth" arrays can get modified after inference, producing misleading metrics.
    """
    if isinstance(x, dict):
        return {k: _deepcopy_for_infer(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deepcopy_for_infer(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_deepcopy_for_infer(v) for v in x)
    # Keep plain strings / numbers as-is.
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    # Copy anything array-like (torch.Tensor, np.ndarray, numpy scalar, etc.)
    try:
        return np.asarray(x).copy()
    except Exception:
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--config", default="robot_32", type=str)
    ap.add_argument("--split", default="test", type=str, help="train|test|all")
    ap.add_argument("--num-episodes", default=10, type=int)
    ap.add_argument("--frames-per-episode", default=1, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--eps", default=0.05, type=float, help="overlap@eps threshold (units match your action space)")
    ap.add_argument(
        "--norm-stats",
        default="dataset",
        type=str,
        help="none|dataset. IMPORTANT: must match how you trained; checkpoint assets are empty for your run.",
    )
    ap.add_argument(
        "--noise",
        default="warm_start",
        type=str,
        help="For PI0/PI05 only: zero|random|none|warm_start. Use 'zero' to make inference deterministic.",
    )
    ap.add_argument(
        "--warm-start-mean",
        default=None,
        type=str,
        help="Path to mean traj .npy (T,32) for warm_start noise (default uses dataset/warm_start_from_dtw.py).",
    )
    ap.add_argument(
        "--warm-start-std",
        default=None,
        type=str,
        help="Path to std traj .npy (T,32) for warm_start noise (default uses dataset/warm_start_from_dtw.py).",
    )
    ap.add_argument(
        "--warm-start-noise-scale",
        default=0.1,
        type=float,
        help="Noise scale for warm_start (multiplied with std_traj).",
    )
    ap.add_argument(
        "--noise-check",
        action="store_true",
        help="If set, run a second inference with a different noise (PI0/PI05) to show sensitivity.",
    )
    ap.add_argument(
        "--indexing",
        default="meta",
        type=str,
        help="meta|scan. meta uses meta/episodes.jsonl prefix-sum (fast). scan does a full pass over all frames (slow).",
    )
    ap.add_argument(
        "--boundary",
        default="avoid",
        type=str,
        help="avoid|allow. If avoid, sample frames away from episode end to reduce padding/repeat due to horizon.",
    )
    ap.add_argument(
        "--exclude-episodes",
        default="",
        type=str,
        help="Comma-separated episode indices to skip, e.g. '0,5,12'. Useful to skip known-bad episodes.",
    )
    ap.add_argument(
        "--min-episode-length",
        default=1,
        type=int,
        help="Skip episodes shorter than this length (based on meta/episodes.jsonl).",
    )
    args = ap.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    ckpt_dir = pathlib.Path(args.ckpt)
    t0 = time.time()
    print(f"[stage] start (t={t0:.3f})", flush=True)

    # Load config and override dataset repo_id to local path.
    cfg0 = train_config_lib.get_config(args.config)
    cfg = dataclasses.replace(cfg0, data=dataclasses.replace(cfg0.data, repo_id=str(dataset_root)))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    # Dataset (raw LeRobot samples, but actions are returned as action chunks due to delta_timestamps).
    print("[stage] creating dataset object (may scan parquet shards metadata)", flush=True)
    ds = data_loader_lib.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    print(f"[stage] dataset ready: len={len(ds)} (t+{time.time()-t0:.2f}s)", flush=True)

    # Pick episodes from split.
    splits = _load_splits(dataset_root)
    exclude_eps: set[int] = set()
    if args.exclude_episodes.strip():
        exclude_eps = {int(x) for x in args.exclude_episodes.split(",") if x.strip()}

    if args.split.lower() == "all":
        episode_pool = list(range(_as_int(json.loads((dataset_root / "meta" / "info.json").read_text())["total_episodes"])))
    else:
        if args.split.lower() not in splits:
            raise ValueError(f"Split {args.split!r} not found in meta/info.json splits: {list(splits.keys())}")
        episode_pool = splits[args.split.lower()]

    # Filter by episode length and explicit excludes.
    episode_lengths_all = _load_episode_lengths(dataset_root)
    episode_pool = [
        ep
        for ep in episode_pool
        if ep not in exclude_eps and int(episode_lengths_all.get(int(ep), 0)) >= int(args.min_episode_length)
    ]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(episode_pool)
    target_episodes = set(episode_pool[: args.num_episodes])
    if not target_episodes:
        raise ValueError("No episodes selected. Check --num-episodes and --split.")

    # Choose global dataset indices for selected episodes.
    # IMPORTANT: avoid full dataset scan because ds[i] is expensive (random parquet access).
    chosen_indices: list[int] = []
    indexing = args.indexing.lower()
    boundary = args.boundary.lower()
    if indexing == "meta":
        episode_lengths = _load_episode_lengths(dataset_root)
        episode_starts = _compute_episode_starts(episode_lengths)

        for ep in sorted(target_episodes):
            if ep not in episode_starts:
                raise ValueError(f"episode {ep} not found in meta/episodes.jsonl")
            start = int(episode_starts[ep])
            length = int(episode_lengths[ep])
            end_exclusive = start + length

            # Avoid sampling too close to episode end to reduce padding effects for action chunks.
            if boundary == "avoid":
                max_start = end_exclusive - cfg.model.action_horizon
            else:
                max_start = end_exclusive - 1
            if max_start < start:
                # Episode shorter than horizon: fall back.
                max_start = end_exclusive - 1

            if args.frames_per_episode <= 1:
                chosen_indices.append(int(rng.integers(low=start, high=max_start + 1)))
            else:
                chosen_indices.extend([int(x) for x in np.linspace(start, max_start, args.frames_per_episode)])

        # Probe a few episodes to ensure the indexing assumption holds.
        for ep in list(sorted(target_episodes))[: min(3, len(target_episodes))]:
            start = int(episode_starts[ep])
            probe = ds[start]
            got = _as_int(probe.get("episode_index"))
            if got != ep:
                raise RuntimeError(
                    f"Indexing assumption failed: ds[start_of_ep]={got} but expected {ep}. "
                    "Re-run with --indexing scan (slow) or investigate dataset ordering."
                )
    elif indexing == "scan":
        print("[stage] indexing=scan: scanning all frames to map episode_index -> frame indices (slow)", flush=True)
        ep_to_indices: dict[int, list[int]] = {ep: [] for ep in target_episodes}
        for i in range(len(ds)):
            item = ds[i]
            ep = _as_int(item.get("episode_index"))
            if ep in ep_to_indices:
                ep_to_indices[ep].append(i)
        for ep, idxs in sorted(ep_to_indices.items()):
            if not idxs:
                continue
            if args.frames_per_episode >= len(idxs):
                picks = idxs
            else:
                picks = [idxs[j] for j in np.linspace(0, len(idxs) - 1, args.frames_per_episode, dtype=int)]
            chosen_indices.extend(picks)
    else:
        raise ValueError(f"Unsupported --indexing: {args.indexing} (expected: meta|scan)")

    if not chosen_indices:
        raise RuntimeError("Failed to find any frame indices for selected episodes.")

    # Norm stats choice (must match training).
    norm_stats = _choose_norm_stats(dataset_root, args.norm_stats)

    warm_start_stats: tuple[np.ndarray, np.ndarray] | None = None
    if args.noise.lower() == "warm_start":
        mean_path = pathlib.Path(args.warm_start_mean) if args.warm_start_mean else DEFAULT_MEAN
        std_path = pathlib.Path(args.warm_start_std) if args.warm_start_std else DEFAULT_STD
        warm_start_stats = load_stats(mean_path, std_path)

    # Create policy.
    # NOTE: Your checkpoint assets/ is empty, so policy_config's default "load from checkpoint assets" would fail.
    print("[stage] creating policy (loads params; first infer will JIT compile)", flush=True)
    policy = policy_config_lib.create_trained_policy(
        cfg,
        ckpt_dir,
        repack_transforms=data_cfg.repack_transforms,
        norm_stats=norm_stats,
    )
    print(f"[stage] policy ready (t+{time.time()-t0:.2f}s)", flush=True)

    action_key = data_cfg.action_sequence_keys[0]  # for robot_32 it's "action"
    delta_mask = _delta_mask_robot32()
    repack_fn = transforms.compose(data_cfg.repack_transforms.inputs)

    # Metrics aggregation.
    abs_mse_list: list[float] = []
    abs_mae_list: list[float] = []
    abs_cos_list: list[float] = []
    abs_ov_list: list[float] = []
    delta_mse_list: list[float] = []  # treating predictions as if they were delta (diagnostic)

    # For PI0/PI05 determinism: fixed noise makes sampling deterministic.
    def make_noise(kind: str) -> np.ndarray | None:
        kind = kind.lower()
        if kind == "none":
            return None
        if kind == "zero":
            return np.zeros((cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)
        if kind == "random":
            return rng.normal(size=(cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
        raise ValueError(f"Unsupported --noise: {kind} (expected: zero|random|none|warm_start)")

    is_diffusion = cfg.model.model_type in (model_lib.ModelType.PI0, model_lib.ModelType.PI05)

    # Evaluate.
    print("=" * 80)
    print(f"config={args.config} split={args.split} episodes={len(target_episodes)} frames={len(chosen_indices)}")
    print(f"dataset_root={dataset_root}")
    print(f"ckpt_dir={ckpt_dir}")
    print(f"model_type={cfg.model.model_type} action_horizon={cfg.model.action_horizon} action_dim={cfg.model.action_dim}")
    print(f"norm_stats={args.norm_stats} (None means no Normalize/Unnormalize)")
    print(f"noise={args.noise} (used only for PI0/PI05)")
    if args.noise.lower() == "warm_start":
        mean_path = pathlib.Path(args.warm_start_mean) if args.warm_start_mean else DEFAULT_MEAN
        std_path = pathlib.Path(args.warm_start_std) if args.warm_start_std else DEFAULT_STD
        print(f"warm_start_mean={mean_path}")
        print(f"warm_start_std={std_path}")
        print(f"warm_start_noise_scale={args.warm_start_noise_scale}")
    print("=" * 80)

    for n, idx in enumerate(chosen_indices):
        # IMPORTANT: take copies BEFORE calling policy.infer, because transforms may mutate inputs in-place.
        raw0 = ds[idx]
        raw_gt = _deepcopy_for_infer(raw0)
        raw_infer = _deepcopy_for_infer(raw0)

        gt_abs = np.asarray(raw_gt[action_key]).copy()
        repacked = repack_fn(raw_gt)
        state_abs = np.asarray(repacked["state"])
        prompt = _as_str(raw_gt.get("prompt", ""))

        if gt_abs.ndim != 2:
            raise ValueError(f"Expected GT action chunk to be rank-2 (H,D), got shape={gt_abs.shape}")

        # Diagnostic: GT delta space (what model was trained on for delta dims).
        gt_delta = _to_delta(gt_abs, state_abs, delta_mask)

        warm_start_idx: int | None = None
        if is_diffusion:
            if args.noise.lower() == "warm_start":
                if warm_start_stats is None:
                    raise RuntimeError("warm_start stats not loaded; check --warm-start-mean/--warm-start-std")
                mean_traj, std_traj = warm_start_stats
                noise, warm_start_idx = warm_start_chunk_from_stats(
                    mean_traj,
                    std_traj,
                    state_abs,
                    chunk_size=cfg.model.action_horizon,
                    noise_scale=args.warm_start_noise_scale,
                    rng=rng,
                )
            else:
                noise = make_noise(args.noise)
        else:
            noise = None
        out1 = policy.infer(raw_infer, noise=noise) if noise is not None else policy.infer(raw_infer)
        pred = np.asarray(out1["actions"])

        # Basic shape check.
        if pred.shape != gt_abs.shape:
            print(f"[WARN] shape mismatch at idx={idx}: pred={pred.shape} gt={gt_abs.shape}")

        # Determinism check (only meaningful if we pass explicit noise for diffusion models).
        if is_diffusion and noise is not None:
            out2 = policy.infer(_deepcopy_for_infer(raw0), noise=noise)
            pred2 = np.asarray(out2["actions"])
            max_diff = float(np.max(np.abs(pred2 - pred)))
            if max_diff != 0.0:
                print(f"[WARN] non-deterministic with fixed noise at idx={idx}: max|Δ|={max_diff:.6g}")

        # Noise sensitivity check (optional, only for diffusion models).
        if args.noise_check and is_diffusion:
            noise_a = np.zeros((cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)
            noise_b = rng.normal(size=(cfg.model.action_horizon, cfg.model.action_dim)).astype(np.float32)
            pa = np.asarray(policy.infer(_deepcopy_for_infer(raw0), noise=noise_a)["actions"])
            pb = np.asarray(policy.infer(_deepcopy_for_infer(raw0), noise=noise_b)["actions"])
            print(
                f"[noise_check idx={idx}] mean|pa-pb|={float(np.mean(np.abs(pa-pb))):.6g} "
                f"max|pa-pb|={float(np.max(np.abs(pa-pb))):.6g}"
            )

        # Metrics in ABS space (expected for final output, matching dataset's absolute pos action).
        abs_mse = float(np.mean((pred - gt_abs) ** 2))
        abs_mae = float(np.mean(np.abs(pred - gt_abs)))
        abs_cos = _cosine_mean(pred, gt_abs)
        abs_ov = _overlap_at_eps(pred, gt_abs, args.eps)

        # Diagnostic: what if pred were actually delta? Compare against gt_delta.
        delta_mse = float(np.mean((pred - gt_delta) ** 2))

        # Diagnostic #2: if pred is delta, convert it to absolute using *raw* state and the known mask.
        # This helps distinguish "missing AbsoluteActions" vs "scale / norm_stats mismatch".
        pred_from_delta = np.array(pred, copy=True)
        dims = delta_mask.shape[0]
        pred_from_delta[..., :dims] += np.expand_dims(np.where(delta_mask, state_abs[:dims], 0.0), axis=-2)
        abs_from_delta_mse = float(np.mean((pred_from_delta - gt_abs) ** 2))

        abs_mse_list.append(abs_mse)
        abs_mae_list.append(abs_mae)
        abs_cos_list.append(abs_cos)
        abs_ov_list.append(abs_ov)
        delta_mse_list.append(delta_mse)

        # Per-dimension aggregates (over horizon).
        # We'll aggregate per-dim MAE/RMSE/overlap@eps across all sampled frames and all horizon steps.
        if n == 0:
            # Initialize once we know D from data.
            D = int(gt_abs.shape[1])
            per_dim_abs_err_sum = np.zeros((D,), dtype=np.float64)
            per_dim_sq_err_sum = np.zeros((D,), dtype=np.float64)
            per_dim_ov_count = np.zeros((D,), dtype=np.int64)
            per_dim_count = 0

        err = (pred - gt_abs)  # (H, D)
        per_dim_abs_err_sum += np.sum(np.abs(err), axis=0)
        per_dim_sq_err_sum += np.sum(err**2, axis=0)
        per_dim_ov_count += np.sum((np.abs(err) < args.eps), axis=0).astype(np.int64)
        per_dim_count += int(err.shape[0])

        # Print details for first few samples.
        if n < 5:
            print("-" * 80)
            print(f"[sample {n+1}/{len(chosen_indices)}] dataset_idx={idx} episode={_as_int(raw_gt['episode_index'])}")
            print(f"prompt={prompt!r}")
            print(_describe_array("state_abs", state_abs))
            print(_describe_array("gt_abs", gt_abs))
            print(_describe_array("gt_delta", gt_delta))
            print(_describe_array("pred", pred))
            if warm_start_idx is not None:
                print(f"warm_start_match_idx={warm_start_idx}")
            print(f"abs:   MSE={abs_mse:.6g} MAE={abs_mae:.6g} cos={abs_cos:.6g} overlap@{args.eps}={abs_ov:.6g}")
            print(f"delta: MSE(pred vs gt_delta)={delta_mse:.6g}  (diagnostic; abs should be smaller)")
            print(f"delta->abs: MSE((pred+state_masked) vs gt_abs)={abs_from_delta_mse:.6g}")
            if abs_from_delta_mse + 1e-12 < abs_mse:
                print("space_check: 预测更像 DELTA（把 pred 按 delta 加回 state 后更接近 GT abs）")
            else:
                print("space_check: 预测更像 ABS（直接 pred 更接近 GT abs；或 delta->abs 不成立）")

    # Aggregate metrics.
    def summarize(name: str, xs: list[float]) -> None:
        arr = np.asarray(xs, dtype=np.float64)
        print(
            f"{name}: mean={float(arr.mean()):.6g} std={float(arr.std()):.6g} "
            f"p50={float(np.percentile(arr, 50)):.6g} p90={float(np.percentile(arr, 90)):.6g} "
            f"min={float(arr.min()):.6g} max={float(arr.max()):.6g}"
        )

    print("=" * 80)
    print("Aggregate over selected frames:")
    summarize("ABS_MSE", abs_mse_list)
    summarize("ABS_MAE", abs_mae_list)
    summarize("ABS_COS", abs_cos_list)
    summarize(f"ABS_overlap@{args.eps}", abs_ov_list)
    summarize("DIAG_delta_MSE(pred vs gt_delta)", delta_mse_list)

    # Per-dimension report with joint names.
    action_names = _load_action_names(dataset_root)
    if "per_dim_count" in locals() and per_dim_count > 0:
        D = per_dim_abs_err_sum.shape[0]
        if action_names is None or len(action_names) != D:
            action_names = [f"dim_{i}" for i in range(D)]

        per_dim_mae = per_dim_abs_err_sum / float(per_dim_count)
        per_dim_rmse = np.sqrt(per_dim_sq_err_sum / float(per_dim_count))
        per_dim_ov = per_dim_ov_count / float(per_dim_count)

        print("-" * 80)
        print(f"Per-dimension metrics over {len(chosen_indices)} frames × {cfg.model.action_horizon} horizon steps:")
        print(f"{'name':<28} {'MAE':>12} {'RMSE':>12} {f'overlap@{args.eps}':>14}")
        for name, mae, rmse, ov in zip(action_names, per_dim_mae, per_dim_rmse, per_dim_ov, strict=True):
            print(f"{name:<28} {mae:>12.6g} {rmse:>12.6g} {ov:>14.6g}")

    print("=" * 80)


if __name__ == "__main__":
    main()

