"""
分析 eepose14 评测日志中的旋转误差：
- 从 per-dim 的 MAE/RMSE 数据重新计算角度误差
- 直接读 parquet 数据对一批帧做四元数角度偏差分析
用法: python analyze_rotation.py --dataset-root ... --ckpt ... --num-episodes 30
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import pathlib
import numpy as np
import sys
import os

# Disable all network calls before importing heavy deps
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# 把 openpi 加入路径
script_dir = pathlib.Path(__file__).parent
openpi_root = script_dir.parent
sys.path.insert(0, str(openpi_root / "src"))

from openpi import transforms
from openpi.models import model as model_lib
from openpi.policies import policy_config as policy_config_lib
from openpi.shared import normalize as normalize_lib
from openpi.training import config as train_config_lib
from openpi.training import data_loader as data_loader_lib


def quat_angle_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> np.ndarray:
    """
    计算两个四元数序列的旋转角度误差（度）。
    q_pred, q_gt: (..., 4) 格式 [qx, qy, qz, qw]
    返回 (...,) 角度误差（degree）
    """
    # 归一化
    q_pred = q_pred / (np.linalg.norm(q_pred, axis=-1, keepdims=True) + 1e-8)
    q_gt   = q_gt   / (np.linalg.norm(q_gt,   axis=-1, keepdims=True) + 1e-8)
    # |dot| -> angle = 2*arccos(|dot|)
    dot = np.abs(np.sum(q_pred * q_gt, axis=-1)).clip(0, 1)
    angle_rad = 2 * np.arccos(dot)
    return np.degrees(angle_rad)


def pos_error_mm(p_pred: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    """L2 位置误差 (mm)"""
    return np.linalg.norm(p_pred - p_gt, axis=-1) * 1000.0


def _deepcopy(x):
    if isinstance(x, dict):  return {k: _deepcopy(v) for k, v in x.items()}
    if isinstance(x, list):  return [_deepcopy(v) for v in x]
    if isinstance(x, tuple): return tuple(_deepcopy(v) for v in x)
    if isinstance(x, (str, int, float, bool)) or x is None: return x
    try: return np.asarray(x).copy()
    except: return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-episodes", default=30, type=int)
    ap.add_argument("--frames-per-episode", default=3, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--label", default="", type=str, help="checkpoint label for output")
    args = ap.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    ckpt_dir = pathlib.Path(args.ckpt)
    label = args.label or ckpt_dir.name

    cfg0 = train_config_lib.get_config("robot_arm14_eepose_action")
    cfg = dataclasses.replace(cfg0, data=dataclasses.replace(cfg0.data, repo_id=str(dataset_root)))
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)

    print(f"[{label}] loading dataset...", flush=True)
    ds = data_loader_lib.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)

    # 读 episode 信息
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    total_episodes = info["total_episodes"]

    ep_lengths: dict[int, int] = {}
    with (dataset_root / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            obj = json.loads(line)
            ep_lengths[int(obj["episode_index"])] = int(obj["length"])

    ep_starts: dict[int, int] = {}
    offset = 0
    for ep in sorted(ep_lengths):
        ep_starts[ep] = offset
        offset += ep_lengths[ep]

    rng = np.random.default_rng(args.seed)
    ep_pool = list(range(total_episodes))
    rng.shuffle(ep_pool)
    chosen_eps = ep_pool[:args.num_episodes]

    chosen_indices = []
    for ep in sorted(chosen_eps):
        start = ep_starts[ep]
        length = ep_lengths[ep]
        max_start = start + length - cfg.model.action_horizon
        if max_start < start:
            max_start = start + length - 1
        if args.frames_per_episode <= 1:
            chosen_indices.append(int(rng.integers(start, max_start + 1)))
        else:
            chosen_indices.extend([int(x) for x in np.linspace(start, max_start, args.frames_per_episode)])

    norm_stats = normalize_lib.load(dataset_root)
    repack_fn = transforms.compose(data_cfg.repack_transforms.inputs)

    print(f"[{label}] loading policy...", flush=True)
    policy = policy_config_lib.create_trained_policy(
        cfg, ckpt_dir,
        repack_transforms=data_cfg.repack_transforms,
        norm_stats=norm_stats,
    )
    print(f"[{label}] policy ready, running {len(chosen_indices)} frames...", flush=True)

    # 收集误差
    left_pos_err_mm_list  = []  # per-frame L2 position error (mm), shape (H,)
    right_pos_err_mm_list = []
    left_rot_err_deg_list  = []  # per-frame angle error (deg), shape (H,)
    right_rot_err_deg_list = []

    noise = np.zeros((cfg.model.action_horizon, cfg.model.action_dim), dtype=np.float32)

    for idx in chosen_indices:
        raw = ds[idx]
        raw_gt = _deepcopy(raw)
        raw_infer = _deepcopy(raw)

        gt = np.asarray(raw_gt["action"]).copy()  # (H, 14)
        out = policy.infer(raw_infer, noise=noise)
        pred = np.asarray(out["actions"])          # (H, 14)

        H = gt.shape[0]
        # layout: left(tx,ty,tz,qx,qy,qz,qw) right(tx,ty,tz,qx,qy,qz,qw)
        left_pos_err_mm_list.append(pos_error_mm(pred[:, 0:3], gt[:, 0:3]))    # (H,)
        right_pos_err_mm_list.append(pos_error_mm(pred[:, 7:10], gt[:, 7:10]))
        left_rot_err_deg_list.append(quat_angle_error_deg(pred[:, 3:7], gt[:, 3:7]))
        right_rot_err_deg_list.append(quat_angle_error_deg(pred[:, 10:14], gt[:, 10:14]))

    # 拼成 (N*H,)
    lp = np.concatenate(left_pos_err_mm_list)
    rp = np.concatenate(right_pos_err_mm_list)
    lr = np.concatenate(left_rot_err_deg_list)
    rr = np.concatenate(right_rot_err_deg_list)

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
            "within_5mm_or_5deg": float(np.mean(arr < 5)),
            "within_10mm_or_10deg": float(np.mean(arr < 10)),
        }

    print(f"\n{'='*70}")
    print(f"Checkpoint: {label}")
    print(f"Frames evaluated: {len(chosen_indices)} × horizon {H} = {len(lp)} samples")
    print(f"{'='*70}")

    for name, arr, unit, t1, t2 in [
        ("left_pos",  lp, "mm",  5,  10),
        ("right_pos", rp, "mm",  5,  10),
        ("left_rot",  lr, "deg", 5,  15),
        ("right_rot", rr, "deg", 5,  15),
    ]:
        s = stats(arr)
        within1 = float(np.mean(arr < t1)) * 100
        within2 = float(np.mean(arr < t2)) * 100
        print(f"\n{name} ({unit}):")
        print(f"  mean={s['mean']:.3f}  median={s['median']:.3f}  p90={s['p90']:.3f}  p95={s['p95']:.3f}  max={s['max']:.3f}")
        print(f"  within {t1}{unit}: {within1:.1f}%   within {t2}{unit}: {within2:.1f}%")

    print(f"\n{'='*70}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
