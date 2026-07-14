"""Compute RECAP N-step advantages from rollout rewards and learned values."""
import argparse
from collections import defaultdict
import pathlib
import pickle

import numpy as np

ROLLOUT_PATH  = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/all_episodes.pkl")
OUTPUT_PATH   = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/labeled_episodes.pkl")
STATS_PATH    = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/advantage_stats.txt")

N_STEP        = 50      # N-step return window
GAMMA         = 1.0     # discount (undiscounted per RECAP paper)
POSITIVE_FRACTION = 0.40

def compute_nstep_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    n: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (advantages, bootstrapped N-step returns) for one episode."""
    episode_length = len(rewards)
    returns = np.zeros(episode_length, dtype=np.float32)
    advantages = np.zeros(episode_length, dtype=np.float32)
    for t in range(episode_length):
        ret = 0.0
        discount = 1.0
        terminal = False
        for k in range(n):
            if t + k >= episode_length:
                break
            ret += discount * rewards[t + k]
            discount *= gamma
            if dones[t + k]:
                terminal = True
                break
        bootstrap_idx = t + n
        if not terminal and bootstrap_idx < episode_length:
            ret += discount * values[bootstrap_idx]
        returns[t] = ret
        advantages[t] = ret - values[t]
    return advantages, returns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=pathlib.Path, default=ROLLOUT_PATH)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_PATH)
    parser.add_argument("--stats", type=pathlib.Path, default=STATS_PATH)
    parser.add_argument("--value-key", default="value_predictions")
    parser.add_argument("--n-step", type=int, default=N_STEP)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--positive-fraction", type=float, default=POSITIVE_FRACTION)
    args = parser.parse_args()
    if not 0.0 < args.positive_fraction < 1.0:
        parser.error("--positive-fraction must be between 0 and 1")

    print(f"Loading rollouts from {args.rollouts}...")
    with open(args.rollouts, "rb") as f:
        episodes = pickle.load(f)

    print(f"  {len(episodes)} episodes loaded")
    n_succ = sum(1 for e in episodes if e["success"])
    print(f"  Success: {n_succ}/{len(episodes)} = {n_succ/len(episodes)*100:.1f}%")

    # ── compute N-step returns per episode ────────────────────────────────────
    print(f"\nComputing {args.n_step}-step advantages (gamma={args.gamma})...")
    all_advantages = []
    all_returns = []
    for ep_idx, ep in enumerate(episodes):
        rewards = ep["rewards"].astype(np.float32)
        dones   = ep["dones"].astype(bool)
        if args.value_key not in ep:
            raise KeyError(
                f"Episode {ep_idx} has no {args.value_key!r}; run learned value inference before labeling"
            )
        values = np.asarray(ep[args.value_key], dtype=np.float32)
        if values.shape != rewards.shape:
            raise ValueError(
                f"Episode {ep_idx} value shape {values.shape} != reward shape {rewards.shape}"
            )

        # terminal reward: +1 success, -1 failure (replace last step)
        # (env already gives +1 at success in done step; failures get 0)
        # add explicit -1 for failed episodes at the end
        if not ep["success"]:
            rewards[-1] = -1.0

        advantages, returns = compute_nstep_advantages(
            rewards, dones, values, args.n_step, args.gamma
        )
        ep["returns"] = returns
        ep["advantages"] = advantages
        all_advantages.append(advantages)
        all_returns.append(returns)

    all_advantages_flat = np.concatenate(all_advantages)
    all_returns_flat = np.concatenate(all_returns)
    print(f"  Returns: min={all_returns_flat.min():.3f} max={all_returns_flat.max():.3f} "
          f"mean={all_returns_flat.mean():.3f} std={all_returns_flat.std():.3f}")

    # ── binarize at 40th percentile ──────��────────────────────────────────────
    threshold_pct = (1.0 - args.positive_fraction) * 100.0
    threshold = float(np.percentile(all_advantages_flat, threshold_pct))
    print(f"\nBinarizing advantages at {threshold_pct:.0f}th percentile = {threshold:.4f}")

    n_positive = 0
    n_total    = 0
    task_stats = defaultdict(lambda: {"pos": 0, "total": 0})

    for ep in episodes:
        labels = (ep["advantages"] > threshold).astype(np.int8)
        ep["advantage_labels"] = labels
        n_positive += labels.sum()
        n_total    += len(labels)
        task_stats[ep["task_id"]]["pos"]   += labels.sum()
        task_stats[ep["task_id"]]["total"] += len(labels)

    pos_rate = n_positive / n_total
    print(f"  Positive labels: {n_positive}/{n_total} = {pos_rate*100:.1f}%")

    # ── per-task stats ────────────────────────────────────────────────────────
    lines = []
    lines.append("=== Advantage Labeling Stats ===")
    lines.append(
        f"N_STEP={args.n_step}, GAMMA={args.gamma}, POSITIVE_FRACTION={args.positive_fraction:.2f}"
    )
    lines.append(f"Threshold value: {threshold:.4f}")
    lines.append(f"Overall positive rate: {pos_rate*100:.1f}% ({n_positive}/{n_total})")
    lines.append("\nPer-task success rate:")
    for tid in sorted(task_stats):
        ep_list = [e for e in episodes if e["task_id"] == tid]
        task_succ = sum(1 for e in ep_list if e["success"])
        s = task_stats[tid]
        name = ep_list[0]["task_name"][:50] if ep_list else f"task_{tid}"
        lines.append(f"  task{tid}: succ={task_succ}/{len(ep_list)} "
                     f"pos_labels={s['pos']}/{s['total']}={s['pos']/s['total']*100:.1f}%  {name}")

    report = "\n".join(lines)
    print("\n" + report)

    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(report)
    print(f"\nStats saved to {args.stats}")

    # ── save labeled episodes ─────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(episodes, f)
    print(f"Labeled episodes saved to {args.output}")
    print(f"  Size: {args.output.stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
