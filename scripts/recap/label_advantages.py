"""
Phase 1B: 优势计算 + 标注
RECAP paper: N-step returns, γ=1.0, 40% quantile binarization
"""
import pickle
import numpy as np
import pathlib
from collections import defaultdict

ROLLOUT_PATH  = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/all_episodes.pkl")
OUTPUT_PATH   = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/labeled_episodes.pkl")
STATS_PATH    = pathlib.Path("/mnt/vepfs/pyten/Programs/code/pi0.6/data/rollouts/advantage_stats.txt")

N_STEP        = 50      # N-step return window
GAMMA         = 1.0     # discount (undiscounted per RECAP paper)
ADV_THRESHOLD = 0.40    # binarize at 40th percentile

def compute_nstep_returns(rewards: np.ndarray, dones: np.ndarray, n: int, gamma: float) -> np.ndarray:
    """Compute N-step returns for each timestep."""
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    for t in range(T):
        ret = 0.0
        discount = 1.0
        for k in range(n):
            if t + k >= T:
                break
            ret += discount * rewards[t + k]
            discount *= gamma
            if dones[t + k]:
                break
        returns[t] = ret
    return returns

def main():
    print(f"Loading rollouts from {ROLLOUT_PATH}...")
    with open(ROLLOUT_PATH, "rb") as f:
        episodes = pickle.load(f)

    print(f"  {len(episodes)} episodes loaded")
    n_succ = sum(1 for e in episodes if e["success"])
    print(f"  Success: {n_succ}/{len(episodes)} = {n_succ/len(episodes)*100:.1f}%")

    # ── compute N-step returns per episode ────────────────────────────────────
    print(f"\nComputing {N_STEP}-step returns (γ={GAMMA})...")
    all_returns = []
    for ep in episodes:
        rewards = ep["rewards"].astype(np.float32)
        dones   = ep["dones"].astype(bool)

        # terminal reward: +1 success, -1 failure (replace last step)
        # (env already gives +1 at success in done step; failures get 0)
        # add explicit -1 for failed episodes at the end
        if not ep["success"]:
            rewards[-1] = -1.0

        returns = compute_nstep_returns(rewards, dones, N_STEP, GAMMA)
        ep["returns"] = returns
        all_returns.append(returns)

    all_returns_flat = np.concatenate(all_returns)
    print(f"  Returns: min={all_returns_flat.min():.3f} max={all_returns_flat.max():.3f} "
          f"mean={all_returns_flat.mean():.3f} std={all_returns_flat.std():.3f}")

    # ── binarize at 40th percentile ──────��────────────────────────────────────
    threshold = float(np.percentile(all_returns_flat, ADV_THRESHOLD * 100))
    print(f"\nBinarizing at {ADV_THRESHOLD*100:.0f}th percentile = {threshold:.4f}")

    n_positive = 0
    n_total    = 0
    task_stats = defaultdict(lambda: {"pos": 0, "total": 0})

    for ep in episodes:
        labels = (ep["returns"] > threshold).astype(np.int8)
        ep["advantage_labels"] = labels
        n_positive += labels.sum()
        n_total    += len(labels)
        task_stats[ep["task_id"]]["pos"]   += labels.sum()
        task_stats[ep["task_id"]]["total"] += len(labels)

    pos_rate = n_positive / n_total
    print(f"  Positive labels: {n_positive}/{n_total} = {pos_rate*100:.1f}%")

    # ── per-task stats ────────────────────────────────────────────────────────
    lines = []
    lines.append(f"=== Advantage Labeling Stats ===")
    lines.append(f"N_STEP={N_STEP}, GAMMA={GAMMA}, THRESHOLD_PCT={ADV_THRESHOLD*100:.0f}%")
    lines.append(f"Threshold value: {threshold:.4f}")
    lines.append(f"Overall positive rate: {pos_rate*100:.1f}% ({n_positive}/{n_total})")
    lines.append(f"\nPer-task success rate:")
    for tid in sorted(task_stats):
        ep_list = [e for e in episodes if e["task_id"] == tid]
        task_succ = sum(1 for e in ep_list if e["success"])
        s = task_stats[tid]
        name = ep_list[0]["task_name"][:50] if ep_list else f"task_{tid}"
        lines.append(f"  task{tid}: succ={task_succ}/{len(ep_list)} "
                     f"pos_labels={s['pos']}/{s['total']}={s['pos']/s['total']*100:.1f}%  {name}")

    report = "\n".join(lines)
    print("\n" + report)

    STATS_PATH.write_text(report)
    print(f"\nStats saved to {STATS_PATH}")

    # ── save labeled episodes ─────────────────────────────────────────────────
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(episodes, f)
    print(f"Labeled episodes saved to {OUTPUT_PATH}")
    print(f"  Size: {OUTPUT_PATH.stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
