import collections
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scripts.recap.train_value_prototype import ValuePrototype

data = dict(np.load("data/recap_v2/value_guided_task4_hard_candidates/initial_candidates.npz"))
image = np.concatenate([data["post_images"], data["post_wrist_images"]], axis=-1)
image = torch.as_tensor(image).permute(0, 3, 1, 2).float() / 255.0
state = torch.as_tensor(data["post_states"]).float()
progress = torch.full((len(state),), 5.0 / 220.0)
task = torch.full((len(state),), 4, dtype=torch.long)
loader = DataLoader(TensorDataset(image, state, progress, task), batch_size=64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions = []
for seed in range(4):
    model = ValuePrototype(8, num_tasks=5).to(device)
    checkpoint = torch.load(f"checkpoints/value_prototype_task4_only_seed{seed}/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            values.append(torch.sigmoid(model(*(x.to(device) for x in batch))).cpu().numpy())
    predictions.append(np.concatenate(values))
score = np.mean(predictions, axis=0)
groups = collections.defaultdict(list)
for i, (state_id, candidate) in enumerate(zip(data["init_state_indices"], data["candidate_ids"], strict=True)):
    groups[(int(state_id), int(candidate))].append(i)
by_state = collections.defaultdict(list)
for (state_id, candidate), indices in groups.items():
    by_state[state_id].append((candidate, float(np.mean(score[indices])), float(np.mean(data["successes"][indices]))))
rng = np.random.default_rng(20260804)
rows = []
for state_id, candidates in sorted(by_state.items()):
    selected = max(candidates, key=lambda x: x[1])
    random_candidate = candidates[int(rng.integers(len(candidates)))]
    baseline = next(row for row in candidates if row[0] == 0)
    rows.append({"state": state_id, "selected": selected[2], "random": random_candidate[2], "candidate0": baseline[2], "selected_candidate": selected[0]})
    print("state", state_id, "scores", [(c, round(s, 4), y) for c, s, y in sorted(candidates, key=lambda x: -x[1])[:8]], "score_range", round(max(x[1] for x in candidates) - min(x[1] for x in candidates), 6))
print(json.dumps({"states": rows, "selected_mean": float(np.mean([r["selected"] for r in rows])), "random_mean": float(np.mean([r["random"] for r in rows])), "candidate0_mean": float(np.mean([r["candidate0"] for r in rows]))}, indent=2))
