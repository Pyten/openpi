import json
from pathlib import Path
import numpy as np

root = Path("data/recap_v2")
paths = [
    root / "counterfactual_task4_cross_candidates/initial_candidates.npz",
    root / "counterfactual_task5_cross_candidates/initial_candidates.npz",
    root / "counterfactual_task9_all_candidates_v2/initial_candidates.npz",
]
data = [dict(np.load(path)) for path in paths]
keys = tuple(data[0])
assert all(tuple(d) == keys for d in data)
out = root / "counterfactual_cross_task_all_candidates"
out.mkdir(parents=True, exist_ok=True)
merged = {key: np.concatenate([d[key] for d in data], axis=0) for key in keys}
np.savez_compressed(out / "initial_candidates.npz", **merged)
manifest = {
    "sources": [str(path) for path in paths],
    "records": int(len(merged["successes"])),
    "tasks": {str(int(task)): int((merged["task_ids"] == task).sum()) for task in np.unique(merged["task_ids"])},
    "groups": int(len(set(zip(merged["task_ids"].tolist(), merged["init_state_indices"].tolist())))),
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
