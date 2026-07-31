import collections
import json
from pathlib import Path

import numpy as np


def analyze(path):
    d = dict(np.load(path))
    candidates = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for i, (task, state, candidate) in enumerate(
        zip(d["task_ids"], d["init_state_indices"], d["candidate_ids"], strict=True)
    ):
        key = (int(task), int(state), int(candidate))
        candidates[key].append(i)
        groups[(int(task), int(state))].append(i)
    candidate_rates = []
    agreements = []
    mixed_group_rates = []
    action_variances = []
    for group, indices in groups.items():
        labels = []
        actions = []
        for candidate in range(24):
            ids = candidates[(group[0], group[1], candidate)]
            y = d["successes"][ids]
            labels.append(float(y.mean()))
            agreements.append(float(y[0] == y[1]))
            actions.append(d["action_chunks"][ids[0]].reshape(-1))
        candidate_rates.extend(labels)
        mixed_group_rates.append(float(np.mean(labels)))
        action_variances.append(float(np.mean(np.var(np.stack(actions), axis=0))))
    return {
        "path": str(path),
        "tasks": sorted(set(map(int, d["task_ids"]))),
        "records": len(d["successes"]),
        "states": len(groups),
        "success_rate": float(d["successes"].mean()),
        "continuation_agreement": float(np.mean(agreements)),
        "candidate_success_rate": float(np.mean(candidate_rates)),
        "candidate_rate_std": float(np.std(candidate_rates)),
        "state_rate_std": float(np.std(mixed_group_rates)),
        "mean_action_variance": float(np.mean(action_variances)),
        "all_zero_or_one_candidate_groups": int(sum(x in (0.0, 1.0) for x in mixed_group_rates)),
    }


paths = [
    Path("data/recap_v2/counterfactual_task4_cross_candidates/initial_candidates.npz"),
    Path("data/recap_v2/counterfactual_task5_cross_candidates/initial_candidates.npz"),
    Path("data/recap_v2/counterfactual_task9_all_candidates_v2/initial_candidates.npz"),
]
print(json.dumps([analyze(path) for path in paths], indent=2))
