import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.recap.train_value_prototype import ValuePrototype, RolloutFrames, load_split_references, predict_outputs, multitask_metrics

root = Path("data/recap_v2/sft_protocol_v1_200_seed1")
meta = root / "metadata_grouped"
tasks = [4]
dataset = RolloutFrames(root, load_split_references(meta, "test", tasks), 10, 112)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logits = []
for seed in range(4):
    model = ValuePrototype(dataset[0]["state"].numel(), num_tasks=5).to(device)
    checkpoint = torch.load(f"checkpoints/value_prototype_task4_only_seed{seed}/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    labels, values, _, task_ids, episode_ids = predict_outputs(model, loader, device)
    logits.append(values)
ensemble = np.mean(logits, axis=0)
report = multitask_metrics(labels, 1.0 / (1.0 + np.exp(-ensemble)), episode_ids, task_ids, {4: float(labels.mean())})
print(json.dumps(report, indent=2, allow_nan=True))
