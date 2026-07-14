# RECAP experiment history

`arfm_eval_history.tsv` preserves the checkpoint sweeps used during early ARFM exploration.

These rows are historical evidence, not publication-ready comparisons. Most were collected with five episodes per task, `replan_steps=10`, and the legacy action RNG encoding. The canonical protocol now lives in `src/openpi/recap/evaluation_protocol.py` and uses:

- LIBERO-Spatial with 50 episodes per task
- five evaluation seeds
- 10 stabilization steps and a 220-step horizon
- `replan_steps=5`, matching the upstream LIBERO evaluator
- collision-free action RNG seeds

Do not compare historical percentages directly against results produced by the canonical protocol. New result summaries should record the checkpoint, training seed, evaluation seed, episode count, protocol version, and per-task successes.
