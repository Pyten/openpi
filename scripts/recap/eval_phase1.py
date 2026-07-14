"""
Phase 1 批量 eval 脚本：支持单卡和多卡并行评测。

单卡用法:
  CUDA_VISIBLE_DEVICES=0 python eval_phase1.py --exp-dir ... --sft-ckpt ...

多卡用法（自动按 GPU 数拆分 10 个 task）:
  python eval_phase1.py --exp-dir ... --sft-ckpt ... --num-gpus 4
"""
import sys as _sys, types as _types
if 'numba' not in _sys.modules:
    _m = _types.ModuleType('numba'); _m.jit = lambda *a,**kw:(lambda f:f); _sys.modules['numba']=_m

import argparse, json, math, os, pathlib, subprocess, sys, tempfile, time
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
os.environ.setdefault("MUJOCO_GL", "osmesa")

from openpi.recap import evaluation_protocol as protocol

ACTION_HORIZON = protocol.ACTION_HORIZON; ACTION_DIM = protocol.ACTION_DIM
REPLAN_STEPS = protocol.REPLAN_STEPS; MAX_STEPS = protocol.MAX_STEPS
SUITE_NAME = protocol.SUITE_NAME; RESIZE = protocol.RESIZE


def _quat2axisangle(q):
    if q[3] > 1: q[3] = 1
    elif q[3] < -1: q[3] = -1
    d = math.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if math.isclose(d, 0) else q[:3] * 2 * math.acos(q[3]) / d


def _prep_obs(raw, r):
    from openpi_client import image_tools
    img = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, r, r))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, r, r))
    state = np.concatenate((raw["robot0_eef_pos"], _quat2axisangle(raw["robot0_eef_quat"]),
                            raw["robot0_gripper_qpos"]))
    return img, wrist, state


def load_model(ckpt_dir, train_config):
    import jax, jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as _model
    model = train_config.model.create(jax.random.PRNGKey(0))
    p = pathlib.Path(ckpt_dir)
    mp = p / "model_params"
    if mp.exists():
        import orbax.checkpoint as ocp
        st = nnx.state(model); ckptr = ocp.StandardCheckpointer()
        nnx.update(model, ckptr.restore(str(mp.resolve()), target=st))
        return model
    raw = _model.restore_params(str(p / "params" if (p / "params").exists() else p),
                                restore_type=np.ndarray)
    gd, st = nnx.split(model); st.replace_by_pure_dict(raw); return nnx.merge(gd, st)


def eval_task(task_id, name, desc, bddl, init_states, infer_fn, tokenize, norm_stats,
              n_ep, seed, label):
    import jax, jax.numpy as jnp
    from libero.libero.envs import OffScreenRenderEnv
    sm = norm_stats["state"].mean.astype(np.float32)
    ss = norm_stats["state"].std.astype(np.float32)
    am = norm_stats["actions"].mean.astype(np.float32)
    astd = norm_stats["actions"].std.astype(np.float32)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(seed)
    succ = 0
    for ep in range(n_ep):
        env.reset(); env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(protocol.NUM_WAIT_STEPS): obs, _, _, _ = env.step([0.] * 6 + [-1.])
        tokens, tmask = tokenize(desc); chunk = None; cs = REPLAN_STEPS
        for si in range(MAX_STEPS):
            if cs >= REPLAN_STEPS or chunk is None:
                img, wrist, state = _prep_obs(obs, RESIZE)
                sn = (state - sm) / (ss + 1e-6)
                sp = (np.concatenate([sn, np.zeros(ACTION_DIM - len(sn))])
                      if len(sn) < ACTION_DIM else sn)
                observation_data = {
                    "images": {
                        "base_0_rgb": jnp.array(img)[None].astype(jnp.float32) / 127.5 - 1,
                        "left_wrist_0_rgb": jnp.array(wrist)[None].astype(jnp.float32) / 127.5 - 1,
                        "right_wrist_0_rgb": jnp.zeros((1, RESIZE, RESIZE, 3), dtype=jnp.float32),
                    },
                    "image_masks": {
                        "base_0_rgb": jnp.ones(1, dtype=bool),
                        "left_wrist_0_rgb": jnp.ones(1, dtype=bool),
                        "right_wrist_0_rgb": jnp.zeros(1, dtype=bool),
                    },
                    "state": jnp.array(sp)[None].astype(jnp.float32),
                    "tokenized_prompt": jnp.array(tokens),
                    "tokenized_prompt_mask": jnp.array(tmask),
                }
                from openpi.models import model as _model
                observation = _model.Observation(**observation_data)
                rng = jax.random.PRNGKey(protocol.action_rng_seed(seed, ep, si))
                raw = np.array(infer_fn(rng, observation)); a7 = raw[0, :, :7]
                chunk = a7 * (astd + 1e-6) + am
                msk = np.array([True] * 6 + [False])
                chunk[:, :7] += np.where(msk, state[:7], 0); cs = 0
            obs, reward, done, info = env.step(chunk[cs].tolist()); cs += 1
            if done: break
        ok = bool(info.get("success", reward > 0.5)); succ += int(ok)
        print(f"  [{label}] task{task_id} ep{ep+1:02d}/{n_ep} "
              f"{'SUCCESS' if ok else 'FAIL'} (total {succ}/{ep+1})", flush=True)
    env.close()
    return succ, n_ep


def run_worker(args_ns, task_ids, gpu_id, result_file):
    """Run eval on specified task_ids on given GPU, write JSON results."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, __file__,
        "--exp-dir", args_ns.exp_dir,
        "--sft-ckpt", args_ns.sft_ckpt,
        "--episodes-per-task", str(args_ns.episodes_per_task),
        "--seed", str(args_ns.seed),
        "--task-ids", ",".join(map(str, task_ids)),
        "--result-file", result_file,
        "--num-gpus", "1",  # worker runs single-GPU
    ]
    if args_ns.steps:
        cmd += ["--steps", args_ns.steps]
    return subprocess.Popen(cmd, env=env)


def eval_single_gpu(args):
    """Single-GPU eval: evaluate specified task_ids for all checkpoints."""
    import jax
    from openpi.training import config as _config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import checkpoints as _ckpts
    from openpi.shared import nnx_utils
    from libero.libero import benchmark as _bench, get_libero_path

    print(f"JAX: {jax.device_count()} x {jax.devices()[0].device_kind}", flush=True)
    jax.config.update("jax_compilation_cache_dir",
                      str(pathlib.Path("~/.cache/jax").expanduser()))

    exp_dir = pathlib.Path(args.exp_dir)
    # Collect checkpoints
    if args.steps:
        step_list = [int(s.strip()) for s in args.steps.split(",")]
        ckpt_dirs = [(s, exp_dir / str(s)) for s in step_list if (exp_dir / str(s)).exists()]
    else:
        step_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                           key=lambda d: int(d.name))
        ckpt_dirs = [(int(d.name), d) for d in step_dirs]

    print(f"Found {len(ckpt_dirs)} checkpoints: {[s for s,_ in ckpt_dirs]}", flush=True)

    train_config = _config.get_config("pi0_libero")
    tok = PaligemmaTokenizer(max_len=train_config.model.max_token_len)
    def tokenize(prompt):
        t, m = tok.tokenize(prompt)
        return np.array(t)[None], np.array(m, dtype=bool)[None]

    norm_stats = _ckpts.load_norm_stats(
        pathlib.Path(args.sft_ckpt) / "assets/physical-intelligence", "libero")

    suite = _bench.get_benchmark_dict()[SUITE_NAME]()
    all_task_ids = list(range(suite.n_tasks))
    # Filter to requested task_ids
    task_ids_filter = (set(int(x) for x in args.task_ids.split(","))
                       if args.task_ids else set(all_task_ids))

    task_infos = []
    for i in range(suite.n_tasks):
        if i not in task_ids_filter:
            continue
        t = suite.get_task(i)
        bddl = str(pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file)
        init = torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder,
                                       t.init_states_file), weights_only=False)
        task_infos.append(dict(id=i, name=t.name, desc=t.language, bddl=bddl, init=init))

    # Results: {step: {task_id: success_rate}}
    results = {}

    for step, ckpt_path in ckpt_dirs:
        print(f"\n{'='*60}\nEvaluating step={step}: {ckpt_path}", flush=True)
        t_start = time.time()
        model = load_model(str(ckpt_path), train_config)
        model.eval()
        infer_fn = nnx_utils.module_jit(model.sample_actions)
        step_results = {}
        for ti in task_infos:
            print(f"\n  Task {ti['id']}: {ti['name']}", flush=True)
            s, total = eval_task(ti['id'], ti['name'], ti['desc'], ti['bddl'], ti['init'],
                                 infer_fn, tokenize, norm_stats,
                                 args.episodes_per_task, args.seed, f"step{step}")
            step_results[ti['id']] = s / total
            print(f"  Task {ti['id']}: {s}/{total} = {s/total*100:.1f}%", flush=True)
        elapsed = time.time() - t_start
        results[step] = step_results
        print(f"\n[step={step}] tasks={list(step_results.keys())} "
              f"avg={sum(step_results.values())/len(step_results)*100:.1f}% "
              f"({elapsed/60:.1f} min)", flush=True)

    if args.result_file:
        with open(args.result_file, "w") as f:
            json.dump(results, f)
        print(f"Results written to {args.result_file}", flush=True)

    return results


def aggregate_and_print(all_results, exp_name, n_tasks=10):
    """Merge results from multiple workers and print summary table."""
    # all_results: list of {step: {task_id: rate}}
    merged = {}
    for worker_res in all_results:
        for step, task_res in worker_res.items():
            step = int(step)
            if step not in merged:
                merged[step] = {}
            merged[step].update({int(k): v for k, v in task_res.items()})

    summary = []
    for step in sorted(merged.keys()):
        task_res = merged[step]
        avg = sum(task_res.values()) / n_tasks
        summary.append((step, avg, task_res))

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL SUMMARY: {exp_name}", flush=True)
    hdr = f"{'Step':>8}  {'AVG':>7}  " + "  ".join(f"T{i:02d}" for i in range(n_tasks))
    print(hdr, flush=True)
    for step, avg, task_res in summary:
        per_task = "  ".join(f"{task_res.get(i, float('nan'))*100:>4.0f}%" for i in range(n_tasks))
        flag = " <-- best" if avg == max(s[1] for s in summary) else ""
        print(f"{step:>8}  {avg*100:>6.1f}%  {per_task}{flag}", flush=True)

    best_step, best_avg, _ = max(summary, key=lambda x: x[1])
    print(f"\nBest: step={best_step}, avg={best_avg*100:.1f}%", flush=True)
    print("Compare against an SFT run produced by the same frozen protocol.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--sft-ckpt", required=True)
    parser.add_argument("--episodes-per-task", type=int, default=protocol.EPISODES_PER_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use in parallel (default: 1)")
    # Internal args used by worker subprocesses
    parser.add_argument("--task-ids", type=str, default="",
                        help="Comma-separated task IDs for this worker")
    parser.add_argument("--result-file", type=str, default="",
                        help="JSON file to write worker results")
    args = parser.parse_args()

    exp_dir = pathlib.Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"ERROR: {exp_dir} does not exist"); sys.exit(1)

    if args.num_gpus <= 1:
        # Single GPU mode
        results = eval_single_gpu(args)
        if not args.result_file:
            # Top-level call: print full summary
            aggregate_and_print([results], exp_dir.name)
    else:
        # Multi-GPU mode: split 10 tasks across GPUs
        n_tasks = 10
        # Build task groups
        base, extra = divmod(n_tasks, args.num_gpus)
        groups, start = [], 0
        for i in range(args.num_gpus):
            size = base + (1 if i < extra else 0)
            groups.append(list(range(start, start + size)))
            start += size

        print(f"Multi-GPU eval: {args.num_gpus} GPUs, task splits: {groups}", flush=True)

        tmpdir = tempfile.mkdtemp()
        procs, result_files = [], []
        for gpu_id, task_group in enumerate(groups):
            if not task_group:
                continue
            rf = os.path.join(tmpdir, f"gpu{gpu_id}.json")
            result_files.append(rf)
            p = run_worker(args, task_group, gpu_id, rf)
            procs.append((gpu_id, p))
            print(f"  GPU {gpu_id}: tasks {task_group} -> PID {p.pid}", flush=True)

        # Wait for all workers
        for gpu_id, p in procs:
            ret = p.wait()
            print(f"  GPU {gpu_id} worker exited: code={ret}", flush=True)

        # Aggregate
        all_results = []
        for rf in result_files:
            if os.path.exists(rf):
                with open(rf) as f:
                    all_results.append(json.load(f))
            else:
                print(f"WARNING: result file missing: {rf}", flush=True)

        aggregate_and_print(all_results, exp_dir.name)


if __name__ == "__main__":
    main()
