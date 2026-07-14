"""Evaluate ARFM fine-tuned policy on LIBERO spatial (standard SFT inference, no CFG)."""
import sys as _sys, types as _types
if 'numba' not in _sys.modules:
    _m = _types.ModuleType('numba'); _m.jit = lambda *a,**kw:(lambda f:f); _sys.modules['numba']=_m

import argparse, math, os, pathlib, sys, time
import jax, jax.numpy as jnp, numpy as np, torch
from flax import nnx
sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))
os.environ.setdefault("MUJOCO_GL","osmesa")

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.shared import nnx_utils
from openpi.recap import evaluation_protocol as protocol

ACTION_HORIZON=protocol.ACTION_HORIZON; ACTION_DIM=protocol.ACTION_DIM
REPLAN_STEPS=protocol.REPLAN_STEPS; MAX_STEPS=protocol.MAX_STEPS
SUITE_NAME=protocol.SUITE_NAME; RESIZE=protocol.RESIZE

def _quat2axisangle(q):
    if q[3]>1: q[3]=1
    elif q[3]<-1: q[3]=-1
    d=math.sqrt(1-q[3]**2)
    return np.zeros(3) if math.isclose(d,0) else q[:3]*2*math.acos(q[3])/d

def _prep_obs(raw,r):
    from openpi_client import image_tools
    img=np.ascontiguousarray(raw["agentview_image"][::-1,::-1])
    wrist=np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1,::-1])
    img=image_tools.convert_to_uint8(image_tools.resize_with_pad(img,r,r))
    wrist=image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist,r,r))
    state=np.concatenate((raw["robot0_eef_pos"],_quat2axisangle(raw["robot0_eef_quat"]),raw["robot0_gripper_qpos"]))
    return img,wrist,state

def load_model(ckpt_dir, train_config):
    model=train_config.model.create(jax.random.PRNGKey(0))
    p=pathlib.Path(ckpt_dir)
    mp=p/"model_params"
    if mp.exists():
        import orbax.checkpoint as ocp
        st=nnx.state(model); ckptr=ocp.StandardCheckpointer()
        nnx.update(model, ckptr.restore(str(mp.resolve()), target=st))
        return model
    raw=_model.restore_params(str(p/"params" if (p/"params").exists() else p), restore_type=np.ndarray)
    gd,st=nnx.split(model); st.replace_by_pure_dict(raw); return nnx.merge(gd,st)

def make_tokenize_fn(train_config):
    from openpi.models.tokenizer import PaligemmaTokenizer
    tok=PaligemmaTokenizer(max_len=train_config.model.max_token_len)
    def f(prompt): t,m=tok.tokenize(prompt); return np.array(t)[None],np.array(m,dtype=bool)[None]
    return f

def eval_task(task_id, name, desc, bddl, init_states, infer_fn, tokenize, norm_stats, n_ep, seed, label):
    from libero.libero.envs import OffScreenRenderEnv
    sm=norm_stats["state"].mean.astype(np.float32); ss=norm_stats["state"].std.astype(np.float32)
    am=norm_stats["actions"].mean.astype(np.float32); astd=norm_stats["actions"].std.astype(np.float32)
    env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256); env.seed(seed)
    succ=0
    for ep in range(n_ep):
        env.reset(); env.set_init_state(init_states[ep%len(init_states)])
        for _ in range(protocol.NUM_WAIT_STEPS): obs,_,_,_=env.step([0.]*6+[-1.])
        tokens,tmask=tokenize(desc); chunk=None; cs=REPLAN_STEPS
        for si in range(MAX_STEPS):
            if cs>=REPLAN_STEPS or chunk is None:
                img,wrist,state=_prep_obs(obs,RESIZE)
                sn=(state-sm)/(ss+1e-6)
                sp=np.concatenate([sn,np.zeros(ACTION_DIM-len(sn))]) if len(sn)<ACTION_DIM else sn
                observation=_model.Observation(
                    images={"base_0_rgb":jnp.array(img)[None].astype(jnp.float32)/127.5-1,
                            "left_wrist_0_rgb":jnp.array(wrist)[None].astype(jnp.float32)/127.5-1,
                            "right_wrist_0_rgb":jnp.zeros((1,RESIZE,RESIZE,3),dtype=jnp.float32)},
                    image_masks={"base_0_rgb":jnp.ones(1,dtype=bool),"left_wrist_0_rgb":jnp.ones(1,dtype=bool),"right_wrist_0_rgb":jnp.zeros(1,dtype=bool)},
                    state=jnp.array(sp)[None].astype(jnp.float32),
                    tokenized_prompt=jnp.array(tokens),tokenized_prompt_mask=jnp.array(tmask))
                rng=jax.random.PRNGKey(protocol.action_rng_seed(seed,ep,si))
                raw=np.array(infer_fn(rng,observation)); a7=raw[0,:,:7]
                chunk=a7*(astd+1e-6)+am; msk=np.array([True]*6+[False]); chunk[:,:7]+=np.where(msk,state[:7],0); cs=0
            obs,reward,done,info=env.step(chunk[cs].tolist()); cs+=1
            if done: break
        ok=bool(info.get("success",reward>0.5)); succ+=int(ok)
        print(f"  [{label}] task{task_id} ep{ep+1:02d}/{n_ep} {'SUCCESS' if ok else 'FAIL'} (total {succ}/{ep+1})",flush=True)
    env.close(); return succ,n_ep

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--arfm-ckpt",required=True)
    parser.add_argument("--sft-ckpt",required=True,help="used only for norm_stats")
    parser.add_argument("--episodes-per-task",type=int,default=protocol.EPISODES_PER_TASK)
    parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()

    print(f"Protocol: suite={SUITE_NAME} episodes/task={args.episodes_per_task} seed={args.seed} "
          f"wait={protocol.NUM_WAIT_STEPS} max_steps={MAX_STEPS} replan={REPLAN_STEPS}",flush=True)

    print(f"JAX: {jax.device_count()} x {jax.devices()[0].device_kind}",flush=True)
    jax.config.update("jax_compilation_cache_dir",str(pathlib.Path("~/.cache/jax").expanduser()))

    from libero.libero import benchmark as _bench
    suite=_bench.get_benchmark_dict()[SUITE_NAME]()
    train_config=_config.get_config("pi0_libero")
    tokenize=make_tokenize_fn(train_config)

    from openpi.training import checkpoints as _ckpts
    norm_stats=_ckpts.load_norm_stats(pathlib.Path(args.sft_ckpt)/"assets/physical-intelligence","libero")

    from libero.libero import get_libero_path; import os as _os
    task_infos=[]
    for i in range(suite.n_tasks):
        t=suite.get_task(i)
        bddl=str(pathlib.Path(get_libero_path("bddl_files"))/t.problem_folder/t.bddl_file)
        init=torch.load(_os.path.join(get_libero_path("init_states"),t.problem_folder,t.init_states_file),weights_only=False)
        task_infos.append(dict(id=i,name=t.name,desc=t.language,bddl=bddl,init=init))

    print(f"\n=== ARFM policy ({args.arfm_ckpt}) ===",flush=True)
    model=load_model(args.arfm_ckpt,train_config); model.eval()
    infer_fn=nnx_utils.module_jit(model.sample_actions)

    results=[]
    for ti in task_infos:
        print(f"\n[ARFM] Task {ti['id']}: {ti['name']}",flush=True)
        s,total=eval_task(ti['id'],ti['name'],ti['desc'],ti['bddl'],ti['init'],infer_fn,tokenize,norm_stats,args.episodes_per_task,args.seed,"ARFM")
        results.append(s/total)
        print(f"  Task {ti['id']}: {s}/{total} = {s/total*100:.1f}%",flush=True)

    print("\n"+"="*55,flush=True)
    print(f"{'Task':<5} {'Name':<35} {'ARFM':>7}",flush=True)
    print("-"*55,flush=True)
    for ti,r in zip(task_infos,results):
        print(f"{ti['id']:<5} {ti['name'][:33]:<35} {r*100:.0f}%",flush=True)
    print("-"*55,flush=True)
    print(f"{'AVG':<5} {'':<35} {sum(results)/len(results)*100:.1f}%",flush=True)
    print("="*55,flush=True)

if __name__=="__main__": main()
