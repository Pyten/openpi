"""Train a group-held-out action-conditioned critic on counterfactual chunks."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(6,32,5,2,2),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.task = nn.Embedding(10,16)
        self.head = nn.Sequential(nn.Linear(128+8+35+8+16,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,image,state,action,post_state,task):
        return self.head(torch.cat([self.vision(image),state,action.flatten(1),post_state,self.task(task)],1)).squeeze(1)


def features(d, idx, device):
    image=np.concatenate([d['images'][idx],d['wrist_images'][idx]],-1)
    return (torch.as_tensor(image,device=device).permute(0,3,1,2).float()/255,
            torch.as_tensor(d['states'][idx],device=device).float(),
            torch.as_tensor(d['action_chunks'][idx],device=device).float(),
            torch.as_tensor(d['post_states'][idx],device=device).float(),
            torch.as_tensor(d['task_ids'][idx],device=device).long())


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=120); p.add_argument('--cv-fold',type=int,default=0); p.add_argument('--num-folds',type=int,default=5); args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); device='cuda'; d=dict(np.load(args.data)); grouped={}
    for i,key in enumerate(zip(d['task_ids'],d['init_state_indices'],d['candidate_ids'],strict=True)): grouped.setdefault(tuple(map(int,key)),[]).append(i)
    records=[]; disagreements=0
    for (task,state,candidate),ids in grouped.items():
        if len(ids)!=2: raise ValueError(f'expected 2 continuation rows, got {len(ids)}')
        y=d['successes'][ids]
        if y[0]!=y[1]: disagreements+=1; continue
        records.append((task,state,candidate,ids[0],int(y[0])))
    groups=sorted({(r[0],r[1]) for r in records}); rng=np.random.default_rng(20260804); rng.shuffle(groups); split={g:('test' if i%args.num_folds==args.cv_fold else 'train') for i,g in enumerate(groups)}
    pairs={x:[] for x in ('train','val','test')}; by={}
    for r in records: by.setdefault((r[0],r[1]),[]).append(r)
    for g,rows in by.items():
        pairs[split[g]] += [(a[3],b[3]) for a in rows if a[4] for b in rows if not b[4]]
    model=Critic().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); order=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order.shuffle(pairs['train']); model.train()
        for start in range(0,len(pairs['train']),32):
            batch=pairs['train'][start:start+32]; pos=np.asarray([x[0] for x in batch]); neg=np.asarray([x[1] for x in batch]); opt.zero_grad(); loss=F.softplus(model(*features(d,neg,device))-model(*features(d,pos,device))).mean(); loss.backward(); opt.step()
    model.eval(); report={'groups':len(groups),'consensus_candidates':len(records),'disagreement_candidates':disagreements,'pairs':{k:len(v) for k,v in pairs.items()}}
    with torch.no_grad():
        for name,ps in pairs.items():
            if ps:
                pos=np.asarray([x[0] for x in ps]); neg=np.asarray([x[1] for x in ps]); report[name]={'pairs':len(ps),'pair_accuracy':float((model(*features(d,pos,device))>model(*features(d,neg,device))).float().mean())}
        for name in ('val','test'):
            selected=[]; random=[]
            for g,rows in by.items():
                if split[g]!=name: continue
                idx=np.asarray([r[3] for r in rows]); y=np.asarray([r[4] for r in rows]); score=model(*features(d,idx,device)).cpu().numpy(); selected.append(float(y[score.argmax()])); random.append(float(y.mean()))
            if selected:
                report.setdefault(name,{}).update({'groups':len(selected),'top1_success':float(np.mean(selected)),'random_success':float(np.mean(random))})
    args.output.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'report':report},args.output/'best.pt'); (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
