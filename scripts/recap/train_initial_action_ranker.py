"""Train a group-held-out pairwise initial-action ranker."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=nn.Sequential(nn.Conv2d(6,32,5,2,2),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.task=nn.Embedding(10,16)
        self.head=nn.Sequential(nn.Linear(128+8+35+16,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,img,state,act,task): return self.head(torch.cat([self.vision(img),state,act.flatten(1),self.task(task)],1)).squeeze(1)

def features(d, idx, device):
    image=np.concatenate([d['images'][idx],d['wrist_images'][idx]],-1)
    return (torch.as_tensor(image,device=device).permute(0,3,1,2).float()/255,
            torch.as_tensor(d['states'][idx],device=device).float(),torch.as_tensor(d['action_chunks'][idx],device=device).float(),torch.as_tensor(d['task_ids'][idx],device=device).long())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=100); a=p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev='cuda'
    d=dict(np.load(a.data)); groups={}
    for i,(t,g) in enumerate(zip(d['task_ids'],d['init_state_indices'],strict=True)): groups.setdefault((int(t),int(g)),[]).append(i)
    ordered=sorted(groups); rng=np.random.default_rng(20260720); rng.shuffle(ordered); split={g:('test' if j%10<2 else 'val' if j%10<4 else 'train') for j,g in enumerate(ordered)}
    pairs={k:[] for k in ('train','val','test')}
    for g,ids in groups.items():
        pos=[i for i in ids if d['successes'][i]]; neg=[i for i in ids if not d['successes'][i]]
        pairs[split[g]] += [(x,y) for x in pos for y in neg]
    model=Ranker().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    train=pairs['train']; rng=np.random.default_rng(a.seed)
    for epoch in range(a.epochs):
        rng.shuffle(train); losses=[]; model.train()
        for start in range(0,len(train),32):
            batch=train[start:start+32]; pi=np.array([x[0] for x in batch]); ni=np.array([x[1] for x in batch])
            opt.zero_grad(); sp=model(*features(d,pi,dev)); sn=model(*features(d,ni,dev)); loss=F.softplus(sn-sp).mean(); loss.backward(); opt.step(); losses.append(loss.item())
        if epoch%20==19: print(json.dumps({'epoch':epoch+1,'loss':float(np.mean(losses))}),flush=True)
    model.eval(); report={}
    with torch.no_grad():
        for name,ps in pairs.items():
            if not ps: continue
            pi=np.array([x[0] for x in ps]); ni=np.array([x[1] for x in ps]); report[name]={'pairs':len(ps),'accuracy':float((model(*features(d,pi,dev))>model(*features(d,ni,dev))).float().mean())}
    a.output.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'report':report},a.output/'best.pt'); (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
