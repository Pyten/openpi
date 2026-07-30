"""Train a group-held-out ranker from continuation-consensus candidates."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(6,32,5,2,2),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.task = nn.Embedding(10,16)
        self.head = nn.Sequential(nn.Linear(128+8+35+16,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self, image, state, action, task):
        return self.head(torch.cat([self.vision(image),state,action.flatten(1),self.task(task)],1)).squeeze(1)


def features(data, index, device):
    image=np.concatenate([data['images'][index],data['wrist_images'][index]],-1)
    return (torch.as_tensor(image,device=device).permute(0,3,1,2).float()/255,
            torch.as_tensor(data['states'][index],device=device).float(),
            torch.as_tensor(data['action_chunks'][index],device=device).float(),
            torch.as_tensor(data['task_ids'][index],device=device).long())


def candidate_records(data):
    required={'candidate_ids','continuation_ids'}
    missing=required-set(data)
    if missing: raise ValueError(f'missing counterfactual metadata: {sorted(missing)}')
    grouped={}
    for i,key in enumerate(zip(data['task_ids'],data['init_state_indices'],data['candidate_ids'],strict=True)):
        grouped.setdefault(tuple(map(int,key)),[]).append(i)
    records=[]; disagreements=0
    for (task,init,candidate), ids in grouped.items():
        if len(ids)!=2: raise ValueError(f'{(task,init,candidate)} has {len(ids)} continuations, expected 2')
        labels=data['successes'][ids]
        if labels[0] != labels[1]:
            disagreements += 1; continue
        records.append((task,init,candidate,ids[0],int(labels[0])))
    return records, disagreements


def pair_list(records, split):
    by_group={}
    for record in records: by_group.setdefault((record[0],record[1]),[]).append(record)
    pairs={name:[] for name in ('train','val','test')}
    for group, rows in by_group.items():
        pos=[row[3] for row in rows if row[4]]; neg=[row[3] for row in rows if not row[4]]
        pairs[split[group]].extend((x,y) for x in pos for y in neg)
    return pairs,by_group


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=100); args=p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); device='cuda'
    data=dict(np.load(args.data)); records, disagreements=candidate_records(data)
    groups=sorted({(row[0],row[1]) for row in records}); split_rng=np.random.default_rng(20260720); split_rng.shuffle(groups)
    split={group:('test' if i%10<2 else 'val' if i%10<4 else 'train') for i,group in enumerate(groups)}
    pairs,by_group=pair_list(records,split)
    if not pairs['train'] or not pairs['test']: raise RuntimeError('insufficient consensus preference pairs')
    model=Ranker().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); rng=np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(pairs['train']); losses=[]; model.train()
        for start in range(0,len(pairs['train']),32):
            batch=pairs['train'][start:start+32]; pos=np.asarray([x[0] for x in batch]); neg=np.asarray([x[1] for x in batch])
            optimizer.zero_grad(); loss=F.softplus(model(*features(data,neg,device))-model(*features(data,pos,device))).mean(); loss.backward(); optimizer.step(); losses.append(loss.item())
        if (epoch+1)%20==0: print(json.dumps({'epoch':epoch+1,'loss':float(np.mean(losses))}),flush=True)
    report={'consensus_candidates':len(records),'disagreement_candidates':disagreements,'groups':len(groups),'pairs':{name:len(value) for name,value in pairs.items()}}
    model.eval()
    with torch.no_grad():
        for name, ps in pairs.items():
            if ps:
                pos=np.asarray([x[0] for x in ps]); neg=np.asarray([x[1] for x in ps])
                report[name]={'pairs':len(ps),'pair_accuracy':float((model(*features(data,pos,device))>model(*features(data,neg,device))).float().mean())}
        for name in ('val','test'):
            selected=[]; random=[]
            for group, rows in by_group.items():
                if split[group] != name: continue
                idx=np.asarray([row[3] for row in rows]); label=np.asarray([row[4] for row in rows])
                score=model(*features(data,idx,device)).detach().cpu().numpy()
                selected.append(float(label[score.argmax()])); random.append(float(label.mean()))
            report[name]['top1_success']=float(np.mean(selected)); report[name]['random_candidate_success']=float(np.mean(random)); report[name]['groups']=len(selected)
    args.output.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'report':report},args.output/'best.pt'); (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
