"""TRAIN-only selection of 3D Gaussian-overlap Soft-NMS; no final TEST access.

Adaptation of Bodla et al., ICCV 2017, arXiv:1704.04503. Gaussian source
similarity replaces box IoU; this is not a calibrated person probability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from evaluate_rescue_replay import match
from train_rescue_tracks import score_model
from train_rescue_generic import registered_extra_cases


def choose(positions,scores,distance,threshold,minimum,scale,sigma):
    scores=scores.copy();selected=[]
    for _ in range(3):
        index=int(np.argmax(scores)) if len(scores) else 0
        if not len(scores) or scores[index]<threshold:break
        selected.append((index,float(scores[index])))
        overlap=np.exp(-.5*(distance[index]/scale)**2)
        scores*=np.exp(-overlap**2/sigma)
        scores[distance[index]<=minimum]=0.
    return [dict(position=positions[i].tolist(),score=s) for i,s in selected]


def metrics(rows,decisions):
    groups={};details=[]
    for row,cues in zip(rows,decisions):
        tp,_=match(cues,row['truth']);fp=len(cues)-tp;n=len(row['truth'])
        for key in [('all',0),('area',row['area']),('nodes',row['nodes'])]:
            a,b,c=groups.get(key,(0,0,0));groups[key]=(a+tp,b+fp,c+n)
        details.append(dict(id=row['id'],matched=tp,false_alarms=fp,people=n,cues=cues))
    tp,fp,n=groups[('all',0)];precision=tp/max(1,tp+fp);recall=tp/max(1,n)
    minimum=min(v for a,b,c in groups.values() for v in ([a/max(1,a+b),a/c] if c else [1. if b==0 else 0.]))
    return (minimum,min(precision,recall),precision+recall),dict(precision=precision,recall=recall,matched=tp,false_alarms=fp,people=n,minimum_stratum=minimum,rows=details)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--train-root',action='append',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    model=json.loads((ROOT/a.model).read_bytes());rows=[]
    for path,case,_,_ in registered_extra_cases(a.train_root):
        mapping=json.loads((path.parent/'map-6000.json').read_bytes());truth=json.loads((path.parent/'truth-6000.json').read_bytes())['victim_positions']
        proposals=mapping['proposals'];scores=np.array([score_model(model,q['features']) for q in proposals]);positions=np.array([q['position'] for q in proposals]).reshape(-1,3)
        rows.append(dict(id=case['id'],area=case['case']['scene'],nodes=case['nodes'],truth=truth,positions=positions,scores=scores))
    best=None;trials=[]
    for height in [1,1.5,2,3]:
        raw_distances=[]
        for row in rows:
            delta=(row['positions'][:,None,:]-row['positions'][None,:,:])*np.array([1,height,1]);raw_distances.append(np.sqrt((delta*delta).sum(2)))
        for radius in [.25,.45,.6]:
            prepared=[]
            for row,distance in zip(rows,raw_distances):
                weight=(distance<=radius)*np.maximum(row['scores'][None,:],1e-9)
                centers=(weight@row['positions'])/np.maximum(weight.sum(1)[:,None],1e-12)
                keep=row['scores']>=.03;centers=centers[keep];scores=row['scores'][keep]
                delta=(centers[:,None,:]-centers[None,:,:])*np.array([1,height,1]);distance=np.sqrt((delta*delta).sum(2))
                prepared.append((centers,scores,distance))
            for scale in [.2,.3,.4,.5]:
              for sigma in [.2,.5,1]:
                for minimum in [.15,.2,.25]:
                  for threshold in [.03,.05,.1,.15,.2,.3,.4,.5]:
                    decisions=[choose(*q,threshold,minimum,scale,sigma) for q in prepared];key,metric=metrics(rows,decisions)
                    config=dict(height=height,radius=radius,power=1,scale=scale,sigma=sigma,minimum=minimum,threshold=threshold)
                    trials.append(dict(**config,**{k:v for k,v in metric.items() if k!='rows'}));key=(*key,-radius,-scale,-minimum,threshold)
                    if best is None or key>best[0]:best=(key,config,metric)
            print(json.dumps(dict(stage='selection',height=height,radius=radius,best=best[1],metrics={k:v for k,v in best[2].items() if k!='rows'})),flush=True)
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use an immutable selection name')
    out.mkdir(parents=True);_,config,metric=best
    (out/'development.json').write_text(json.dumps(dict(purpose='TRAIN-only Soft-NMS selection; not VALID or final acceptance',test_opened=False,model=a.model,
        model_sha256=hashlib.sha256((ROOT/a.model).read_bytes()).hexdigest(),train_roots=a.train_root,training_worlds=len(rows),chosen=config,metrics=metric,trials=trials,
        reference='https://arxiv.org/abs/1704.04503',adaptation='Gaussian 3D source overlap instead of 2D box IoU'),indent=2)+'\n')


if __name__=='__main__':main()
