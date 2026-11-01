"""Fit angular sensor likelihood on TRAIN reference positions only; no references at inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--no-range',action='store_true');a=p.parse_args()
    records=[];audit=[];rng=np.random.default_rng(20261031)
    for path in sorted((ROOT/'experiments/raw/rescue/generalization-v35/train').glob('*/complete.json')):
        people=np.asarray(json.loads((path.parent/'truth-6000.json').read_bytes())['victim_positions'])
        if not len(people):continue
        labels={}
        for i,line in enumerate((path.parent/'labels.jsonl').open(encoding='utf-8-sig')):
            if i%4!=3:continue
            row=json.loads(line);labels[row['step']]={r['id']:r for r in row['robots']}
        samples=[]
        for i,line in enumerate((path.parent/'packets.jsonl').open(encoding='utf-8-sig')):
            if i%4!=3:continue
            packet=json.loads(line)
            for scan in packet['scans']:
                reference=labels[packet['sim_step']][scan['node_id']]
                if not reference['human'] or scan['pose_sigma_m']>.15:continue
                delta=people-np.array(reference['position']);distance=np.linalg.norm(delta,axis=1)
                if np.min(distance)<.08:continue
                direction=np.asarray(scan['bearing']);direction/=max(1e-12,np.linalg.norm(direction))
                predicted_az=np.arctan2(delta[:,0],delta[:,2]);observed_az=np.arctan2(direction[0],direction[2])
                az=predicted_az-observed_az;az=np.arctan2(np.sin(az),np.cos(az))
                predicted_el=np.arcsin(np.clip(delta[:,1]/distance,-1,1));observed_el=np.arcsin(np.clip(direction[1],-1,1))
                projection=np.cos((predicted_el+observed_el)/2)
                samples.append((az*projection,predicted_el-observed_el,projection,math.radians(scan['bearing_sigma_deg'])))
        if len(samples)>500:samples=[samples[i] for i in rng.choice(len(samples),500,replace=False)]
        records+=samples;audit.append(dict(id=path.parent.name,samples=len(samples)))
    n=len(records);az=np.zeros((n,3));el=np.zeros((n,3));projection=np.zeros((n,3));sigma=np.zeros(n);mask=np.zeros((n,3),bool)
    for i,(aa,ee,pp,ss) in enumerate(records):
        az[i,:len(aa)]=aa;el[i,:len(aa)]=ee;projection[i,:len(aa)]=pp;sigma[i]=ss;mask[i,:len(aa)]=True
    def loss(v):
        sh,sv,mu,direct=v;h=sigma[:,None]*sh;vertical=sigma[:,None]*sv;m=mu*projection
        horizontal=logsumexp(np.stack((-.5*(az/h)**2+np.log(direct),
                        -.5*((az-m)/h)**2+np.log((1-direct)/2),-.5*((az+m)/h)**2+np.log((1-direct)/2))),axis=0)-np.log(h)
        score=horizontal-.5*(el/vertical)**2-np.log(vertical)
        score[~mask]=-np.inf
        return float(-np.mean(logsumexp(score,axis=1)-np.log(mask.sum(axis=1))))
    fit=minimize(loss,[.6,.4,math.radians(24),.5],method='Nelder-Mead',bounds=[(.1,1.5),(.1,1.5),(0,.8),(.05,.95)],options={'maxiter':1000,'xatol':1e-6})
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable model name')
    out.mkdir(parents=True);model=json.loads((ROOT/'models/generic-v35-peaks-v2/model.json').read_bytes())
    model['bearing_likelihood']=fit.x.tolist()
    if a.no_range:model['log_range_weights']=[]
    model['train_maps']=sorted(set(model['train_maps'])|{r['id'] for r in audit})
    (out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    report=dict(purpose='Angular sensor calibration on TRAIN references; not person detection accuracy',parameters=fit.x.tolist(),
                samples=n,worlds=len(audit),negative_log_likelihood=loss(fit.x),converged=bool(fit.success),
                reference_positions_at_inference=False,test_opened=False,train_audit=audit,no_range_prior=a.no_range)
    (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['parameters','samples','worlds','converged']}))


if __name__=='__main__':main()
