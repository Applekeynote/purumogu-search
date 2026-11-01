"""Calibrate instantaneous and temporal audio separately on registered TRAIN only.

References identify the dominant source class offline. True positions and world
identity never enter a classifier feature. This is not a person-detection score.
"""
import argparse
from collections import deque,Counter
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from rescue_dataset import world_group


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(folder,window,cache):
    complete=folder/'complete.json';case=json.loads(complete.read_bytes());packets=folder/'packets.jsonl';labels=folder/'labels.jsonl'
    signature=hashlib.sha256(complete.read_bytes()+str((window,packets.stat().st_size,packets.stat().st_mtime_ns,labels.stat().st_size,labels.stat().st_mtime_ns)).encode()).hexdigest()
    target=cache/(signature+'.npz');meta_path=target.with_suffix('.json')
    if target.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_bytes())
        if meta['cache_sha256']!=sha(target):raise ValueError('Numeric training cache changed')
        with np.load(target) as d:return case,{k:d[k] for k in d.files},meta
    references={r['step']:{n['id']:int(n['human']) for n in r['robots']} for r in map(json.loads,labels.read_text(encoding='utf-8-sig').splitlines())}
    instant=[];truth=[];mean=[];majority=[];histories={};digest=hashlib.sha256()
    with packets.open('rb') as stream:
        for line in stream:
            digest.update(line);packet=json.loads(line)
            for scan in packet.get('scans',[]):
                y=references[packet['sim_step']][scan['node_id']]
                x=[scan['acoustic_snr_db']/20,scan['voice_band_ratio'],scan['modulation'],scan['temperature_delta_c']/2]
                instant.append(x+[scan['bearing_sigma_deg']/90]);truth.append(y)
                history=histories.setdefault(scan['node_id'],deque(maxlen=window));history.append((x,y))
                if len(history)>=4:
                    mean.append(np.mean([v[0] for v in history],axis=0));majority.append(int(np.mean([v[1] for v in history])>=.5))
    rng=np.random.default_rng(int(hashlib.sha256(case['id'].encode()).hexdigest()[:16],16));values={}
    for name,x,y in [('instant',instant,truth),('temporal',mean,majority)]:
        selected=np.sort(rng.choice(len(x),min(2000,len(x)),replace=False));values[name+'_x']=np.asarray(x)[selected];values[name+'_y']=np.asarray(y)[selected]
    np.savez_compressed(target,**values);meta=dict(id=case['id'],packets_sha256=digest.hexdigest(),labels_sha256=sha(labels),cache_sha256=sha(target),window=window,position_features=False)
    meta_path.write_text(json.dumps(meta,indent=2)+'\n');return case,values,meta


def polynomial(x):
    return np.column_stack([x]+[(x[:,i]*x[:,j])[:,None] for i in range(x.shape[1]) for j in range(i,x.shape[1])])


def probability(x,weights):return 1/(1+np.exp(-np.clip(weights[0]+x@np.array(weights[1:]),-40,40)))


def score(records,kind,dimensions,weights):
    details=[]
    for case,data,_,held in records:
        if not held:continue
        y=data[kind+'_y'];features=data[kind+'_x'][:,:dimensions]
        if len(weights)!=dimensions+1:features=polynomial(features)
        p=np.clip(probability(features,weights),1e-9,1-1e-9)
        details.append(dict(id=case['id'],area=case['case']['scene'],log_loss=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),brier=float(np.mean((p-y)**2)),accuracy=float(np.mean((p>=.5)==y))))
    areas={str(a):float(np.mean([r['log_loss'] for r in details if r['area']==a])) for a in {r['area'] for r in details}}
    return dict(macro_world_log_loss=float(np.mean([r['log_loss'] for r in details])),worst_area_log_loss=max(areas.values()),macro_world_brier=float(np.mean([r['brier'] for r in details])),macro_world_accuracy=float(np.mean([r['accuracy'] for r in details])),areas=areas)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--train-root',action='append',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    base=json.loads((ROOT/a.base).read_bytes());window=base['acoustic_window_frames'];records=[];cache=ROOT/'tmp/generic-audio-calibration';cache.mkdir(exist_ok=True)
    ids=set()
    for source in a.train_root:
        root=ROOT/source;collection=json.loads((root/'collection.json').read_bytes())
        if collection['status']!='complete' or not collection['split'].startswith('train'):raise ValueError('Complete registered TRAIN only')
        expected={c['id'] for c in collection['plan']['cases']};paths=sorted(root.glob('*/complete.json'))
        if {json.loads(p.read_bytes())['id'] for p in paths}!=expected:raise ValueError('Training collection differs from plan')
        for path in paths:
            case,data,meta=extract(path.parent,window,cache)
            if case['id'] in ids:raise ValueError('Duplicate TRAIN trajectory')
            ids.add(case['id']);held=int(hashlib.sha256(world_group(case['id']).encode()).hexdigest()[:8],16)%5==0
            records.append((case,data,meta,held))
            if len(records)%10==0:print(json.dumps(dict(stage='extract',cases=len(records))),flush=True)
    results={};trials=[]
    with threadpool_limits(limits=2):
      for kind in ['instant','temporal']:
        baseline=score(records,kind,4,base['acoustic_weights']);best=None
        for dimensions in ([4,5] if kind=='instant' else [4]):
            for c in [.001,.01,.1,1]:
                subset=[r for r in records if not r[3]];counts=Counter(r[0]['case']['scene'] for r in subset)
                x=np.concatenate([polynomial(d[kind+'_x'][:,:dimensions]) for _,d,_,_ in subset]);y=np.concatenate([d[kind+'_y'] for _,d,_,_ in subset])
                weight=np.concatenate([np.full(len(d[kind+'_y']),1/(len(d[kind+'_y'])*counts[case['case']['scene']])) for case,d,_,_ in subset]);weight*=len(y)/weight.sum()
                fit=LogisticRegression(C=c,max_iter=1000,tol=1e-6);fit.fit(x,y,sample_weight=weight);weights=np.r_[fit.intercept_,fit.coef_[0]].tolist()
                metric=score(records,kind,dimensions,weights);trial=dict(kind=kind,dimensions=dimensions,c=c,**metric);trials.append(trial)
                key=(metric['worst_area_log_loss'],metric['macro_world_log_loss'])
                if best is None or key<best[0]:best=(key,dimensions,c,metric)
        _,dimensions,c,metric=best;counts=Counter(r[0]['case']['scene'] for r in records)
        x=np.concatenate([polynomial(d[kind+'_x'][:,:dimensions]) for _,d,_,_ in records]);y=np.concatenate([d[kind+'_y'] for _,d,_,_ in records])
        weight=np.concatenate([np.full(len(d[kind+'_y']),1/(len(d[kind+'_y'])*counts[case['case']['scene']])) for case,d,_,_ in records]);weight*=len(y)/weight.sum()
        fit=LogisticRegression(C=c,max_iter=1000,tol=1e-6);fit.fit(x,y,sample_weight=weight)
        results[kind]=dict(dimensions=dimensions,c=c,weights=np.r_[fit.intercept_,fit.coef_[0]].tolist(),baseline=baseline,held_out_train=metric,samples=len(y));print(json.dumps(dict(stage='calibration',kind=kind,baseline=baseline,chosen=metric)),flush=True)
    out=ROOT/'models'/a.name;out.mkdir(exist_ok=False)
    report=dict(purpose='TRAIN-world-held-out calibration; not mission detection accuracy',test_opened=False,base=a.base,base_sha256=sha(ROOT/a.base),window=window,roots=a.train_root,
        results=results,trials=trials,train_audit=[dict(id=c['id'],held=h,source=m) for c,_,m,h in records],features=['SNR/20','voice_band','modulation','temperature/2','bearing_sigma/90 (instantaneous candidate only)'],true_positions_at_inference=False)
    (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
