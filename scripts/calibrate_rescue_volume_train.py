"""Fit one global logit intercept on TRAIN, then measure untouched VALID labels."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from rescue_volume_network import build_network
from rescue_local_features import truth_mask
from rescue_range_moments import input_grid
from rescue_validation_data import validation_cases
from rescue_dataset import assert_disjoint_worlds,scene_index

STATE=threading.local()


def initialize(checkpoint):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    STATE.net=build_network(state).eval();STATE.net.load_state_dict(state['state'])


def area(identifier):
    match=re.search(r'-S(\d+)-W',identifier)
    if match:return int(match[1])
    match=re.search(r'-([EMHC]\d{2})-',identifier)
    if not match:raise ValueError('Registered fitting area required')
    return scene_index(match[1])


def best_train_bias(rows,biases):
    if not rows or any(r.get('held',False) or r['source'].startswith(('valid','test')) for r in rows):raise ValueError('Only fitting TRAIN can determine the intercept')
    if not biases or any(not math.isfinite(b) or abs(b)>.5 for b in biases):raise ValueError('Bounded global intercepts required')
    trials=[]
    for index,bias in enumerate(biases):
        groups=defaultdict(list)
        for row in rows:groups[row['group']].append(row['ious'][index])
        worlds={group:float(np.mean(values)) for group,values in groups.items()}
        grouped=defaultdict(list)
        for row in rows:
            grouped[area(row['id'])].append((row['group'],worlds[row['group']]))
        areas={str(a):float(np.mean(list(dict(values).values()))) for a,values in grouped.items()}
        trials.append(dict(bias=bias,minimum_train_area_iou=min(areas.values()),world_mean_train_iou=float(np.mean(list(worlds.values()))),areas=areas))
    return max(trials,key=lambda r:(r['minimum_train_area_iou'],r['world_mean_train_iou'],-abs(r['bias']))),trials


def fit_row(job):
    row,checkpoint_sha,biases=job;source=ROOT/'tmp/generic-volume-input'/('range90-'+row['sha256']+'.npz')
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    cache=ROOT/'tmp/train-intercept-predictions'/checkpoint_sha;cache.mkdir(parents=True,exist_ok=True)
    target=cache/(row['sha256']+'.npz')
    with np.load(source) as data:y=data['y'][0,:32,:21,:70]>.5
    if target.exists():
        with np.load(target) as data:
            if str(data['input_sha256'])!=digest:raise ValueError('Cached calibration input changed')
            logits=data['logits']
    else:
        with np.load(source) as data:x=torch.tensor(data['x']).unsqueeze(0)
        with torch.inference_mode():logits=STATE.net(x)[0,0,:32,:21,:70].numpy()
        temporary=target.with_name(target.stem+f'.{os.getpid()}.partial.npz')
        np.savez_compressed(temporary,logits=logits,input_sha256=np.array(digest));os.replace(temporary,target)
    threshold=math.log(.6/.4);ious=[]
    for bias in biases:
        pred=logits+bias>threshold;ious.append(int((pred&y).sum())/max(1,int((pred|y).sum())))
    return dict(row,ious=ious,input_archive_sha256=digest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',action='append',required=True)
    p.add_argument('--geometry-pool',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    out=ROOT/a.output
    if out.exists() or not 1<=a.workers<=4:raise ValueError('Fresh report and bounded CPU workers required')
    checkpoint=ROOT/a.checkpoint;checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    development=json.loads((checkpoint.parent/'development.json').read_bytes())
    if development['test_opened'] or not (checkpoint.parent/'history.json').exists():raise ValueError('Completed TRAIN-only parent required')
    training=[row for row in development['train_audit'] if not row['held']]
    if any(row['source'].startswith(('valid','test')) for row in training):raise ValueError('Evaluation labels cannot calibrate the intercept')
    biases=[round(v/10,1) for v in range(-5,6)]
    torch.set_num_threads(2)
    with ThreadPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(checkpoint),)) as pool:
        fitted=[]
        for index,row in enumerate(pool.map(fit_row,[(r,checkpoint_sha,biases) for r in training]),1):
            fitted.append(row)
            if index%40==0 or index==len(training):print(json.dumps(dict(stage='TRAIN-only-intercept',completed=index,total=len(training))),flush=True)
    chosen,trials=best_train_bias(fitted,biases)
    # Freeze the TRAIN choice before looking up any validation labels.
    receipt=out.with_suffix('.train-choice.json')
    if receipt.exists():raise ValueError('TRAIN choice receipt already exists')
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(dict(chosen=chosen,checkpoint_sha256=checkpoint_sha,train_cases=len(fitted),independent_train_worlds=len({r['group'] for r in fitted}),validation_labels_read=False),indent=2)+'\n')
    print(json.dumps(dict(stage='frozen-TRAIN-choice',chosen=chosen)),flush=True)
    sources=list(validation_cases(a.data,18000));assert_disjoint_worlds([r['id'] for r in sources],{r['id'] for r in training})
    manifest=json.loads((ROOT/a.geometry_pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or manifest['test_opened'] or set(manifest['case_ids'])!={r['id'] for r in sources}:raise ValueError('Exact complete VALID preparation required')
    initialize(str(checkpoint));rows=[]
    with torch.inference_mode():
        for index,row in enumerate(sources,1):
            source=ROOT/a.geometry_pool/row['source']/row['id']/'map-18000.json';x=torch.tensor(np.pad(input_grid(dict(map=source),90),((0,0),(0,0),(0,3),(0,2)))).unsqueeze(0)
            logits=STATE.net(x)[0,0,:32,:21,:70].numpy();truth=truth_mask(row['truth']['occupied_keys']).reshape(32,21,70)>.5
            metrics={}
            for name,bias in [('baseline',0),('TRAIN_intercept',chosen['bias'])]:
                pred=logits+bias>math.log(.6/.4);metrics[name]=int((pred&truth).sum())/max(1,int((pred|truth).sum()))
            rows.append(dict(id=row['id'],domain=row['source'],area=area(row['id']),**metrics))
            if index%18==0 or index==len(sources):print(json.dumps(dict(stage='VALID-after-frozen-intercept',completed=index,total=len(sources))),flush=True)
    summary={}
    for name in ['baseline','TRAIN_intercept']:
        areas={str(ar):float(np.mean([r[name] for r in rows if r['area']==ar])) for ar in sorted({r['area'] for r in rows})}
        summary[name]=dict(minimum_area_iou=min(areas.values()),macro_iou=float(np.mean([r[name] for r in rows])),areas=areas)
    if checkpoint_sha!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise ValueError('Parent weights changed')
    out.write_text(json.dumps(dict(purpose='One global logit intercept fitted on TRAIN only, then frozen and measured on VALID',final_test_opened=False,closed_loop_acceptance=False,
        threshold=.6,checkpoint=a.checkpoint,checkpoint_sha256=checkpoint_sha,chosen=chosen,trials=trials,train_audit=fitted,validation_rows=rows,summary=summary,
        source_labels_used_only_after_sensor_forward=True,no_area_specific_parameters=True),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
