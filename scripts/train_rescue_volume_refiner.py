"""Fit a local residual classifier on TRAIN worlds held out by a frozen volume model.

Formal VALID is used only to select hyperparameters. No absolute coordinates or
ground-truth masks enter the feature extraction; labels are separate arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from scipy.ndimage import shift
from rescue_volume_network import VolumeNet
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask
from train_rescue_generic import cases,export_tree


def features(row,net,channels):
    fine=input_grid(row,74)
    x=input_grid(row,channels)
    with torch.no_grad():
        grid=np.pad(x,((0,0),(0,0),(0,3),(0,2)))
        pred=net(torch.tensor(grid,device='cuda').unsqueeze(0)).sigmoid()[0,0,:32,:21,:70].cpu().numpy()
    values=[fine]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for dz in [-1,0,1]:
                values.append(shift(pred,(dx,dy,dz),order=0,mode='constant',cval=.5)[None])
    return np.concatenate(values).reshape(101,-1).T.copy(),truth_mask(row['truth']['occupied_keys']).astype(np.int32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    checkpoint=ROOT/a.checkpoint;out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use immutable new name')
    state=torch.load(checkpoint,map_location='cuda',weights_only=True)
    net=VolumeNet(state['width'],state['residual'],state['channels']).cuda();net.load_state_dict(state['state']);net.eval();torch.set_num_threads(4)
    audit=json.loads((checkpoint.parent/'development.json').read_bytes())
    held={r['id'] for r in audit['train_audit'] if r['held'] and not r['source'].startswith('valid')}
    train=[r for r in cases() if r['id'] in held]
    if len(train)<10:raise ValueError('Insufficient untouched TRAIN calibration worlds')
    validation=[]
    for split in ['valid-known','valid-unknown']:
        for path in sorted((ROOT/'experiments/raw/rescue/generalization-v35'/split).glob('*/complete.json')):
            case=json.loads(path.read_bytes())
            validation.append(dict(id=case['id'],map=path.parent/'map-6000.json',source=split,truth=json.loads((path.parent/'truth-6000.json').read_bytes())))
    rng=np.random.default_rng(20261101);xs=[];ys=[];ws=[]
    for row in train:
        x,y=features(row,net,state['channels'])
        positive=np.where(y>0)[0];negative=np.where(y==0)[0]
        negatives=rng.choice(negative,min(len(negative),10000),replace=False)
        index=np.concatenate((positive,negatives));xs.append(x[index]);ys.append(y[index])
        ws.append(np.concatenate((np.ones(len(positive)),np.full(len(negatives),len(negative)/len(negatives)))))
    x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws)
    valid=[(r,*features(r,net,state['channels'])) for r in validation]
    best=None;trials=[]
    with threadpool_limits(limits=4):
        for leaves in [7,15,31]:
            fit=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,min_samples_leaf=80,l2_regularization=5,learning_rate=.06,early_stopping=False,random_state=20261101)
            fit.fit(x,y,sample_weight=weight)
            predictions=[(r,fit.predict_proba(v)[:,1],truth) for r,v,truth in valid]
            for threshold in [.35,.4,.45,.5,.55,.6,.65,.7]:
                rows=[]
                for r,pred,truth in predictions:
                    occupied=pred>threshold;rows.append(dict(id=r['id'],source=r['source'],iou=float(np.sum(occupied&truth.astype(bool))/max(1,np.sum(occupied|truth.astype(bool))))))
                domains={s:float(np.mean([r['iou'] for r in rows if r['source']==s])) for s in ['valid-known','valid-unknown']}
                record=dict(leaves=leaves,threshold=threshold,domains=domains);trials.append(record);print(json.dumps(record),flush=True)
                key=(min(domains.values()),sum(domains.values()))
                if best is None or key>best[0]:best=(key,fit,record,rows)
    _,fit,chosen,rows=best;out.mkdir(parents=True)
    (out/'refiner.json').write_text(json.dumps(dict(feature_map='volume-fine-local101-v1',bias=float(fit._baseline_prediction[0,0]),trees=[export_tree(t[0]) for t in fit._predictors],threshold=chosen['threshold']),separators=(',',':'))+'\n')
    (out/'development.json').write_text(json.dumps(dict(purpose='TRAIN-held-out local residual calibration; formal VALID selection only',checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),train_cases=[r['id'] for r in train],coordinate_inputs=False,test_opened=False,chosen=chosen,trials=trials,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
