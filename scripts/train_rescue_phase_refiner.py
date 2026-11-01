"""Fit local geometry refinement on TRAIN coordinate augmentations only.

The base volume model may have fitted these physical TRAIN worlds. Its unseen
coordinate phases are calibration inputs, not independent held-out worlds.
Formal VALID selects capacity, never fits trees; the 0.6 acceptance cutoff stays fixed.
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
from scipy.ndimage import shift
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from rescue_volume_network import VolumeNet
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask,iou
from train_rescue_generic import export_tree


def features(path,net):
    measured=input_grid(dict(map=path),74)
    with torch.no_grad():
        grid=np.pad(measured,((0,0),(0,0),(0,3),(0,2)))
        pred=net(torch.tensor(grid,device='cuda').unsqueeze(0)).sigmoid()[0,0,:32,:21,:70].cpu().numpy()
    values=[measured]
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for dz in [-1,0,1]:values.append(shift(pred,(dx,dy,dz),order=0,mode='constant',cval=.5)[None])
    return np.concatenate(values).reshape(101,-1).T.copy(),pred.reshape(-1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--subcell-pool',required=True)
    p.add_argument('--validation-root',required=True);p.add_argument('--geometry-pool',required=True)
    p.add_argument('--step',type=int,choices=[6000,18000],default=18000);p.add_argument('--name',required=True);a=p.parse_args()
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use an immutable model name')
    checkpoint=ROOT/a.checkpoint;state=torch.load(checkpoint,map_location='cuda',weights_only=True);torch.set_num_threads(4)
    if state['channels']!=74:raise ValueError('Refiner prototype requires measured 74-channel inputs')
    net=VolumeNet(state['width'],state['residual'],74,state.get('profiles',False),state.get('normalization','group')).cuda()
    net.load_state_dict(state['state']);net.eval()
    pool=ROOT/a.subcell_pool;manifest=json.loads((pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or manifest['split']!='train' or manifest['test_opened']:raise ValueError('TRAIN-only augmentation required')
    development=json.loads((checkpoint.parent/'development.json').read_bytes())
    fitted={r['id'] for r in development['train_audit'] if not r['held']}
    if not set(manifest['case_ids'])<=fitted:raise ValueError('Unregistered calibration world')
    rng=np.random.default_rng(20261031);xs=[];ys=[];ws=[];audit=[]
    for number,name in enumerate(manifest['case_ids'],1):
        record=json.loads((pool/name/'complete.json').read_bytes())
        if not record['zero_translation_exact']:raise ValueError('Original geometry parity is required')
        for view in record['views']:
            directory=pool/name/f"view-{view['index']}";path=directory/'map-6000.json';labels=directory/'truth-6000.json'
            for file,key in [(path,'map_sha256'),(labels,'truth_sha256'),(directory/'range-features.npz','features_sha256')]:
                if hashlib.sha256(file.read_bytes()).hexdigest()!=view[key]:raise ValueError('Calibration artifact changed')
            x,pred=features(path,net);y=truth_mask(json.loads(labels.read_bytes())['occupied_keys']).astype(np.int32)
            positive=np.flatnonzero(y);hard=np.flatnonzero((y==0)&(pred>.03));easy=np.flatnonzero((y==0)&(pred<=.03))
            sample=rng.choice(easy,min(1200,len(easy)),replace=False);index=np.r_[positive,hard,sample]
            weight=np.r_[np.ones(len(positive)+len(hard)),np.full(len(sample),len(easy)/max(1,len(sample)))]/len(y)
            xs.append(x[index]);ys.append(y[index]);ws.append(weight)
            audit.append(dict(id=name,view=view['index'],map_sha256=view['map_sha256'],truth_sha256=view['truth_sha256']))
        if number%12==0:print(json.dumps(dict(stage='train-features',worlds=number,total=len(manifest['case_ids']))),flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys);weights=np.concatenate(ws);weights*=len(weights)/weights.sum();del xs,ys,ws
    valid=[]
    for split in ['valid-known','valid-unknown']:
        root=ROOT/a.validation_root/split;collection=json.loads((root/'collection.json').read_bytes())
        if collection['status']!='complete' or collection['steps']<a.step:raise ValueError('Complete matching VALID required')
        for case in collection['plan']['cases']:
            directory=root/case['id'];path=ROOT/a.geometry_pool/split/case['id']/f'map-{a.step}.json'
            matrix,base=features(path,net);truth=truth_mask(json.loads((directory/f'truth-{a.step}.json').read_bytes())['occupied_keys'])
            valid.append((dict(id=case['id'],area=case['scene'],nodes=case['nodes'],domain=split),matrix,truth,base))
    best=None;trials=[]
    with threadpool_limits(limits=4):
        for leaves in [7,15,31]:
            fit=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,learning_rate=.06,min_samples_leaf=80,l2_regularization=5,early_stopping=False,random_state=20261031)
            fit.fit(x,y,sample_weight=weights);rows=[]
            for row,matrix,truth,base in valid:
                metric=iou(fit.predict_proba(matrix)[:,1],truth)
                rows.append(dict(row,**metric,base_iou=iou(base,truth)['iou']))
            areas={str(s):float(np.mean([r['iou'] for r in rows if r['area']==s])) for s in sorted({r['area'] for r in rows})}
            domains={s:float(np.mean([r['iou'] for r in rows if r['domain']==s])) for s in ['valid-known','valid-unknown']}
            record=dict(leaves=leaves,rounds=160,minimum_area=min(areas.values()),areas=areas,domains=domains)
            trials.append(record);print(json.dumps(record),flush=True);key=(record['minimum_area'],sum(domains.values()))
            if best is None or key>best[0]:best=(key,fit,record,rows)
    _,fit,chosen,rows=best;out.mkdir(parents=True)
    (out/'refiner.json').write_text(json.dumps(dict(feature_map='volume-fine-local101-v1',bias=float(fit._baseline_prediction[0,0]),trees=[export_tree(t[0]) for t in fit._predictors],acceptance_threshold=.6),separators=(',',':'))+'\n')
    (out/'development.json').write_text(json.dumps(dict(purpose='TRAIN phase calibration from already fitted worlds; VALID capacity selection, not closed-loop acceptance',test_opened=False,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),coordinate_inputs=False,train_audit=audit,validation_root=a.validation_root,evaluation_step=a.step,chosen=chosen,trials=trials,rows=rows,arguments=vars(a)),indent=2)+'\n')


if __name__=='__main__':main()
