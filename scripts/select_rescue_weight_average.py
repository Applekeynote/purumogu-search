"""One global checkpoint interpolation on formal VALID; one network at runtime."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from rescue_volume_network import build_network,extend_correction_inputs
from rescue_weight_average import interpolate
from rescue_dataset import assert_disjoint_worlds
from rescue_validation_data import validation_cases
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--name',required=True)
    p.add_argument('--data',action='append',required=True);p.add_argument('--geometry-pool',required=True);a=p.parse_args()
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable model name')
    torch.set_num_threads(2);states=[];developments=[];parents=[]
    for source in [a.left,a.right]:
        path=ROOT/source
        if not (path.parent/'history.json').exists():raise ValueError('Completed parent training required')
        states.append(torch.load(path,map_location='cpu',weights_only=True));developments.append(json.loads((path.parent/'development.json').read_bytes()))
        parents.append(dict(checkpoint=source,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    left,right=states
    keys=['channels','width','residual','profiles','normalization','backbone_channels','fine_width','fine_supervision','plane_registration_v1','axis_plane_registration_v2','range_tsdf_v1']
    if any(left.get(k)!=right.get(k) for k in keys):raise ValueError('Common feature and architecture semantics required')
    if left['architecture']=='subvoxel-unet-refinement-v3' and right['architecture']=='subvoxel-phase-refinement-v4':
        left=copy.deepcopy(left);left['architecture']=right['architecture'];left['state']=extend_correction_inputs(left['state'],build_network(right),8)
    if left['architecture']!=right['architecture']:raise ValueError('Architectures differ without a function-preserving conversion')
    audit={};trained=set()
    for d,s in zip(developments,states):
        if d['test_opened']:raise ValueError('TEST-informed checkpoint prohibited')
        for row in d['train_audit']:
            if row['id'] in audit and audit[row['id']]['held']!=row['held']:raise ValueError('TRAIN/VALID roles differ between parents')
            audit[row['id']]=row
            if not row['held']:
                if row['source'].startswith(('valid','test')):raise ValueError('Evaluation labels fitted by a parent')
                trained.add(row['id'])
        trained.update(s.get('pretraining_train_maps',[]))
    records=list(validation_cases(a.data,18000));assert_disjoint_worlds([r['id'] for r in records],trained)
    manifest=json.loads((ROOT/a.geometry_pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or manifest['test_opened'] or set(manifest['case_ids'])!={r['id'] for r in records}:raise ValueError('Exact complete VALID feature preparation required')
    inputs=[]
    for row in records:
        source=ROOT/a.geometry_pool/row['source']/row['id']/'map-18000.json'
        x=torch.tensor(np.pad(input_grid(dict(map=source),left['channels']),((0,0),(0,0),(0,3),(0,2)))).unsqueeze(0)
        inputs.append((row,x,truth_mask(row['truth']['occupied_keys']).reshape(32,21,70)>.5))
    best=None;trials=[];net=build_network(left).eval()
    for weight in [0,.25,.5,.75,1]:
        state=interpolate(left['state'],right['state'],weight);net.load_state_dict(state);rows=[]
        with torch.no_grad():
            for row,x,truth in inputs:
                pred=net(x).sigmoid()[0,0,:32,:21,:70].numpy()>.6
                area=int(re.search(r'-S(\d+)-W',row['id'])[1]);rows.append(dict(id=row['id'],source=row['source'],area=area,iou=int((pred&truth).sum())/max(1,int((pred|truth).sum()))))
        areas={str(area):float(np.mean([r['iou'] for r in rows if r['area']==area])) for area in sorted({r['area'] for r in rows})}
        trial=dict(right_weight=weight,macro_iou=float(np.mean([r['iou'] for r in rows])),minimum_area_iou=min(areas.values()),areas=areas,
            domains={role:float(np.mean([r['iou'] for r in rows if r['source']==role])) for role in ['valid-known','valid-unknown']})
        key=(trial['minimum_area_iou'],trial['macro_iou']);trials.append(trial);print(json.dumps(trial),flush=True)
        if best is None or key>best[0]:best=(key,trial,rows)
    chosen=best[1];selected=copy.deepcopy(left);selected.update(state=interpolate(left['state'],right['state'],chosen['right_weight']),epoch=0,pretraining_train_maps=sorted(trained),weight_interpolation=dict(parents=parents,right_weight=chosen['right_weight']))
    out.mkdir(parents=True);torch.save(selected,out/'volume.pt')
    report=dict(purpose='One global checkpoint interpolation selected on formal VALID, no new gradient fitting',test_opened=False,closed_loop_acceptance=False,
        architecture=left['architecture'],train_audit=list(audit.values()),pretraining_train_maps=sorted(trained),pretraining=parents,
        weight_interpolation=dict(parents=parents,right_weight=chosen['right_weight']),epoch_zero_means_no_new_training=True,
        best=dict(chosen,epoch=0),rows=best[2],history=trials,arguments=dict(resume=a.left,weight_interpolation_right=a.right),
        primary_reference='https://proceedings.mlr.press/v162/wortsman22a.html',reference_scope='Checkpoint weight averaging principle; original image-classification results are not rescue performance evidence')
    (out/'development.json').write_text(json.dumps(report,indent=2)+'\n');(out/'history.json').write_text(json.dumps(trials,indent=2)+'\n')
    print(json.dumps(dict(chosen=chosen)),flush=True)


if __name__=='__main__':main()
