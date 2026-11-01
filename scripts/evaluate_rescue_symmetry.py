"""Fixed-threshold symmetry averaging on development data, never final TEST."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from rescue_volume_network import build_network,rotate_evidence,reflect_evidence
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask,iou

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--geometry-pool',required=True)
    p.add_argument('--validation-root',default='experiments/raw/rescue/generalization-v41-mission360');p.add_argument('--extra-validation-root',action='append')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--views',type=int,choices=[2,4,8],default=8)
    p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);device=a.device;path=ROOT/a.checkpoint;state=torch.load(path,map_location=device,weights_only=True)
    net=build_network(state).to(device);net.load_state_dict(state['state']);net.eval()
    selections={'identity':[(0,False)],'halfturn2':[(0,False),(2,False)]}
    if a.views>=4:selections['reflections4']=[(0,False),(0,True),(2,False),(2,True)]
    if a.views==8:selections['dihedral8']=[(turn,flip) for turn in range(4) for flip in [False,True]]
    transforms=sorted(set(v for selection in selections.values() for v in selection));results={k:[] for k in selections}
    from rescue_validation_data import validation_cases
    from rescue_dataset import assert_disjoint_worlds
    development=json.loads((path.parent/'development.json').read_bytes());trained={r['id'] for r in development['train_audit'] if not r['held']}|set(state.get('pretraining_train_maps',[]))
    records=list(validation_cases([a.validation_root]+(a.extra_validation_root or []),18000));assert_disjoint_worlds([r['id'] for r in records],trained)
    manifest=json.loads((ROOT/a.geometry_pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or manifest['test_opened'] or not {r['id'] for r in records}<=set(manifest['case_ids']):raise ValueError('Matching development feature preparation required')
    for completed,record in enumerate(records,1):
            split=record['source'];case=json.loads((record['map'].parent/'complete.json').read_bytes())['case']
            source=ROOT/a.geometry_pool/split/case['id']/'map-18000.json'
            grid=np.pad(input_grid(dict(map=source),state['channels']),((0,0),(0,0),(0,3),(0,2)))
            x=torch.tensor(grid,device=device).unsqueeze(0);predictions={}
            with torch.no_grad():
                for turn,flip in transforms:
                    transformed=rotate_evidence(x.clone(),turn)
                    if flip:transformed=reflect_evidence(transformed,2)
                    prediction=net(transformed).sigmoid()
                    if flip:prediction=prediction.flip(2)
                    prediction=torch.rot90(prediction,-turn,(2,4))
                    predictions[(turn,flip)]=prediction[0,0,:32,:21,:70].cpu().numpy().reshape(-1)
            # The same global transformations apply to every world. Labels are only scores.
            truth=truth_mask(record['truth']['occupied_keys'])
            for name,indices in selections.items():
                probability=np.mean([predictions[i] for i in indices],axis=0)
                results[name].append(dict(id=case['id'],split=split,area=case['scene'],nodes=case['nodes'],**iou(probability,truth)))
            if completed%9==0 or completed==len(records):print(json.dumps(dict(completed=completed,total=len(records),views=len(transforms))),flush=True)
    summaries={}
    for name,rows in results.items():
        areas={str(area):float(np.mean([r['iou'] for r in rows if r['area']==area])) for area in sorted({r['area'] for r in rows})}
        summaries[name]=dict(domains={split:float(np.mean([r['iou'] for r in rows if r['split']==split])) for split in ['valid-known','valid-unknown']},areas=areas,minimum_area_iou=min(areas.values()))
    report=dict(purpose='Development symmetry averaging; not closed-loop acceptance',test_opened=False,checkpoint=a.checkpoint,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),threshold=.6,step=18000,summaries=summaries,rows=results,
        validation_roots=[a.validation_root]+(a.extra_validation_root or []),device=a.device,evaluated_views=len(transforms),
        primary_method_reference='https://arxiv.org/abs/1807.07356',reference_scope='Segmentation test-time augmentation, not evidence of rescue accuracy')
    target=ROOT/a.output
    if target.exists():raise ValueError('Use a new report')
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries))

if __name__=='__main__':main()
