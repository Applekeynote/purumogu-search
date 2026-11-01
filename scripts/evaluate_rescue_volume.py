"""Evaluate the 3D completion network on actual observed worlds, never truth inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from rescue_local_features import evidence,truth_mask,iou
from rescue_volume_network import build_network,tiled_logits
from rescue_generic_metrics import tables
from train_rescue_tracks import choose


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--tracks',required=True)
    p.add_argument('--split',choices=['train','valid-known','valid-unknown','test-known','test-unknown','test-validation-families'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--final-test',action='store_true');p.add_argument('--bias',type=float,default=0)
    p.add_argument('--geometry-pool');p.add_argument('--proposal-pool')
    p.add_argument('--step',type=int,choices=[6000,18000],default=6000);p.add_argument('--native-proposals',action='store_true')
    p.add_argument('--data-root',default='experiments/raw/rescue/generalization-v35');p.add_argument('--plan-root',default='experiments/generalization-v35');a=p.parse_args()
    if a.split.startswith('test') and not a.final_test:raise ValueError('Frozen TEST needs explicit --final-test')
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=ROOT/a.checkpoint;state=torch.load(checkpoint,map_location=device,weights_only=True)
    model=build_network(state).to(device);model.load_state_dict(state['state']);model.eval()
    tracks=json.loads((ROOT/a.tracks).read_text(encoding='utf-8'));rows=[]
    for key in ['plane_registration_v1','axis_plane_registration_v2']:
        if tracks.get(key,False)!=state.get(key,False):raise ValueError('Geometry and person pose contracts differ')
    if state.get('plane_registration_v1') and not a.geometry_pool:raise ValueError('Registered observations required')
    if tracks.get('acoustic_peaks_v2') and not (a.proposal_pool or a.native_proposals):raise ValueError('Matched native proposal pool required')
    if a.native_proposals:
        collection=json.loads((ROOT/a.data_root/a.split/'collection.json').read_bytes())
        if collection['status']!='complete' or collection['steps']<a.step:raise ValueError('Complete native acquisition required')
        source=ROOT/collection['plan']['model']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=collection['plan']['model_sha256']:raise ValueError('Acquisition model changed')
        source=json.loads(source.read_bytes())
        from rescue_dataset import acoustic_feature_signature
        if acoustic_feature_signature(source)!=acoustic_feature_signature(tracks) or source.get('track_feature_map')!=tracks.get('track_feature_map'):raise ValueError('Native proposal feature contract differs')
    if a.proposal_pool:
        manifest=json.loads((ROOT/a.proposal_pool/'manifest.json').read_bytes())
        if manifest['status']!='complete':raise ValueError('Incomplete proposal replay')
        proposal_model=json.loads((ROOT/manifest['model']).read_bytes())
        from rescue_dataset import acoustic_feature_signature
        if acoustic_feature_signature(proposal_model)!=acoustic_feature_signature(tracks):raise ValueError('Proposal feature contract differs')
    paths=sorted((ROOT/a.data_root/a.split).glob('*/complete.json'))
    if not paths:raise ValueError('No completed cases')
    with torch.no_grad():
        for path in paths:
            case=json.loads(path.read_text(encoding='utf-8'));data=json.loads((path.parent/f'map-{a.step}.json').read_text(encoding='utf-8'))
            truth=json.loads((path.parent/f'truth-{a.step}.json').read_text(encoding='utf-8'))
            if a.proposal_pool:data['proposals']=json.loads((ROOT/a.proposal_pool/a.split/path.parent.name/f'map-{a.step}.json').read_bytes())['proposals']
            if tracks.get('track_feature_map')=='thermal16-v1' and any(len(p['features'])!=16 for p in data['proposals']):
                from rescue_thermal_features import extend
                data['proposals']=extend(path.parent,a.step,data['proposals'])
            geometry=ROOT/a.geometry_pool/a.split/path.parent.name/f'map-{a.step}.json' if a.geometry_pool else path.parent/f'map-{a.step}.json'
            geometry_data=json.loads(geometry.read_bytes()) if a.geometry_pool else data
            grid=evidence(geometry_data['cells']).transpose(3,0,1,2);grid=np.pad(grid,((0,0),(0,0),(0,3),(0,2)))
            if state.get('channels',2)>2:
                from rescue_range_moments import input_grid
                grid=input_grid(dict(map=geometry),state['channels'])
                grid=np.pad(grid,((0,0),(0,0),(0,3),(0,2)))
            tensor=torch.tensor(grid,device=device).unsqueeze(0)
            logits=tiled_logits(model,tensor) if state.get('tiled',False) else model(tensor)
            probability=(logits+a.bias).sigmoid()[0,0,:32,:21,:70].cpu().numpy().reshape(-1)
            metric=iou(probability,truth_mask(truth['occupied_keys']))
            decision=choose([dict(map_id=case['id'],nodes=case['nodes'],proposals=data['proposals'],victim_positions=truth['victim_positions'])],tracks,tracks['track_threshold'],tracks['track_nms_m'])['rows'][0]
            labels=next((r for r in map(json.loads,(path.parent/'labels.jsonl').read_text().splitlines()) if r['step']==a.step),None)
            if labels is None:raise ValueError('Matching node-state checkpoint required')
            rows.append(dict(id=case['id'],scene=case['scene'],nodes=case['nodes'],lost_nodes=sum(r['lost'] for r in labels['robots']),
                             baseline_iou=truth['metrics']['global_iou'],**metric,**{k:decision[k] for k in ['matched','false_alarms','people']}))
    plan=json.loads((ROOT/a.plan_root/f'{a.split}.json').read_text(encoding='utf-8'))
    result=dict(split=a.split,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                track_model=a.tracks,bias=a.bias,purpose='Actual recorded observations; 3D completion and temporal evidence classifier',
                geometry_pool=a.geometry_pool,proposal_pool=a.proposal_pool,native_proposals=a.native_proposals,evaluation_step=a.step,seconds=a.step*.02,closed_loop_acceptance=False,
                expected_cases=len(plan['cases']),collection_complete=len(paths)==len(plan['cases']),test_opened=a.split.startswith('test'),
                **tables(rows),rows=rows)
    result['all_95'] &= result['collection_complete']
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(split=a.split,summary=result['summary'],all_95=result['all_95'],collection_complete=result['collection_complete'])))


if __name__=='__main__':main()
