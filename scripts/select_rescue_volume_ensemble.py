"""Select reflection averaging/ensemble calibration on VALID only; never opens TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
import torch
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask,iou
from rescue_volume_network import VolumeNet,reflect_evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);device='cuda';networks=[];rows=[];predictions=[]
    for name in a.checkpoint:
        path=ROOT/name;state=torch.load(path,map_location=device,weights_only=True)
        if state.get('tiled'):raise ValueError('This comparison expects full-volume models')
        net=VolumeNet(state['width'],state['residual'],state['channels'],state.get('profiles',False),state.get('normalization','group')).to(device);net.load_state_dict(state['state']);net.eval()
        cache=ROOT/'tmp/generic-reflection-predictions'/hashlib.sha256(path.read_bytes()).hexdigest();cache.mkdir(parents=True,exist_ok=True)
        networks.append((net,state['channels'],cache))
    paths=[(split,path) for split in ['valid-known','valid-unknown'] for path in sorted((ROOT/'experiments/raw/rescue/generalization-v35'/split).glob('*/complete.json'))]
    with torch.no_grad():
        for split,path in paths:
            case=json.loads(path.read_bytes());truth=truth_mask(json.loads((path.parent/'truth-6000.json').read_bytes())['occupied_keys'])
            fields=[]
            for net,channels,cache in networks:
                target=cache/(case['id']+'.npz')
                if target.exists():probability=np.load(target)['probability']
                else:
                    grid=np.pad(input_grid(dict(map=path.parent/'map-6000.json'),channels),((0,0),(0,0),(0,3),(0,2)))
                    x=torch.tensor(grid,device=device).unsqueeze(0);images=[]
                    for flip_x,flip_z in [(False,False),(True,False),(False,True),(True,True)]:
                        transformed=x.clone()
                        if flip_x:transformed=reflect_evidence(transformed,2)
                        if flip_z:transformed=reflect_evidence(transformed,4)
                        prediction=net(transformed).sigmoid()
                        if flip_z:prediction=prediction.flip(4)
                        if flip_x:prediction=prediction.flip(2)
                        images.append(prediction[0,0,:32,:21,:70].cpu().numpy().reshape(-1))
                    probability=np.stack(images);np.savez_compressed(target,probability=probability)
                fields.append(probability)
            rows.append(dict(id=case['id'],scene=case['scene'],nodes=case['nodes'],split=split,truth=truth))
            predictions.append(fields)
    trials=[]
    # Every candidate uses the same observed input; no training/validation world IDs are model features.
    combinations=[([i],None) for i in range(len(networks))]+[(list(range(len(networks))),None)]
    for selected,_ in combinations:
      for views in [1,4]:
       for threshold in [.4,.45,.5,.55,.6,.65,.7,.75,.8,.85]:
        metrics=[]
        for row,fields in zip(rows,predictions):
            probability=np.mean([fields[i][:views].mean(axis=0) for i in selected],axis=0)
            metrics.append(dict(id=row['id'],scene=row['scene'],nodes=row['nodes'],split=row['split'],**iou(probability,row['truth'],threshold)))
        domains={s:float(np.mean([r['iou'] for r in metrics if r['split']==s])) for s in ['valid-known','valid-unknown']}
        trials.append(dict(models=[a.checkpoint[i] for i in selected],views=views,threshold=threshold,domains=domains,
                           minimum_case_iou=min(r['iou'] for r in metrics),rows=metrics))
    best=max(trials,key=lambda t:(min(t['domains'].values()),sum(t['domains'].values())))
    result=dict(purpose='Formal VALID ensemble selection, not final acceptance',test_opened=False,best=best,
                trials=[{k:v for k,v in trial.items() if k!='rows'} for trial in trials])
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in best.items() if k!='rows'}))


if __name__=='__main__':main()
