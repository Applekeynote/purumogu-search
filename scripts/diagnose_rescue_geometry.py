"""Attribute development occupancy errors after inference; never feed labels back."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from rescue_volume_network import build_network
from rescue_range_moments import input_grid
from rescue_local_features import truth_mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);device='cuda';state=torch.load(ROOT/a.checkpoint,map_location=device,weights_only=True)
    net=build_network(state).to(device);net.load_state_dict(state['state']);net.eval();rows=[]
    for area,split in [(25,'valid-known'),(26,'valid-known'),(29,'valid-unknown')]:
        for folder in sorted((ROOT/'experiments/raw/rescue/generalization-v41-mission360'/split).glob(f'g41-*-S{area:02d}-*')):
            source=ROOT/'experiments/raw/rescue/generalization-v41-mission360/registered-y74-final360'/split/folder.name/'map-18000.json'
            grid=input_grid(dict(map=source),state['channels']);x=np.pad(grid,((0,0),(0,0),(0,3),(0,2)))
            with torch.no_grad():pred=net(torch.tensor(x,device=device).unsqueeze(0)).sigmoid()[0,0,:32,:21,:70].cpu().numpy()>.6
            # Labels are opened only after producing predictions.
            truth=json.loads((folder/'truth-18000.json').read_bytes());y=truth_mask(truth['occupied_keys']).reshape(32,21,70)>0
            fp=pred&~y;fn=~pred&y;seen=grid[1]>0
            rows.append(dict(id=folder.name,area=area,false_occupied=int(fp.sum()),missed_occupied=int(fn.sum()),
                unobserved_missed=int((fn&~seen).sum()),unobserved_false=int((fp&~seen).sum()),
                height=[dict(y_index=i-3,true_occupied=int(y[:,i,:].sum()),fp=int(fp[:,i,:].sum()),fn=int(fn[:,i,:].sum())) for i in range(21)]))
    result=dict(purpose='Development error attribution, labels accessed after inference',checkpoint=a.checkpoint,test_opened=False,rows=rows)
    path=ROOT/a.output
    if path.exists():raise ValueError('Use an immutable new report')
    path.write_text(json.dumps(result,indent=2)+'\n')
    for area in [25,26,29]:
        group=[r for r in rows if r['area']==area]
        print(json.dumps(dict(area=area,fp=sum(r['false_occupied'] for r in group),fn=sum(r['missed_occupied'] for r in group),unobserved_fn=sum(r['unobserved_missed'] for r in group),
            worst_heights=sorted([dict(y=i-3,errors=sum(r['height'][i]['fp']+r['height'][i]['fn'] for r in group)) for i in range(21)],key=lambda r:-r['errors'])[:6])))

if __name__=='__main__':main()
