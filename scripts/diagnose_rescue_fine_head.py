"""Read auxiliary fine-grid predictions without altering the model's eval behavior."""
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
from torch.nn import functional as F
from rescue_volume_network import build_network
from rescue_validation_data import validation_cases
from rescue_range_moments import input_grid
from rescue_dataset import assert_disjoint_worlds
from rescue_local_features import truth_mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--geometry-pool',required=True)
    p.add_argument('--data',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=ROOT/a.output
    if out.exists():raise ValueError('Use a fresh diagnostic report')
    torch.set_num_threads(2);checkpoint=ROOT/a.checkpoint;digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);net=build_network(state).eval();net.load_state_dict(state['state'])
    if not hasattr(net,'fine_out') or not net.fine_unet:raise ValueError('Supervised fine U-Net required')
    development=json.loads((checkpoint.parent/'development.json').read_bytes())
    trained={r['id'] for r in development['train_audit'] if not r['held']}|set(state.get('pretraining_train_maps',[]))
    records=list(validation_cases(a.data,18000));assert_disjoint_worlds([r['id'] for r in records],trained)
    manifest=json.loads((ROOT/a.geometry_pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or manifest['test_opened'] or set(manifest['case_ids'])!={r['id'] for r in records}:raise ValueError('Exact complete VALID preparation required')
    captured={};original=net.fine_backbone.features
    def capture(x):
        value=original(x);captured['fine']=value
        return value
    net.fine_backbone.features=capture
    rows=[]
    with torch.no_grad():
        for index,row in enumerate(records,1):
            source=ROOT/a.geometry_pool/row['source']/row['id']/'map-18000.json'
            grid=input_grid(dict(map=source),state['channels']);x=torch.tensor(np.pad(grid,((0,0),(0,0),(0,3),(0,2)))).unsqueeze(0)
            coarse=net(x).sigmoid()[0,0,:32,:21,:70].numpy()>.6
            # Diagnostic only. The shipped v3 forward does not use this auxiliary
            # output; max pooling is a hypothesis, not a measured coarse surface.
            fine=F.max_pool3d(net.fine_out(captured.pop('fine')),2).sigmoid()[0,0,:32,:21,:70].numpy()>.6
            truth=truth_mask(row['truth']['occupied_keys']).reshape(32,21,70)>.5
            def iou(pred):return int((pred&truth).sum())/max(1,int((pred|truth).sum()))
            error=coarse!=truth;fixed=error&(fine==truth);spoiled=~error&(fine!=truth)
            rows.append(dict(id=row['id'],domain=row['source'],coarse_iou=iou(coarse),fine_max_iou=iou(fine),
                coarse_errors=int(error.sum()),coarse_errors_fine_would_correct=int(fixed.sum()),coarse_correct_cells_fine_would_spoil=int(spoiled.sum())))
            if index%9==0 or index==len(records):print(json.dumps(dict(completed=index,total=len(records))),flush=True)
    if digest!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():raise ValueError('Checkpoint changed during diagnostic')
    summary={role:{name:float(np.mean([r[name] for r in rows if r['domain']==role])) for name in ['coarse_iou','fine_max_iou']} for role in ['valid-known','valid-unknown']}
    report=dict(purpose='Fixed VALID fine-head diagnostic; no model fitting or changed inference',checkpoint=a.checkpoint,checkpoint_sha256=digest,
        threshold=.6,fine_resolution_m=.025,scoring_resolution_m=.05,summary=summary,rows=rows,
        oracle_caveat='Correction/spoil counts compare labels only as a diagnostic; they are not an implementable inference rule or achieved performance',
        final_test_opened=False,closed_loop_acceptance=False,eval_mode_preserved=True,device='cpu')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
