"""Initialize a wider GroupNorm U-Net and verify its initial function on TRAIN."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
import torch
from rescue_volume_network import VolumeNet
from rescue_range_moments import input_grid


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    p.add_argument('--probe',required=True);a=p.parse_args()
    probe=ROOT/a.probe
    if 'train' not in probe.parts or any(part.startswith(('valid','test')) for part in probe.parts):raise ValueError('TRAIN-only numerical probe required')
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable initialization name')
    torch.set_num_threads(4);torch.manual_seed(20261031)
    original=torch.load(ROOT/a.checkpoint,map_location='cpu',weights_only=True)
    if original.get('normalization','group')!='group':raise ValueError('GroupNorm-only widening')
    width=original['width'];channels=original['channels'];profiles=original.get('profiles',False)
    small=VolumeNet(width,original['residual'],channels,profiles).eval();small.load_state_dict(original['state'])
    wide=VolumeNet(width*2,original['residual'],channels,profiles).eval();state={}
    for key,target in wide.state_dict().items():
        value=original['state'][key]
        if value.shape==target.shape:state[key]=value.clone();continue
        if value.ndim==1 and target.shape[0]==value.shape[0]*2:
            state[key]=value.repeat_interleave(2,0)
        elif value.ndim==5:
            factors=[target.shape[d]//value.shape[d] for d in [0,1]]
            if any(f not in [1,2] for f in factors) or any(target.shape[d]!=value.shape[d]*factors[d] for d in [0,1]):raise ValueError('Unexpected convolution shape')
            expanded=value.repeat_interleave(factors[0],0).repeat_interleave(factors[1],1)/factors[1]
            state[key]=expanded
        else:raise ValueError('Unsupported widening: '+key)
    # Opposite outgoing perturbations cancel for identical duplicated features,
    # while allowing their gradients to diverge on the first training update.
    weight=state['out.weight'];noise=torch.randn_like(weight[:,::2])*float(weight.std())*.01
    weight[:,::2]+=noise;weight[:,1::2]-=noise
    wide.load_state_dict(state)
    grid=input_grid(dict(map=probe),channels)
    x=torch.tensor(np.pad(grid,((0,0),(0,0),(0,3),(0,2)))).unsqueeze(0)
    with torch.no_grad():before=small(x);after=wide(x)
    difference=float((before-after).abs().max());changed=int(((before.sigmoid()>.6)!=(after.sigmoid()>.6)).sum())
    if difference>2e-4 or changed:raise ValueError(f'Widening changed initialization: {difference}, {changed}')
    out.mkdir(parents=True)
    result=dict(original,width=width*2,state=state,initialization='GroupNorm-preserving doubled channels with paired outgoing perturbations')
    torch.save(result,out/'volume.pt')
    report=dict(source=a.checkpoint,source_sha256=hashlib.sha256((ROOT/a.checkpoint).read_bytes()).hexdigest(),
        probe=a.probe,maximum_logit_difference=difference,threshold_decisions_changed=changed,
        old_parameters=sum(p.numel() for p in small.parameters()),new_parameters=sum(p.numel() for p in wide.parameters()),
        acceptance=False,reference='https://arxiv.org/abs/1511.05641')
    (out/'initialization.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
