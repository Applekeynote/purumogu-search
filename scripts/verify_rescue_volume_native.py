"""Verify the whole exported feature tensor and standalone CPU Rust inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
import torch
from rescue_volume_network import build_network
from rescue_range_moments import input_grid
from rescue_local_features import MINIMUM

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--server',default='tmp/generic-tsdf-rust-target/release/complete_map.exe');a=p.parse_args()
    folder=ROOT/a.model;export=json.loads((folder/'export.json').read_bytes())
    checkpoint=ROOT/export['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=export['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);torch.set_num_threads(2)
    net=build_network(state);net.load_state_dict(state['state']);net.eval()
    grid=input_grid(dict(map=ROOT/export['map']),state['channels'])
    x=torch.tensor(np.pad(grid,((0,0),(0,0),(0,3),(0,2)))).unsqueeze(0)
    with torch.no_grad():expected=net(x).sigmoid()[0,0,:32,:21,:70].numpy()
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='native-volume-parity-') as temporary:
        output=Path(temporary)/'map.json'
        run=subprocess.run([str(ROOT/a.server),str(folder/'model.json'),
            str(ROOT/export['probe_map']),str(output)],check=True,capture_output=True,text=True)
        cells=json.loads(output.read_bytes())['cells'];actual=np.full(expected.shape,.5,dtype=np.float32);errors=[]
        for cell in cells:
            key=tuple(np.array(cell['key'])-MINIMUM);actual[key]=cell['occupancy'];errors.append(abs(float(expected[key])-cell['occupancy']))
        disagreements=int(np.sum((actual>.6)!=(expected>.6)));maximum=max(errors)
        if maximum>1e-3 or disagreements:raise AssertionError(f'Native parity failed: {maximum}/{disagreements}')
    report=dict(status='pass',model=a.model,checkpoint_sha256=export['checkpoint_sha256'],onnx_sha256=export['onnx_sha256'],
        architecture=export['architecture'],channels=state['channels'],maximum_probability_error=maximum,class_disagreements=disagreements,
        truth_to_native=False,requires_python_for_app=False,requires_cuda_for_app=False,
        native_binary=a.server,native_binary_sha256=hashlib.sha256((ROOT/a.server).read_bytes()).hexdigest(),**json.loads(run.stdout))
    path=ROOT/a.output
    if path.exists():raise ValueError('Use a new report')
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
