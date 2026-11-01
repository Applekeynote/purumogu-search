"""Compare the exported standalone Rust inference against the fitted Python checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('OMP_NUM_THREADS','8')
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import rescue_local_features as local


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--occupancy',required=True)
    p.add_argument('--map',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--binary',default='tmp/generic-rust-target/release/complete_map.exe');a=p.parse_args()
    model_path=ROOT/a.model;model=json.loads(model_path.read_text(encoding='utf-8'));local.configure(model['geometry_feature_map'])
    fit=joblib.load(ROOT/a.occupancy);rows=[]
    with tempfile.TemporaryDirectory(prefix='generic-parity-',dir=ROOT/'tmp') as folder,threadpool_limits(limits=8):
        for i,name in enumerate(a.map):
            source=ROOT/name;destination=Path(folder)/f'{i}.json'
            completed=subprocess.run([str(ROOT/a.binary),str(model_path),str(source),str(destination)],check=True,capture_output=True,text=True)
            runtime=json.loads(destination.read_bytes());data=json.loads(source.read_bytes())
            x=local.features(local.evidence(data['cells']));probability=fit.predict_proba(x)[:,1]
            logit=np.log(np.clip(probability,1e-15,1-1e-15)/(1-np.clip(probability,1e-15,1-1e-15)))
            probability=1/(1+np.exp(-np.clip(logit+model['geometry_bias']-float(fit._baseline_prediction[0,0]),-40,40)))
            values=[]
            for cell in runtime['cells']:
                index=np.ravel_multi_index(tuple(np.array(cell['key'])-local.MINIMUM),local.SHAPE)
                values.append((probability[index],cell['occupancy']))
            values=np.array(values);error=float(np.max(np.abs(values[:,0]-values[:,1])))
            disagreement=int(np.sum((values[:,0]>.6)!=(values[:,1]>.6)))
            if error>1e-6 or disagreement:raise AssertionError(f'Runtime parity failed: {error}/{disagreement}')
            rows.append(dict(map=name,map_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cells=len(values),
                             maximum_probability_error=error,class_disagreements=disagreement,**json.loads(completed.stdout)))
    report=dict(status='pass',model=a.model,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                binary_sha256=hashlib.sha256((ROOT/a.binary).read_bytes()).hexdigest(),rows=rows)
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
