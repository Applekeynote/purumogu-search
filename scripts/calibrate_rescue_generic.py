"""Calibrate occupancy decisions on the base model's internal held-out TRAIN worlds."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('OMP_NUM_THREADS','8')
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import rescue_local_features as local
from train_rescue_generic import cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    source=ROOT/a.base;out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable model name')
    model=json.loads((source/'model.json').read_text(encoding='utf-8'));fit=joblib.load(source/'occupancy.joblib')
    local.configure(model['geometry_feature_map']);ids=set(model['valid_maps']);predictions=[]
    with threadpool_limits(limits=8):
        for row in cases():
            if row['id'] not in ids:continue
            data=json.loads(row['map'].read_bytes());x=local.features(local.evidence(data['cells']))
            probability=fit.predict_proba(x)[:,1];probability[~np.any(x[:,1::2]>0,axis=1)]=0
            predictions.append((row,probability,local.truth_mask(row['truth']['occupied_keys'])))
    trials=[]
    for threshold in [.25,.3,.35,.4,.45,.5,.55,.6,.65,.7]:
        rows=[dict(id=r['id'],source=r['source'],**local.iou(probability,truth,threshold)) for r,probability,truth in predictions]
        domains={s:float(np.mean([r['iou'] for r in rows if r['source']==s])) for s in {r['source'] for r in rows}}
        trials.append(dict(threshold=threshold,macro_iou=float(np.mean([r['iou'] for r in rows])),minimum_domain_iou=min(domains.values()),domains=domains,rows=rows))
    chosen=max(trials,key=lambda t:(t['minimum_domain_iou'],t['macro_iou']))
    threshold=chosen['threshold'];delta=math.log(.6/.4)-math.log(threshold/(1-threshold))
    model['geometry_bias']=float(fit._baseline_prediction[0,0])+delta
    # Preserve the acoustic pretraining lineage as well as the new geometry lineage.
    sensor=json.loads((ROOT/'models/swarm-v34-local/model.json').read_text(encoding='utf-8'))
    model['train_maps']=sorted(set(model['train_maps'])|set(sensor['train_maps']))
    model['valid_maps']=sorted(set(model['valid_maps'])|set(sensor['valid_maps']))
    if set(model['train_maps'])&set(model['valid_maps']):raise ValueError('Split leakage')
    out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    report=dict(purpose='Internal TRAIN world holdout calibration only; acceptance threshold remains .6',
                source=a.base,source_sha256=hashlib.sha256((source/'model.json').read_bytes()).hexdigest(),
                chosen=chosen,logit_adjustment=delta,trials=trials,test_opened=False,
                occupancy_checkpoint=str(source.relative_to(ROOT)/'occupancy.joblib'))
    (out/'calibration.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(chosen_threshold=threshold,internal_macro_iou=chosen['macro_iou'],domains=chosen['domains'])))


if __name__=='__main__':main()
