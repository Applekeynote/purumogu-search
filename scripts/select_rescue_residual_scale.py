"""Select one global residual strength on formal VALID; no refitting or TEST."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from threadpoolctl import threadpool_limits
from rescue_dataset import assert_disjoint_worlds
from rescue_validation_data import validation_cases
from rescue_residual_head import forest_logits
from train_rescue_generic_tracks import fast_metric
from train_rescue_tracks import choose


def scaled_model(base,model,strength):
    count=len(base['track_trees'])
    if not 0<=strength<=2 or model['track_trees'][:count]!=base['track_trees']:raise ValueError('Exact frozen prefix and bounded strength required')
    result=copy.deepcopy(model);result['track_bias']=base['track_bias']+strength*(model['track_bias']-base['track_bias'])
    for tree in result['track_trees'][count:]:
        for node in tree['nodes']:node['value']*=strength
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--model',action='append',required=True)
    p.add_argument('--data',action='append',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable selection name')
    base=json.loads((ROOT/a.base).read_bytes());records=[]
    for record in validation_cases(a.data,18000):
        data=json.loads(record['map'].read_bytes());case=json.loads((record['map'].parent/'complete.json').read_bytes())['case']
        if any(len(q['features'])!=32 for q in data['proposals']):raise ValueError('Recorded spectral32 prefix required')
        records.append(dict(map_id=record['id'],domain=record['source'],nodes=case['nodes'],area=case['scene'],victim_positions=record['truth']['victim_positions'],proposals=data['proposals'],cells=data['cells']))
    best=None;trials=[];parents=[]
    for source in a.model:
        model=json.loads((ROOT/source).read_bytes());scaled_model(base,model,1)
        assert_disjoint_worlds([r['map_id'] for r in records],model['train_maps'])
        parents.append(dict(path=source,sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()))
        rows=[]
        for record in records:
            row=dict(record)
            if model['track_feature_map']=='thermal-spectral-map40-v4':
                from rescue_map_context import extend
                row['proposals']=extend(record)
            elif model['track_feature_map']=='thermal-spectral-teacher33-v5':
                from rescue_residual_head import teacher_features
                if model['track_teacher_trees']!=len(base['track_trees']) or model['track_teacher_bias']!=base['track_bias']:raise ValueError('Frozen teacher metadata differs')
                row['proposals']=[dict(q,features=features.tolist()) for q,features in zip(record['proposals'],teacher_features(base,[q['features'] for q in record['proposals']]))] if record['proposals'] else []
            elif model['track_feature_map']!='thermal-spectral32-v3':raise ValueError('Unsupported residual feature map')
            row.pop('cells');matrix=np.array([q['features'] for q in row['proposals']])
            original=forest_logits(base,matrix) if len(matrix) else np.array([])
            correction=forest_logits(model,matrix)-original if len(matrix) else np.array([])
            rows.append((row,original,correction))
        for strength in [0,.25,.5,.75,1,1.25,1.5,2]:
            ranked=[]
            for row,original,correction in rows:
                scores=1/(1+np.exp(-np.clip(original+strength*correction,-40,40)))
                ranked.append((row,sorted([dict(position=q['position'],score=float(s)) for q,s in zip(row['proposals'],scores)],key=lambda q:-q['score'])))
            for threshold in [.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]:
                for nms in [.25,.3,.35,.4,.45,.5,.6]:
                    for height in [1,1.5,2,3]:
                        metric=fast_metric(ranked,threshold,nms,height)
                        precision=metric['human_precision'];recall=metric['human_recall']
                        key=(metric['minimum_stratum_criterion'],min(precision,recall),precision+recall,-abs(strength-1))
                        trial=dict(source=source,strength=strength,threshold=threshold,nms=nms,height=height,**metric);trials.append(trial)
                        if best is None or key>best[0]:best=(key,trial,scaled_model(base,model,strength),[r for r,_,_ in rows])
        print(json.dumps(dict(source=source,best=best[1])),flush=True)
    _,chosen,model,selected_rows=best
    model.update(model_id=a.name,track_threshold=chosen['threshold'],track_nms_m=chosen['nms'],track_height_scale=chosen['height'],track_refine_radius_m=0,track_refine_score_power=1)
    model['valid_maps']=sorted(set(model['valid_maps'])|{r['map_id'] for r in records})
    model['data_sha256']=hashlib.sha256(json.dumps(dict(parents=parents,chosen=chosen),sort_keys=True).encode()).hexdigest()
    metric=choose(selected_rows,model,chosen['threshold'],chosen['nms'])
    out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    (out/'development.json').write_text(json.dumps(dict(purpose='Global residual-strength hyperparameter selected on VALID; no new tree fitting and no final TEST',
        final_test_opened=False,closed_loop_acceptance=False,parents=parents,base=a.base,data=a.data,chosen=chosen,metrics=metric,trials=trials),indent=2)+'\n')
    print(json.dumps(dict(chosen=chosen,metrics={k:metric[k] for k in ['matched','people','false_alarms','human_precision','human_recall']})),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
