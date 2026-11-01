"""Offline completion/track evaluation; acquisition control is unchanged by these two heads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS','8');sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
import rescue_local_features as local
from train_rescue_tracks import choose
from evaluate_rescue_replay import match


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--occupancy',required=True)
    p.add_argument('--split',choices=['train','valid-known','valid-unknown','test-known','test-unknown'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--final-test',action='store_true');a=p.parse_args()
    if a.split.startswith('test') and not a.final_test:raise ValueError('Frozen TEST needs --final-test')
    model_path=ROOT/a.model;model=json.loads(model_path.read_text(encoding='utf-8'));local.configure(model['geometry_feature_map'])
    occupancy_path=ROOT/a.occupancy;fit=joblib.load(occupancy_path);rows=[]
    cache=ROOT/'tmp/generic-probability-cache'/hashlib.sha256(occupancy_path.read_bytes()).hexdigest()
    cache.mkdir(parents=True,exist_ok=True)
    paths=sorted((ROOT/'experiments/raw/rescue/generalization-v35'/a.split).glob('*/complete.json'))
    if not paths:raise ValueError('No completed cases')
    with threadpool_limits(limits=8):
        for path in paths:
            case=json.loads(path.read_text(encoding='utf-8'));data=json.loads((path.parent/'map-6000.json').read_text(encoding='utf-8'))
            truth=json.loads((path.parent/'truth-6000.json').read_text(encoding='utf-8'))
            if a.split!='train' and case['id'] in model['train_maps']:raise ValueError('Training/validation collision')
            cache_path=cache/(hashlib.sha256((path.parent/'map-6000.json').read_bytes()).hexdigest()+'.npz')
            if cache_path.exists():
                stored=np.load(cache_path);raw=stored['probability'];known=stored['context_known']
            else:
                x=local.features(local.evidence(data['cells']));raw=fit.predict_proba(x)[:,1];known=np.any(x[:,1::2]>0,axis=1)
                np.savez_compressed(cache_path,probability=raw,context_known=known)
            delta=model['geometry_bias']-float(fit._baseline_prediction[0,0])
            probability=1/(1+np.exp(-np.clip(np.log(np.clip(raw,1e-15,1-1e-15)/(1-np.clip(raw,1e-15,1-1e-15)))+delta,-40,40)))
            probability[~known]=.5
            metric=local.iou(probability,local.truth_mask(truth['occupied_keys']))
            if model.get('track_trees') or model.get('track_weights'):
                decision=choose([dict(map_id=case['id'],nodes=case['nodes'],proposals=data['proposals'],victim_positions=truth['victim_positions'])],model,model['track_threshold'],model['track_nms_m'])['rows'][0]
            else:
                matched,_=match(data['cues'],truth['victim_positions']);decision=dict(matched=matched,false_alarms=len(data['cues'])-matched,people=len(truth['victim_positions']))
            row=dict(id=case['id'],scene=case['scene'],nodes=case['nodes'],lost_nodes=case['lost_nodes'],
                     baseline_iou=case['global_iou'],**metric,**{k:decision[k] for k in ['matched','false_alarms','people']})
            rows.append(row)
    def aggregate(group):
        tp=sum(r['matched'] for r in group);fp=sum(r['false_alarms'] for r in group);people=sum(r['people'] for r in group)
        return dict(cases=len(group),macro_iou=float(np.mean([r['iou'] for r in group])),
                    micro_iou=sum(r['intersection'] for r in group)/max(1,sum(r['union'] for r in group)),
                    minimum_case_iou=min(r['iou'] for r in group),human_precision=tp/(tp+fp) if tp+fp else None,
                    human_recall=tp/people if people else None,matched=tp,false_alarms=fp,people=people)
    result=dict(split=a.split,model=a.model,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                purpose='Recorded actual observations, generic local completion and temporal classification; same acquisition policy',
                test_opened=a.split.startswith('test'),summary=aggregate(rows),
                by_node_count={str(n):aggregate([r for r in rows if r['nodes']==n]) for n in sorted({r['nodes'] for r in rows})},
                by_area={s:aggregate([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})},rows=rows)
    result['all_95']=all(v['macro_iou']>=.95 and all(v[k] is None or v[k]>=.95 for k in ['human_precision','human_recall'])
                         and (v['people']>0 or v['false_alarms']==0)
                         for v in [result['summary'],*result['by_node_count'].values(),*result['by_area'].values()])
    dest=ROOT/a.output;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(split=a.split,summary=result['summary'],all_95=result['all_95'])),flush=True)


if __name__=='__main__':main()
