"""Research-only global convex blend of TRAIN-fitted proposal heads on VALID."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from rescue_dataset import assert_disjoint_worlds
from rescue_residual_head import forest_logits,teacher_features
from rescue_validation_data import validation_cases
from train_rescue_generic_tracks import fast_metric
from train_rescue_tracks import choose


def blend(base,left,right,weight):
    count=len(base['track_trees'])
    if not 0<=weight<=1 or any(model['track_trees'][:count]!=base['track_trees'] for model in [left,right]):raise ValueError('Exact shared teacher prefix and convex weights required')
    for model in [left,right]:
        if model['track_feature_map'] not in ['thermal-spectral32-v3','thermal-spectral-teacher33-v5']:raise ValueError('Only the frozen native32/33 feature contract is supported')
        if model['track_feature_map']=='thermal-spectral-teacher33-v5' and (model['track_teacher_trees']!=count or model['track_teacher_bias']!=base['track_bias']):raise ValueError('Shared independent teacher metadata required')
    result=copy.deepcopy(right)
    result.update(track_bias=weight*left['track_bias']+(1-weight)*right['track_bias'],track_trees=copy.deepcopy(base['track_trees']),
        track_feature_map='thermal-spectral-teacher33-v5',track_teacher_trees=count,track_teacher_bias=base['track_bias'])
    for model,strength in [(left,weight),(right,1-weight)]:
        if not strength:continue
        trees=copy.deepcopy(model['track_trees'][count:])
        for tree in trees:
            for node in tree['nodes']:node['value']*=strength
        result['track_trees'].extend(trees)
    result['train_maps']=sorted(set(left['train_maps'])|set(right['train_maps']))
    result['valid_maps']=sorted(set(left['valid_maps'])|set(right['valid_maps']))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--data',action='append',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a fresh immutable selection name')
    base,left,right=[json.loads((ROOT/path).read_bytes()) for path in [a.base,a.left,a.right]]
    blend(base,left,right,.5);records=[]
    for row in validation_cases(a.data,18000):
        mapping=json.loads(row['map'].read_bytes());case=json.loads((row['map'].parent/'complete.json').read_bytes())['case']
        proposals=mapping['proposals']
        if any(len(q['features'])!=32 for q in proposals):raise ValueError('Original native32 proposals required')
        enriched=[dict(q,features=f.tolist()) for q,f in zip(proposals,teacher_features(base,[q['features'] for q in proposals]))] if proposals else []
        records.append(dict(map_id=row['id'],domain=row['source'],area=case['scene'],nodes=case['nodes'],victim_positions=row['truth']['victim_positions'],proposals=enriched))
    for model in [left,right]:assert_disjoint_worlds([r['map_id'] for r in records],model['train_maps'])
    logits=[]
    for row in records:
        matrix=np.array([q['features'] for q in row['proposals']]);logits.append((forest_logits(left,matrix),forest_logits(right,matrix)) if len(matrix) else (np.array([]),np.array([])))
    trials=[];best=None
    for weight in [0,.25,.5,.75,1]:
        ranked=[]
        for row,(l,r) in zip(records,logits):
            scores=1/(1+np.exp(-np.clip(weight*l+(1-weight)*r,-40,40)))
            ranked.append((row,sorted([dict(position=q['position'],score=float(s)) for q,s in zip(row['proposals'],scores)],key=lambda q:-q['score'])))
        for threshold in [.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95,.97,.99]:
            for nms in [.25,.3,.35,.4,.45,.5,.6]:
                for height in [1,1.5,2,3]:
                    metric=fast_metric(ranked,threshold,nms,height);precision=metric['human_precision'];recall=metric['human_recall']
                    key=(metric['minimum_stratum_criterion'],min(precision,recall),precision+recall,int(weight in [0,1]))
                    trial=dict(left_weight=weight,threshold=threshold,nms=nms,height=height,**metric);trials.append(trial)
                    if best is None or key>best[0]:best=(key,trial)
        print(json.dumps(dict(weight=weight,current_best=best[1])),flush=True)
    chosen=best[1];model=blend(base,left,right,chosen['left_weight'])
    model.update(model_id=a.name,track_threshold=chosen['threshold'],track_nms_m=chosen['nms'],track_height_scale=chosen['height'],track_refine_radius_m=0,track_refine_score_power=1)
    model['valid_maps']=sorted(set(model['valid_maps'])|{r['map_id'] for r in records})
    metric=choose(records,model,chosen['threshold'],chosen['nms']);trees=len(model['track_trees'])
    out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    report=dict(purpose='Global two-head logit blend selected on formal VALID, not final acceptance',final_test_opened=False,closed_loop_acceptance=False,
        parents=[dict(path=path,sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()) for path in [a.base,a.left,a.right]],
        chosen=chosen,metrics=metric,trials=trials,exported_trees=trees,current_native_tree_limit=256,
        native_support_requires_separate_validation=trees>256,data=a.data,no_tree_refitting=True,no_per_area_or_count_parameters=True)
    (out/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(chosen=chosen,metrics={k:metric[k] for k in ['matched','people','false_alarms','human_precision','human_recall']},native_supported=trees<=256)),flush=True)


if __name__=='__main__':main()
