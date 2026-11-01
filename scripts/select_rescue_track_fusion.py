"""Select global candidate-position fusion on development VALID, never TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from threadpoolctl import threadpool_limits
from train_rescue_tracks import score_model
from evaluate_rescue_replay import match


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--name',required=True)
    p.add_argument('--research-vertical-caps',action='store_true',help='Research only: active caps are not supported by the current native model')
    p.add_argument('--step',type=int,default=6000)
    p.add_argument('--extra-data',action='append');p.add_argument('--export-model',action='store_true')
    a=p.parse_args()
    model=json.loads((ROOT/a.model).read_bytes());rows=[]
    from rescue_validation_data import validation_cases
    from rescue_dataset import assert_disjoint_worlds
    if a.step<=0 or a.step%1500:raise ValueError('Recorded development checkpoint required')
    seen=set()
    for source in [a.data]+(a.extra_data or []):
        for record in validation_cases([source],a.step,seen):
            seen.add(record['id']);folder=record['map'].parent;item=json.loads((folder/'complete.json').read_bytes())['case'];split=record['source']
            data=json.loads(record['map'].read_bytes());truth=record['truth']['victim_positions']
            if model.get('track_feature_map')=='thermal-spectral-teacher33-v5' and data['proposals']:
                from rescue_residual_head import teacher_features
                teacher=dict(track_bias=model['track_teacher_bias'],track_trees=model['track_trees'][:model['track_teacher_trees']])
                values=teacher_features(teacher,[q['features'] for q in data['proposals']])
                data['proposals']=[dict(q,features=f.tolist()) for q,f in zip(data['proposals'],values)]
            elif model.get('track_feature_map')=='thermal-spectral-map40-v4':
                from rescue_map_context import extend
                data['proposals']=extend(data)
            candidates=sorted(data['proposals'],key=lambda q:-score_model(model,q['features']))
            positions=np.array([q['position'] for q in candidates]).reshape(-1,3);scores=np.array([score_model(model,q['features']) for q in candidates])
            rows.append(dict(id=item['id'],nodes=item['nodes'],area=item['scene'],domain=split,truth=truth,positions=positions,scores=scores))
    assert_disjoint_worlds(seen,set(model['train_maps'])|set(model.get('calibration_train_maps',[])))
    if a.export_model and a.research_vertical_caps:raise ValueError('Research-only vertical caps cannot be exported')
    def metrics(decisions):
        groups={};details=[]
        for row,cues in zip(rows,decisions):
            count,_=match([dict(position=q.tolist()) for q in cues],row['truth']);fp=len(cues)-count;people=len(row['truth'])
            for key in [('all','all'),('nodes',row['nodes']),('area',row['area']),('domain',row['domain'])]:
                tp,alarm,n=groups.get(key,(0,0,0));groups[key]=(tp+count,alarm+fp,n+people)
            details.append(dict(id=row['id'],matched=count,false_alarms=fp,people=people,cues=[q.tolist() for q in cues]))
        tp,fp,people=groups[('all','all')];precision=tp/max(1,tp+fp);recall=tp/max(1,people)
        minimum=min(v for tp,fp,n in groups.values() for v in ([tp/max(1,tp+fp),tp/n] if n else [1. if fp==0 else 0.]))
        return (minimum,min(precision,recall),precision+recall),dict(precision=precision,recall=recall,matched=tp,false_alarms=fp,people=people,minimum_stratum=minimum,rows=details)
    best=None;trials=[]
    for height,vertical in [(h,v) for h in [1,1.5,2,3] for v in ([.15,.2,.25,.35,100] if a.research_vertical_caps else [100])]:
        distances=[]
        for row in rows:
            raw=row['positions'][:,None,:]-row['positions'][None,:,:];vectors=raw*np.array([1,height,1]);distance=np.sqrt((vectors*vectors).sum(2));distance[np.abs(raw[:,:,1])>vertical]=np.inf;distances.append(distance)
        for radius in [0,.25,.35,.45,.6]:
          for power in ([1] if radius==0 else [.5,1,2]):
            centers=[]
            for row,d in zip(rows,distances):
                if not len(d):centers.append(row['positions']);continue
                weight=(d<=radius)*np.maximum(row['scores'][None,:],1e-9)**power
                # Weak alternative peaks can move a maximum, but cannot create a new seed.
                centers.append((weight@row['positions'])/weight.sum(1)[:,None])
            for nms in [.25,.3,.35,.4,.45,.5,.6]:
              if radius>nms:continue
              for threshold in [.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]:
                decisions=[]
                for row,d,points in zip(rows,distances,centers):
                    selected=[]
                    for i,score in enumerate(row['scores']):
                        if score<threshold:break
                        if all(d[i,j]>nms for j in selected):selected.append(i)
                        if len(selected)==3:break
                    decisions.append(points[selected])
                key,metric=metrics(decisions);key=(*key,threshold,-radius,height,-vertical,-abs(power-1));record=dict(radius=radius,power=power,nms=nms,height=height,vertical=vertical,threshold=threshold,**{k:v for k,v in metric.items() if k!='rows'})
                trials.append(record)
                if best is None or key>best[0]:best=(key,record,metric)
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new immutable selection name')
    out.mkdir(parents=True);_,chosen,metric=best
    (out/'development.json').write_text(json.dumps(dict(purpose='Global VALID hyperparameter selection of candidate position fusion; not final acceptance',
        test_opened=False,source=a.model,source_sha256=hashlib.sha256((ROOT/a.model).read_bytes()).hexdigest(),data=a.data,chosen=chosen,metrics=metric,trials=trials,
        native_supported=not a.research_vertical_caps,fusion_radius_not_above_suppression=True,evaluation_step=a.step,extra_data=a.extra_data,validation_cases=len(rows)),indent=2)+'\n')
    if a.export_model:
        model.update(track_threshold=chosen['threshold'],track_nms_m=chosen['nms'],track_height_scale=chosen['height'],track_refine_radius_m=chosen['radius'],track_refine_score_power=chosen['power'])
        (out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    print(json.dumps(chosen),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
