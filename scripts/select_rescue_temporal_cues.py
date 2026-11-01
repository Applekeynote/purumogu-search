"""Research-only causal confirmation policy, calibrated globally on formal VALID."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from rescue_dataset import assert_disjoint_worlds
from rescue_validation_data import validation_cases
from rescue_residual_head import forest_logits
from rescue_temporal_cues import TemporalCues
from train_rescue_generic_tracks import fast_metric
from evaluate_rescue_replay import match


def summarize(records,outputs):
    metric=fast_metric([(row,cues) for row,cues in zip(records,outputs)],0,1e-6,1)
    rows=[]
    for row,cues in zip(records,outputs):
        matched,_=match(cues,row['victim_positions']);rows.append(dict(id=row['map_id'],domain=row['domain'],area=row['area'],nodes=row['nodes'],matched=matched,false_alarms=len(cues)-matched,people=len(row['victim_positions']),cues=cues))
    metric.update(matched=sum(r['matched'] for r in rows),false_alarms=sum(r['false_alarms'] for r in rows),people=sum(r['people'] for r in rows),rows=rows)
    return metric


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',action='append',required=True);p.add_argument('--output',required=True)
    p.add_argument('--include-current',action='store_true',help='Permit current provisional observations while retaining confirmed stationary hypotheses');a=p.parse_args()
    out=ROOT/a.output
    if out.exists():raise ValueError('Use a fresh immutable research report')
    path=ROOT/a.model;model=json.loads(path.read_bytes())
    if model['track_feature_map']!='thermal-spectral32-v3' or model.get('track_refine_radius_m',0):raise ValueError('Frozen native32 unrefined source head required')
    records=[];sequences=[];inputs=[];steps=[1500,3000,4500,6000,9000,12000,15000,18000]
    sources=list(validation_cases(a.data,18000));assert_disjoint_worlds([r['id'] for r in sources],model['train_maps'])
    for index,row in enumerate(sources,1):
        case=json.loads((row['map'].parent/'complete.json').read_bytes())['case'];sequence=[]
        for step in steps:
            source=row['map'].with_name(f'map-{step}.json');mapping=json.loads(source.read_bytes())
            if mapping['sim_step']!=step or any(len(q['features'])!=32 for q in mapping['proposals']):raise ValueError('Causal native32 snapshot required')
            proposals=mapping['proposals'];x=np.array([q['features'] for q in proposals]);scores=1/(1+np.exp(-np.clip(forest_logits(model,x),-40,40))) if proposals else []
            ranked=sorted([dict(position=q['position'],score=float(s)) for q,s in zip(proposals,scores)],key=lambda q:-q['score']);cues=[]
            def distance(l,r):return math.sqrt(sum((v-w)**2*(model.get('track_height_scale',1)**2 if i==1 else 1) for i,(v,w) in enumerate(zip(l,r))))
            for cue in ranked:
                if cue['score']<model['track_threshold']:break
                if all(distance(cue['position'],q['position'])>model['track_nms_m'] for q in cues):cues.append(cue)
                if len(cues)==3:break
            sequence.append(cues);inputs.append(dict(id=row['id'],step=step,map_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        records.append(dict(map_id=row['id'],domain=row['source'],area=case['scene'],nodes=case['nodes'],victim_positions=row['truth']['victim_positions']));sequences.append(sequence)
        if index%9==0 or index==len(sources):print(json.dumps(dict(stage='frozen-head-snapshots',completed=index,total=len(sources))),flush=True)
    baseline=summarize(records,[s[-1] for s in sequences]);trials=[];best=None
    for radius,retention in itertools.product([.15,.2,.25],[6000,18000]):
        trackers=[]
        for sequence in sequences:
            tracker=TemporalCues(radius,retention)
            for step,cues in zip(steps,sequence):tracker.update(step,cues)
            trackers.append(tracker)
        for support,threshold,support_rank,nms in itertools.product([2,3],[.4,.6,.8,.9],[False,True],[.2,.3,.4]):
            outputs=[t.confirmed(support,threshold,nms,support_rank,a.include_current) for t in trackers]
            metric=summarize(records,outputs);precision=metric['human_precision'];recall=metric['human_recall']
            key=(metric['minimum_stratum_criterion'],min(precision,recall),precision+recall,-support)
            policy=dict(radius_m=radius,inactive_retention_steps=retention,minimum_support=support,mean_threshold=threshold,support_rank=support_rank,nms_m=nms)
            trials.append(dict(policy=policy,**{k:v for k,v in metric.items() if k!='rows'}))
            if best is None or key>best[0]:best=(key,policy,metric)
        print(json.dumps(dict(stage='policy-selection',radius=radius,retention=retention,best=best[1],precision=best[2]['human_precision'],recall=best[2]['human_recall'],minimum_stratum=best[2]['minimum_stratum_criterion'])),flush=True)
    chosen=best[1];prefix_outputs=[]
    for sequence in sequences:
        tracker=TemporalCues(chosen['radius_m'],chosen['inactive_retention_steps'])
        for step,cues in zip(steps[:4],sequence[:4]):tracker.update(step,cues)
        prefix_outputs.append(tracker.confirmed(chosen['minimum_support'],chosen['mean_threshold'],chosen['nms_m'],chosen['support_rank'],a.include_current))
    report=dict(purpose='Causal temporal confirmation policy selected on fixed formal VALID snapshots; not native or closed-loop acceptance',
        final_test_opened=False,closed_loop_acceptance=False,native_support_verified=False,model=a.model,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        assumption='Sources remain stationary within a mission; inactive hypotheses expire, persistent tracks retain all supporting observations',
        schedule_steps=steps,inputs=inputs,baseline360=baseline,baseline120=summarize(records,[s[3] for s in sequences]),chosen=chosen,metrics360=best[2],metrics120=summarize(records,prefix_outputs),trials=trials,include_current=a.include_current,
        labels_used_for_scoring_only=True,no_gradient_fitting=True,no_scene_or_count_specific_parameters=True)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(chosen=chosen,baseline={k:baseline[k] for k in ['matched','false_alarms','people']},temporal={k:best[2][k] for k in ['matched','false_alarms','people','human_precision','human_recall','minimum_stratum_criterion']})),flush=True)


if __name__=='__main__':main()
