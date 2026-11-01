"""Diagnose proposal recall ceiling on registered development worlds, not TEST."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def oracle_match(proposals,people):
    if len(people)>8:raise ValueError('Diagnostic supports at most eight source positions')
    states={0:0.}
    for proposal in proposals:
        updated=dict(states)
        for mask,error in states.items():
            for j,person in enumerate(people):
                if mask&(1<<j):continue
                distance=math.dist(proposal['position'],person)
                if distance<.25:
                    key=mask|(1<<j);updated[key]=min(updated.get(key,math.inf),error+distance)
        states=updated
    mask,error=max(states.items(),key=lambda item:(item[0].bit_count(),-item[1]))
    return mask.bit_count(),error


def audit(plan_path,allow_incomplete=False):
    plan=json.loads(plan_path.read_bytes())
    if plan['split'] not in ['train','valid-known','valid-unknown'] or plan.get('test_opened',False):raise ValueError('Registered development-only plan required')
    rows=[];pending=[]
    for case in plan['cases']:
        folder=Path(plan['output'])/case['id'];complete=folder/'complete.json'
        if not complete.exists():
            if not allow_incomplete:raise ValueError('Incomplete cohort')
            pending.append(case['id']);continue
        record=json.loads(complete.read_bytes())
        if record['case']!=case or record['truth_to_server'] or record['steps']!=plan['steps']:raise ValueError('Case provenance mismatch')
        mapping=folder/f"map-{plan['steps']}.json";data=json.loads(mapping.read_bytes())
        if data['sim_step']!=plan['steps']:raise ValueError('Observation time mismatch')
        proposals=data['proposals'];truth=json.loads((folder/f"truth-{plan['steps']}.json").read_bytes())['victim_positions']
        matched,error=oracle_match(proposals,truth)
        rows.append(dict(id=case['id'],area=case['scene'],nodes=case['nodes'],proposals=len(proposals),people=len(truth),oracle_matched=matched,oracle_matching_error_m=error,
            nearest_proposal_m=[min((math.dist(p['position'],v) for p in proposals),default=None) for v in truth],acquisition_matched=record['matched'],map_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest()))
    people=sum(row['people'] for row in rows);matched=sum(row['oracle_matched'] for row in rows)
    return dict(purpose='Upper bound for classification-only changes on this fixed proposal pool; not achieved model accuracy',status='partial-diagnostic' if pending else 'complete-diagnostic',test_opened=False,
        split=plan['split'],plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),cases=len(rows),pending=pending,people=people,oracle_matched=matched,oracle_recall=matched/people if people else None,rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    result=audit(ROOT/a.plan,a.allow_incomplete);output=ROOT/a.output
    if output.exists():raise ValueError('Use an immutable diagnostic report')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:result[key] for key in ['status','cases','people','oracle_matched','oracle_recall']}))


if __name__=='__main__':main()
