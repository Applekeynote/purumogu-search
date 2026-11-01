"""Evaluate saved live Unity/Rust outcomes, using labels only after acquisition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from evaluate_rescue_replay import match
from rescue_generic_metrics import tables
from rescue_local_features import MINIMUM,SHAPE,iou,truth_mask


def evaluate(root,requested_step=None):
    collection=json.loads((root/'collection.json').read_bytes())
    if collection['status']!='complete':raise ValueError('Incomplete acquisition')
    if collection['split'].startswith('test'):raise ValueError('Development evaluator never opens final TEST')
    step=requested_step or collection['steps']
    if step>collection['steps'] or step<=0 or step%1500:raise ValueError('A recorded evaluation checkpoint is required')
    rows=[]
    for item in collection['plan']['cases']:
        folder=root/item['id'];case=json.loads((folder/'complete.json').read_bytes())
        if case['case']!=item or case['truth_to_server']:raise ValueError('Acquisition provenance mismatch')
        mapping=json.loads((folder/f'map-{step}.json').read_bytes())
        truth=json.loads((folder/f'truth-{step}.json').read_bytes())
        probability=np.full(tuple(SHAPE),.5)
        for cell in mapping['cells']:
            index=np.array(cell['key'])-MINIMUM
            if np.all((index>=0)&(index<SHAPE)):probability[tuple(index)]=cell['occupancy']
        metric=iou(probability.reshape(-1),truth_mask(truth['occupied_keys']))
        matched,_=match(mapping['cues'],truth['victim_positions'])
        labels=next(r for r in map(json.loads,(folder/'labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['step']==step)
        heights=Counter(int(np.floor(r['position'][1]/.1)) for r in labels['robots'] if r['released'] and not r['lost'])
        rows.append(dict(id=item['id'],scene=case['scene'],nodes=item['nodes'],**metric,
            matched=matched,false_alarms=len(mapping['cues'])-matched,people=len(truth['victim_positions']),
            lost_nodes=sum(r['lost'] for r in labels['robots']),active_height_10cm_bins=dict(sorted(heights.items())),
            model_version=mapping.get('model_version'),map_sha256=hashlib.sha256((folder/f'map-{step}.json').read_bytes()).hexdigest()))
    return dict(purpose='Actual closed-loop development acquisition; no fixed-trajectory substitution',
        test_opened=False,collection_complete=True,seconds=step*.02,evaluation_step=step,acceptance_deadline=step==collection['steps'],
        acquisition=collection['plan'],**tables(rows),rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--step',type=int);a=p.parse_args()
    report=evaluate(ROOT/a.root,a.step);out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(root=a.root,summary=report['summary'],all_95=report['all_95'])))


if __name__=='__main__':main()
