"""Evaluate a head on fixed development proposals through the native classifier."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from evaluate_rescue_replay import match
from rescue_dataset import acoustic_feature_signature

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--root',required=True)
    p.add_argument('--step',type=int,default=6000);p.add_argument('--output',required=True)
    p.add_argument('--binary',default='tmp/generic-rust-target/release/classify_proposals.exe');a=p.parse_args()
    root=ROOT/a.root;collection=json.loads((root/'collection.json').read_bytes())
    if collection['status']!='complete' or collection['split'].startswith('test'):raise ValueError('Complete development data required; no final TEST')
    if a.step<=0 or a.step>collection['steps'] or a.step%1500:raise ValueError('Recorded checkpoint required')
    acquisition=ROOT/collection['plan']['model'];head=json.loads((ROOT/a.model).read_bytes())
    if hashlib.sha256(acquisition.read_bytes()).hexdigest()!=collection['plan']['model_sha256']:raise ValueError('Acquisition model changed')
    source=json.loads(acquisition.read_bytes())
    if acoustic_feature_signature(head)!=acoustic_feature_signature(source) or head.get('track_feature_map')!=source.get('track_feature_map'):raise ValueError('Native proposal feature contracts differ')
    with tempfile.TemporaryDirectory(prefix='head-eval-',dir=ROOT/'tmp') as work:
        def evaluate(case):
            folder=root/case['id'];output=Path(work)/(case['id']+'.json');source=folder/f'map-{a.step}.json'
            subprocess.run([str(ROOT/a.binary),str(ROOT/a.model),str(source),str(output)],check=True,capture_output=True)
            cues=json.loads(output.read_bytes())['cues']
            truth=json.loads((folder/f'truth-{a.step}.json').read_bytes())['victim_positions']
            tp,_=match(cues,truth)
            return dict(id=case['id'],area=case['scene'],nodes=case['nodes'],matched=tp,false_alarms=len(cues)-tp,people=len(truth),map_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        with ThreadPoolExecutor(max_workers=4) as workers:rows=list(workers.map(evaluate,collection['plan']['cases']))
    def metric(rows):
        tp=sum(r['matched'] for r in rows);fp=sum(r['false_alarms'] for r in rows);n=sum(r['people'] for r in rows)
        return dict(precision=tp/(tp+fp) if tp+fp else None,recall=tp/n if n else None,matched=tp,false_alarms=fp,people=n)
    report=dict(purpose='Fixed development proposals; not closed-loop acceptance',test_opened=False,split=collection['split'],step=a.step,
        model=a.model,model_sha256=hashlib.sha256((ROOT/a.model).read_bytes()).hexdigest(),summary=metric(rows),
        by_area={str(s):metric([r for r in rows if r['area']==s]) for s in sorted({r['area'] for r in rows})},rows=rows)
    destination=ROOT/a.output;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('Use a new report path')
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary']))


if __name__=='__main__':main()
