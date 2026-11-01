"""Freeze a complete, pre-ordered TRAIN pass while later repeats are collecting.

No score filtering: the first registered pass must contain every TRAIN family
exactly once. Later full-cohort fits replace this staged root rather than count
the same worlds twice. Original sensor/label files remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-plan',required=True);p.add_argument('--families',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    parent=ROOT/a.parent_plan;plan=json.loads(parent.read_bytes())
    if plan['split']!='train' or plan['test_opened']:raise ValueError('Registered TRAIN only')
    if any(sha(ROOT/plan[k])!=plan[k+'_sha256'] for k in ['model','server']):raise ValueError('Acquisition artifact changed')
    cases=plan['cases'][:a.families]
    if len(cases)!=a.families or len({c['scene'] for c in cases})!=a.families or {c['scene'] for c in cases}!={c['scene'] for c in plan['cases']}:raise ValueError('Exactly one first-pass world per registered family required')
    source=Path(plan['output']).resolve();target=(ROOT/a.output).resolve()
    raw=(ROOT/'experiments/raw/rescue').resolve()
    if not source.is_relative_to(raw) or not target.is_relative_to(raw) or source==target:raise ValueError('Separate workspace raw-data roots required')
    if target.exists():raise ValueError('Staging root is immutable; use a fresh one')
    for case in cases:
        path=source/case['id']/'complete.json'
        if not path.exists():raise ValueError('First registered pass is not complete: '+case['id'])
        record=json.loads(path.read_bytes())
        if record['id']!=case['id'] or record['case']!=case or record['steps']!=plan['steps'] or record['split']!='train' or record['truth_to_server']:raise ValueError('Source case provenance mismatch')
    target.mkdir(parents=True);files={}
    for case in cases:
        destination=target/case['id'];destination.mkdir()
        for path in sorted((source/case['id']).iterdir()):
            if not path.is_file() or path.suffix not in ('.json','.jsonl'):raise ValueError('Unexpected source data file')
            digest=sha(path);shutil.copyfile(path,destination/path.name)
            if sha(destination/path.name)!=digest:raise ValueError('Staged copy mismatch')
            files[str((destination/path.name).relative_to(target))]=digest
    staging=dict(purpose='Complete first registered TRAIN pass; no outcome-based selection',parent_plan=a.parent_plan,
        parent_plan_sha256=sha(parent),parent_output=str(source.relative_to(ROOT)),families=a.families,
        independent_worlds=len(cases),test_opened=False,files=files,
        replacement_rule='Use full parent cohort instead of this root in later fits; never include both')
    staged_plan=dict(plan,cases=cases,output=str(target),staging=staging)
    (target/'collection.json').write_text(json.dumps(dict(status='complete',split='train',steps=plan['steps'],plan=staged_plan,truth_to_server=False,staged=True),indent=2)+'\n')
    (target/'staging-receipt.json').write_text(json.dumps(staging,indent=2)+'\n')
    print(json.dumps(dict(status='complete',cases=len(cases),output=a.output,parent_collection_may_still_be_running=True)))

if __name__=='__main__':main()
