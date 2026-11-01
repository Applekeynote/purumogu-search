"""Register dense labels for static TRAIN worlds, retaining official 5cm scoring."""
import argparse
import hashlib
import json
from pathlib import Path
from collect_rescue_registered import write_once
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',required=True,action='append');p.add_argument('--output',required=True)
    p.add_argument('--exclude-pool');a=p.parse_args()
    excluded={};exclusion=None
    if a.exclude_pool:
        pool=ROOT/a.exclude_pool;manifest=pool/'manifest.json';record=json.loads(manifest.read_bytes())
        if record['status']!='complete' or record['split']!='train' or record['test_opened'] or not record['all_coarse_geometry_exact'] or record['fine_cell_m']!=.025 or record['acceptance_cell_m']!=.05:raise ValueError('Complete matching TRAIN auxiliary pool required')
        excluded={identifier:json.loads((pool/identifier/'complete.json').read_bytes()) for identifier in record['case_ids']}
        exclusion=dict(manifest=str(manifest.relative_to(ROOT)),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    cases=[];sources=[];seen=set()
    for source in a.train_root:
        root=ROOT/source;collection=json.loads((root/'collection.json').read_bytes())
        if collection['status']!='complete' or collection['split']!='train':raise ValueError('Complete registered TRAIN required')
        sources.append(dict(root=source,sha256=hashlib.sha256((root/'collection.json').read_bytes()).hexdigest()))
        for case in collection['plan']['cases']:
            if case['id'] in seen:raise ValueError('Duplicate TRAIN case')
            seen.add(case['id'])
            if 10<=case['scene']<15:continue
            if case['scene'] in [17,19,22,23,27,28,29,34,35]:raise ValueError('Held family in TRAIN')
            truth=root/case['id']/f"truth-{collection['steps']}.json"
            if case['id'] in excluded:
                if hashlib.sha256(truth.read_bytes()).hexdigest()!=excluded[case['id']]['source_truth_sha256']:raise ValueError('Existing fine labels do not match this TRAIN geometry')
                continue
            cases.append(dict(case,source_truth=str(truth),source_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest()))
    out=ROOT/a.output
    write_once(out/'fine-truth-plan.json',dict(split='train',test_opened=False,purpose='2.5cm auxiliary TRAIN occupancy labels; official acceptance remains 5cm/0.42',
        output=str(out/'truth'),cases=cases,source_collections=sources,exclusion=exclusion))
    print(json.dumps(dict(static_training_cases=len(cases),auxiliary_cell_m=.025,acceptance_cell_m=.05)))

if __name__=='__main__':main()
