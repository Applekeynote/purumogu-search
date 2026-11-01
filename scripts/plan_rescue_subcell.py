"""Register TRAIN-only continuous coordinate translations before preparing labels."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from collect_rescue_registered import write_once

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=ROOT/a.train_root;collection=json.loads((source/'collection.json').read_bytes())
    if collection['status']!='complete' or collection['split']!='train':raise ValueError('Complete registered TRAIN required')
    cases=[]
    for case in collection['plan']['cases']:
        if 10<=case['scene']<15:continue  # Moving rubble has no saved continuous checkpoint.
        rng=random.Random('subcell-v1-'+case['id']);truth=source/case['id']/'truth-6000.json'
        cases.append(dict(case,source_truth=str(truth),source_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
            views=[dict(index=i,translation_m=[rng.uniform(-.024,.024) for _ in range(3)]) for i in range(2)]))
    out=ROOT/a.output
    write_once(out/'subcell-plan.json',dict(split='train',purpose='Coordinate gauge augmentation; sensor errors and 5cm/0.42 label definition unchanged',
        output=str(out/'truth'),source_collection=a.train_root,source_sha256=hashlib.sha256((source/'collection.json').read_bytes()).hexdigest(),
        source_geometry=collection['plan']['geometry_sources'],cases=cases,test_opened=False))
    print(json.dumps(dict(static_training_cases=len(cases),views=len(cases)*2)))


if __name__=='__main__':main()
