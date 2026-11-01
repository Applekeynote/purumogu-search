"""Register coarse/fine TRAIN coordinate-phase augmentation without changing sensors."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from collect_rescue_registered import write_once
from rescue_dataset import world_group

ROOT=Path(__file__).resolve().parents[1]
HELD={17,19,22,23,27,28,29,34,35}


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-root',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cases=[];sources=[];seen=set()
    for source in a.train_root:
        directory=ROOT/source;path=directory/'collection.json';collection=json.loads(path.read_bytes())
        if collection['status']!='complete' or collection['split']!='train' or collection['steps']!=18000:raise ValueError('Complete registered360-second TRAIN required')
        sources.append(dict(root=source,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),geometry=collection['plan']['geometry_sources']))
        for case in collection['plan']['cases']:
            if case['scene'] in HELD:raise ValueError('Held geometry entered TRAIN augmentation')
            if 10<=case['scene']<15:continue
            group=world_group(case['id'])
            if group in seen:raise ValueError('TRAIN world duplicated')
            seen.add(group);record=json.loads((directory/case['id']/'complete.json').read_bytes())
            if record['case']!=case or record['split']!='train' or record['steps']!=18000:raise ValueError('TRAIN provenance differs')
            truth=directory/case['id']/'truth-18000.json';rng=random.Random('joint-subcell-v2-'+case['id'])
            cases.append(dict(case,source_truth=str(truth),source_truth_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
                views=[dict(index=i,translation_m=[rng.uniform(-.024,.024) for _ in range(3)]) for i in range(2)]))
    folder=ROOT/a.output
    write_once(folder/'subcell-plan.json',dict(split='train',steps=18000,snapshots=[6000],fine_supervision=True,
        purpose='Continuous coordinate-phase augmentation with matching coarse and2.5cm auxiliary labels; not additional independent worlds',
        output=str(folder/'truth'),source_collections=sources,cases=cases,
        current_geometry_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'unity/Assets/Scripts').glob('Rescue*.cs')},
        unchanged=['Raw range/bearing/temperature/audio readings','Odometry increments and pose error','5cm acceptance cells and .42 half extent','360-second deadline'],
        ground_truth_to_native=False,test_opened=False))
    print(json.dumps(dict(static_train_worlds=len(cases),coordinate_views=2*len(cases),sensor_snapshots=4*len(cases))),flush=True)


if __name__=='__main__':main()
