"""Explicit, complete development-only validation cohorts, never TRAIN or TEST."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def validation_cases(roots,step,train_ids=()):
    seen=set(train_ids)
    for source in roots:
        root=ROOT/source
        for split in ['valid-known','valid-unknown']:
            collection=json.loads((root/split/'collection.json').read_bytes())
            if collection['status']!='complete' or collection['split']!=split or collection['steps']<step:raise ValueError('Complete matching VALID cohort required')
            expected={case['id']:case for case in collection['plan']['cases']}
            if len(expected)!=len(collection['plan']['cases']):raise ValueError('Duplicate registered VALID IDs')
            paths=sorted((root/split).glob('*/complete.json'))
            if {path.parent.name for path in paths}!=set(expected):raise ValueError('VALID collection differs from registered case set')
            for path in paths:
                case=json.loads(path.read_bytes());identifier=case['id']
                if identifier!=path.parent.name or identifier in seen or case['split']!=split or case['case']!=expected[identifier] or case.get('truth_to_server',True):raise ValueError('Mixed roles, duplicate world or invalid VALID provenance')
                seen.add(identifier)
                mapping=path.parent/f'map-{step}.json';truth=path.parent/f'truth-{step}.json'
                if json.loads(mapping.read_bytes())['sim_step']!=step:raise ValueError('VALID observation time mismatch')
                yield dict(id=identifier,group=f"S{case['case']['scene']}-W{case['case']['seed']}",map=mapping,truth=json.loads(truth.read_bytes()),source=split,evaluation_step=step,acquisition_root=str(source))
