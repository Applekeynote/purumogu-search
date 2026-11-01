"""Compose disjoint immutable TRAIN auxiliary masks; no inference input labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True,action='append');p.add_argument('--output',required=True);p.add_argument('--expected-cases',type=int,required=True);a=p.parse_args()
    output=(ROOT/a.output).resolve()
    if output.exists() or not output.is_relative_to(ROOT/'experiments/raw/rescue'):raise ValueError('Fresh workspace auxiliary pool required')
    cases={};sources=[]
    for source in a.pool:
        pool=ROOT/source;manifest=pool/'manifest.json';record=json.loads(manifest.read_bytes())
        if record['status']!='complete' or record['split']!='train' or record['test_opened'] or not record['all_coarse_geometry_exact'] or record['fine_cell_m']!=.025 or record['acceptance_cell_m']!=.05:raise ValueError('Matching complete TRAIN-only auxiliary masks required')
        sources.append(dict(manifest=str(manifest.relative_to(ROOT)),sha256=sha(manifest)))
        for identifier in record['case_ids']:
            if identifier in cases:raise ValueError('Duplicate auxiliary world')
            path=pool/identifier;data=json.loads((path/'complete.json').read_bytes())
            if not data['coarse_geometry_exact'] or sha(path/'fine-truth.npz')!=data['labels_sha256']:raise ValueError('Auxiliary mask changed')
            cases[identifier]=path
    if len(cases)!=a.expected_cases:raise ValueError('Exact auxiliary count required')
    output.mkdir(parents=True)
    for identifier,path in cases.items():
        shutil.copytree(path,output/identifier)
        for name in ['fine-truth.npz','complete.json']:
            if sha(path/name)!=sha(output/identifier/name):raise ValueError('Auxiliary copy differs')
    (output/'manifest.json').write_text(json.dumps(dict(status='complete',split='train',test_opened=False,truth_to_native=False,cases=len(cases),case_ids=sorted(cases),fine_cell_m=.025,acceptance_cell_m=.05,all_coarse_geometry_exact=True,sources=sources),indent=2)+'\n')
    print(json.dumps(dict(status='complete',cases=len(cases),truth_to_native=False)))


if __name__=='__main__':main()
