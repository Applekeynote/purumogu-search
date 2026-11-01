"""Run isolated hidden Unity label export, then prepare TRAIN-only auxiliary masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--project',default='tmp/g52-fine-truth');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    plan_path=ROOT/a.plan;plan=json.loads(plan_path.read_bytes());output=(ROOT/a.output).resolve()
    if plan['split']!='train' or plan['test_opened']:raise ValueError('TRAIN-only label plan required')
    if not output.is_relative_to(ROOT/'experiments/raw/rescue') or output.exists():raise ValueError('Fresh workspace raw output required')
    for case in plan['cases']:
        if sha(Path(case['source_truth']))!=case['source_truth_sha256']:raise ValueError('Recorded coarse truth changed')
    if not a.prepare_only:
        project=(ROOT/a.project).resolve()
        if not project.is_relative_to(ROOT/'tmp') or project.exists():raise ValueError('Fresh isolated workspace project required')
        project.mkdir(parents=True)
        for directory in ['Assets','Packages','ProjectSettings']:shutil.copytree(ROOT/'unity'/directory,project/directory)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        command=['C:/Program Files/Unity/Hub/Editor/6000.4.1f1/Editor/Unity.exe','-batchmode','-nographics','-projectPath',str(project),
            '-executeMethod','Purumogu.Editor.RescueFineTruth.Export','-purumoguPlan',str(plan_path),'-logFile',str(ROOT/'tmp'/f'{project.name}-unity.log')]
        subprocess.run(command,check=True,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    raw=Path(plan['output']);collection=raw/'collection.json';record=json.loads(collection.read_bytes())
    if record['status']!='complete' or record['split']!='train' or record['cases']!=len(plan['cases']):raise ValueError('Incomplete fine export')
    output.mkdir(parents=True);rows=[]
    for case in plan['cases']:
        folder=raw/case['id'];complete=json.loads((folder/'complete.json').read_bytes())
        if not complete['coarse_geometry_exact'] or complete['truth_to_inference']:raise ValueError('Coarse reconstruction parity required')
        source=folder/'fine-truth.json';truth=json.loads(source.read_bytes())
        if truth['shape']!=[64,42,140] or truth['minimum_key']!=[-32,-6,-12] or truth['cell_m']!=.025:raise ValueError('Fine grid contract differs')
        key=np.asarray(truth['occupied_keys'],dtype=np.int32)-np.array([-32,-6,-12])
        if not np.all((key>=0)&(key<np.array([64,42,140]))):raise ValueError('Fine label outside domain')
        y=np.zeros((1,64,48,144),dtype=np.uint8);y[0,key[:,0],key[:,1],key[:,2]]=1
        target=output/case['id'];target.mkdir();np.savez_compressed(target/'fine-truth.npz',y=y)
        row=dict(id=case['id'],split='train',coarse_geometry_exact=True,source_truth_sha256=case['source_truth_sha256'],source_fine_sha256=sha(source),labels_sha256=sha(target/'fine-truth.npz'))
        (target/'complete.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    manifest=dict(status='complete',split='train',test_opened=False,truth_to_native=False,cases=len(rows),case_ids=[r['id'] for r in rows],
        fine_cell_m=.025,acceptance_cell_m=.05,plan=a.plan,plan_sha256=sha(plan_path),source_collection_sha256=sha(collection),all_coarse_geometry_exact=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
