"""Reprocess registered TRAIN and VALID sensor logs with the native Mapper.

Only numerical workers are used. No TEST data is opened; no labels reach Rust.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from rescue_range_moments import audit_path
from train_rescue_generic import cases


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--server',default='tmp/generic-registration-target/release/prepare_geometry.exe')
    p.add_argument('--extra-train-root',action='append');p.add_argument('--validation-root',default='experiments/raw/rescue/generalization-v35')
    p.add_argument('--snapshots',default='');p.add_argument('--only-extra-train',action='store_true')
    p.add_argument('--validation-only',action='store_true');p.add_argument('--steps',type=int,choices=[6000,18000],default=6000)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Use one to four numerical preparation workers')
    if a.validation_only and a.only_extra_train:raise ValueError('Choose a single preparation role')
    binary=ROOT/a.server;model=ROOT/a.model;out=ROOT/a.output
    channels=json.loads(model.read_bytes())['volume_channels']
    snapshots=sorted({int(s) for s in a.snapshots.split(',') if s})
    if any(s<=0 or s>=a.steps or s%25 for s in snapshots):raise ValueError('Intermediate sensor steps required')
    rows=[] if a.validation_only else [dict(row,split='train') for row in cases(a.extra_train_root) if not a.only_extra_train or row['source']=='physical-3.5-train']
    if a.only_extra_train and not a.extra_train_root:raise ValueError('Additional registered TRAIN root required')
    if not a.only_extra_train:
        from rescue_validation_data import validation_cases
        for row in validation_cases([a.validation_root],a.steps,{r['id'] for r in rows}):rows.append(dict(row,split=row['source']))
    lineage=dict(model=a.model,model_sha256=sha(model),server=a.server,server_sha256=sha(binary),truth_to_server=False,
                 script_sha256=sha(Path(__file__)),validation_contract_sha256=sha(ROOT/'scripts/rescue_validation_data.py'))
    if snapshots:lineage['snapshots']=snapshots
    lineage['final_step']=a.steps
    lineage['training_deadlines']='actual source map time per TRAIN case'
    lineage['actual_final_step_filenames']=True
    def compress(directory):
        raw=directory/'moments-final.f32'
        features=np.fromfile(raw,dtype='<f4').reshape(32,21,70,channels-2).transpose(3,0,1,2)
        np.savez_compressed(directory/'range-features.npz',features=features)
        raw.unlink()  # Only the explicitly named intermediate created by this task.
    def work(row):
        directory=out/row['split']/row['id'];complete=directory/'complete.json';final_step=row.get('evaluation_step',a.steps)
        case_snapshots=[s for s in snapshots if s<final_step]
        if complete.exists():
            previous=json.loads(complete.read_bytes())
            if previous['lineage']!=lineage:raise ValueError('Preparation provenance changed')
            return row['id']
        if directory.exists():raise ValueError('Retain partial preparation and use a new output directory')
        directory.mkdir(parents=True);audit=audit_path(row)
        command=[str(binary),str(model),str(audit),str(directory)]
        command.extend([','.join(map(str,case_snapshots)) if case_snapshots else '-',str(final_step)])
        subprocess.run(command,check=True,capture_output=True)
        compress(directory);intermediate=[]
        for step in case_snapshots:
            target=directory/f'step-{step}';compress(target);mapping=target/f'map-{step}.json'
            if json.loads(mapping.read_bytes())['sim_step']!=step:raise ValueError('Wrong sensor snapshot')
            intermediate.append(dict(step=step,map_sha256=sha(mapping),features_sha256=sha(target/'range-features.npz')))
        complete.write_text(json.dumps(dict(id=row['id'],lineage=lineage,evaluation_step=final_step,source_audit=str(audit.relative_to(ROOT)),source_sha256=sha(audit),map_sha256=sha(directory/f'map-{final_step}.json'),features_sha256=sha(directory/'range-features.npz'),intermediate=intermediate),indent=2)+'\n')
        return row['id']
    with ThreadPoolExecutor(max_workers=a.workers) as workers:
        for i,name in enumerate(workers.map(work,rows),1):
            if i%10==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),case=name)),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',cases=len(rows),case_ids=[r['id'] for r in rows],test_opened=False,lineage=lineage),indent=2)+'\n')


if __name__=='__main__':main()
