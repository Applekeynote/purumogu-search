"""Append the native sensor-only TSDF band to immutable 74-channel inputs.

The shared Rust integrator preserves the first 74 channels. A full-Mapper parity
probe must be completed before fitting this corpus. TEST is never enumerated.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np

BASES=[('experiments/raw/rescue/generalization-v40-mixed/registered-temporal-y74',['train']),
       ('experiments/raw/rescue/generalization-v44-long-train/registered-temporal-y74',['train']),
       ('experiments/raw/rescue/generalization-v41-mission360/registered-y74-final360',['valid-known','valid-unknown'])]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int);p.add_argument('--train-base',action='append')
    p.add_argument('--validation-base',action='append');p.add_argument('--only-validation',action='store_true')
    p.add_argument('--only-extra-train',action='store_true');p.add_argument('--expected-cases',type=int);a=p.parse_args()
    if a.only_extra_train and (not a.train_base or not a.expected_cases):raise ValueError('Explicit additional TRAIN sources and count required')
    if a.only_validation and (not a.validation_base or not a.expected_cases or a.only_extra_train or a.train_base):raise ValueError('Explicit VALID-only sources and count required')
    if a.only_extra_train and a.validation_base:raise ValueError('VALID cannot enter TRAIN-only preparation')
    model=ROOT/'models/generic-v46-tsdf-preparation/model.json'
    binary=ROOT/'tmp/generic-tsdf-rust-target/release/prepare_tsdf_band.exe'
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    params=json.loads(model.read_bytes());rows=[];sources=[]
    bases=([] if a.only_extra_train or a.only_validation else BASES)+[(base,['train']) for base in a.train_base or []]+[(base,['valid-known','valid-unknown']) for base in a.validation_base or []]
    for base,splits in bases:
        manifest=ROOT/base/'manifest.json';record=json.loads(manifest.read_bytes())
        if record['status']!='complete' or record['test_opened']:raise ValueError('Development-only source required')
        profile=ROOT/record['lineage']['model']
        if sha(profile)!=record['lineage']['model_sha256']:raise ValueError('Source profile changed')
        source=json.loads(profile.read_bytes())
        if source['volume_channels']!=74 or any(source.get(k,False)!=params.get(k,False) for k in ['plane_registration_v1','axis_plane_registration_v2']):raise ValueError('Incompatible base features')
        sources.append(dict(manifest=str(manifest.relative_to(ROOT)),sha256=sha(manifest)))
        for split in splits:
            candidates=sorted((ROOT/base/split).glob('*/complete.json'))
            if not {path.parent.name for path in candidates}<=set(record['case_ids']):raise ValueError('Unregistered prepared case')
            rows.extend((split,path) for path in candidates)
    expected=a.expected_cases or 380
    if len(rows)!=expected or len({path.parent.name for _,path in rows})!=expected:raise ValueError('Expected exact registered case set')
    lineage=dict(model=str(model.relative_to(ROOT)),model_sha256=sha(model),server=str(binary.relative_to(ROOT)),server_sha256=sha(binary),
                 truth_to_server=False,actual_final_step_filenames=True,composition='immutable base 74 plus shared native TSDF suffix 16',sources=sources,script_sha256=sha(Path(__file__)))
    if a.limit:rows=rows[:a.limit]
    def work(row):
        split,path=row;base=path.parent;source=json.loads(path.read_bytes());target=out/split/base.name
        done=target/'complete.json'
        if done.exists():
            if json.loads(done.read_bytes())['lineage']!=lineage:raise ValueError('Changed provenance')
            return base.name
        if target.exists():raise ValueError('Partial target retained; use a fresh output')
        target.mkdir(parents=True)
        mapping=next(base.glob('map-*.json'));final_step=json.loads(mapping.read_bytes())['sim_step']
        if sha(mapping)!=source['map_sha256']:raise ValueError('Changed base map')
        audit=ROOT/source['source_audit']
        if sha(audit)!=source['source_sha256']:raise ValueError('Changed sensor audit')
        shots=[s['step'] for s in source['intermediate']]
        subprocess.run([str(binary),str(model),str(audit),str(target),','.join(map(str,shots)) if shots else '-',str(final_step)],check=True,capture_output=True)
        def combine(src,dst,step,map_hash,feature_hash):
            srcmap=next(src.glob('map-*.json'))
            if sha(srcmap)!=map_hash or sha(src/'range-features.npz')!=feature_hash:raise ValueError('Changed base tensor')
            with np.load(src/'range-features.npz') as data:old=data['features']
            band=np.fromfile(dst/'tsdf.f32',dtype='<f4').reshape(32,21,70,16).transpose(3,0,1,2)
            if old.shape!=(72,32,21,70) or not np.isfinite(band).all():raise ValueError('Invalid tensor')
            merged=np.concatenate([old,band],axis=0)
            if not np.array_equal(merged[:72],old):raise ValueError('Changed original inputs')
            np.savez_compressed(dst/'range-features.npz',features=merged)
            shutil.copyfile(srcmap,dst/f'map-{step}.json')
            (dst/'tsdf.f32').unlink()
            return dict(step=step,map_sha256=sha(dst/f'map-{step}.json'),features_sha256=sha(dst/'range-features.npz'))
        intermediate=[combine(base/f'step-{s["step"]}',target/f'step-{s["step"]}',s['step'],s['map_sha256'],s['features_sha256']) for s in source['intermediate']]
        final=combine(base,target,final_step,source['map_sha256'],source['features_sha256'])
        done.write_text(json.dumps(dict(id=base.name,lineage=lineage,evaluation_step=final_step,source_audit=source['source_audit'],source_sha256=source['source_sha256'],
                                       base_complete=str(path.relative_to(ROOT)),base_complete_sha256=sha(path),map_sha256=final['map_sha256'],features_sha256=final['features_sha256'],intermediate=intermediate),indent=2)+'\n')
        return base.name
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(work,row) for row in rows]
        for i,future in enumerate(as_completed(futures),1):
            name=future.result()
            if i%10==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),case=name)),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='partial-probe' if a.limit else 'complete',cases=len(rows),case_ids=[p.parent.name for _,p in rows],test_opened=False,lineage=lineage),indent=2)+'\n')

if __name__=='__main__':main()
