"""Translate TRAIN sensor coordinate frames and extract native geometry, without labels in inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default='models/generic-v35-plane-preparation/model.json')
    p.add_argument('--binary',default='tmp/generic-footprint-rust-target/release/prepare_geometry.exe');a=p.parse_args()
    truth=ROOT/a.truth;out=ROOT/a.output;collection=json.loads((truth/'collection.json').read_bytes())
    if collection['status']!='complete' or collection['split']!='train' or collection['test_opened']:raise ValueError('Complete TRAIN-only labels required')
    profile=json.loads((ROOT/a.model).read_bytes());channels=profile['volume_channels']
    lineage=dict(model=a.model,model_sha256=sha(ROOT/a.model),binary=a.binary,binary_sha256=sha(ROOT/a.binary),truth_to_inference=False,final_step=6000)
    def work(case):
        name=case['id'];record=json.loads((truth/name/'complete.json').read_bytes())
        if not record['zero_translation_exact']:raise ValueError('Reconstructed geometry failed original mask parity')
        source=Path(case['source_truth']);audit=source.parent/'packets.jsonl'
        if sha(source)!=case['source_truth_sha256']:raise ValueError('Original TRAIN label changed')
        directory=out/name;complete=directory/'complete.json'
        if complete.exists():
            if json.loads(complete.read_bytes())['lineage']!=lineage:raise ValueError('Changed preparation lineage')
            return name
        if directory.exists():raise ValueError('Retain partial preparation')
        directory.mkdir(parents=True);views=[]
        for view in record['views']:
            index=view['index'];delta=view['translation_m'];target=directory/f'view-{index}';target.mkdir()
            with tempfile.TemporaryDirectory(prefix='subcell-sensor-',dir=ROOT/'tmp') as temporary:
                if not Path(temporary).resolve().is_relative_to(ROOT/'tmp'):raise ValueError('Workspace temporary required')
                translated=Path(temporary)/'packets.jsonl'
                with audit.open('r',encoding='utf-8') as raw,translated.open('w',encoding='utf-8',newline='\n') as stream:
                    for line in raw:
                        packet=json.loads(line)
                        if packet['sim_step']>6000:break
                        for scan in packet.get('scans',[]):scan['deployment_anchor']=[v+d for v,d in zip(scan['deployment_anchor'],delta)]
                        stream.write(json.dumps(packet,separators=(',',':'))+'\n')
                sensor_sha=sha(translated)
                result=subprocess.run([str(ROOT/a.binary),str(ROOT/a.model),str(translated),str(target),'-','6000'],capture_output=True)
                if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
            raw=target/'moments-final.f32';features=np.fromfile(raw,dtype='<f4').reshape(32,21,70,channels-2).transpose(3,0,1,2)
            np.savez_compressed(target/'range-features.npz',features=features);raw.unlink()
            labels=truth/name/view['truth'];(target/'truth-6000.json').write_bytes(labels.read_bytes())
            views.append(dict(index=index,translation_m=delta,sensor_audit_sha256=sensor_sha,map_sha256=sha(target/'map-6000.json'),features_sha256=sha(target/'range-features.npz'),truth_sha256=sha(target/'truth-6000.json')))
        complete.write_text(json.dumps(dict(id=name,split='train',lineage=lineage,source_audit=str(audit.relative_to(ROOT)),source_audit_sha256=sha(audit),zero_translation_exact=True,views=views),indent=2)+'\n')
        return name
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,name in enumerate(pool.map(work,collection['plan']['cases']),1):
            if i%12==0:print(json.dumps(dict(completed=i,total=len(collection['plan']['cases']),case=name)),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',split='train',test_opened=False,case_ids=[c['id'] for c in collection['plan']['cases']],views_per_case=2,lineage=lineage),indent=2)+'\n')


if __name__=='__main__':main()
