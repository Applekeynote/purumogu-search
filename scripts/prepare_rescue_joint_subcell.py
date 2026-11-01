"""Matching coarse/fine TRAIN coordinate phases; raw sensor errors are preserved."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from collect_rescue_registered import hidden


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def simulator_precision_plan(plan):
    result=copy.deepcopy(plan)
    for case in result['cases']:
        for view in case['views']:view['translation_m']=[float(np.float32(v)) for v in view['translation_m']]
    return result


def translate_packets(source,target,delta,deadline):
    if len(delta)!=3 or not all(np.isfinite(d) and abs(d)<.025 for d in delta):raise ValueError('Bounded coordinate translation required')
    last=None
    with source.open(encoding='utf-8-sig') as raw,target.open('w',encoding='utf-8',newline='\n') as output:
        for line in raw:
            packet=json.loads(line)
            if packet['sim_step']>deadline:break
            for scan in packet['scans']:scan['deployment_anchor']=[v+d for v,d in zip(scan['deployment_anchor'],delta)]
            output.write(json.dumps(packet,separators=(',',':'))+'\n');last=packet['sim_step']
    if last!=deadline:raise ValueError('Sensor deadline missing')


def fine_tensor(source):
    data=json.loads(source.read_bytes())
    if data['shape']!=[64,42,140] or data['minimum_key']!=[-32,-6,-12] or data['cell_m']!=.025:raise ValueError('Fine-label grid differs')
    keys=np.array(data['occupied_keys'],dtype=np.int32).reshape(-1,3)-np.array([-32,-6,-12])
    if np.any((keys<0)|(keys>=np.array([64,42,140]))):raise ValueError('Fine-label key outside domain')
    y=np.zeros((1,64,48,144),dtype=np.uint8);y[0,keys[:,0],keys[:,1],keys[:,2]]=1
    return y


def main():
    p=argparse.ArgumentParser();p.add_argument('--truth',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);p.add_argument('--binary',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    truth=ROOT/a.truth;out=(ROOT/a.output).resolve();source=truth/'collection.json';collection=json.loads(source.read_bytes());plan=collection['plan']
    if collection['status']!='complete' or collection['split']!='train' or collection['test_opened'] or plan['steps']!=18000 or plan['snapshots']!=[6000] or not plan['fine_supervision']:raise ValueError('Complete joint coarse/fine TRAIN plan required')
    if out.exists() or not out.is_relative_to(ROOT/'experiments/raw/rescue') or not 1<=a.workers<=4:raise ValueError('Fresh workspace output and bounded numerical workers required')
    model=ROOT/a.model;binary=ROOT/a.binary;profile=json.loads(model.read_bytes())
    if profile['volume_channels']!=90 or not profile['range_tsdf_v1']:raise ValueError('Native90 signed-distance feature preparation required')
    lineage=dict(model=a.model,model_sha256=sha(model),binary=a.binary,binary_sha256=sha(binary),truth_to_inference=False,
        final_step=18000,snapshots=[6000],fine_supervision=True,truth_collection_sha256=sha(source),script_sha256=sha(Path(__file__)))
    out.mkdir(parents=True)
    def compress(directory):
        path=directory/'moments-final.f32';features=np.fromfile(path,dtype='<f4').reshape(32,21,70,88).transpose(3,0,1,2)
        np.savez_compressed(directory/'range-features.npz',features=features);path.unlink()
    def work(case):
        record=json.loads((truth/case['id']/'complete.json').read_bytes())
        original=Path(case['source_truth']);audit=original.parent/'packets.jsonl'
        if not record['zero_translation_exact'] or sha(original)!=case['source_truth_sha256'] or case['scene'] in {17,19,22,23,27,28,29,34,35} or 10<=case['scene']<15:raise ValueError('Unchanged static TRAIN geometry required')
        directory=out/case['id'];directory.mkdir();views=[]
        for view in record['views']:
            index=view['index'];target=directory/f'view-{index}';target.mkdir()
            with tempfile.TemporaryDirectory(prefix='joint-subcell-sensor-',dir=ROOT/'tmp') as temporary:
                translated=Path(temporary)/'packets.jsonl';translate_packets(audit,translated,view['translation_m'],18000)
                sensor_sha=sha(translated)
                subprocess.run([str(binary),str(model),str(translated),str(target),'6000','18000'],check=True,capture_output=True,**hidden())
            # Labels are added only after native sensor extraction has finished.
            coarse=truth/case['id']/view['truth'];fine=truth/case['id']/view['fine_truth']
            np.savez_compressed(target/'fine-truth.npz',y=fine_tensor(fine));fine_sha=sha(target/'fine-truth.npz')
            for step,prepared in [(6000,target/'step-6000'),(18000,target)]:
                compress(prepared);mapping=prepared/f'map-{step}.json'
                if json.loads(mapping.read_bytes())['sim_step']!=step:raise ValueError('Coordinate-view sensor time differs')
                label=prepared/f'truth-{step}.json';label.write_bytes(coarse.read_bytes())
                views.append(dict(index=index,translation_m=view['translation_m'],evaluation_step=step,directory=prepared.relative_to(directory).as_posix(),
                    sensor_audit_sha256=sensor_sha,map_sha256=sha(mapping),features_sha256=sha(prepared/'range-features.npz'),truth_sha256=sha(label),
                    fine_labels=(target/'fine-truth.npz').relative_to(directory).as_posix(),fine_labels_sha256=fine_sha,source_fine_sha256=sha(fine)))
        (directory/'complete.json').write_text(json.dumps(dict(id=case['id'],split='train',lineage=lineage,source_truth_sha256=case['source_truth_sha256'],source_audit_sha256=sha(audit),zero_translation_exact=True,views=views),indent=2)+'\n')
        return case['id']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,name in enumerate(pool.map(work,plan['cases']),1):
            if i%12==0 or i==len(plan['cases']):print(json.dumps(dict(completed=i,total=len(plan['cases']),case=name)),flush=True)
    manifest=dict(status='complete',split='train',test_opened=False,fine_supervision=True,case_ids=[c['id'] for c in plan['cases']],views_per_case=4,
        independent_worlds=len(plan['cases']),augmentation_is_not_new_worlds=True,lineage=lineage)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
