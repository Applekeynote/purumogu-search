"""Paired TRAIN sensor replay: changing map weights must preserve frozen exploration."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
from collect_rescue_registered import hidden
from rescue_dataset import control_signature

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--server',required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=6000);a=p.parse_args()
    case=(ROOT/a.case).resolve();source=ROOT/a.model;binary=ROOT/a.server;output=ROOT/a.output
    if not case.is_relative_to(ROOT/'experiments/raw/rescue') or 'train' not in case.parts or any(s.startswith(('test','valid')) for s in case.parts):raise ValueError('TRAIN-only sensor replay required')
    record=json.loads((case/'complete.json').read_bytes())
    if record['split']!='train' or not 0<a.steps<=record['steps'] or a.steps%100:raise ValueError('Complete recorded TRAIN checkpoint required')
    model=json.loads(source.read_bytes())
    if model.get('navigation_file')!='navigation.onnx' or model['navigation_sha256']==model['volume_sha256']:raise ValueError('Two different bundled map and exploration networks required')
    if sha(source.parent/'navigation.onnx')!=model['navigation_sha256']:raise ValueError('Exploration artifact changed')
    if output.exists():raise ValueError('Use a new report path')
    artifacts=output.parent/(output.stem+'-artifacts');reference=artifacts/'reference';reference.mkdir(parents=True,exist_ok=False)
    descriptor=dict(model,volume_channels=74,volume_sha256=model['navigation_sha256'],volume_logit_bias=model['navigation_logit_bias'],range_tsdf_v1=False)
    for key in ['navigation_file','navigation_sha256','navigation_channels','navigation_logit_bias']:descriptor.pop(key)
    if control_signature(model)!=control_signature(descriptor):raise ValueError('Reference and separated model have different exploration contracts')
    shutil.copy2(source.parent/'navigation.onnx',reference/'volume.onnx')
    export=json.loads((source.parent/'export.json').read_bytes())
    for name in ['onnxruntime.dll','onnxruntime_providers_shared.dll',*export['microsoft_crt']['files']]:shutil.copy2(source.parent/name,reference/name)
    (reference/'model.json').write_text(json.dumps(descriptor,separators=(',',':'))+'\n')
    results={}
    for name,profile in [('reference',reference/'model.json'),('independent',source)]:
        with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        frames=[]
        with (artifacts/(name+'.log')).open('w') as error:
            process=subprocess.Popen([str(binary),'--port',str(port),'--model',str(profile)],stdout=subprocess.DEVNULL,stderr=error,**hidden())
            try:
                for _ in range(400):
                    if process.poll() is not None:raise RuntimeError('CPU inference failed to start')
                    try:
                        with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                    except OSError:time.sleep(.05)
                else:raise TimeoutError('CPU inference unavailable')
                with socket.create_connection(('127.0.0.1',port),timeout=600) as client,client.makefile('rwb') as stream:
                    for line in (case/'packets.jsonl').open(encoding='utf-8-sig'):
                        packet=json.loads(line)
                        if packet['sim_step']>a.steps:break
                        packet['request_diagnostics']=True
                        if packet['sim_step']==a.steps:packet['request_map']=True
                        stream.write(json.dumps(packet,separators=(',',':')).encode()+b'\n');stream.flush()
                        raw=stream.readline(8388609)
                        if not raw or len(raw)>8388608:raise RuntimeError('Invalid CPU inference reply')
                        reply=json.loads(raw)
                        if reply.get('type')=='error':raise RuntimeError(reply)
                        frames.append(dict(step=reply['sim_step'],goals=reply['goals'],nodes=reply['nodes'],cues=reply['cues'],observed_cells=reply['observed_cells']))
                        if packet['sim_step']==a.steps:(artifacts/(name+'-map.json')).write_bytes(raw)
                if not frames or frames[-1]['step']!=a.steps:raise ValueError('Missing requested sensor checkpoint')
                results[name]=frames
            finally:process.terminate();process.wait(timeout=15)
    if results['reference']!=results['independent']:
        first=next(((left,right) for left,right in zip(results['reference'],results['independent']) if left!=right),None)
        (artifacts/'control-mismatch.json').write_text(json.dumps(dict(first=first),indent=2)+'\n')
        raise AssertionError('Map decoder update changed frozen-explorer control, poses or candidates')
    left=json.loads((artifacts/'reference-map.json').read_bytes());right=json.loads((artifacts/'independent-map.json').read_bytes())
    left_cells={tuple(c['key']):c['occupancy'] for c in left['cells']};right_cells={tuple(c['key']):c['occupancy'] for c in right['cells']}
    changed=sum(abs(left_cells.get(key,.5)-right_cells.get(key,.5))>1e-3 for key in left_cells.keys()|right_cells.keys())
    if not changed:raise AssertionError('Probe did not exercise different map outputs')
    report=dict(status='pass',purpose='Control equivalence on fixed TRAIN sensor replay, not accuracy or closed-loop acceptance',
        case=record['id'],steps=a.steps,frames=len(results['reference']),all_goals_identical=True,all_estimated_poses_identical=True,all_cues_identical=True,
        unchanged_raw_observation_counts=True,changed_output_cells=changed,model_sha256=sha(source),navigation_sha256=model['navigation_sha256'],
        server_sha256=sha(binary),sensor_packets_sha256=sha(case/'packets.jsonl'),truth_to_native=False,final_test_opened=False,closed_loop_acceptance=False)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
