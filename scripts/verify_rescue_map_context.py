"""Check native raw-map features against Python on TRAIN sensor packets only."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from collect_rescue_registered import hidden
from rescue_map_context import MapContext,VERSION

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--model',required=True)
    p.add_argument('--server',required=True);p.add_argument('--output',required=True)
    p.add_argument('--teacher-base',help='Compare the frozen teacher confidence feature instead of measured-map context');a=p.parse_args()
    version='thermal-spectral-teacher33-v5' if a.teacher_base else VERSION
    directory=(ROOT/a.case).resolve();output=ROOT/a.output
    if not directory.is_relative_to(ROOT/'experiments/raw/rescue') or 'train' not in directory.parts or any(part.startswith(('valid','test')) for part in directory.parts):raise ValueError('Registered TRAIN sensor case required')
    case=json.loads((directory/'complete.json').read_bytes())
    if case['steps']!=18000 or not case['id'].startswith('g60-train-'):raise ValueError('Full layered-collapse TRAIN case required for this parity probe')
    if output.exists():raise ValueError('Use a new report path')
    artifacts=output.parent/(output.stem+'-artifacts');artifacts.mkdir(parents=True,exist_ok=False)
    binary=ROOT/a.server;source=ROOT/a.model;model=json.loads(source.read_bytes())
    if model['track_feature_map']!='thermal-spectral32-v3':raise ValueError('A validated spectral32 forest is required to compare the unchanged prefix')
    model.update(volume_file='',volume_sha256='')
    teacher=json.loads((ROOT/a.teacher_base).read_bytes()) if a.teacher_base else None
    if teacher and model['track_trees'][:len(teacher['track_trees'])]!=teacher['track_trees']:raise ValueError('Teacher is not the exact frozen prefix')
    snapshots=[1500,6000,18000];results={}
    for mode in ['thermal-spectral32-v3',version]:
        descriptor=dict(model,track_feature_map=mode)
        if teacher and mode==version:descriptor.update(track_teacher_trees=len(teacher['track_trees']),track_teacher_bias=teacher['track_bias'])
        profile=artifacts/(mode+'.json');profile.write_text(json.dumps(descriptor,separators=(',',':'))+'\n')
        with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        with (artifacts/(mode+'.log')).open('w') as error:
            process=subprocess.Popen([str(binary),'--port',str(port),'--model',str(profile)],stdout=subprocess.DEVNULL,stderr=error,**hidden())
            try:
                for _ in range(200):
                    if process.poll() is not None:raise RuntimeError('Native mapper failed to start')
                    try:
                        with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                    except OSError:time.sleep(.05)
                else:raise TimeoutError('Native mapper unavailable')
                saved={}
                with socket.create_connection(('127.0.0.1',port),timeout=600) as client,client.makefile('rwb') as stream:
                    for line in (directory/'packets.jsonl').open(encoding='utf-8-sig'):
                        packet=json.loads(line);packet['request_diagnostics']=True
                        if packet['sim_step'] in snapshots:packet['request_map']=True
                        stream.write(json.dumps(packet,separators=(',',':')).encode()+b'\n');stream.flush()
                        data=stream.readline(8388609)
                        if not data or len(data)>8388608:raise RuntimeError('Invalid native frame')
                        reply=json.loads(data)
                        if reply.get('type')=='error':raise RuntimeError(reply)
                        if packet['sim_step'] in snapshots:
                            saved[packet['sim_step']]=reply
                            (artifacts/f'{mode}-{packet["sim_step"]}.json').write_bytes(data)
                if set(saved)!=set(snapshots):raise ValueError('Missing sensor checkpoint')
                results[mode]=saved
            finally:process.terminate();process.wait(timeout=15)
    rows=[]
    for step in snapshots:
        baseline=results['thermal-spectral32-v3'][step];extended=results[version][step]
        if baseline['cells']!=extended['cells'] or baseline['goals']!=extended['goals']:raise AssertionError('Feature extension changed raw maps or control for the same forest')
        left=baseline['proposals'];right=extended['proposals']
        if not left or len(left)!=len(right):raise AssertionError('Proposal count changed or probe is empty')
        maximum=0.
        if teacher:
            from rescue_residual_head import teacher_features
            expected=teacher_features(teacher,[q['features'] for q in left])[:,32:].tolist()
        else:
            context=MapContext(extended['cells']);expected=[context.features(q['position']) for q in right]
        for old,new,features in zip(left,right,expected):
            if old['position']!=new['position'] or old['features']!=new['features'][:32] or len(new['features'])!=32+len(features):raise AssertionError('Spectral prefix or proposal position changed')
            error=max(abs(a-b) for a,b in zip(features,new['features'][32:]))
            maximum=max(maximum,error)
        if maximum>1e-12:raise AssertionError(f'Raw measured-context mismatch: {maximum}')
        rows.append(dict(step=step,proposals=len(right),maximum_context_error=maximum,unchanged_prefix=True,unchanged_control=True))
    report=dict(status='pass',case=case['id'],sensor_packets_sha256=hashlib.sha256((directory/'packets.jsonl').read_bytes()).hexdigest(),
        source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        feature_map=version,rows=rows,truth_to_native=False,final_test_opened=False,closed_loop_acceptance=False)
    if teacher:report['teacher_base_sha256']=hashlib.sha256((ROOT/a.teacher_base).read_bytes()).hexdigest()
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
