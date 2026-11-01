"""Collect a pre-registered TRAIN/VALID plan in isolated Unity processes.

These are numerical simulation workers, not AI agents. Each process has its own
project, inference server, local port and log. Completed cases are immutable; no
low accuracy result is retried or dropped. TEST needs an explicit frozen artifact.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path,value):
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('Immutable artifact changed: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')


def workspace(path):
    path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Output/project must remain in the workspace')
    return path


def hidden():
    if os.name!='nt':return {}
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    return dict(startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--model',required=True)
    p.add_argument('--server',default='tmp/generic-rust-target/release/rescue_server.exe')
    p.add_argument('--unity',default='C:/Program Files/Unity/Hub/Editor/6000.4.1f1/Editor/Unity.exe')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--project-prefix',required=True);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Use one to four numerical workers')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.project_prefix):raise ValueError('Use a plain project prefix')
    plan_path=ROOT/a.plan;plan=json.loads(plan_path.read_bytes())
    if plan['split'].startswith('test'):raise ValueError('Development collector never opens final TEST')
    if not plan['split'].startswith(('train','valid')):raise ValueError('Registered TRAIN/VALID role required')
    binary=(ROOT/a.server).resolve();model=(ROOT/a.model).resolve();output=workspace(Path(plan['output']));output.mkdir(parents=True,exist_ok=True)
    if sha(model)!=plan['model_sha256'] or sha(binary)!=plan['server_sha256']:raise ValueError('Frozen acquisition artifact mismatch')
    ids=[c['id'] for c in plan['cases']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate cases')
    for name,digest in plan.get('geometry_sources',{}).items():
        if sha(ROOT/'unity/Assets/Scripts'/name)!=digest:raise ValueError('Physical geometry changed since registration')
    for name,digest in plan.get('acquisition_sources',{}).items():
        if sha(ROOT/name)!=digest:raise ValueError('Acquisition source changed since registration: '+name)
    if sha(ROOT/'unity/Assets/Scripts/RescueRobot.cs')!=plan['sensor_source_sha256']:raise ValueError('Sensor/motor source changed')
    receipt=dict(plan_sha256=sha(plan_path),model_sha256=sha(model),server_sha256=sha(binary),workers=a.workers,truth_to_server=False)
    write_once(output/'acquisition-receipt.json',receipt)
    projects=[]
    for i in range(a.workers):
        project=workspace(ROOT/'tmp'/f'{a.project_prefix}-{i}');project.mkdir(parents=True,exist_ok=True)
        if (project/'Temp/UnityLockfile').exists():raise ValueError('Simulation project is already active')
        for name in ['Assets','Packages','ProjectSettings']:
            shutil.copytree(ROOT/'unity'/name,project/name,dirs_exist_ok=True)
        projects.append(project)
    jobs=[]
    for i,project in enumerate(projects):
        child=dict(plan,cases=plan['cases'][i::a.workers],collection_file=f'collection-worker-{i}.json')
        child_path=workspace(ROOT/'tmp'/f'{a.project_prefix}-plan-{i}.json');write_once(child_path,child);jobs.append((i,project,child_path,child))
    def work(job):
        i,project,child_path,child=job
        with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        log=ROOT/'tmp'/f'{a.project_prefix}-server-{i}.log'
        with log.open('w',encoding='utf-8') as error:
            server=subprocess.Popen([str(binary),'--port',str(port),'--model',str(model)],stdout=subprocess.DEVNULL,stderr=error,**hidden())
            process=None
            try:
                for _ in range(200):
                    if server.poll() is not None:raise RuntimeError('Inference server failed')
                    try:
                        with socket.create_connection(('127.0.0.1',port),timeout=.1):break
                    except OSError:time.sleep(.05)
                else:raise TimeoutError('Inference server unavailable')
                unity_log=ROOT/'tmp'/f'{a.project_prefix}-unity-{i}.log'
                command=[a.unity,'-batchmode','-nographics','-job-worker-count','4','-projectPath',str(project),'-executeMethod','Purumogu.Editor.ProjectTools.Run','-purumoguTask','rescue-generalization-collect','-purumoguPlan',str(child_path),'-purumoguPort',str(port),'-logFile',str(unity_log)]
                process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**hidden())
                last=-1
                while process.poll() is None:
                    complete=sum((output/c['id']/'complete.json').exists() for c in child['cases'])
                    if complete!=last:print(json.dumps(dict(worker=i,completed=complete,total=len(child['cases']))),flush=True);last=complete
                    time.sleep(1)
                if process.returncode:raise RuntimeError(f'Unity worker {i} failed; see {unity_log}')
                if not all((output/c['id']/'complete.json').exists() for c in child['cases']):raise ValueError('Worker did not complete its registered cases')
                return i
            finally:
                # Stop only children created by this worker, including on failed polling.
                for owned in [process,server]:
                    if owned is not None and owned.poll() is None:
                        owned.terminate()
                        try:owned.wait(timeout=15)
                        except subprocess.TimeoutExpired:owned.kill();owned.wait(timeout=15)
    with ThreadPoolExecutor(max_workers=a.workers) as workers:list(workers.map(work,jobs))
    outcomes=[]
    for case in plan['cases']:
        data=json.loads((output/case['id']/'complete.json').read_bytes())
        if data['case']!=case or data['truth_to_server']:raise ValueError('Case provenance mismatch')
        outcomes.append(data)
    write_once(output/'collection.json',dict(status='complete',split=plan['split'],steps=plan['steps'],plan=plan,truth_to_server=False,outcomes=outcomes,receipt=receipt))
    print(json.dumps(dict(status='complete',cases=len(outcomes),output=str(output))),flush=True)


if __name__=='__main__':main()
