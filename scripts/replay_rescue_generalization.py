"""Replay SENSOR packets for developmental candidate extraction, not closed-loop acceptance.

Changing acoustic peaks changes guidance, so fresh physical validation is still required.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--splits',default='train,valid-known,valid-unknown');p.add_argument('--workers',type=int,default=3)
    p.add_argument('--server',default='tmp/generic-rust-target/release/rescue_server.exe')
    p.add_argument('--data-root',default='experiments/raw/rescue/generalization-v35');p.add_argument('--extra-train-root');a=p.parse_args()
    splits=a.splits.split(',')
    if any(s.startswith('test') for s in splits):raise ValueError('This development tool does not open TEST')
    rows=[(split,path.parent) for split in splits for path in sorted((ROOT/a.data_root/split).glob('*/complete.json'))]
    if a.extra_train_root:
        collection=json.loads((ROOT/a.extra_train_root/'collection.json').read_bytes())
        if collection['status']!='complete' or not collection['split'].startswith('train'):raise ValueError('Extra cases must be registered TRAIN')
        rows.extend(('train',path.parent) for path in sorted((ROOT/a.extra_train_root).glob('*/complete.json')))
    model=ROOT/a.model;binary=ROOT/a.server;out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    error=(out/'server.log').open('w',encoding='utf-8')
    process=subprocess.Popen([str(binary),'--port',str(port),'--model',str(model)],stdout=subprocess.DEVNULL,stderr=error)
    try:
        for _ in range(200):
            if process.poll() is not None:raise RuntimeError('Inference failed to start')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1):break
            except OSError:time.sleep(.05)
        else:raise TimeoutError('Inference unavailable')
        def run(row):
            split,directory=row;destination=out/split/directory.name
            if (destination/'complete.json').exists():return directory.name
            destination.mkdir(parents=True,exist_ok=True)
            failures=[]
            for attempt in range(3):
              try:
                replay(directory,destination,port)
                break
              except (OSError,RuntimeError) as failure:
                failures.append(str(failure));print(json.dumps(dict(retry=attempt+1,case=directory.name,error=str(failure))),flush=True)
                if attempt==2:raise
            record=dict(source_case=str(directory.relative_to(ROOT)),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                        server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),transient_retries=failures,
                        method='fixed-trajectory sensor replay; not validation of changed guidance',truth_to_server=False)
            (destination/'complete.json').write_text(json.dumps(record,indent=2)+'\n');return directory.name
        def replay(directory,destination,port):
            with socket.create_connection(('127.0.0.1',port),timeout=600) as client,client.makefile('rwb') as stream:
                for line in (directory/'packets.jsonl').open(encoding='utf-8-sig'):
                    packet=json.loads(line);packet['request_diagnostics']=True
                    stream.write(json.dumps(packet,separators=(',',':')).encode()+b'\n');stream.flush()
                    data=stream.readline(8_388_609)
                    if not data or len(data)>8_388_608:raise RuntimeError('Invalid mapping frame')
                    reply=json.loads(data)
                    if reply.get('type')=='error':raise RuntimeError(reply)
                    if packet['request_map']:(destination/f"map-{packet['sim_step']}.json").write_bytes(data)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i,name in enumerate(pool.map(run,rows),1):
                if i%5==0 or i==len(rows):print(json.dumps(dict(completed=i,total=len(rows),case=name)),flush=True)
    finally:
        process.terminate();process.wait(timeout=15);error.close()
    (out/'manifest.json').write_text(json.dumps(dict(status='complete',cases=len(rows),splits=splits,truth_to_server=False,
                model=a.model,data_root=a.data_root,extra_train_root=a.extra_train_root,model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),closed_loop_acceptance=False),indent=2)+'\n')


if __name__=='__main__':main()
