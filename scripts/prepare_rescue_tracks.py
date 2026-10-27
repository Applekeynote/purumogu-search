"""Export sensor-only track proposals; ground truth stays in the offline training file."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import math
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['train','valid'],required=True)
    parser.add_argument('--model',default='models/rescue-v9/model.json')
    parser.add_argument('--server',default='rust/target/release/rescue_server.exe')
    parser.add_argument('--output',default='experiments/raw/rescue/track-proposals')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    args=parser.parse_args()
    raw=ROOT/'experiments/raw/rescue'
    ground_path=raw/f'grids-{args.split}.jsonl'
    ground=[json.loads(line) for line in ground_path.read_text(encoding='utf-8-sig').splitlines()]
    model_path=ROOT/args.model; model=json.loads(model_path.read_text(encoding='utf-8'))
    assert all(r['split']==args.split for r in ground)
    assert all((r['map_id'] in model['train_maps'])==(args.split=='train') for r in ground)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    binary=ROOT/args.server
    process=subprocess.Popen([str(binary),'--port',str(port),'--model',str(model_path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Mapping server failed')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2):break
            except OSError:time.sleep(.05)
        else:raise TimeoutError('Mapping server unavailable')
        def replay(row):
            destination=out/f"{row['run_id']}.json"
            audit=raw/f"{row['run_id']}.jsonl"
            audit_hash=hashlib.sha256(audit.read_bytes()).hexdigest()
            if destination.exists():
                cached=json.loads(destination.read_text(encoding='utf-8'))
                if cached.get('audit_sha256')==audit_hash and cached.get('server_sha256')==hashlib.sha256(binary.read_bytes()).hexdigest():
                    print('CACHED',row['map_id'],flush=True);return cached
            reply=None
            with socket.create_connection(('127.0.0.1',port),timeout=60) as client,client.makefile('rwb') as stream,audit.open(encoding='utf-8') as source:
                for line in source:
                    if '"scans":' not in line:continue
                    packet=json.loads(line);packet['request_diagnostics']=True
                    stream.write(json.dumps(packet,separators=(',',':')).encode()+b'\n');stream.flush()
                    data=stream.readline(8388609);assert data and len(data)<=8388608
                    reply=json.loads(data);assert reply.get('type')!='error',reply
            assert reply is not None and 'proposals' in reply
            proposals=reply['proposals']
            # TRAIN/VALID labels are added here, after inference, and never sent to Rust.
            for item in proposals:
                distance=min((math.dist(item['position'],p) for p in row['victim_positions']),default=None)
                item['distance_to_person_m']=distance
                item['label']=int(distance is not None and distance<.25)
            result=dict(map_id=row['map_id'],run_id=row['run_id'],scene=row['scene'],split=args.split,
                        victim_positions=row['victim_positions'],proposals=proposals,audit_sha256=audit_hash,
                        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            destination.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
            print('PROPOSALS',row['map_id'],len(proposals),sum(p['label'] for p in proposals),flush=True)
            return result
        with ThreadPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(replay,ground))
    finally:process.terminate();process.wait(timeout=10)
    manifest=dict(split=args.split,maps=len(rows),truth_sha256=hashlib.sha256(ground_path.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),server_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  rows=[dict(map_id=r['map_id'],run_id=r['run_id'],audit_sha256=r['audit_sha256']) for r in rows])
    (out/f'manifest-{args.split}.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
