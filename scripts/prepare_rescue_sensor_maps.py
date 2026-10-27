"""Replay acquisition paths without a model to avoid calibrating already calibrated maps."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from rescue_dataset import complete_jsonl,final_acquisition_reply

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'valid'], required=True)
    parser.add_argument('--raw', default='experiments/raw/rescue')
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=4)
    parser.add_argument('--watch-expected',type=int,default=0)
    parser.add_argument('--server',default='rust/target/release/rescue_server.exe')
    args = parser.parse_args()
    raw = ROOT / args.raw
    ground_path = raw / f'grids-{args.split}.jsonl'
    ground = complete_jsonl(ground_path)
    assert args.watch_expected or len(ground) >= 15 and len(ground) % 15 == 0
    assert len({row['map_id'] for row in ground}) == len(ground)
    out = raw / 'sensor-maps'; out.mkdir(exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    binary = ROOT / args.server
    binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    process = subprocess.Popen([str(binary), '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None: raise RuntimeError('Mapping server failed')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2): break
            except OSError: time.sleep(.05)
        else: raise TimeoutError('Mapping server unavailable')

        def replay(row):
            audit = raw / f"{row['run_id']}.jsonl"
            destination = out / f"{row['run_id']}.json"
            metadata=out/f"{row['run_id']}.meta.json"
            deadline=time.monotonic()+10
            while final_acquisition_reply(audit) is None:
                if time.monotonic()>deadline:raise RuntimeError(f'Incomplete acquisition audit: {audit}')
                time.sleep(.1)
            if destination.exists() and metadata.exists():
                entry=json.loads(metadata.read_bytes())
                if entry.get('server_sha256')==binary_hash and entry['sensor_map_sha256']==hashlib.sha256(destination.read_bytes()).hexdigest() and entry['audit_sha256']==hashlib.sha256(audit.read_bytes()).hexdigest():return entry
            reply = None
            with socket.create_connection(('127.0.0.1', port), timeout=30) as client, client.makefile('rwb') as stream, audit.open(encoding='utf-8') as source:
                for line in source:
                    if '"scans":' not in line: continue
                    stream.write(line.replace('"request_diagnostics":true','"request_diagnostics":false').encode('utf-8')); stream.flush()
                    data = stream.readline(8_388_609)
                    assert data and len(data) <= 8_388_608
                    reply = json.loads(data); assert reply.get('type') != 'error', reply
            assert reply and reply['cells'] and reply['model_version'] == 'sensor-baseline'
            assert all(cell['source'] != 'learned_prior' for cell in reply['cells'])
            data = json.dumps(reply, separators=(',', ':')).encode('utf-8')
            destination.write_bytes(data)
            print('RAW_SENSOR_MAP', row['map_id'], flush=True)
            entry=dict(map_id=row['map_id'], run_id=row['run_id'], server_sha256=binary_hash,sensor_map_sha256=hashlib.sha256(data).hexdigest(), audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest())
            metadata.write_text(json.dumps(entry)+'\n',encoding='utf-8',newline='\n');return entry

        completed={}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                ground=complete_jsonl(ground_path)
                pending=[r for r in ground if r['run_id'] not in completed]
                for row in pool.map(replay,pending):completed[row['run_id']]=row
                if not args.watch_expected or len(completed)>=args.watch_expected:break
                time.sleep(2)
        rows=[completed[r['run_id']] for r in ground]
    finally:
        process.terminate(); process.wait(timeout=10)
    manifest = dict(split=args.split, maps=len(rows), model_applied=False, trajectory='Recorded acquisition trajectory; no new physics or guidance', rows=rows)
    (out / f'manifest-{args.split}.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
