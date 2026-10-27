"""Replay recorded sensor packets to Rust; labels are read only by this evaluator."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:yield json.loads(line)

def match(cues,people):
    # Small exact maximum-cardinality matching, minimizing error for ties.
    best=(0,0.)
    def visit(i,used,count,error):
        nonlocal best
        if i==len(cues):
            if count>best[0] or count==best[0] and error<best[1]:best=(count,error)
            return
        visit(i+1,used,count,error)
        for j,p in enumerate(people):
            if j in used:continue
            distance=math.dist(cues[i]['position'],p)
            if distance<.25:visit(i+1,used|{j},count+1,error+distance)
    visit(0,set(),0,0.)
    return best

def wilson(success,total):
    if not total:return None
    p=success/total;z=1.95996398454;den=1+z*z/total
    center=(p+z*z/(2*total))/den;half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-half),min(1,center+half)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--split',choices=['valid','test'],default='valid');parser.add_argument('--output',required=True);parser.add_argument('--final-test',action='store_true');parser.add_argument('--server',default='rust/target/release/rescue_server.exe');parser.add_argument('--raw',default='experiments/raw/rescue');args=parser.parse_args()
    assert args.split!='test' or args.final_test,'Explicit --final-test is required to open TEST'
    model_path=(ROOT/args.model).resolve();model=json.loads(model_path.read_text(encoding='utf-8'))
    raw=ROOT/args.raw;truth_path=raw/f'grids-{args.split}.jsonl';truth=list(lines(truth_path))
    ids={r['map_id'] for r in truth};assert not ids&set(model['train_maps'])
    if args.split=='test':assert not ids&set(model['valid_maps'])
    with socket.socket() as probe:probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    binary=(ROOT/args.server).resolve();binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    process=subprocess.Popen([str(binary),'--port',str(port),'--model',str(model_path)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    rows=[]
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Mapping server failed to start')
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2):break
            except OSError:time.sleep(.05)
        else:raise TimeoutError('Mapping server unavailable')
        for ground in truth:
            audit=raw/f"{ground['run_id']}.jsonl";reply=None;history=[]
            with socket.create_connection(('127.0.0.1',port),timeout=30) as client,client.makefile('rwb') as stream,audit.open(encoding='utf-8') as source:
                for line in source:
                    if '"scans":' not in line:continue
                    stream.write(line.encode('utf-8'));stream.flush();data=stream.readline(8388609)
                    assert data and len(data)<=8388608,'Invalid map frame'
                    reply=json.loads(data);assert reply.get('type')!='error',reply
                    if reply['sim_step']%100==0:
                        found,_=match(reply['cues'],ground['victim_positions'])
                        history.append(dict(sim_step=reply['sim_step'],matched=found,cues=reply['cues']))
            assert reply is not None
            predicted={tuple(c['key']) for c in reply['cells'] if c['occupancy']>.6}
            actual={tuple(c) for c in ground['occupied_keys']};intersection=len(predicted&actual);union=len(predicted|actual)
            tp,error=match(reply['cues'],ground['victim_positions']);false=len(reply['cues'])-tp
            row={'scene':ground['scene'],'map_id':ground['map_id'],'iou':intersection/max(1,union),'intersection':intersection,'union':union,'matched':tp,'false_alarms':false,'people':len(ground['victim_positions']),'localization_error_m':error/tp if tp else None,'observed_cells':reply['observed_cells'],'completion_family':reply['completion_family'],'completion_confidence':reply['completion_confidence'],'candidates':reply['cues']}
            row['max_simultaneously_matched']=max((h['matched'] for h in history),default=0)
            row['last_matching_step']=max((h['sim_step'] for h in history if h['matched']),default=None)
            row['cue_history']=history
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='cue_history'}),flush=True)
    finally:
        process.terminate();process.wait(timeout=10)
    tp=sum(r['matched'] for r in rows);fp=sum(r['false_alarms'] for r in rows);people=sum(r['people'] for r in rows)
    macro=sum(r['iou'] for r in rows)/len(rows);micro=sum(r['intersection'] for r in rows)/max(1,sum(r['union'] for r in rows));precision=tp/max(1,tp+fp);recall=tp/max(1,people)
    result={'split':args.split,'server_sha256':binary_hash,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'truth_sha256':hashlib.sha256(truth_path.read_bytes()).hexdigest(),'maps':len(rows),'map_iou_macro':macro,'map_iou_micro':micro,'human_precision':precision,'human_recall':recall,'human_precision_wilson95':wilson(tp,tp+fp),'human_recall_wilson95':wilson(tp,people),'people':people,'matched':tp,'false_alarms':fp,'target':.95,'target_met':min(macro,micro,precision,recall)>=.95,'rows':rows}
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
