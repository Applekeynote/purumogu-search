"""Replay recorded sensor packets to Rust; labels are read only by this evaluator."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from rescue_dataset import assert_disjoint_worlds,control_signature,node_count,validate_model_splits,validate_paired_rows,world_group

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
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--split',choices=['valid','test'],default='valid');parser.add_argument('--output',required=True);parser.add_argument('--final-test',action='store_true');parser.add_argument('--server',default='rust/target/release/rescue_server.exe');parser.add_argument('--raw',default='experiments/raw/rescue');parser.add_argument('--rollout-report');parser.add_argument('--workers',type=int,choices=[1,2,3],default=3);args=parser.parse_args()
    assert args.split!='test' or args.final_test,'Explicit --final-test is required to open TEST'
    model_path=(ROOT/args.model).resolve();model=json.loads(model_path.read_text(encoding='utf-8'))
    validate_model_splits(model)
    raw=ROOT/args.raw;truth_path=raw/f'grids-{args.split}.jsonl';truth=list(lines(truth_path))
    ids={r['map_id'] for r in truth};assert not ids&set(model['train_maps'])
    assert len(ids)==len(truth) and all(r['split']==args.split for r in truth)
    assert_disjoint_worlds(ids,set(model['train_maps'])|set(model.get('calibration_train_maps',[])))
    if args.split=='test':assert not ids&set(model['valid_maps'])
    if args.split=='test':assert_disjoint_worlds(ids,model['valid_maps'])
    if model.get('distributed_swarm'):
        validate_paired_rows(truth)
    if args.rollout_report:
        provenance_path=ROOT/args.rollout_report/'rollout-provenance.json'
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text(encoding='utf-8-sig'))
            acquisition_path=ROOT/provenance['model'];acquisition=json.loads(acquisition_path.read_bytes())
            assert hashlib.sha256(acquisition_path.read_bytes()).hexdigest()==provenance['model_sha256']
            assert control_signature(acquisition,provenance.get('diagnostics',False))==control_signature(model,provenance.get('diagnostics',False)),'Replay cannot verify changed control on the old trajectories'
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
        def replay(ground):
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
            row={'scene':ground['scene'],'map_id':ground['map_id'],'sim_steps':reply['sim_step'],'iou':intersection/max(1,union),'intersection':intersection,'union':union,'matched':tp,'false_alarms':false,'people':len(ground['victim_positions']),'localization_error_m':error/tp if tp else None,'observed_cells':reply['observed_cells'],'completion_family':reply['completion_family'],'completion_confidence':reply['completion_confidence'],'candidates':reply['cues']}
            row['nodes']=node_count(ground);row['world_group_id']=world_group(ground['map_id'])
            row['max_simultaneously_matched']=max((h['matched'] for h in history),default=0)
            row['last_matching_step']=max((h['sim_step'] for h in history if h['matched']),default=None)
            row['cue_history']=history
            print(json.dumps({k:v for k,v in row.items() if k!='cue_history'}),flush=True)
            return row
        # Each connection owns an independent Mapper. Keep one server permit spare.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows=list(pool.map(replay,truth))
    finally:
        process.terminate();process.wait(timeout=10)
    tp=sum(r['matched'] for r in rows);fp=sum(r['false_alarms'] for r in rows);people=sum(r['people'] for r in rows)
    macro=sum(r['iou'] for r in rows)/len(rows);micro=sum(r['intersection'] for r in rows)/max(1,sum(r['union'] for r in rows));precision=tp/max(1,tp+fp);recall=tp/max(1,people)
    result={'split':args.split,'server_sha256':binary_hash,'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'truth_sha256':hashlib.sha256(truth_path.read_bytes()).hexdigest(),'maps':len(rows),'map_iou_macro':macro,'map_iou_micro':micro,'human_precision':precision,'human_recall':recall,'human_precision_wilson95':wilson(tp,tp+fp),'human_recall_wilson95':wilson(tp,people),'people':people,'matched':tp,'false_alarms':fp,'target':.95,'target_met':min(macro,micro,precision,recall)>=.95,'rows':rows}
    result['by_node_count']={}
    result['world_groups']=len({r['world_group_id'] for r in rows})
    result['interval_note']='Wilson intervals are descriptive references. Paired node counts share the same world and sources; pooled detections are not independent trials.'
    for nodes in sorted({r['nodes'] for r in rows}):
        group=[r for r in rows if r['nodes']==nodes];a=sum(r['matched'] for r in group);b=sum(r['false_alarms'] for r in group);n=sum(r['people'] for r in group)
        metrics=dict(maps=len(group),matched=a,false_alarms=b,people=n,map_iou_macro=sum(r['iou'] for r in group)/len(group),map_iou_micro=sum(r['intersection'] for r in group)/max(1,sum(r['union'] for r in group)),human_precision=a/max(1,a+b),human_recall=a/max(1,n))
        metrics['target_met']=min(metrics[k] for k in ['map_iou_macro','map_iou_micro','human_precision','human_recall'])>=.95
        result['by_node_count'][str(nodes)]=metrics
    result['target_met']=result['target_met'] and all(bucket['target_met'] for bucket in result['by_node_count'].values())
    steps={r['sim_steps'] for r in rows};assert len(steps)==1,'Mixed mission budgets cannot be pooled'
    result['mission_steps']=steps.pop();result['mission_seconds']=result['mission_steps']*.02
    if args.rollout_report:
        report_path=ROOT/args.rollout_report;manifest=json.loads((report_path/'manifest.json').read_text(encoding='utf-8-sig'));outcomes=list(json.loads((report_path/'outcomes.json').read_text(encoding='utf-8-sig')))
        assert manifest['status']=='complete' and manifest['steps']==result['mission_steps'] and {r['map_id'] for r in outcomes}==ids
        result['lost_nodes']=sum(r['lost_nodes'] for r in outcomes);result['rollout_manifest_sha256']=hashlib.sha256((report_path/'manifest.json').read_bytes()).hexdigest()
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__':main()
