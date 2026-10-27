"""Fit track evidence in Rust on TRAIN; select thresholds on VALID, never TEST."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
from evaluate_rescue_replay import match

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['mean_score','peak_score','log_hits_over_5','observers_over_12','position_spread_over_25cm',
          'radius_m','age_over_9000','span_over_9000','nearest_observer_distance_over_2',
          'mean_human_score','mean_temperature_over_2','mean_log_range_residual_over_3']

def choose(rows,weights,threshold,nms):
    tp=fp=people=0;details=[]
    for row in rows:
        candidates=[]
        for p in row['proposals']:
            z=weights[0]+sum(a*b for a,b in zip(weights[1:],p['features']))
            score=1/(1+math.exp(-max(-40,min(40,z))))
            if score>=threshold:candidates.append(dict(position=p['position'],score=score))
        candidates.sort(key=lambda p:-p['score']);cues=[]
        for p in candidates:
            if all(math.dist(p['position'],q['position'])>nms for q in cues):cues.append(p)
            if len(cues)==3:break
        count,_=match(cues,row['victim_positions']);false=len(cues)-count
        tp+=count;fp+=false;people+=len(row['victim_positions'])
        details.append(dict(map_id=row['map_id'],matched=count,false_alarms=false,people=len(row['victim_positions']),cues=cues))
    return dict(matched=tp,false_alarms=fp,people=people,human_precision=tp/max(1,tp+fp),human_recall=tp/max(1,people),rows=details)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='models/rescue-v9/model.json');parser.add_argument('--name',default='rescue-v11');parser.add_argument('--ridge',type=float,default=.001);parser.add_argument('--raw',default='experiments/raw/rescue');parser.add_argument('--report',default='reports/rescue/valid');parser.add_argument('--proposal-model');args=parser.parse_args()
    raw=ROOT/args.raw;pool=raw/'track-proposals';base=ROOT/args.base;model=json.loads(base.read_bytes())
    manifest=json.loads((pool/'manifest-train.json').read_bytes())
    proposal_model_path=ROOT/args.proposal_model if args.proposal_model else base
    proposal_model=json.loads(proposal_model_path.read_bytes())
    assert manifest['maps']==len(model['train_maps']) and manifest['model_sha256']==hashlib.sha256(proposal_model_path.read_bytes()).hexdigest()
    for field in ['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma']:
        assert proposal_model.get(field)==model.get(field),'Proposal-generating acoustic parameters must match; geometry refitting alone may be transferred'
    train=[json.loads((pool/f"{r['run_id']}.json").read_bytes()) for r in manifest['rows']]
    assert {r['map_id'] for r in train}==set(model['train_maps']) and all(r['split']=='train' for r in train)
    valid_manifest=json.loads((ROOT/args.report/'manifest.json').read_text(encoding='utf-8-sig'))
    assert valid_manifest['status']=='complete' and valid_manifest['controller_revision']=='progress-3cm-v2'
    valid=[]
    for line in (raw/'grids-valid.jsonl').read_text(encoding='utf-8-sig').splitlines():
        row=json.loads(line);audit=raw/f"{row['run_id']}.jsonl";last=None
        with audit.open(encoding='utf-8') as stream:
            for packet in stream:
                if '"proposals":' in packet:last=json.loads(packet)
        assert last is not None and last['cells'],'Final map diagnostics are required'
        valid.append(dict(map_id=row['map_id'],split='valid',victim_positions=row['victim_positions'],proposals=last['proposals']))
    assert {r['map_id'] for r in valid}==set(model['valid_maps']) and not set(model['train_maps'])&{r['map_id'] for r in valid}
    train_path=pool/'training.json';train_path.write_text(json.dumps(train,separators=(',',':')),encoding='utf-8')
    train_hash=hashlib.sha256(train_path.read_bytes()).hexdigest();source_hash=hashlib.sha256(base.read_bytes()+train_path.read_bytes()).hexdigest()
    output=ROOT/'models'/args.name;output.mkdir(parents=True,exist_ok=True);model_path=output/'model.json'
    subprocess.run([str(ROOT/'rust/target/release/train_tracks.exe'),str(train_path),str(base),str(model_path),str(args.ridge),source_hash],check=True)
    fitted=json.loads(model_path.read_bytes());weights=fitted['track_weights']
    grid=[]
    for nms in [.25,.35,.45]:
        for i in range(1,40):
            threshold=i/40;result=choose(valid,weights,threshold,nms)
            grid.append(dict(threshold=threshold,nms_m=nms,**{k:v for k,v in result.items() if k!='rows'}))
    selected=max(grid,key=lambda r:(min(r['human_precision'],r['human_recall']),r['human_precision']+r['human_recall'],-r['false_alarms'],r['threshold']))
    fitted['track_threshold']=selected['threshold'];fitted['track_nms_m']=selected['nms_m']
    data=json.dumps(fitted,separators=(',',':')).encode();model_path.write_bytes(data);model_hash=hashlib.sha256(data).hexdigest()
    (output/'SHA256SUMS').write_text(model_hash+'  model.json\n',encoding='ascii',newline='\n')
    oracle=sum(sum(any(math.dist(p['position'],v)<.25 for p in row['proposals']) for v in row['victim_positions']) for row in valid)
    report=dict(method='Rust IRLS Bernoulli maximum likelihood with L2 penalty; no absolute position features',ridge=args.ridge,features=FEATURES,
                train_maps=len(train),valid_maps=len(valid),train_proposals=sum(len(r['proposals']) for r in train),
                train_labels='Offline distance <0.25m to any TRAIN person source; duplicate proposals allowed before NMS',
                parent_model_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),proposal_model_sha256=hashlib.sha256(proposal_model_path.read_bytes()).hexdigest(),train_proposal_sha256=train_hash,model_sha256=model_hash,
                valid_controller_revision=valid_manifest['controller_revision'],test_accessed=False,valid_proposal_oracle_matched=oracle,
                selected=selected,valid=choose(valid,weights,selected['threshold'],selected['nms_m']),train=choose(train,weights,selected['threshold'],selected['nms_m']),selection_grid=grid,
                note='Python proposal scoring is a selection diagnostic. Acceptance requires independent actual Rust replay and final unopened TEST.')
    (output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['train','valid','selection_grid']},indent=2))

if __name__=='__main__':main()
