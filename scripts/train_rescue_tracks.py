"""Fit track evidence in Rust on TRAIN; select thresholds on VALID, never TEST."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
from evaluate_rescue_replay import match
from rescue_dataset import control_signature,node_count,validate_model_splits

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['mean_score','peak_score','log_hits_over_5','observers_over_12','position_spread_over_25cm',
          'radius_m','age_over_9000','span_over_9000','nearest_observer_distance_over_2',
          'mean_human_score','mean_temperature_over_2','mean_log_range_residual_over_3']

def score_model(model,features):
    if model.get('track_trees'):
        z=model['track_bias']
        for tree in model['track_trees']:
            index=0
            while tree['nodes'][index]['feature'] is not None:
                node=tree['nodes'][index];index=node['left'] if features[node['feature']]<node['threshold'] else node['right']
            z+=tree['nodes'][index]['value']
    else:
        weights=model['track_weights'];z=weights[0]+sum(a*b for a,b in zip(weights[1:],features))
    return 1/(1+math.exp(-max(-40,min(40,z))))

def choose(rows,model,threshold,nms):
    tp=fp=people=0;details=[]
    for row in rows:
        candidates=[]
        for p in row['proposals']:
            score=score_model(model,p['features'])
            if score>=threshold:candidates.append(dict(position=p['position'],score=score))
        candidates.sort(key=lambda p:-p['score']);cues=[]
        for p in candidates:
            if all(math.dist(p['position'],q['position'])>nms for q in cues):cues.append(p)
            if len(cues)==3:break
        count,_=match(cues,row['victim_positions']);false=len(cues)-count
        tp+=count;fp+=false;people+=len(row['victim_positions'])
        details.append(dict(map_id=row['map_id'],nodes=node_count(row),matched=count,false_alarms=false,people=len(row['victim_positions']),cues=cues))
    by_count={}
    for nodes in sorted({r['nodes'] for r in details}):
        group=[r for r in details if r['nodes']==nodes];a=sum(r['matched'] for r in group);b=sum(r['false_alarms'] for r in group);n=sum(r['people'] for r in group)
        by_count[str(nodes)]=dict(matched=a,false_alarms=b,people=n,human_precision=a/max(1,a+b),human_recall=a/max(1,n))
    return dict(matched=tp,false_alarms=fp,people=people,human_precision=tp/max(1,tp+fp),human_recall=tp/max(1,people),by_node_count=by_count,rows=details)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='models/rescue-v9/model.json');parser.add_argument('--name',default='rescue-v11');parser.add_argument('--ridge',type=float,default=.001);parser.add_argument('--raw',default='experiments/raw/rescue');parser.add_argument('--report',default='reports/rescue/valid');parser.add_argument('--proposal-model');parser.add_argument('--train-pool');parser.add_argument('--boost',action='store_true');parser.add_argument('--depth',type=int,choices=[1,2,3,4],default=2);parser.add_argument('--rounds',type=int,default=60);parser.add_argument('--trainer',default='rust/target/release/train_tracks.exe');args=parser.parse_args()
    raw=ROOT/args.raw;pool=ROOT/args.train_pool if args.train_pool else raw/'track-proposals';base=ROOT/args.base;model=json.loads(base.read_bytes())
    validate_model_splits(model)
    manifest=json.loads((pool/'manifest-train.json').read_bytes())
    proposal_model_path=ROOT/args.proposal_model if args.proposal_model else base
    proposal_model=json.loads(proposal_model_path.read_bytes())
    assert model.get('distributed_swarm',False)==proposal_model.get('distributed_swarm',False),'Proposal observer normalization must match the fitted model'
    assert manifest['maps']==len(model['train_maps']) and manifest['model_sha256']==hashlib.sha256(proposal_model_path.read_bytes()).hexdigest()
    for field in ['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma']:
        assert proposal_model.get(field)==model.get(field),'Proposal-generating acoustic parameters must match; geometry refitting alone may be transferred'
    train=[json.loads((pool/f"{r['run_id']}.json").read_bytes()) for r in manifest['rows']]
    assert {r['map_id'] for r in train}==set(model['train_maps']) and all(r['split']=='train' for r in train)
    valid_manifest=json.loads((ROOT/args.report/'manifest.json').read_text(encoding='utf-8-sig'))
    assert valid_manifest['status']=='complete' and valid_manifest['controller_revision'] in ['progress-3cm-v2','whole-scene-drop-dense-v3','whole-scene-spherical-env-v31']
    valid_provenance_path=ROOT/args.report/'rollout-provenance.json'
    valid_provenance=json.loads(valid_provenance_path.read_text(encoding='utf-8-sig')) if valid_provenance_path.exists() else None
    if valid_provenance:
        acquisition_path=ROOT/valid_provenance['model'];acquisition=json.loads(acquisition_path.read_bytes())
        assert hashlib.sha256(acquisition_path.read_bytes()).hexdigest()==valid_provenance['model_sha256']
        for field in ['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma']:
            assert acquisition.get(field)==model.get(field),'Recorded VALID paths require unchanged control-generating acoustic parameters'
        assert control_signature(acquisition,valid_provenance.get('diagnostics',False))==control_signature(model,valid_provenance.get('diagnostics',False)),'Acquisition control changed; collect new VALID rollouts'
    valid=[]
    for line in (raw/'grids-valid.jsonl').read_text(encoding='utf-8-sig').splitlines():
        row=json.loads(line);audit=raw/f"{row['run_id']}.jsonl"
        with audit.open('rb') as stream:
            stream.seek(max(0,audit.stat().st_size-8_388_609));last=json.loads(stream.read().splitlines()[-1])
        assert last['sim_step']==valid_manifest['steps'] and last['cells'] and 'proposals' in last,'Final complete map diagnostics are required'
        valid.append(dict(map_id=row['map_id'],nodes=node_count(row),split='valid',victim_positions=row['victim_positions'],proposals=last['proposals']))
    assert {r['map_id'] for r in valid}==set(model['valid_maps']) and not set(model['train_maps'])&{r['map_id'] for r in valid}
    train_path=pool/'training.json';train_path.write_text(json.dumps(train,separators=(',',':')),encoding='utf-8')
    train_hash=hashlib.sha256(train_path.read_bytes()).hexdigest();source_hash=hashlib.sha256(base.read_bytes()+train_path.read_bytes()).hexdigest()
    output=ROOT/'models'/args.name;output.mkdir(parents=True,exist_ok=False);model_path=output/'model.json'
    command=[str(ROOT/args.trainer),str(train_path),str(base),str(model_path),str(args.ridge),source_hash]
    if args.boost:command+=['track-boost',str(args.depth),str(args.rounds)]
    subprocess.run(command,check=True)
    fitted=json.loads(model_path.read_bytes())
    grid=[]
    for nms in [.25,.35,.45]:
        for i in range(1,40):
            threshold=i/40;result=choose(valid,fitted,threshold,nms)
            grid.append(dict(threshold=threshold,nms_m=nms,**{k:v for k,v in result.items() if k!='rows'}))
    def selection_key(row):
        worst=min(min(bucket['human_precision'],bucket['human_recall']) for bucket in row['by_node_count'].values())
        return worst,min(row['human_precision'],row['human_recall']),row['human_precision']+row['human_recall'],-row['false_alarms'],row['threshold']
    selected=max(grid,key=selection_key)
    fitted['track_threshold']=selected['threshold'];fitted['track_nms_m']=selected['nms_m']
    data=json.dumps(fitted,separators=(',',':')).encode();model_path.write_bytes(data);model_hash=hashlib.sha256(data).hexdigest()
    (output/'SHA256SUMS').write_text(model_hash+'  model.json\n',encoding='ascii',newline='\n')
    oracle=sum(sum(any(math.dist(p['position'],v)<.25 for p in row['proposals']) for v in row['victim_positions']) for row in valid)
    features=FEATURES.copy()
    if model.get('distributed_swarm'):features[3]='fraction_of_known_released_observers'
    report=dict(method='Rust shallow Newton tree boosting for Bernoulli loss' if args.boost else 'Rust IRLS Bernoulli maximum likelihood with L2 penalty; no absolute position features',ridge=None if args.boost else args.ridge,boost=dict(depth=args.depth,rounds=args.rounds,learning_rate=.1,leaf_lambda=5,min_leaf=12,cut_budget_per_feature=31) if args.boost else None,features=features,
                train_maps=len(train),valid_maps=len(valid),train_proposals=sum(len(r['proposals']) for r in train),
                train_labels='Offline distance <0.25m to any TRAIN person source; duplicate proposals allowed before NMS',
                parent_model_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),proposal_model_sha256=hashlib.sha256(proposal_model_path.read_bytes()).hexdigest(),train_proposal_sha256=train_hash,model_sha256=model_hash,
                valid_controller_revision=valid_manifest['controller_revision'],test_accessed=False,valid_proposal_oracle_matched=oracle,
                valid_guidance_policy=valid_provenance.get('guidance_policy') if valid_provenance else 'See merged source provenance',
                selected=selected,valid=choose(valid,fitted,selected['threshold'],selected['nms_m']),train=choose(train,fitted,selected['threshold'],selected['nms_m']),selection_grid=grid,
                note='Python proposal scoring is a selection diagnostic. Acceptance requires independent actual Rust replay and final unopened TEST.')
    (output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['train','valid','selection_grid']},indent=2))

if __name__=='__main__':main()
