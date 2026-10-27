"""Refit long-mission geometry evidence in Rust, preserving acoustic control parameters."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
from rescue_dataset import validate_model_splits,validate_paired_rows,world_group
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True);parser.add_argument('--report',required=True);parser.add_argument('--base',default='models/rescue-v11/model.json');parser.add_argument('--name',default='rescue-v12-geometry');parser.add_argument('--samples-per-map',type=int,default=3000);parser.add_argument('--trainer',default='rust/target/release/train_tracks.exe');parser.add_argument('--replace-splits',action='store_true');args=parser.parse_args()
    raw=ROOT/args.raw;report=ROOT/args.report;base=ROOT/args.base;model=json.loads(base.read_bytes());manifest=json.loads((report/'manifest.json').read_text(encoding='utf-8-sig'));assert manifest['status']=='complete' and manifest['split']=='train'
    ground=[json.loads(line) for line in (raw/'grids-train.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    if args.replace_splits:
        assert manifest['generation']=='rescue-3.0' and model['distributed_swarm']
        valid=[json.loads(line) for line in (raw/'grids-valid.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        valid_manifest=json.loads((report.parent/'valid/manifest.json').read_text(encoding='utf-8-sig'))
        assert valid_manifest['status']=='complete' and valid_manifest['expected_maps']==len(valid)
        validate_paired_rows(ground);validate_paired_rows(valid)
        train_provenance=json.loads((report/'rollout-provenance.json').read_text(encoding='utf-8-sig'))
        valid_provenance=json.loads((report.parent/'valid/rollout-provenance.json').read_text(encoding='utf-8-sig'))
        assert train_provenance['source_sha256']==valid_provenance['source_sha256'] and train_provenance['protocol_design_sha256']==valid_provenance['protocol_design_sha256']
        assert all(r['split']=='train' for r in ground) and all(r['split']=='valid' for r in valid)
        model['calibration_train_maps']=model.get('calibration_train_maps') or model['train_maps']
        model['train_maps']=[r['map_id'] for r in ground];model['valid_maps']=[r['map_id'] for r in valid]
    validate_model_splits(model)
    assert {r['map_id'] for r in ground}==set(model['train_maps']) and len(ground)==manifest['expected_maps']
    sensor_manifest=json.loads((raw/'sensor-maps/manifest-train.json').read_bytes());assert sensor_manifest['model_applied'] is False and sensor_manifest['maps']==len(ground)
    hashes={r['run_id']:r['sensor_map_sha256'] for r in sensor_manifest['rows']};families={};truths={};groups={}
    for row in ground:
        grid=np.zeros(47040,dtype=np.uint8)
        for x,y,z in row['occupied_keys']:grid[((x+16)*21+y+3)*70+z+6]=1
        truths[row['map_id']]=grid;families.setdefault(row['scene'],[]).append(grid);groups.setdefault(world_group(row['map_id']),[]).append(grid)
    group_sums={k:np.sum(v,axis=0) for k,v in groups.items()}
    sums={k:np.sum(v,axis=0) for k,v in families.items()};sensor_sums={k:np.zeros(47040) for k in families};sensor_counts={k:np.zeros(47040) for k in families}
    occupancy=[];fusion=[]
    for index,row in enumerate(ground):
        path=raw/'sensor-maps'/f"{row['run_id']}.json";data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==hashes[row['run_id']]
        cells=json.loads(data)['cells'];truth=truths[row['map_id']];family=row['scene'];group=world_group(row['map_id'])
        remaining=len(families[family])-len(groups[group]);assert remaining>0
        prior=(sums[family]-group_sums[group]+.02)/(remaining+.04)
        assert all(c['source']=='sensor' for c in cells)
        indices=np.array([((c['key'][0]+16)*21+c['key'][1]+3)*70+c['key'][2]+6 for c in cells],dtype=int)
        p=np.array([c['occupancy'] for c in cells]);counts=np.array([c['observations'] for c in cells]);sensor_sums[family][indices]+=p;sensor_counts[family][indices]+=1
        log=np.log(np.clip(p,1e-9,1-1e-9)/(1-np.clip(p,1e-9,1-1e-9)));q=np.clip(prior[indices],1e-6,1-1e-6);logprior=np.log(q/(1-q))
        chosen=np.sort(np.random.default_rng(73000+index).choice(len(cells),size=min(args.samples_per_map,len(cells)),replace=False))
        occ=[dict(features=[float(log[j]),float(np.log1p(counts[j]))],label=int(truth[indices[j]])) for j in chosen]
        fused=[dict(features=example['features']+[float(logprior[j])],label=example['label']) for example,j in zip(occ,chosen)]
        occupancy.append(dict(map_id=row['map_id'],split='train',examples=occ));fusion.append(dict(map_id=row['map_id'],split='train',examples=fused))
    model['prototypes']=[dict(family=family,occupancy=np.round((sums[family]+.02)/(len(families[family])+.04),6).tolist(),sensor_occupancy=np.round((sensor_sums[family]+1)/(sensor_counts[family]+2),6).tolist()) for family in sorted(families)]
    out=ROOT/'models'/args.name;out.mkdir(parents=True,exist_ok=False);model_path=out/'model.json';model_path.write_text(json.dumps(model,separators=(',',':')),encoding='utf-8')
    training=raw/'geometry-training';training.mkdir(exist_ok=True);model_hash=None
    for mode,rows in [('occupancy',occupancy),('fusion',fusion)]:
        source=training/(mode+'.json');source.write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8');digest=hashlib.sha256(base.read_bytes()+source.read_bytes()).hexdigest()
        subprocess.run([str(ROOT/args.trainer),str(source),str(model_path),str(model_path),'.001',digest,mode],check=True)
    data=model_path.read_bytes();model_hash=hashlib.sha256(data).hexdigest();(out/'SHA256SUMS').write_text(model_hash+'  model.json\n',encoding='ascii')
    fitted=json.loads(data);result=dict(method='Rust IRLS, L2=.001, deterministic uniform sampling within each TRAIN map',train_maps=len(ground),train_world_groups=len(groups),calibration_train_maps=len(model.get('calibration_train_maps',[])),samples_per_map=args.samples_per_map,occupancy_weights=fitted['occupancy_weights'],fusion_weights=fitted['fusion_weights'],fusion_prior='leave-one-world-group-out TRAIN ground truth; all paired node counts excluded together',parent_model_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),sensor_manifest_sha256=hashlib.sha256((raw/'sensor-maps/manifest-train.json').read_bytes()).hexdigest(),model_sha256=model_hash,acoustic_parameters_changed=False,control_policy_changed=False,test_accessed=False)
    (out/'training-report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
