"""Train sensor calibration and geometry priors using TRAIN only; report VALID separately."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import deque

ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import numpy as np

def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:yield json.loads(line)

def fit(features,labels,ridge):
    x=np.column_stack([np.ones(len(features)),np.asarray(features,dtype=float)])
    y=np.asarray(labels,dtype=float)
    assert set(y)=={0.,1.},'Both classes are required'
    w=np.zeros(x.shape[1])
    # Penalized Bernoulli maximum likelihood, Newton/IRLS; no TEST access.
    penalty=np.eye(x.shape[1])*ridge;penalty[0,0]=0
    for iteration in range(100):
        p=1/(1+np.exp(-np.clip(x@w,-40,40)))
        grad=x.T@(p-y)/len(y)+penalty@w
        hessian=(x.T*(p*(1-p)))@x/len(y)+penalty+np.eye(x.shape[1])*1e-8
        delta=np.linalg.solve(hessian,grad);w-=delta
        if np.linalg.norm(delta)<1e-7:break
    return w

def metrics(rows,weights):
    if not rows:return None
    x=np.column_stack([np.ones(len(rows)),np.array([r['features'] for r in rows])]);y=np.array([r['label'] for r in rows]);p=1/(1+np.exp(-np.clip(x@weights,-40,40)));pred=p>=.5
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    return {'samples':len(rows),'accuracy':float(np.mean(pred==y)),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'log_loss':float(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12)))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ridge',type=float,default=.001);parser.add_argument('--name',default='rescue-v1');parser.add_argument('--fusion',action='store_true');parser.add_argument('--window',type=int,default=1);parser.add_argument('--sensor-mix',type=float,default=1);parser.add_argument('--raw-sensor-maps',action='store_true');parser.add_argument('--signed-prior',action='store_true');parser.add_argument('--geometry-scale',type=float,default=.04);parser.add_argument('--range-angle',action='store_true');args=parser.parse_args()
    raw=ROOT/'experiments/raw/rescue'
    train=list(lines(raw/'features-train.jsonl'));valid=list(lines(raw/'features-valid.jsonl')) if (raw/'features-valid.jsonl').exists() else []
    sensor_manifests=[]
    if args.raw_sensor_maps:
        def replace_occupancy(rows,split):
            result=[row for row in rows if row['kind']=='acoustic']
            manifest_path=raw/'sensor-maps'/('manifest-'+split+'.json');manifest=json.loads(manifest_path.read_text(encoding='utf-8'));assert manifest['model_applied'] is False
            entries={row['run_id']:row for row in manifest['rows']}
            for ground in lines(raw/('grids-'+split+'.jsonl')):
                path=raw/'sensor-maps'/(ground['run_id']+'.json');data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==entries[ground['run_id']]['sensor_map_sha256']
                occupied={tuple(key) for key in ground['occupied_keys']}
                for cell in json.loads(data)['cells']:
                    result.append(dict(kind='occupancy',map_id=ground['map_id'],split=split,features=[math_logit(cell['occupancy']),math_log1p(cell['observations'])],label=int(tuple(cell['key']) in occupied)))
            sensor_manifests.append(manifest_path.read_bytes());return result
        train=replace_occupancy(train,'train')
        if valid:valid=replace_occupancy(valid,'valid')
    train_maps=sorted({r['map_id'] for r in train});valid_maps=sorted({r['map_id'] for r in valid});assert not set(train_maps)&set(valid_maps)
    generations={mid.split('-')[1] for mid in train_maps+valid_maps};assert len(generations)==1,'TRAIN and VALID generation mismatch; finish both collections first'
    if generations=={'2.5'}:
        for split,maps in [('train',train_maps),('valid',valid_maps)]:
            manifest=json.loads((ROOT/'reports/rescue'/split/'manifest.json').read_text(encoding='utf-8-sig'))
            assert manifest['status']=='complete' and manifest['expected_maps']==len(maps),'Incomplete collection must not be fitted or selected'
    weights={};report={}
    for kind in ['acoustic','occupancy']:
        rows=[r for r in train if r['kind']==kind];w=fit([r['features'] for r in rows],[r['label'] for r in rows],args.ridge);weights[kind]=w.tolist()
        report[kind]={'train':metrics(rows,w),'valid':metrics([r for r in valid if r['kind']==kind],w),'weights':w.tolist()}
    families={}
    for row in lines(raw/'grids-train.jsonl'):
        assert row['map_id'] in train_maps and row['split']=='train'
        grid=np.zeros((32,21,70),dtype=np.uint8)
        for x,y,z in row['occupied_keys']:grid[x+16,y+3,z+6]=1
        families.setdefault(row['scene'],[]).append(grid)
    assert len(families)==15
    prototypes=[]
    for family,grids in sorted(families.items()):
        # Beta(.02,.02) posterior mean; variability is retained, never filled from TEST.
        prior=(np.sum(grids,axis=0)+.02)/(len(grids)+.04)
        prototypes.append({'family':family,'occupancy':np.round(prior.flatten(),6).tolist()})
    fusion_weights=[];range_weights=[];range_sigma=.7
    if args.fusion or args.window>1:
        first={};acoustic_by_map={}
        for row in train:
            if row['kind']=='acoustic':
                acoustic_by_map.setdefault(row['map_id'],[]).append(row)
                first.setdefault(row['map_id'],tuple(round(v,8) for v in row['features']))
        signatures={v:k for k,v in first.items()};assert len(signatures)==len(train_maps)
        paths={}
        for ground in lines(raw/'grids-train.jsonl'):
            path=raw/(ground['run_id']+'.jsonl')
            with path.open(encoding='utf-8') as stream:packet=json.loads(stream.readline())
            assert packet['run_id']==ground['run_id'] and packet.get('scans')
            scan=packet['scans'][0];signature=tuple(round(v,8) for v in [scan['acoustic_snr_db']/20,scan['voice_band_ratio'],scan['modulation'],scan['temperature_delta_c']/2])
            assert signatures[signature]==ground['map_id'];paths[ground['map_id']]=path
        assert set(paths)==set(train_maps),'Raw TRAIN audit association failed'
        fusion_x=[];fusion_y=[];temporal_x=[];temporal_y=[];range_x=[];range_y=[];sensor_sums={family:np.zeros((32,21,70)) for family in families};sensor_counts={family:np.zeros((32,21,70)) for family in families}
        for ground in lines(raw/'grids-train.jsonl'):
            path=paths[ground['map_id']];grids=families[ground['scene']]
            current=np.zeros((32,21,70),dtype=np.uint8)
            for x,y,z in ground['occupied_keys']:current[x+16,y+3,z+6]=1
            prior=(np.sum(grids,axis=0)-current+.02)/(len(grids)-1+.04)
            if args.raw_sensor_maps:final=json.loads((raw/'sensor-maps'/(ground['run_id']+'.json')).read_bytes())
            else:
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-8388608));final=json.loads(stream.read().splitlines()[-1])
            for cell in final['cells']:
                x,y,z=cell['key'];q=float(prior[x+16,y+3,z+6]);p=cell['occupancy']
                sensor_sums[ground['scene']][x+16,y+3,z+6]+=p;sensor_counts[ground['scene']][x+16,y+3,z+6]+=1
                fusion_x.append([math_logit(p),math_log1p(cell['observations']),math_logit(q) if args.signed_prior else max(0,math_logit(q))]);fusion_y.append(int(current[x+16,y+3,z+6]))
            if args.window>1:
                labels=iter(acoustic_by_map[ground['map_id']]);histories={};poses={}
                for packet in lines(path):
                    if 'scans' not in packet:continue
                    for scan in packet['scans']:
                        row=next(labels);features=[scan['acoustic_snr_db']/20,scan['voice_band_ratio'],scan['modulation'],scan['temperature_delta_c']/2]
                        position=poses.setdefault(scan['node_id'],np.array(scan['deployment_anchor']));position+=np.array(scan['odometry_delta'])
                        assert max(abs(a-b) for a,b in zip(features,row['features']))<1e-8
                        history=histories.setdefault(scan['node_id'],deque(maxlen=args.window));history.append((features,row['label']))
                        if len(history)>=4:
                            averaged=np.mean([h[0] for h in history],axis=0);temporal_x.append(averaged);temporal_y.append(int(np.mean([h[1] for h in history])>=.5))
                            if row['label']==1 and len(ground.get('victim_positions',[]))==1:
                                distance=np.linalg.norm(position-np.array(ground['victim_positions'][0]));range_x.append(features+[scan['bearing_sigma_deg']/90] if args.range_angle else averaged);range_y.append(float(np.log(distance+.04)))
            print('Prepared leave-one-map-out fusion',ground['map_id'],flush=True)
        if args.fusion:
            fusion_weights=fit(fusion_x,fusion_y,args.ridge).tolist();report['fusion']={'samples':len(fusion_y),'weights':fusion_weights,'prior':'leave-one-map-out TRAIN prior, never the same map or VALID'}
        if args.window>1:
            temporal_weights=fit(temporal_x,temporal_y,args.ridge);weights['acoustic']=temporal_weights.tolist();report['temporal_acoustic']={'window_frames':args.window,'samples':len(temporal_y),'weights':weights['acoustic']}
        for prototype in prototypes:
            family=prototype['family'];sensor=(sensor_sums[family]+1)/(sensor_counts[family]+2);prototype['sensor_occupancy']=np.round(sensor.flatten(),6).tolist()
        if range_x:
            x=np.column_stack([np.ones(len(range_x)),np.array(range_x)]);y=np.array(range_y);penalty=np.eye(x.shape[1])*args.ridge;penalty[0,0]=0;w=np.linalg.solve(x.T@x/len(y)+penalty,x.T@y/len(y));range_weights=w.tolist();range_sigma=max(.3,float(np.std(x@w-y)));report['acoustic_range']={'samples':len(y),'weights':range_weights,'log_sigma':range_sigma,'instantaneous_features_and_bearing_uncertainty':args.range_angle,'labels':'TRAIN single-source humans only, source position supervised offline; no source position at inference'}
    source_hash=hashlib.sha256((raw/'features-train.jsonl').read_bytes()+(raw/'grids-train.jsonl').read_bytes()+b''.join(sensor_manifests[:1])).hexdigest()
    model={'version':'rescue-trained-v1','cell_m':.05,'acoustic_weights':weights['acoustic'],'occupancy_weights':weights['occupancy'],'fusion_weights':fusion_weights,'acoustic_window_frames':args.window,'log_range_weights':range_weights,'log_range_sigma':range_sigma,'sensor_likelihood_mix':args.sensor_mix,'geometry_log_likelihood_scale':args.geometry_scale,'signed_geometry_prior':args.signed_prior,'prototypes':prototypes,'train_maps':train_maps,'valid_maps':valid_maps,'data_sha256':source_hash}
    output=ROOT/'models'/args.name
    output.mkdir(parents=True,exist_ok=False)
    data=json.dumps(model,separators=(',',':'),ensure_ascii=False).encode('utf-8');(output/'model.json').write_bytes(data)
    (output/'SHA256SUMS').write_text(hashlib.sha256(data).hexdigest()+'  model.json\n',encoding='ascii',newline='\n')
    report.update({'train_maps':len(train_maps),'valid_maps':len(valid_maps),'ridge':args.ridge,'raw_sensor_maps':args.raw_sensor_maps,'training_method':'IRLS Bernoulli classifiers + learned per-family Beta occupancy priors','target_note':'Scan classifier scores are NOT the scene-level IoU/precision/recall acceptance metrics.','test_accessed':False,'model_sha256':hashlib.sha256(data).hexdigest()})
    (output/'training-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

def math_logit(p):
    p=max(1e-9,min(1-1e-9,p));return float(np.log(p/(1-p)))
def math_log1p(n):return float(np.log1p(n))

if __name__=='__main__':main()
