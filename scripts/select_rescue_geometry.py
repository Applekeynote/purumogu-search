"""VALID-only selection of geometry likelihood mixture/scale; all fitted weights stay TRAIN-only."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/media-runtime').exists():sys.path.insert(0,str(ROOT/'tmp/media-runtime'))
import numpy as np


def main():
    source=ROOT/'models/rescue-v5/model.json';model=json.loads(source.read_bytes())
    raw=ROOT/'experiments/raw/rescue';ground=[json.loads(line) for line in (raw/'grids-valid.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    assert len(ground)==15 and not {row['map_id'] for row in ground}&set(model['train_maps'])
    priors=np.array([p['occupancy'] for p in model['prototypes']]);sensor=np.array([p['sensor_occupancy'] for p in model['prototypes']])
    prepared=[]
    for row in ground:
        record=json.loads((raw/'sensor-maps'/(row['run_id']+'.json')).read_bytes())
        keys=np.array([c['key'] for c in record['cells']],dtype=int);index=((keys[:,0]+16)*21+keys[:,1]+3)*70+keys[:,2]+6
        observed=np.array([c['occupancy'] for c in record['cells']]);count=np.array([c['observations'] for c in record['cells']])
        truth=np.zeros(47040,dtype=bool)
        for x,y,z in row['occupied_keys']:truth[((x+16)*21+y+3)*70+z+6]=True
        prepared.append((row,index,observed,count,truth))
    results=[]
    for mixture in [0.,.25,.5,.75,1.]:
        likelihood=np.clip((1-mixture)*priors+mixture*sensor,.05,.95)
        for scale in [.01,.04,.1,.25,1.]:
            rows=[]
            for ground,index,observed,count,truth in prepared:
                q=likelihood[:,index];certainty=np.abs(2*observed-1)
                scores=scale*np.sum(certainty*(observed*np.log(q)+(1-observed)*np.log(1-q)),axis=1)
                weights=np.exp(scores-np.max(scores));weights/=weights.sum();prior=weights@priors
                w=model['fusion_weights'];logodds=np.log(np.clip(observed,1e-12,1-1e-12)/(1-np.clip(observed,1e-12,1-1e-12)))
                p=np.clip(prior[index],.000001,.999999)
                logits=w[0]+w[1]*logodds+w[2]*np.log1p(count)+w[3]*np.maximum(0,np.log(p/(1-p)))
                predicted=prior>.65;predicted[index]=logits>np.log(.6/.4)
                intersection=int(np.sum(predicted&truth));union=int(np.sum(predicted|truth))
                rows.append(dict(scene=ground['scene'],iou=intersection/union,intersection=intersection,union=union,family=model['prototypes'][int(np.argmax(weights))]['family'],confidence=float(np.max(weights))))
            macro=float(np.mean([r['iou'] for r in rows]));micro=sum(r['intersection'] for r in rows)/sum(r['union'] for r in rows)
            results.append(dict(sensor_likelihood_mix=mixture,geometry_log_likelihood_scale=scale,map_iou_macro=macro,map_iou_micro=micro,rows=rows))
    results.sort(key=lambda r:min(r['map_iou_macro'],r['map_iou_micro']),reverse=True)
    report=dict(split='valid',source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),test_accessed=False,selection='maximum minimum of macro/micro global IoU; thresholds and domain fixed',candidates=results)
    (ROOT/'reports/rescue/geometry-selection-v6.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    best=results[0];model['sensor_likelihood_mix']=best['sensor_likelihood_mix'];model['geometry_log_likelihood_scale']=best['geometry_log_likelihood_scale']
    out=ROOT/'models/rescue-v6';out.mkdir(exist_ok=False)
    data=json.dumps(model,separators=(',',':'),ensure_ascii=False).encode('utf-8');(out/'model.json').write_bytes(data)
    digest=hashlib.sha256(data).hexdigest();(out/'SHA256SUMS').write_text(digest+'  model.json\n',encoding='ascii',newline='\n')
    (out/'selection-report.json').write_text(json.dumps(dict(parent_model='rescue-v5',model_sha256=digest,fit_split='train',selection_split='valid',test_accessed=False,selected=best),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(best,indent=2))


if __name__=='__main__':main()
