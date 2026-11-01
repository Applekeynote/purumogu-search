"""Learn whether two acoustic hypotheses describe one source, using TRAIN only.

A fixed suppression radius can merge separate people on adjacent floors. The
pair model uses relative separation and observation features, never world IDs or
absolute coordinates. VALID selects its threshold; TEST is never read.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from train_rescue_tracks import score_model
from train_rescue_generic import export_tree
from train_rescue_generic_tracks import fast_metric
from evaluate_rescue_replay import match


def features(anchor,candidate):
    delta=np.array(anchor['position'])-candidate['position'];horizontal=math.hypot(delta[0],delta[2])
    a=np.asarray(anchor['features']);b=np.asarray(candidate['features'])
    return np.concatenate(([np.linalg.norm(delta)/.5,horizontal/.5,abs(delta[1])/.25],b,a,np.abs(b-a))).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--proposal-pool',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    model=json.loads((ROOT/a.base).read_bytes());pool=ROOT/a.proposal_pool;manifest=json.loads((pool/'manifest.json').read_bytes())
    if manifest['status']!='complete' or any(s.startswith('test') for s in manifest['splits']):raise ValueError('Complete development pool only')
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new model name')
    rng=np.random.default_rng(20261101);x=[];y=[];validation=[]
    for split in ['train','valid-known','valid-unknown']:
        for complete in sorted((ROOT/'experiments/raw/rescue/generalization-v35'/split).glob('*/complete.json')):
            case=json.loads(complete.read_bytes())
            for step in ([1500,3000,4500,6000] if split=='train' else [6000]):
                proposals=json.loads((pool/split/complete.parent.name/f'map-{step}.json').read_bytes())['proposals']
                truth=json.loads((complete.parent/f'truth-{step}.json').read_bytes())['victim_positions']
                ranked=sorted([dict(q,score=score_model(model,q['features'])) for q in proposals],key=lambda q:-q['score'])
                if split!='train':
                    validation.append(dict(map_id=case['id'],nodes=case['nodes'],area=case['case']['scene'],domain=split,victim_positions=truth,proposals=ranked[:48]));continue
                if not truth:continue
                nearest=[min(range(len(truth)),key=lambda i:math.dist(q['position'],truth[i])) for q in ranked]
                labels=[i if math.dist(q['position'],truth[i])<.25 else -1 for q,i in zip(ranked,nearest)]
                anchors=set(range(min(6,len(ranked))))
                for person in range(len(truth)):
                    anchors.update(i for i,l in enumerate(labels) if l==person and sum(v==person for v in labels[:i])<2)
                for anchor in sorted(anchors):
                    candidates=[i for i,q in enumerate(ranked) if i!=anchor and math.dist(q['position'],ranked[anchor]['position'])<1.]
                    if len(candidates)>40:candidates=rng.choice(candidates,40,replace=False).tolist()
                    for i in candidates:
                        duplicate=labels[anchor]>=0 and (labels[i]==labels[anchor] or labels[i]<0)
                        x.append(features(ranked[anchor],ranked[i]));y.append(int(duplicate))
    if len(x)<100:raise ValueError('Insufficient TRAIN pairs')
    print(json.dumps(dict(stage='fit',training_pairs=len(x),positive_pairs=sum(y),features=len(x[0]))),flush=True)
    best=None;trials=[]
    with threadpool_limits(limits=4):
        for leaves in [7,15,31]:
          for rounds in [60,120]:
            fit=HistGradientBoostingClassifier(max_iter=rounds,max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=2,learning_rate=.06,early_stopping=False,random_state=20261101)
            fit.fit(np.array(x),np.array(y));fields=[]
            for row in validation:
                ranked=row['proposals'];pairs=[(i,j) for i,p in enumerate(ranked) for j,q in enumerate(ranked) if i<j and math.dist(p['position'],q['position'])<1.]
                probability=fit.predict_proba(np.array([features(ranked[i],ranked[j]) for i,j in pairs]))[:,1] if pairs else []
                fields.append(dict(zip(pairs,probability)))
            for score_threshold in [.03,.05,.1,.15,.2,.3]:
              for duplicate_threshold in [.3,.4,.5,.6,.7,.8,.9]:
                rows=[];strata={};details=[]
                for row,field in zip(validation,fields):
                    selected=[]
                    for i,q in enumerate(row['proposals']):
                        if q['score']<score_threshold:break
                        if any(field.get((j,i),0)>=duplicate_threshold for j in selected):continue
                        selected.append(i)
                        if len(selected)==3:break
                    cues=[row['proposals'][i] for i in selected];tp,_=match(cues,row['victim_positions']);fp=len(cues)-tp;n=len(row['victim_positions'])
                    for key in [('domain',row['domain']),('area',row['area']),('nodes',row['nodes'])]:
                        v=strata.setdefault(key,[0,0,0]);v[0]+=tp;v[1]+=fp;v[2]+=n
                    details.append(dict(id=row['map_id'],matched=tp,false_alarms=fp,people=n,cues=[dict(position=q['position'],score=q['score']) for q in cues]))
                criteria=[v for tp,fp,n in strata.values() for v in ([tp/max(1,tp+fp),tp/n] if n else [1. if fp==0 else 0.])]
                tp=sum(r['matched'] for r in details);fp=sum(r['false_alarms'] for r in details);n=sum(r['people'] for r in details)
                record=dict(leaves=leaves,rounds=rounds,score_threshold=score_threshold,duplicate_threshold=duplicate_threshold,precision=tp/max(1,tp+fp),recall=tp/n,minimum_stratum=min(criteria))
                trials.append(record);key=(record['minimum_stratum'],min(record['precision'],record['recall']),record['precision']+record['recall'])
                if best is None or key>best[0]:best=(key,fit,record,details)
            print(json.dumps(dict(leaves=leaves,rounds=rounds,best=best[2])),flush=True)
    _,fit,chosen,details=best;out.mkdir()
    artifact=dict(feature_map='relative-source-pair99-v1',bias=float(fit._baseline_prediction[0,0]),trees=[export_tree(t[0]) for t in fit._predictors],**chosen)
    (out/'pairs.json').write_text(json.dumps(artifact,separators=(',',':'))+'\n')
    (out/'development.json').write_text(json.dumps(dict(purpose='TRAIN pair supervision; formal VALID selection, not closed-loop acceptance',test_opened=False,base=a.base,base_sha256=hashlib.sha256((ROOT/a.base).read_bytes()).hexdigest(),proposal_pool=a.proposal_pool,training_pairs=len(x),chosen=chosen,rows=details,trials=trials),indent=2)+'\n')


if __name__=='__main__':main()
