"""Fit local volumetric occupancy from TRAIN worlds; no coordinate templates or TEST reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '8')
sys.path.insert(0, str(ROOT/'tmp/learning-runtime'))
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from rescue_local_features import FEATURES, VERSION, evidence, features, iou, rotate_features, truth_mask
import rescue_local_features as local_features


def registered_extra_cases(roots):
    if not roots:return
    if isinstance(roots,(str,Path)):roots=[roots]
    seen=set()
    for index,source in enumerate(roots):
        root=ROOT/source;collection=json.loads((root/'collection.json').read_bytes())
        if collection['status']!='complete' or not collection['split'].startswith('train'):raise ValueError('Complete registered TRAIN collection required')
        paths=sorted(root.glob('*/complete.json'));expected={c['id']:c for c in collection['plan']['cases']}
        if len(paths)!=len(expected) or {json.loads(p.read_bytes())['id'] for p in paths}!=set(expected):raise ValueError('TRAIN collection does not match the registered plan')
        for path in paths:
            row=json.loads(path.read_bytes())
            if not row['split'].startswith('train') or row['id'] in seen:raise ValueError('Non-TRAIN or duplicate trajectory in fitting input')
            if row['id']!=path.parent.name or 'scene' in expected[row['id']] and row.get('case')!=expected[row['id']]:raise ValueError('TRAIN case provenance differs from plan')
            if row.get('case',{}).get('scene') in [17,19,22,23,27,28,29,34,35]:raise ValueError('Held geometry entered TRAIN')
            seen.add(row['id']);yield path,row,collection,index==len(roots)-1


def cases(extra_train_root=None):
    raw = ROOT/'experiments/raw/rescue/swarm-v31'
    for line in (raw/'grids-train.jsonl').read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        path = raw/'sensor-maps'/f"{row['run_id']}.json"
        if path.exists():
            from rescue_dataset import scene_index
            if scene_index(row['scene']) in [17,19,22,23,27,28,29,34,35]:raise ValueError('Held geometry entered legacy TRAIN')
            step=int(json.loads(path.read_bytes())['sim_step'])
            yield dict(id=row['map_id'], group=row['world_group_id'], map=path, truth=row,
                       source='legacy-3.1-train', scene=row['scene'],evaluation_step=step,legacy_prepared_alias_step=6000)
    for path in sorted((ROOT/'experiments/raw/rescue/generalization-v35/train').glob('*/complete.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        yield dict(id=row['id'], group=f"S{row['case']['scene']}-W{row['case']['seed']}",
                   map=path.parent/'map-6000.json', truth=json.loads((path.parent/'truth-6000.json').read_text(encoding='utf-8')),
                   source='physical-3.4-train', scene=row['scene'],evaluation_step=int(row['steps']))
    from rescue_dataset import world_group
    for path,row,collection,latest in registered_extra_cases(extra_train_root):
        step=int(collection['steps'])
        yield dict(id=row['id'],group=world_group(row['id']),map=path.parent/f'map-{step}.json',truth=json.loads((path.parent/f'truth-{step}.json').read_bytes()),source='physical-3.5-train',scene=row['scene'],latest_policy=latest,evaluation_step=step)


def cache_case(row, cache):
    # Cache is immutable by content; raw sensor fields, not the old model's probabilities.
    digest = hashlib.sha256(row['map'].read_bytes()).hexdigest()
    target = cache/(VERSION+'-'+digest+'.npz')
    if not target.exists():
        data = json.loads(row['map'].read_bytes())
        grid = evidence(data['cells']); y = truth_mask(row['truth']['occupied_keys'])
        x = features(grid)
        rng = np.random.default_rng(20261031)
        negatives = np.flatnonzero(y == 0)
        selected_negatives = rng.choice(negatives, min(len(negatives), 7000), replace=False)
        indices = np.r_[np.flatnonzero(y == 1), selected_negatives]
        weights = np.where(y[indices] == 1, 1, len(negatives)/len(selected_negatives)).astype(np.float32)
        np.savez_compressed(target, x=x[indices], y=y[indices], weights=weights)
    return target, digest


def export_tree(predictor):
    return dict(nodes=[dict(feature=None if n['is_leaf'] else int(n['feature_idx']),
                            # sklearn uses <=; Rust uses <. This makes equality behavior identical.
                            threshold=float(np.nextafter(n['num_threshold'], np.inf)),
                            left=int(n['left']), right=int(n['right']), value=float(n['value']) if n['is_leaf'] else 0.0)
                       for n in predictor.nodes])


def main():
    p = argparse.ArgumentParser();p.add_argument('--name', required=True);p.add_argument('--iterations',type=int,default=256)
    p.add_argument('--leaves',type=int,default=31);p.add_argument('--augment',action='store_true');p.add_argument('--legacy-limit',type=int,default=80)
    p.add_argument('--wide-context',action='store_true')
    args=p.parse_args();out=ROOT/'models'/args.name
    global FEATURES, VERSION
    if args.wide_context:
        local_features.configure('relative-rays-322-v2');FEATURES=local_features.FEATURES;VERSION=local_features.VERSION
    if out.exists():raise ValueError('Use a new immutable model name')
    out.mkdir(parents=True);cache=ROOT/'tmp/generic-v35-features';cache.mkdir(parents=True,exist_ok=True)
    rows=list(cases());legacy=[r for r in rows if r['source'].startswith('legacy')][:args.legacy_limit]
    rows=legacy+[r for r in rows if r['source'].startswith('physical')]
    if not rows:raise ValueError('No TRAIN worlds')
    # A group hash, independent of labels, reserves training-internal holdout worlds.
    held={r['group'] for r in rows if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%5==0}
    train=[r for r in rows if r['group'] not in held];valid=[r for r in rows if r['group'] in held]
    if not train or not valid:raise ValueError('Need both TRAIN and internal held-out worlds')
    xs=[];ys=[];ws=[];audit=[]
    for row in train:
        path,digest=cache_case(row,cache);data=np.load(path);x=data['x'];y=data['y'];w=data['weights']
        if args.augment:
            # Each sample receives a deterministic local yaw rotation; no extra world labels.
            turns=int(digest[:8],16)%4;x=rotate_features(x,turns)
        xs.append(x);ys.append(y);ws.append(w);audit.append(dict(id=row['id'],group=row['group'],sha256=digest))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);del xs,ys,ws
    print(json.dumps(dict(stage='fit',worlds=len(train),internal_valid=len(valid),samples=len(y),features=FEATURES)),flush=True)
    with threadpool_limits(limits=8):
        model=HistGradientBoostingClassifier(max_iter=args.iterations,max_leaf_nodes=args.leaves,learning_rate=.08,
                     min_samples_leaf=60,l2_regularization=5,early_stopping=False,random_state=20261031)
        model.fit(x,y,sample_weight=w)
        joblib.dump(model,out/'occupancy.joblib',compress=3)
        rows_out=[]
        for row in valid:
            data=json.loads(row['map'].read_bytes());gx=features(evidence(data['cells']));gy=truth_mask(row['truth']['occupied_keys'])
            metric=iou(model.predict_proba(gx)[:,1],gy);metric.update(id=row['id'],scene=row['scene'],source=row['source'])
            rows_out.append(metric)
    base=json.loads((ROOT/'models/swarm-v34-local/model.json').read_text(encoding='utf-8'))
    pretrained_train=base['train_maps'];pretrained_valid=base['valid_maps']
    base.update(geometry_feature_map=VERSION,geometry_trees=[export_tree(t[0]) for t in model._predictors],
                geometry_bias=float(model._baseline_prediction[0,0]),
                train_maps=sorted(set(pretrained_train)|{r['id'] for r in train}),
                valid_maps=sorted(set(pretrained_valid)|{r['id'] for r in valid}),calibration_train_maps=[],
                data_sha256=hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest())
    (out/'model.json').write_text(json.dumps(base,separators=(',',':'))+'\n',encoding='utf-8')
    report=dict(purpose='TRAIN-internal world-held-out model development, not final acceptance',test_opened=False,
                feature_map=VERSION,absolute_coordinates=False,scene_templates=False,model_selection_on_train_only=True,
                train_worlds=len({r['group'] for r in train}),internal_validation_worlds=len({r['group'] for r in valid}),
                train_cases=len(train),internal_validation_cases=len(valid),train_audit=audit,rows=rows_out,
                macro_iou=float(np.mean([r['iou'] for r in rows_out])),arguments=vars(args))
    (out/'development.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['train_worlds','internal_validation_worlds','macro_iou']}),flush=True)


if __name__=='__main__':main()
