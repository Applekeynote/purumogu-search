"""Select temporal evidence classifiers with world-held-out TRAIN only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS','4');sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from train_rescue_generic import export_tree
from train_rescue_tracks import choose
from evaluate_rescue_replay import match


def snapshot_weights(records,long_share=0):
    if any(r['domain']!='train' for r in records):raise ValueError('Evaluation snapshots cannot fit classifier weights')
    counts=Counter(r['group'] for r in records)
    long_groups={r['group'] for r in records if r['final_step']>6000};old_groups=set(counts)-long_groups
    if not 0<=long_share<1 or long_share and (not long_groups or not old_groups):raise ValueError('Both acquisition durations required for weighted training')
    return {group:((long_share/len(long_groups) if group in long_groups else (1-long_share)/len(old_groups)) if long_share else 1/len(counts))/n for group,n in counts.items()}


def fit_head(x,target,weights,leaves,rounds,soft=False,hard_weight=0):
    def create():
        return (HistGradientBoostingRegressor if soft else HistGradientBoostingClassifier)(max_iter=rounds,max_leaf_nodes=leaves,learning_rate=.08,min_samples_leaf=20,l2_regularization=.5,early_stopping=False,random_state=20261031)
    fit=create();fit.fit(x,target,sample_weight=weights)
    if hard_weight:
        # Only the current fitting fold supplies these residuals. Held worlds do
        # not train this first fit, its error weights, or the second fit.
        probability=fit.predict_proba(x)[:,1];focused=weights*(1+hard_weight*(target-probability)**2)
        focused*=len(target)/focused.sum();fit=create();fit.fit(x,target,sample_weight=focused)
    return fit


def fast_metric(rows,threshold,nms,height=1):
    tp=fp=people=0;domains={};strata={}
    for row,ranked in rows:
        cues=[]
        for proposal in ranked:
            if proposal['score']<threshold:break
            if all(math.sqrt(sum((a-b)**2*(height*height if i==1 else 1) for i,(a,b) in enumerate(zip(proposal['position'],q['position']))))>nms for q in cues):cues.append(proposal)
            if len(cues)==3:break
        count,_=match(cues,row['victim_positions']);tp+=count;fp+=len(cues)-count;people+=len(row['victim_positions'])
        domain=row.get('domain','internal');a,b,c=domains.get(domain,(0,0,0));domains[domain]=(a+count,b+len(cues)-count,c+len(row['victim_positions']))
        for group in [('domain',domain),('nodes',row['nodes']),('area',row['area'])]:
            a,b,c=strata.get(group,(0,0,0));strata[group]=(a+count,b+len(cues)-count,c+len(row['victim_positions']))
    criteria=[v for a,b,c in domains.values() for v in [a/max(1,a+b),a/max(1,c)]]
    stratum_criteria=[v for a,b,c in strata.values() for v in ([a/max(1,a+b),a/c] if c else [1. if b==0 else 0.])]
    return dict(human_precision=tp/max(1,tp+fp),human_recall=tp/max(1,people),minimum_domain_criterion=min(criteria),minimum_stratum_criterion=min(stratum_criteria))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--name',required=True)
    p.add_argument('--radii',default='.25');p.add_argument('--thermal',action='store_true')
    p.add_argument('--selection-valid',action='store_true');p.add_argument('--proposal-pool')
    p.add_argument('--thermal-local',action='store_true');p.add_argument('--spectral-local',action='store_true')
    p.add_argument('--worst-strata',action='store_true');p.add_argument('--data-root',default='experiments/raw/rescue/generalization-v35');p.add_argument('--extra-train-root',action='append')
    p.add_argument('--extra-acquisition-model',action='append');p.add_argument('--validation-root');p.add_argument('--validation-acquisition-model')
    p.add_argument('--extra-validation-root',action='append');p.add_argument('--extra-validation-acquisition-model',action='append')
    p.add_argument('--stale-augmentation',action='store_true')
    p.add_argument('--refit-all-train',action='store_true',help='Select with TRAIN world holdout, then refit the selected tree configuration on all registered TRAIN')
    p.add_argument('--leaves',help='Comma separated positive tree capacities')
    p.add_argument('--rounds',default='60,120,200')
    p.add_argument('--hard-example-weight',type=float,default=0)
    p.add_argument('--validation-step',type=int,choices=[6000,18000],default=6000)
    p.add_argument('--long-world-share',type=float,default=0)
    p.add_argument('--soft-localization',action='store_true')
    p.add_argument('--residual-logit',action='store_true',help='Keep the base forest and fit bounded Bernoulli Newton corrections on TRAIN')
    p.add_argument('--map-context',action='store_true',help='Research: append eight measured-occupancy context features; native support must be verified separately')
    p.add_argument('--teacher-feature',action='store_true',help='Append frozen TRAIN forest confidence to bounded corrective trees')
    p.add_argument('--person-balanced',action='store_true',help='Balance positive TRAIN proposal mass per person within each world snapshot')
    p.add_argument('--positive-mass',type=float,default=.5)
    p.add_argument('--unique-person-targets',action='store_true',help='TRAIN-only one-to-one source targets and normalized positive/negative budgets')
    p.add_argument('--neighbor-context',action='store_true',help='Research: relative context among inferred proposals; native parity required')
    a=p.parse_args()
    if a.neighbor_context:a.teacher_feature=True;a.spectral_local=True
    if a.map_context:a.spectral_local=True
    if a.spectral_local:a.thermal_local=True
    if a.thermal_local:a.thermal=True
    if a.teacher_feature and (not a.residual_logit or not a.spectral_local or a.map_context or a.stale_augmentation):raise ValueError('Frozen spectral32 teacher requires direct residual fitting without map-context or stale augmentation')
    if not math.isfinite(a.positive_mass) or not 0<a.positive_mass<1:raise ValueError('Bounded positive TRAIN mass required')
    if a.person_balanced and (a.stale_augmentation or a.refit_all_train or a.soft_localization):raise ValueError('Person weighting requires direct binary TRAIN fitting')
    if a.unique_person_targets and (a.person_balanced or a.stale_augmentation or a.refit_all_train or a.soft_localization or not a.residual_logit):raise ValueError('Unique source targets require direct residual binary fitting without another target weighting mode')
    if a.refit_all_train and a.selection_valid:raise ValueError('Refitting is for TRAIN-internal world selection only')
    if a.validation_step!=6000 and not a.selection_valid:raise ValueError('Long mission validation requires the explicit formal VALID role')
    if not math.isfinite(a.hard_example_weight) or not 0<=a.hard_example_weight<=100 or a.hard_example_weight and a.soft_localization:raise ValueError('Bounded classification error weighting required')
    leaf_sizes=[int(v) for v in a.leaves.split(',')] if a.leaves else ([7,15,31] if a.thermal and a.selection_valid else [7,15])
    round_counts=[int(v) for v in a.rounds.split(',')]
    leaf_limit=32 if a.thermal else 16
    if any(v<2 or v>leaf_limit for v in leaf_sizes) or any(v<1 or v>256 for v in round_counts):raise ValueError('Capacities exceed the native head budget')
    base=json.loads((ROOT/a.base/'model.json').read_text(encoding='utf-8'));out=ROOT/'models'/a.name
    if a.residual_logit:
        if a.soft_localization or a.hard_example_weight or a.refit_all_train:raise ValueError('Residual head requires direct binary TRAIN fitting')
        if not base.get('track_trees') or len(base['track_trees'])+max(round_counts)>256:raise ValueError('Frozen base plus corrections exceed native tree budget')
        if max(round_counts)>56:raise ValueError('Residual correction budget is 56 trees')
    base.update(track_refine_radius_m=0,track_refine_score_power=1)
    if a.proposal_pool:
        manifest=json.loads((ROOT/a.proposal_pool/'manifest.json').read_bytes())
        if manifest['status']!='complete' or manifest['closed_loop_acceptance']:raise ValueError('Expected complete developmental replay')
        source=json.loads((ROOT/manifest['model']).read_bytes())
        for key in ['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma','bearing_likelihood','acoustic_probe','acoustic_peaks_v2','joint_acoustic_fusion','plane_registration_v1','axis_plane_registration_v2','learned_navigation_v1','scout_inferred_routes_v1','footprint_safe_routes_v1']:
            if key in source:base[key]=source[key]
    if out.exists():raise ValueError('Use a new model name')
    records=[]
    sources=['train','valid-known','valid-unknown'] if a.selection_valid else ['train']
    paths=[(split,path) for split in sources for path in sorted((ROOT/(a.validation_root if split!='train' and a.validation_root else a.data_root)/split).glob('*/complete.json'))]
    raw_validation_ids=set();validation_lineage=None
    if a.validation_root:
        if not a.selection_valid or not a.validation_acquisition_model:raise ValueError('Explicit VALID role and acquisition model required')
        acquisition_path=ROOT/a.validation_acquisition_model;acquisition=json.loads(acquisition_path.read_bytes())
        from rescue_dataset import acoustic_feature_signature
        if acoustic_feature_signature(base)!=acoustic_feature_signature(acquisition):raise ValueError('Native VALID feature contract differs')
        for split in ['valid-known','valid-unknown']:
            collection=json.loads((ROOT/a.validation_root/split/'collection.json').read_bytes())
            if collection['status']!='complete' or collection['plan']['model_sha256']!=hashlib.sha256(acquisition_path.read_bytes()).hexdigest():raise ValueError('Incomplete or changed VALID acquisition')
            expected={c['id'] for c in collection['plan']['cases']}
            if {json.loads(p.read_bytes())['id'] for role,p in paths if role==split}!=expected:raise ValueError('VALID case set differs from plan')
            raw_validation_ids.update(expected)
        validation_lineage=dict(root=a.validation_root,model=a.validation_acquisition_model,model_sha256=hashlib.sha256(acquisition_path.read_bytes()).hexdigest())
    extra_validation_lineage=[]
    if a.extra_validation_root:
        if not a.selection_valid or not a.extra_validation_acquisition_model or len(a.extra_validation_root)!=len(a.extra_validation_acquisition_model):raise ValueError('One explicit acquisition model per additional formal VALID root required')
        from rescue_validation_data import validation_cases
        from rescue_dataset import acoustic_feature_signature
        for root,model_source in zip(a.extra_validation_root,a.extra_validation_acquisition_model):
            acquisition_path=ROOT/model_source;acquisition=json.loads(acquisition_path.read_bytes());digest=hashlib.sha256(acquisition_path.read_bytes()).hexdigest()
            if acoustic_feature_signature(base)!=acoustic_feature_signature(acquisition):raise ValueError('Additional VALID feature contract differs')
            for split in ['valid-known','valid-unknown']:
                collection=json.loads((ROOT/root/split/'collection.json').read_bytes())
                if collection['plan']['model_sha256']!=digest:raise ValueError('Additional VALID acquisition changed')
            for row in validation_cases([root],a.validation_step,{path.parent.name for _,path in paths}):
                paths.append((row['source'],row['map'].parent/'complete.json'));raw_validation_ids.add(row['id'])
            extra_validation_lineage.append(dict(root=root,model=model_source,model_sha256=digest))
    extra_ids=set();extra_lineage=[]
    if a.extra_train_root:
        if not a.extra_acquisition_model or len(a.extra_train_root)!=len(a.extra_acquisition_model):raise ValueError('One acquisition model per additional TRAIN root required')
        from train_rescue_generic import registered_extra_cases
        from rescue_dataset import acoustic_feature_signature
        for root,model_source in zip(a.extra_train_root,a.extra_acquisition_model):
            collection=json.loads((ROOT/root/'collection.json').read_bytes());acquisition_path=ROOT/model_source
            if hashlib.sha256(acquisition_path.read_bytes()).hexdigest()!=collection['plan']['model_sha256']:raise ValueError('Extra acquisition model changed')
            acquisition=json.loads(acquisition_path.read_bytes())
            if acoustic_feature_signature(base)!=acoustic_feature_signature(acquisition):raise ValueError('Additional raw observations have a different feature contract')
            for key in ['learned_navigation_v1','scout_inferred_routes_v1','footprint_safe_routes_v1']:base[key]=acquisition.get(key,False)
            extra_lineage.append(dict(root=root,acquisition_model=model_source,model_sha256=collection['plan']['model_sha256']))
        for path,case,_,_ in registered_extra_cases(a.extra_train_root):
            extra_ids.add(case['id']);paths.append(('train',path))
    for split,path in paths:
        case=json.loads(path.read_text(encoding='utf-8'));group=f"S{case['case']['scene']}-W{case['case']['seed']}"
        held=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%5==0
        if a.selection_valid:held=split!='train'
        steps=[a.validation_step] if split!='train' else [s for s in [1500,3000,4500,6000,9000,12000,15000,18000] if s<=case['steps']]
        local=None
        if a.thermal_local:
            from rescue_thermal_features import local_by_step
            pool=ROOT/a.proposal_pool/split/path.parent.name if a.proposal_pool and case['id'] not in extra_ids|raw_validation_ids else path.parent
            proposals_by_step={str(step):json.loads((pool/f'map-{step}.json').read_bytes())['proposals'] for step in steps}
            if any(len(p['features'])!=(32 if a.spectral_local else 24) for proposals in proposals_by_step.values() for p in proposals):
                if base.get('plane_registration_v1'):raise ValueError('Registered native proposal features required; raw-pose thermal cache cannot be substituted')
                local=local_by_step(path.parent,proposals_by_step,a.spectral_local)
        for step in steps:
            map_path=ROOT/a.proposal_pool/split/path.parent.name/f'map-{step}.json' if a.proposal_pool and case['id'] not in extra_ids|raw_validation_ids else path.parent/f'map-{step}.json'
            data=json.loads(map_path.read_bytes())
            if local is not None:
                data['proposals']=[dict(p,features=p['features'][:12]+extra) for p,extra in zip(data['proposals'],local[str(step)])]
            elif a.thermal and any(len(p['features'])!=(32 if a.spectral_local else 24 if a.thermal_local else 16) for p in data['proposals']):
                from rescue_thermal_features import extend
                data['proposals']=extend(path.parent,step,data['proposals'])
            if a.map_context:
                from rescue_map_context import extend as extend_map
                data['proposals']=extend_map(data)
            truth=json.loads((path.parent/f'truth-{step}.json').read_text(encoding='utf-8'))['victim_positions']
            records.append(dict(map_id=case['id'],group=group,held=held,step=step,nodes=case['nodes'],domain=split,area=case['case']['scene'],
                                proposals=data['proposals'],victim_positions=truth,final_step=case['steps']))
    if a.neighbor_context:
        from rescue_residual_head import teacher_features
        from rescue_neighbor_features import extend_teacher
        values=teacher_features(base,[q['features'] for row in records for q in row['proposals']]);start=0
        for row in records:
            count=len(row['proposals']);prefix=[dict(q,features=f.tolist()) for q,f in zip(row['proposals'],values[start:start+count])];start+=count
            row['proposals']=extend_teacher(prefix,row['step'])
    training=[r for r in records if not r['held']];validation=[r for r in records if r['held'] and r['step']==a.validation_step]
    if len(validation)<2:raise ValueError('Need internal held-out TRAIN worlds')
    x=[];distances=[];weights=[];world_weight=snapshot_weights(training,a.long_world_share)
    for row in training:
        for proposal in row['proposals']:
            x.append(proposal['features']);distances.append(min((math.dist(proposal['position'],v) for v in row['victim_positions']),default=99))
            weights.append(world_weight[row['group']]/max(1,len(row['proposals'])))
    if a.stale_augmentation:
        # A trapped simulated source stays at its position. Revisit old evidence
        # later without inventing another observation or increasing its support.
        rng=np.random.default_rng(20261031);original=len(x)
        for i in range(original):
            older=list(x[i]);older[6]+=float(rng.uniform(0,12000/9000))
            x.append(older);distances.append(distances[i]);weights.append(weights[i])
    selected=None;trials=[]
    unique_scores=None;unique_audit=[]
    if a.unique_person_targets:
        from rescue_residual_head import teacher_features
        unique_scores=1/(1+np.exp(-10*teacher_features(base,[q['features'][:32] for row in training for q in row['proposals']])[:,-1]))
    if a.teacher_feature and not a.neighbor_context:
        from rescue_residual_head import teacher_features
        x=teacher_features(base,x)
        validation=[dict(row,proposals=[dict(q,features=features.tolist()) for q,features in zip(row['proposals'],teacher_features(base,[q['features'] for q in row['proposals']]))]) if row['proposals'] else row for row in validation]
    with threadpool_limits(limits=4):
        for radius in [float(v) for v in a.radii.split(',')]:
          distance=np.array(distances);y=(distance<radius).astype(np.int32)
          if a.unique_person_targets:
            from rescue_unique_targets import targets as unique_targets
            labels=[];weights=[];start=0;unique_audit=[]
            for row in training:
                count=len(row['proposals']);labels_row,weights_row,audit_row=unique_targets(row,unique_scores[start:start+count],world_weight[row['group']],radius,a.positive_mass)
                start+=count;labels.extend(labels_row);weights.extend(weights_row);unique_audit.append(dict(map_id=row['map_id'],step=row['step'],**audit_row))
            y=np.array(labels,dtype=np.int32)
          if a.person_balanced:
            from rescue_person_weights import snapshot_weights as person_weights
            weights=[w for row in training for w in person_weights(row,world_weight[row['group']],radius,a.positive_mass)]
            if len(weights)!=len(y):raise ValueError('Person weights and TRAIN proposals differ')
          soft=np.clip(np.exp(-.5*(distance/.15)**2),.002,.95);target=np.log(soft/(1-soft)) if a.soft_localization else y
          for leaves in leaf_sizes:
            for rounds in round_counts:
                if a.residual_logit:
                    from rescue_residual_head import ResidualHead
                    fit=ResidualHead(base,rounds,leaves).fit(np.array(x),target,np.array(weights)*len(y)/sum(weights))
                else:fit=fit_head(np.array(x),target,np.array(weights)*len(y)/sum(weights),leaves,rounds,a.soft_localization,a.hard_example_weight)
                print(json.dumps(dict(stage='head-fit',leaves=leaves,rounds=rounds,train_proposals=len(y))),flush=True)
                model=dict(base);model.update(track_weights=[],track_trees=fit.trees if a.residual_logit else [export_tree(t[0]) for t in fit._predictors],
                                             track_feature_map='thermal-spectral-neighbor45-v6' if a.neighbor_context else 'thermal-spectral-teacher33-v5' if a.teacher_feature else 'thermal-spectral-map40-v4' if a.map_context else 'thermal-spectral32-v3' if a.spectral_local else 'thermal24-v2' if a.thermal_local else 'thermal16-v1' if a.thermal else '',
                                             track_bias=fit.bias if a.residual_logit else float(fit._baseline_prediction[0,0]))
                if a.teacher_feature:model.update(track_teacher_trees=len(base['track_trees']),track_teacher_bias=base['track_bias'])
                ranked=[]
                for row in validation:
                    if row['proposals']:
                        matrix=np.array([q['features'] for q in row['proposals']]);scores=1/(1+np.exp(-np.clip(fit.predict(matrix),-40,40))) if a.soft_localization else fit.predict_proba(matrix)[:,1]
                        ranked.append((row,sorted([dict(position=q['position'],score=float(s)) for q,s in zip(row['proposals'],scores)],key=lambda q:-q['score'])))
                    else:ranked.append((row,[]))
                for threshold in ([.001,.005,.01] if a.unique_person_targets else [])+[.03,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9]+([.95,.97,.99,.995] if a.person_balanced else []):
                    for nms in ([.1,.15,.2] if a.unique_person_targets else [])+[.25,.3,.35,.4,.45,.5,.6]:
                      for height in ([1,1.5,2,3] if a.selection_valid else [1]):
                        metric=fast_metric(ranked,threshold,nms,height)
                        precision=metric['human_precision'];recall=metric['human_recall']
                        # The worse criterion dominates; no favorable averaging hides missed people.
                        key=(metric['minimum_stratum_criterion'] if a.worst_strata else metric['minimum_domain_criterion'],min(precision,recall),2*precision*recall/max(1e-9,precision+recall),-rounds,-leaves)
                        trials.append(dict(radius=radius,leaves=leaves,rounds=rounds,threshold=threshold,nms=nms,height=height,
                                           precision=precision,recall=recall,minimum_stratum=metric['minimum_stratum_criterion'],minimum_domain=metric['minimum_domain_criterion']))
                        if selected is None or key>selected[0]:selected=(key,dict(model),threshold,nms,leaves,rounds,radius,int(y.sum()),height)
    _,model,threshold,nms,leaves,rounds,radius,positives,height=selected
    model['track_height_scale']=height
    metric=choose(validation,model,threshold,nms)
    selection_sha=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    if a.refit_all_train:
        # The held worlds are registered TRAIN. Retain their out-of-fold scores
        # above, then fit this fixed configuration for the later fresh VALID run.
        if a.soft_localization or a.stale_augmentation:raise ValueError('Refit currently supports the unchanged classification features')
        training=records
        x=[];distances=[];weights=[];world_weight=snapshot_weights(training,a.long_world_share)
        for row in training:
            for proposal in row['proposals']:
                x.append(proposal['features']);distances.append(min((math.dist(proposal['position'],v) for v in row['victim_positions']),default=99))
                weights.append(world_weight[row['group']]/max(1,len(row['proposals'])))
        y=(np.array(distances)<radius).astype(np.int32);positives=int(y.sum())
        with threadpool_limits(limits=4):
            fit=fit_head(np.array(x),y,np.array(weights)*len(y)/sum(weights),leaves,rounds,hard_weight=a.hard_example_weight)
        model.update(track_trees=[export_tree(t[0]) for t in fit._predictors],track_bias=float(fit._baseline_prediction[0,0]))
    model.update(track_threshold=threshold,track_nms_m=nms,
                 train_maps=sorted(set(base['train_maps'])|{r['map_id'] for r in training}),
                 valid_maps=sorted(set(base['valid_maps'])|{r['map_id'] for r in validation}))
    if a.selection_valid or a.refit_all_train:
        from train_rescue_generic import cases
        registered_train={r['id'] for r in cases(a.extra_train_root)}
        model['valid_maps']=sorted((set(model['valid_maps'])-registered_train)|({r['map_id'] for r in validation} if a.selection_valid else set()))
    if set(model['train_maps'])&set(model['valid_maps']):raise ValueError('Split leakage')
    out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    report=dict(purpose='Formal VALID selection; weights fitted on all TRAIN worlds, not final acceptance' if a.selection_valid else 'Temporal classifier selected on TRAIN-internal world holdout, not acceptance',test_opened=False,
                training_worlds=len({r['group'] for r in training}),validation_worlds=len({r['group'] for r in validation}),validation_cases=len(validation),samples=len(x),
                positive_samples=positives,chosen=dict(radius=radius,leaves=leaves,rounds=rounds,threshold=threshold,nms=nms,height=height),
                metrics=metric,trials=trials,features_exclude=['coordinates','scene ID','true positions'],proposal_pool=a.proposal_pool,
                worst_strata_selection=a.worst_strata,closed_loop_acceptance=False if a.proposal_pool else None,extra_train=extra_lineage,validation=validation_lineage,
                stale_augmentation=a.stale_augmentation,label_method='TRAIN distance Gaussian logit ranking; validation scores are not calibrated rescue probabilities' if a.soft_localization else 'TRAIN distance threshold classification')
    report.update(refit_all_registered_train=a.refit_all_train,selection_model_sha256=selection_sha,metrics_measured_before_refit=a.refit_all_train)
    report.update(arguments=vars(a),training_map_ids=sorted({r['map_id'] for r in training}),validation_map_ids=sorted({r['map_id'] for r in validation}))
    report['extra_validation']=extra_validation_lineage
    report['map_context']=a.map_context
    report['teacher_feature']=a.teacher_feature
    report['person_balanced']=a.person_balanced
    report['unique_person_targets']=a.unique_person_targets
    if a.unique_person_targets:report['unique_supervision']=dict(positive_mass=a.positive_mass,ignore_radius_m=.08,method='Maximum-cardinality one-to-one TRAIN assignment, then frozen teacher preference and normalized world budgets; no extra inference features',reference='https://arxiv.org/abs/2005.12872',reference_scope='Bipartite matching principle only; this is a tree residual classifier, not DETR',audit=unique_audit)
    report['neighbor_context']=a.neighbor_context
    if a.neighbor_context:
        report['features_exclude']=['absolute coordinates','scene ID','true source positions']
        report['native_support_requires_separate_validation']=True
    if a.person_balanced:report['source_weighting']=dict(positive_mass=a.positive_mass,method='Equal positive mass per represented TRAIN person within each snapshot; remaining mass to negatives; unchanged normalized world budgets',truth_used_only_for_training_labels_and_weights=True)
    if a.map_context:report['native_support_requires_separate_validation']=True
    if a.residual_logit:report.update(label_method='Binary TRAIN Bernoulli Newton residuals; frozen base forest plus <=56 correction trees',frozen_base_sha256=hashlib.sha256((ROOT/a.base/'model.json').read_bytes()).hexdigest(),frozen_base_trees=len(base['track_trees']),exported_trees=len(model['track_trees']))
    (out/'track-development.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['training_worlds','validation_worlds','samples','chosen']}))
    print(json.dumps({k:metric[k] for k in ['human_precision','human_recall']}))


if __name__=='__main__':main()
