"""Learn volumetric completion with grouped TRAIN holdout, without scene/coordinate inputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from torch.nn import functional as F
from rescue_local_features import evidence, truth_mask
from rescue_volume_network import VolumeNet,LocalSurfaceNet,SubvoxelRefinementNet,rotate_evidence,reflect_evidence,extend_correction_inputs
from rescue_losses import occupancy_lovasz,area_sampling_mass,area_risk_weights
from train_rescue_generic import cases

def area_key(row):
    match=re.search(r'-S(\d+)-W',row['id'])
    if match:return int(match[1])
    match=re.search(r'-([EMHC]\d{2})-',row['id'])
    if not match:raise ValueError('Unregistered TRAIN area')
    from rescue_dataset import scene_index
    return scene_index(match[1])


def pretraining_lineage(checkpoint,seen=None):
    """Keep every fitted world in the lineage even when later batches differ."""
    seen=set() if seen is None else seen
    path=(ROOT/checkpoint).resolve()
    if not path.is_relative_to(ROOT) or path in seen:raise ValueError('Invalid or cyclic checkpoint lineage')
    seen.add(path);worlds=set();ancestors=[];parent=None
    development=path.parent/'development.json';initialization=path.parent/'initialization.json'
    if development.exists():
        record=json.loads(development.read_bytes())
        if record['test_opened']:raise ValueError('TEST-informed pretraining is prohibited')
        for row in record['train_audit']:
            if not row['held']:
                if row['source'].startswith(('valid','test')):raise ValueError('Validation world fitted in pretraining')
                worlds.add(row['id'])
        parent=record['arguments'].get('resume')
    elif initialization.exists():parent=json.loads(initialization.read_bytes())['source']
    else:raise ValueError('Pretraining audit missing: '+str(path))
    if parent:
        earlier,earlier_worlds=pretraining_lineage(parent,seen);ancestors.extend(earlier);worlds.update(earlier_worlds)
    ancestors.append(dict(checkpoint=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return ancestors,worlds


def load(row,cache,channels=2):
    raw=row['map'].read_bytes();data=json.loads(raw)
    if 'evaluation_step' in row and data['sim_step']!=row['evaluation_step']:raise ValueError('Prepared input and label checkpoint differ')
    signature=hashlib.sha256(raw)
    signature.update(json.dumps(row['truth']['occupied_keys'],separators=(',',':')).encode())
    native=row['map'].parent/'range-features.npz'
    if native.exists():signature.update(native.read_bytes())
    digest=signature.hexdigest();path=cache/((('range'+str(channels)+'-' if channels>2 else '')+digest)+'.npz')
    if not path.exists():
        x=evidence(data['cells']).transpose(3,0,1,2)
        if channels>2:
            from rescue_range_moments import input_grid
            x=input_grid(row,channels)
        y=truth_mask(row['truth']['occupied_keys']).reshape(1,32,21,70)
        x=np.pad(x,((0,0),(0,0),(0,3),(0,2)));y=np.pad(y,((0,0),(0,0),(0,3),(0,2)))
        temporary=path.with_name(path.stem+f'.{os.getpid()}.partial.npz')
        np.savez_compressed(temporary,x=x,y=y)
        try:
            if not path.exists():os.replace(temporary,path)
        except PermissionError:
            if not path.exists():raise
        finally:
            if temporary.exists():temporary.unlink()
    with np.load(path) as data:return data['x'],data['y'],digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=240)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--resume');p.add_argument('--width',type=int,default=16)
    p.add_argument('--residual',action='store_true');p.add_argument('--boundary-weight',type=float,default=1)
    p.add_argument('--lovasz-weight',type=float,default=0)
    p.add_argument('--warp',action='store_true');p.add_argument('--range-moments',action='store_true')
    p.add_argument('--local-patches',action='store_true');p.add_argument('--all-train',action='store_true')
    p.add_argument('--local-surface',action='store_true')
    p.add_argument('--subvoxel-refinement',action='store_true');p.add_argument('--fine-width',type=int,default=12)
    p.add_argument('--unfreeze-backbone',action='store_true')
    p.add_argument('--fine-plane-profiles',action='store_true')
    p.add_argument('--fine-unet',action='store_true')
    p.add_argument('--fine-occupancy-phases',action='store_true')
    p.add_argument('--fine-truth-pool');p.add_argument('--fine-loss-weight',type=float,default=.5)
    p.add_argument('--backbone-channels',type=int,choices=[74,90],default=74)
    p.add_argument('--cutmix',action='store_true');p.add_argument('--range-covariance',action='store_true')
    p.add_argument('--range-fine',action='store_true')
    p.add_argument('--range-free',action='store_true')
    p.add_argument('--range-tsdf',action='store_true')
    p.add_argument('--geometry-pool')
    p.add_argument('--extra-geometry-pool');p.add_argument('--temporal-training',action='store_true')
    p.add_argument('--stream-temporal',action='store_true')
    p.add_argument('--subcell-pool')
    p.add_argument('--subcell-share',type=float,default=.35)
    p.add_argument('--validation-geometry-pool');p.add_argument('--validation-step',type=int,choices=[6000,18000],default=6000)
    p.add_argument('--onpolicy-step-min',type=int)
    p.add_argument('--onpolicy-share',type=float,default=0)
    p.add_argument('--extra-train-root',action='append');p.add_argument('--validation-root',default='experiments/raw/rescue/generalization-v35')
    p.add_argument('--extra-validation-root',action='append')
    p.add_argument('--new-train-only',action='store_true');p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--worst-area',action='store_true');p.add_argument('--hard-reweight',action='store_true');p.add_argument('--weight-decay',type=float,default=.0001)
    p.add_argument('--axis-profiles',action='store_true');p.add_argument('--normalization',choices=['group','batch','channel'],default='group');a=p.parse_args()
    if a.range_free and a.range_tsdf:raise ValueError('Choose free-space evidence or signed-distance evidence')
    if not math.isfinite(a.lovasz_weight) or not 0<=a.lovasz_weight<=2:raise ValueError('Bounded IoU surrogate weight required')
    if a.range_free or a.range_tsdf:a.range_fine=True
    if a.range_tsdf and not a.geometry_pool:raise ValueError('Native signed-distance preparation required')
    if a.local_surface:
        if a.axis_profiles:raise ValueError('Local surface inference cannot use global plane profiles')
        a.normalization='channel'
    if a.range_covariance and a.range_fine:raise ValueError('Choose covariance or fine octants')
    if a.range_covariance or a.range_fine:a.range_moments=True
    channels=90 if a.range_free or a.range_tsdf else 74 if a.range_fine else 18 if a.range_covariance else 9 if a.range_moments else 2
    if a.subvoxel_refinement and (a.local_surface or not a.resume or a.local_patches or channels not in (74,90) or a.backbone_channels>channels):raise ValueError('Full-volume pretrained fine-octant backbone required')
    if a.unfreeze_backbone and not a.subvoxel_refinement:raise ValueError('Backbone adaptation requires the subvoxel architecture')
    if a.subvoxel_refinement and not 4<=a.fine_width<=32:raise ValueError('Bounded fine-surface width required')
    if a.fine_plane_profiles and not a.subvoxel_refinement:raise ValueError('Fine-plane profiles require the subvoxel correction architecture')
    if a.fine_unet and (not a.fine_plane_profiles or a.fine_width%4):raise ValueError('Fine U-Net requires height context and width divisible by four')
    if a.fine_occupancy_phases and not (a.fine_unet and a.fine_truth_pool):raise ValueError('Fine occupancy phases require the supervised fine U-Net')
    if a.fine_truth_pool and (not a.subvoxel_refinement or a.local_patches or a.warp or a.cutmix):raise ValueError('Static fine labels require the full-volume subvoxel model and matching coordinate views')
    if not math.isfinite(a.subcell_share) or not 0<a.subcell_share<=1:raise ValueError('Bounded coordinate-view sampling share required')
    if not math.isfinite(a.fine_loss_weight) or not 0<=a.fine_loss_weight<=2:raise ValueError('Bounded auxiliary loss weight required')
    if a.warp and a.range_moments:raise ValueError('Sub-voxel moments require exact rotations/reflections; affine warps are not enabled')
    pose_configuration=dict(plane_registration_v1=False,axis_plane_registration_v2=False,range_tsdf_v1=False)
    if a.geometry_pool:
        preparation=json.loads((ROOT/a.geometry_pool/'manifest.json').read_bytes())
        if preparation['status']!='complete' or preparation['test_opened']:raise ValueError('Complete development-only geometry preparation required')
        profile=ROOT/preparation['lineage']['model']
        if hashlib.sha256(profile.read_bytes()).hexdigest()!=preparation['lineage']['model_sha256']:raise ValueError('Preparation profile changed')
        parameters=json.loads(profile.read_bytes())
        if parameters.get('range_tsdf_v1',False)!=a.range_tsdf or parameters['volume_channels']!=channels:raise ValueError('Preparation channels or feature semantics differ')
        pose_configuration={key:parameters.get(key,False) for key in pose_configuration}
    if a.extra_geometry_pool:
        if not a.extra_train_root or not a.geometry_pool:raise ValueError('Additional geometry requires registered additional TRAIN')
        extra_preparation=json.loads((ROOT/a.extra_geometry_pool/'manifest.json').read_bytes())
        if extra_preparation['status']!='complete' or extra_preparation['test_opened']:raise ValueError('Complete TRAIN geometry required')
        extra_profile=ROOT/extra_preparation['lineage']['model']
        if hashlib.sha256(extra_profile.read_bytes()).hexdigest()!=extra_preparation['lineage']['model_sha256']:raise ValueError('Additional preparation profile changed')
        extra_parameters=json.loads(extra_profile.read_bytes())
        if any(extra_parameters.get(key,False)!=value for key,value in pose_configuration.items()):raise ValueError('Cannot mix pose-registration contracts')
    if a.validation_geometry_pool:
        validation_preparation=json.loads((ROOT/a.validation_geometry_pool/'manifest.json').read_bytes())
        if validation_preparation['status']!='complete' or validation_preparation['test_opened']:raise ValueError('Complete development-only validation preparation required')
        lineage=validation_preparation['lineage'];profile=ROOT/lineage['model']
        if hashlib.sha256(profile.read_bytes()).hexdigest()!=lineage['model_sha256']:raise ValueError('Validation preparation changed')
        parameters=json.loads(profile.read_bytes())
        if any(parameters.get(k,False)!=v for k,v in pose_configuration.items()):raise ValueError('Validation pose registration differs')
    if a.temporal_training and not a.geometry_pool:raise ValueError('Native intermediate geometry required')
    if a.stream_temporal and not (a.temporal_training or a.subcell_pool):raise ValueError('Temporal or subcell alternatives required for streaming')
    if not 0<=a.onpolicy_share<1 or a.onpolicy_share and not a.extra_train_root:raise ValueError('Additional TRAIN and a share below one required')
    if a.onpolicy_step_min is not None and (a.onpolicy_step_min<=0 or a.onpolicy_step_min%25 or not a.onpolicy_share):raise ValueError('A causal frame threshold and on-policy sampling share required')
    ancestry,pretrained_worlds=pretraining_lineage(a.resume) if a.resume else ([],set())
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use a new model name')
    out.mkdir(parents=True);cache=ROOT/'tmp/generic-volume-input';cache.mkdir(parents=True,exist_ok=True)
    random.seed(20261031);np.random.seed(20261031);torch.manual_seed(20261031);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('GPU training requested; CUDA unavailable')
    device=torch.device('cuda');torch.backends.cudnn.benchmark=True
    rows=list(cases(a.extra_train_root));train=[];valid=[];audit=[];temporal={};subcell={};subcell_ids=set()
    fine_targets={};fine_audit=[];fine_ids=set()
    if a.fine_truth_pool:
        manifest=json.loads((ROOT/a.fine_truth_pool/'manifest.json').read_bytes())
        if manifest['status']!='complete' or manifest['split']!='train' or manifest['test_opened'] or not manifest['all_coarse_geometry_exact'] or manifest['fine_cell_m']!=.025 or manifest['acceptance_cell_m']!=.05:raise ValueError('Matching TRAIN-only auxiliary labels required')
        fine_ids=set(manifest['case_ids'])
        if not fine_ids<={row['id'] for row in rows}:raise ValueError('Auxiliary labels outside registered TRAIN')
    def alternate(x,y,digest):
        if a.stream_temporal:
            from rescue_tensor_storage import disk_pair
            return disk_pair(cache/'temporal-npy',digest,x,y)
        return torch.tensor(x),torch.tensor(y,dtype=torch.float32)
    if a.extra_geometry_pool and not set(extra_preparation['case_ids'])<={r['id'] for r in rows}:raise ValueError('Additional geometry contains non-TRAIN cases')
    if a.subcell_pool:
        subcell_manifest=json.loads((ROOT/a.subcell_pool/'manifest.json').read_bytes())
        if subcell_manifest['status']!='complete' or subcell_manifest['split']!='train' or subcell_manifest['test_opened']:raise ValueError('Complete TRAIN-only subcell preparation required')
        if a.fine_truth_pool and not subcell_manifest.get('fine_supervision',False):raise ValueError('Coordinate augmentation needs matching fine labels for auxiliary supervision')
        subcell_ids=set(subcell_manifest['case_ids'])
        if not subcell_ids<={r['id'] for r in rows}:raise ValueError('Subcell views outside registered TRAIN')
        lineage=subcell_manifest['lineage'];profile=ROOT/lineage['model']
        if hashlib.sha256(profile.read_bytes()).hexdigest()!=lineage['model_sha256']:raise ValueError('Subcell preparation profile changed')
        profile=json.loads(profile.read_bytes())
        if any(profile.get(k,False)!=v for k,v in pose_configuration.items()):raise ValueError('Subcell pose registration differs')
    if a.new_train_only:rows=[row for row in rows if row['source'].startswith('physical-')]
    if a.all_train:
        from rescue_validation_data import validation_cases
        rows.extend(validation_cases([a.validation_root]+(a.extra_validation_root or []),a.validation_step,{row['id'] for row in rows}))
    elif a.extra_validation_root:raise ValueError('Additional formal VALID requires --all-train')
    for row in rows:
        original_map=row['map']
        if row['id'] in fine_ids:
            target=ROOT/a.fine_truth_pool/row['id'];record=json.loads((target/'complete.json').read_bytes())
            source=original_map.parent/f"truth-{row['evaluation_step']}.json"
            if not record['coarse_geometry_exact'] or hashlib.sha256(source.read_bytes()).hexdigest()!=record['source_truth_sha256'] or hashlib.sha256((target/'fine-truth.npz').read_bytes()).hexdigest()!=record['labels_sha256']:raise ValueError('Auxiliary label provenance changed')
            with np.load(target/'fine-truth.npz') as data:labels=data['y']
            if labels.shape!=(1,64,48,144) or labels.dtype!=np.uint8 or not np.all(labels<=1):raise ValueError('Invalid fine-label tensor')
            fine_targets[row['id']]=torch.tensor(labels)
            fine_audit.append(dict(id=row['id'],labels_sha256=record['labels_sha256'],source_truth_sha256=record['source_truth_sha256']))
        if a.geometry_pool:
            split=row['source'] if row['source'].startswith('valid-') else 'train'
            pool=a.validation_geometry_pool if row['source'].startswith('valid-') and a.validation_geometry_pool else a.extra_geometry_pool if a.extra_geometry_pool and row['id'] in extra_preparation['case_ids'] else a.geometry_pool
            selected_manifest=validation_preparation if row['source'].startswith('valid-') and a.validation_geometry_pool else extra_preparation if a.extra_geometry_pool and row['id'] in extra_preparation['case_ids'] else preparation
            storage_step=row.get('evaluation_step',6000)
            if row.get('legacy_prepared_alias_step') and not selected_manifest['lineage'].get('actual_final_step_filenames',False):storage_step=row['legacy_prepared_alias_step']
            row=dict(row,map=ROOT/pool/split/row['id']/f'map-{storage_step}.json')
        x,y,digest=load(row,cache,channels);item=(row,torch.tensor(x,device=device),torch.tensor(y,dtype=torch.float32,device=device))
        held=int(hashlib.sha256(row['group'].encode()).hexdigest()[:8],16)%5==0
        if a.all_train:held=row['source'].startswith('valid-')
        (valid if held else train).append(item);audit.append(dict(id=row['id'],group=row['group'],held=held,sha256=digest,source=row['source'],evaluation_step=row.get('evaluation_step',6000)))
        if a.temporal_training and not held:
            case_preparation=json.loads((row['map'].parent/'complete.json').read_bytes());intermediate=case_preparation.get('intermediate',[])
            if not intermediate:raise ValueError('Temporal TRAIN preparation missing: '+row['id'])
            alternatives=[]
            for snapshot in intermediate:
                step=snapshot['step'];path=row['map'].parent/f'step-{step}'/f'map-{step}.json'
                labels=original_map.parent/f'truth-{step}.json'
                if labels.exists():snapshot_truth=json.loads(labels.read_bytes());label_source=str(labels.relative_to(ROOT))
                elif 10<=area_key(row)<15:
                    # Moving debris after the aftershock needs its own time label.
                    # Keep this world's final example; do not substitute a future mask.
                    continue
                else:snapshot_truth=row['truth'];label_source='static-environment final mask'
                if hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot['map_sha256']:raise ValueError('Intermediate map changed')
                if hashlib.sha256((path.parent/'range-features.npz').read_bytes()).hexdigest()!=snapshot['features_sha256']:raise ValueError('Intermediate features changed')
                sx,sy,sdigest=load(dict(row,map=path,truth=snapshot_truth,evaluation_step=step),cache,channels)
                # One random time per world update. Larger corpora can stream losslessly.
                alternatives.append(alternate(sx,sy,sdigest))
                audit[-1].setdefault('temporal',[]).append(dict(step=step,sha256=sdigest,label_source=label_source))
            temporal[row['id']]=alternatives
        if row['id'] in subcell_ids:
            if held:raise ValueError('Subcell TRAIN label used for evaluation')
            directory=ROOT/a.subcell_pool/row['id'];record=json.loads((directory/'complete.json').read_bytes())
            if not record['zero_translation_exact'] or record['lineage']!=subcell_manifest['lineage']:raise ValueError('Invalid subcell provenance')
            alternatives=[]
            from rescue_subcell_training import views as subcell_views
            for view in subcell_views(directory,record,bool(a.fine_truth_pool)):
                sx,sy,sdigest=load(dict(row,map=view['map'],truth=view['truth'],evaluation_step=view['evaluation_step']),cache,channels)
                sfine=torch.tensor(view['fine']) if view['fine'] is not None else None
                alternatives.append((*alternate(sx,sy,sdigest),sfine))
                audit[-1].setdefault('subcell_labels',[]).append(view['audit'])
                audit[-1].setdefault('subcell',[]).append(dict(view['audit'],sha256=sdigest))
            subcell[row['id']]=alternatives
        if len(audit)%20==0 or len(audit)==len(rows):print(json.dumps(dict(stage='inputs',loaded=len(audit),total=len(rows))),flush=True)
    architecture='subvoxel-phase-refinement-v4' if a.fine_occupancy_phases else 'subvoxel-unet-refinement-v3' if a.fine_unet else 'subvoxel-plane-refinement-v2' if a.fine_plane_profiles else 'subvoxel-refinement-v1' if a.subvoxel_refinement else 'local-surface-v1' if a.local_surface else 'volume-unet-v1'
    model=(SubvoxelRefinementNet(a.width,a.residual,channels,a.axis_profiles,a.normalization,a.backbone_channels,a.fine_width,not a.unfreeze_backbone,a.fine_plane_profiles,bool(a.fine_truth_pool),a.fine_unet,a.fine_occupancy_phases) if a.subvoxel_refinement else LocalSurfaceNet(a.width,a.residual,channels) if a.local_surface else VolumeNet(a.width,a.residual,channels,a.axis_profiles,a.normalization)).to(device)
    if a.resume:
        prior=torch.load(ROOT/a.resume,map_location=device,weights_only=True);previous=prior['state']
        initializing_fine=a.subvoxel_refinement and prior.get('architecture','volume-unet-v1')=='volume-unet-v1'
        initializing_profiles=a.fine_plane_profiles and not a.fine_unet and prior.get('architecture')=='subvoxel-refinement-v1'
        initializing_aux=bool(a.fine_truth_pool) and not prior.get('fine_supervision',False)
        initializing_phases=a.fine_occupancy_phases and prior.get('architecture')=='subvoxel-unet-refinement-v3'
        if prior.get('architecture','volume-unet-v1')!=architecture and not (initializing_fine or initializing_profiles or initializing_phases):raise ValueError('Use fresh training for a different architecture')
        if a.subvoxel_refinement and not initializing_fine:
            if prior['channels']!=channels or prior.get('backbone_channels',74)!=a.backbone_channels or prior.get('range_tsdf_v1',False)!=a.range_tsdf:raise ValueError('Frozen correction input contract differs')
            if any(prior.get(key,False)!=pose_configuration[key] for key in ['plane_registration_v1','axis_plane_registration_v2']):raise ValueError('Frozen correction registration differs')
        if initializing_fine:
            if prior['channels']!=a.backbone_channels or prior.get('profiles',False)!=a.axis_profiles:raise ValueError('Backbone input contract differs')
            for key in ['plane_registration_v1','axis_plane_registration_v2']:
                if prior.get(key,False)!=pose_configuration[key]:raise ValueError('Backbone registration differs')
            if a.backbone_channels==90 and prior.get('range_tsdf_v1',False)!=a.range_tsdf:raise ValueError('Backbone distance semantics differ')
            model.backbone.load_state_dict(previous)
        elif initializing_profiles:
            old=previous['correction.0.weight'];new=torch.zeros_like(model.state_dict()['correction.0.weight'])
            if new.shape[1]!=old.shape[1]+16:raise ValueError('Fine-plane extension contract differs')
            new[:,:old.shape[1]]=old;previous['correction.0.weight']=new
            result=model.load_state_dict(previous,strict=False)
            if result.unexpected_keys or any(not key.startswith('fine_plane_profile.') and not (initializing_aux and key.startswith('fine_out.')) for key in result.missing_keys):raise ValueError('Unexpected fine-plane initialization mismatch')
        elif initializing_phases:
            if not prior.get('fine_supervision',False):raise ValueError('A trained fine occupancy head is required for phase initialization')
            previous=extend_correction_inputs(previous,model,8)
        elif initializing_aux:
            result=model.load_state_dict(previous,strict=False)
            if result.unexpected_keys or set(result.missing_keys)!={'fine_out.weight','fine_out.bias'}:raise ValueError('Unexpected auxiliary-head initialization mismatch')
        elif not a.local_surface and not a.subvoxel_refinement and previous['e0.0.weight'].shape[1]!=model.state_dict()['e0.0.weight'].shape[1]:
            old=previous['e0.0.weight'];new=torch.zeros_like(model.state_dict()['e0.0.weight'])
            if prior.get('profiles') and a.axis_profiles:
                raw=prior['channels'];new[:,:raw]=old[:,:raw];new[:,channels:channels+24]=old[:,raw:raw+24]
            else:new[:,:old.shape[1]]=old
            previous['e0.0.weight']=new
        if not (initializing_fine or initializing_profiles or initializing_aux):model.load_state_dict(previous,strict=a.local_surface or a.subvoxel_refinement or not a.axis_profiles and a.normalization=='group')
    groups=sorted({area_key(row) for row,_,_ in train});group_loss={g:.15 for g in groups}
    sampling=np.full(len(train),1/len(train))
    if a.onpolicy_share:
        new=np.array([row['source']=='physical-3.5-train' and row.get('evaluation_step',6000)>=a.onpolicy_step_min if a.onpolicy_step_min else row.get('latest_policy',False) for row,_,_ in train]);old=~new
        if not new.any() or not old.any():raise ValueError('Both original and on-policy TRAIN worlds required')
        sampling=np.where(new,a.onpolicy_share/new.sum(),(1-a.onpolicy_share)/old.sum())
    group_mass=area_sampling_mass([area_key(row) for row,_,_ in train],sampling)
    optim=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optim,a.epochs,eta_min=.00003)
    scaler=torch.amp.GradScaler('cuda');best=(-1.,-1.);history=[];started=time.monotonic();last_best=0
    print(json.dumps(dict(stage='fit',gpu=torch.cuda.get_device_name(),train_cases=len(train),valid_cases=len(valid),parameters=sum(v.numel() for v in model.parameters()))),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();indices=np.random.permutation(len(train));losses=[]
        if a.onpolicy_share:
            indices=np.random.choice(len(train),len(train),replace=True,p=sampling)
        for offset in range(0,len(indices),a.batch):
            chosen=[train[i] for i in indices[offset:offset+a.batch]]
            fine_y=None;fine_available=None
            if a.fine_truth_pool:
                empty=torch.zeros((1,64,48,144),device=device)
                fine_y=torch.stack([fine_targets[row['id']].to(device,dtype=torch.float32) if row['id'] in fine_targets else empty for row,_,_ in chosen])
                fine_available=torch.tensor([row['id'] in fine_targets for row,_,_ in chosen],device=device,dtype=torch.float32)
            if a.temporal_training or a.subcell_pool:
                from rescue_subcell_training import choose_training_views
                chosen=choose_training_views(chosen,subcell,temporal,a.subcell_share,device,fine_y,fine_available)
            if a.local_patches:
                inputs=[];targets=[]
                for row,gx,gy in chosen:
                    start_y=random.randrange(9);start_z=random.randrange(41)
                    inputs.append(gx[:, :,start_y:start_y+16,start_z:start_z+32])
                    targets.append(gy[:, :,start_y:start_y+16,start_z:start_z+32])
                x=torch.stack(inputs);y=torch.stack(targets)
            else:x=torch.stack([r[1] for r in chosen]);y=torch.stack([r[2] for r in chosen])
            # Rotate around gravity only. Random translations prevent the network using fixed floor coordinates.
            turns=random.randrange(4);x=rotate_evidence(x,turns);y=torch.rot90(y,turns,(2,4))
            if fine_y is not None:fine_y=torch.rot90(fine_y,turns,(2,4))
            if random.random()<.5:
                x=reflect_evidence(x,2);y=y.flip(2)
                if fine_y is not None:fine_y=fine_y.flip(2)
            if random.random()<.5:
                x=reflect_evidence(x,4);y=y.flip(4)
                if fine_y is not None:fine_y=fine_y.flip(4)
            shifts=[random.randint(-3,3),random.randint(-2,2),random.randint(-3,3)]
            x=torch.roll(x,shifts,(2,3,4));y=torch.roll(y,shifts,(2,3,4))
            for dimension,shift in zip((2,3,4),shifts):
                if not shift:continue
                sl=[slice(None)]*5;sl[dimension]=slice(0,shift) if shift>0 else slice(shift,None)
                x[tuple(sl)]=0;y[tuple(sl)]=0
            if fine_y is not None:
                fine_y=torch.roll(fine_y,[2*s for s in shifts],(2,3,4))
                for dimension,shift in zip((2,3,4),shifts):
                    if shift:
                        sl=[slice(None)]*5;sl[dimension]=slice(0,2*shift) if shift>0 else slice(2*shift,None)
                        fine_y[tuple(sl)]=0
            if a.cutmix and len(chosen)>1 and random.random()<.7:
                # Compose TRAIN fragments into new local structures; never mix validation/test labels.
                donor=torch.randperm(len(chosen),device=device)
                axis=random.choice([2,3,4]);start=random.randrange(max(1,x.shape[axis]//2))
                end=min(x.shape[axis],start+random.randint(4,max(4,x.shape[axis]//2)))
                sl=[slice(None)]*5;sl[axis]=slice(start,end)
                source_x=x[donor].clone();source_y=y[donor].clone()
                x[tuple(sl)]=source_x[tuple(sl)];y[tuple(sl)]=source_y[tuple(sl)]
            if a.warp:
                # TRAIN-only domain randomization: vary corridor angles, widths and storey spacing.
                theta=torch.eye(3,4,device=device).unsqueeze(0).repeat(len(chosen),1,1)
                theta[:,0,0]=torch.empty(len(chosen),device=device).uniform_(.87,1.13)
                theta[:,1,1]=torch.empty(len(chosen),device=device).uniform_(.9,1.1)
                theta[:,2,2]=torch.empty(len(chosen),device=device).uniform_(.87,1.13)
                theta[:,0,2]=torch.empty(len(chosen),device=device).uniform_(-.18,.18)
                theta[:,2,0]=torch.empty(len(chosen),device=device).uniform_(-.18,.18)
                warp=F.affine_grid(theta,x.shape,align_corners=False)
                x=F.grid_sample(x,warp,mode='nearest',align_corners=False)
                y=F.grid_sample(y,warp,mode='nearest',align_corners=False)
                # Missing patches simulate incomplete range acquisition; labels are never filled into inputs.
                if random.random()<.6:
                    start=[random.randrange(max(1,x.shape[d]-6)) for d in (2,3,4)]
                    x[:,:,start[0]:start[0]+6,start[1]:start[1]+4,start[2]:start[2]+6]=0
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(x);probability=logits.sigmoid()
                element=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(2.,device=device),reduction='none')
                boundary=(F.max_pool3d(y,3,1,1)+F.max_pool3d(-y,3,1,1))>0
                bce=(element*(1+boundary*a.boundary_weight)).mean((1,2,3,4))
                intersection=(probability*y).sum((1,2,3,4));union=(probability+y-probability*y).sum((1,2,3,4))
                individual=bce+1-(intersection+1)/(union+1)
                if a.lovasz_weight:individual=individual+a.lovasz_weight*occupancy_lovasz(logits,y)
                coarse_risk=individual.detach().float() if a.hard_reweight else None
                if fine_y is not None:
                    fine_logits=model.fine_logits.float();fine_probability=fine_logits.sigmoid()
                    fine_bce=F.binary_cross_entropy_with_logits(fine_logits,fine_y,pos_weight=torch.tensor(2.,device=device),reduction='none').mean((1,2,3,4))
                    fine_intersection=(fine_probability*fine_y).sum((1,2,3,4));fine_union=(fine_probability+fine_y-fine_probability*fine_y).sum((1,2,3,4))
                    individual=individual+a.fine_loss_weight*(fine_bce+1-(fine_intersection+1)/(fine_union+1))*fine_available
                if a.hard_reweight:
                    # Soft worst-group risk, balanced by TRAIN area. VALID labels
                    # never update these weights; half the risk stays uniform.
                    # Correct for the actual long/old sampling mixture. Only
                    # coarse occupancy errors update group risk, so availability
                    # of auxiliary labels does not masquerade as a hard area.
                    weight=torch.tensor(area_risk_weights([area_key(row) for row,_,_ in chosen],group_mass,group_loss),device=device,dtype=individual.dtype)
                    loss=(individual*weight).mean()
                else:loss=individual.mean()
            if a.hard_reweight:
                for (row,_,_),value in zip(chosen,coarse_risk.cpu().tolist()):
                    g=area_key(row);group_loss[g]=.95*group_loss[g]+.05*value
            scaler.scale(loss).backward();scaler.unscale_(optim);torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optim);scaler.update();losses.append(float(loss.detach()))
        schedule.step()
        if epoch%5 and epoch!=1:continue
        model.eval();details=[]
        with torch.no_grad():
            for row,x,y in valid:
                if a.local_patches:
                    from rescue_volume_network import tiled_logits
                    probability=tiled_logits(model,x.unsqueeze(0)).sigmoid()[0,0,:32,:21,:70]
                else:probability=model(x.unsqueeze(0)).sigmoid()[0,0,:32,:21,:70]
                truth=y[0,:32,:21,:70]>.5;pred=probability>.6
                intersection=int((truth&pred).sum());union=int((truth|pred).sum())
                details.append(dict(id=row['id'],source=row['source'],area=area_key(row),iou=intersection/max(1,union)))
        domains={s:float(np.mean([r['iou'] for r in details if r['source']==s])) for s in {r['source'] for r in details}}
        areas={str(g):float(np.mean([r['iou'] for r in details if r['area']==g])) for g in {r['area'] for r in details}}
        macro=float(np.mean([r['iou'] for r in details]));key=(min(areas.values()) if a.worst_area else min(domains.values()),macro)
        record=dict(epoch=epoch,loss=float(np.mean(losses)),macro_iou=macro,domains=domains,minimum_area_iou=min(areas.values()),areas=areas,seconds=time.monotonic()-started)
        history.append(record);print(json.dumps(record),flush=True)
        if key>best:
            best=key;last_best=epoch
            torch.save(dict(state=model.state_dict(),epoch=epoch,architecture=architecture,feature_map='range-tsdf-v6' if a.range_tsdf else 'range-free-v5' if channels==90 else 'range-octants-v4' if channels==74 else 'range-covariance-v3' if channels==18 else 'range-moments-v2' if channels==9 else 'observed-logodds-count-v1',channels=channels,width=a.width,residual=a.residual,tiled=a.local_patches,profiles=a.axis_profiles,normalization=a.normalization,backbone_channels=a.backbone_channels,fine_width=a.fine_width,freeze_backbone=a.subvoxel_refinement and not a.unfreeze_backbone,fine_supervision=bool(a.fine_truth_pool),pretraining_train_maps=sorted(pretrained_worlds),**pose_configuration),out/'volume.pt')
            (out/'development.json').write_text(json.dumps(dict(purpose='Formal VALID model selection, not final acceptance' if a.all_train else 'TRAIN-internal world-held-out selection, not acceptance',
                     test_opened=False,coordinate_inputs=False,shape_templates=False,acquisition='actual Unity/Rust observations',
                     train_audit=audit,fine_training_audit=fine_audit,pretraining=ancestry,pretraining_train_maps=sorted(pretrained_worlds),history=history,best=record,rows=details,arguments=vars(a)),indent=2)+'\n')
        if epoch-last_best>=80:break
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')


if __name__=='__main__':main()
