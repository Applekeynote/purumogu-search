"""Relative context among inferred hypotheses; no true sources or absolute coordinates."""
import math
import numpy as np
from rescue_residual_head import teacher_features

VERSION='thermal-spectral-neighbor45-v6'
FEATURES=['local_max_teacher','local_mean_teacher','local_density_log5','local_max_voice','local_max_heat_fit',
    'rival_max_teacher','rival_mean_distance_over65cm','rival_mean_height_difference_over65cm','teacher_contrast','heat_fit_contrast','voice_contrast','local_min_age']


def extend_teacher(proposals,step):
    if not proposals:return []
    if any(len(q['features'])!=33 or q.get('first_seen_step',0)>q.get('last_seen_step',step) or q.get('last_seen_step',step)>step for q in proposals):raise ValueError('Causal spectral33 proposal prefix required')
    x=np.array([q['features'] for q in proposals],dtype=np.float64);positions=np.array([q['position'] for q in proposals],dtype=np.float64)
    if positions.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(positions).all():raise ValueError('Finite inferred proposals required')
    delta=positions[:,None,:]-positions[None,:,:];squared=np.sum(delta*delta,axis=2);distance=np.sqrt(squared)
    local=squared<=.2**2+1e-12;shell=(~local)&(squared<=.65**2+1e-12);has=shell.any(axis=1)
    teacher=x[:,32];local_weight=np.exp(-.5*squared/.15**2)*local;local_den=local_weight.sum(axis=1)
    rival_max=np.where(shell,teacher[None,:],-np.inf).max(axis=1);rival_max=np.where(has,rival_max,teacher)
    rival_weight=np.exp(np.clip(8*(teacher[None,:]-rival_max[:,None]),-80,0))*shell;rival_den=np.maximum(rival_weight.sum(axis=1),1e-12)
    def local_max(feature):return np.where(local,x[None,:,feature],-np.inf).max(axis=1)
    def rival_mean(values,default):return np.where(has,(rival_weight*values).sum(axis=1)/rival_den,default)
    extra=np.column_stack((local_max(32),(local_weight*teacher[None,:]).sum(axis=1)/local_den,np.log1p(local.sum(axis=1)-1)/5,
        local_max(27),local_max(16),rival_max,rival_mean(distance/.65,1),rival_mean(np.abs(delta[:,:,1])/.65,0),
        teacher-rival_max,x[:,16]-rival_mean(x[None,:,16],x[:,16]),x[:,27]-rival_mean(x[None,:,27],x[:,27]),
        np.where(local,x[None,:,6],np.inf).min(axis=1)))
    if extra.shape!=(len(x),12) or not np.isfinite(extra).all():raise ValueError('Invalid relative neighborhood features')
    return [dict(q,features=q['features']+f.tolist()) for q,f in zip(proposals,extra)]


def extend(base,proposals,step):
    if not proposals:return []
    values=teacher_features(base,[q['features'] for q in proposals])
    return extend_teacher([dict(q,features=f.tolist()) for q,f in zip(proposals,values)],step)
