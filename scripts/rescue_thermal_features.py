"""Sensor-only spatial thermal support for each temporal acoustic proposal."""
import hashlib
import json
from pathlib import Path
import numpy as np
from functools import lru_cache
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def query(samples,positions,local=False):
    if not positions:return []
    chosen=[(key,v) for key,v in sorted(samples.items()) if v[2]>=3]
    if not chosen:return np.zeros((len(positions),12 if local else 4)).tolist()
    points=(np.array([k for k,_ in chosen],dtype=float)+.5)*.03
    weight=np.array([min(v[1],12) for _,v in chosen]);temperature=np.array([v[0]/v[1] for _,v in chosen])
    distance=np.linalg.norm(np.array(positions)[:,None,:]-points[None,:,:],axis=2)
    kernel=np.exp(-7*distance)
    with threadpool_limits(limits=4):
        et=kernel@(weight*temperature);ee=(kernel*kernel)@weight
    tt=np.sum(weight*temperature*temperature)
    valid=(ee>=1e-12)&(tt>=1e-12);result=np.zeros((len(positions),4))
    result[valid,0]=np.clip(et[valid]/np.sqrt(ee[valid]*tt),-1,1)
    result[valid,1]=np.clip(et[valid]/ee[valid]/4,-4,4)
    positive=temperature>.12
    if np.any(positive):
        result[valid,2]=np.minimum(2,np.min(distance[valid][:,positive],axis=1))/2
        with threadpool_limits(limits=4):result[valid,3]=np.log1p(kernel[valid][:,positive]@weight[positive])/5
    else:result[valid,2]=1
    if local:
        parts=[result]
        for radius in [.18,.4]:
            # Local fitting prevents a warmer source on a different floor from
            # dominating the normalization of a weak, nearby heat signature.
            local_weight=np.exp(-.5*(distance/radius)**2)*weight
            et=np.sum(local_weight*kernel*temperature,axis=1)
            ee=np.sum(local_weight*kernel*kernel,axis=1)
            tt=np.sum(local_weight*temperature*temperature,axis=1)
            den=np.sum(local_weight,axis=1);fit=np.zeros((len(positions),4))
            fit[:,0]=np.clip(et/np.sqrt(np.maximum(ee*tt,1e-24)),-1,1)
            fit[:,1]=np.clip(et/np.maximum(ee,1e-12)/4,-4,4)
            fit[:,2]=np.sum(local_weight*temperature,axis=1)/np.maximum(den,1e-12)/4
            fit[:,3]=np.log1p(den)/5
            parts.append(fit)
        result=np.concatenate(parts,axis=1)
    return result.tolist()


def local_by_step(directory,proposals_by_step,spectral=False):
    """Additional sensor-only features for arbitrary immutable replay proposals."""
    audit=directory/'packets.jsonl'
    signature=hashlib.sha256(audit.read_bytes())
    signature.update(b'spectral32-v3' if spectral else b'thermal24-v2')
    signature.update(json.dumps(proposals_by_step,sort_keys=True).encode())
    folder=ROOT/'tmp/generic-thermal-local';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(signature.hexdigest()+'.json')
    if target.exists():return json.loads(target.read_bytes())
    poses={};samples={};spectra={};result={}
    for line in audit.open(encoding='utf-8-sig'):
        packet=json.loads(line)
        for scan in packet['scans']:
            node=scan['node_id'];origin=poses.get(node,np.array(scan['deployment_anchor']))+scan['odometry_delta'];poses[node]=origin
            if scan['pose_sigma_m']>.15:continue
            key=tuple(np.floor(origin/.03).astype(int));weight=1/(1+(scan['pose_sigma_m']/.05)**2)
            v=samples.setdefault(key,[0.,0.,0]);v[0]+=scan['temperature_delta_c']*weight;v[1]+=weight;v[2]+=1
            if spectral:
                s=spectra.setdefault(key,np.zeros(4));voice=scan['voice_band_ratio'];mod=scan['modulation']
                s+=np.array([scan['acoustic_snr_db']/20,voice,mod,voice*mod])*weight
        step=str(packet['sim_step'])
        if step in proposals_by_step:
            result[step]=query(samples,[p['position'] for p in proposals_by_step[step]],local=True)
            if spectral and proposals_by_step[step]:
                chosen=[(key,v) for key,v in sorted(samples.items()) if v[2]>=3]
                if chosen:
                    points=(np.array([k for k,_ in chosen])+.5)*.03
                    distance=np.linalg.norm(np.array([p['position'] for p in proposals_by_step[step]])[:,None,:]-points[None,:,:],axis=2)
                    weight=np.array([min(v[1],12) for _,v in chosen])
                    values=np.array([spectra[k]/v[1] for k,v in chosen])
                    parts=[]
                    for radius in [.18,.4]:
                        w=np.exp(-.5*(distance/radius)**2)*weight
                        with threadpool_limits(limits=4):parts.append((w@values)/np.maximum(w.sum(axis=1,keepdims=True),1e-12))
                    extra=np.concatenate(parts,axis=1)
                else:extra=np.zeros((len(proposals_by_step[step]),8))
                result[step]=np.concatenate((result[step],extra),axis=1).tolist()
    target.write_text(json.dumps(result,separators=(',',':'))+'\n');return result


@lru_cache(maxsize=256)
def cached(directory):
    audit=directory/'packets.jsonl';signature=hashlib.sha256()
    signature.update(str(audit.resolve()).encode());signature.update(str((audit.stat().st_size,audit.stat().st_mtime_ns)).encode())
    maps={s:json.loads((directory/f'map-{s}.json').read_bytes()) for s in [1500,3000,4500,6000]}
    for step,data in maps.items():signature.update(json.dumps(data['proposals'],sort_keys=True).encode())
    folder=ROOT/'tmp/generic-thermal-features';folder.mkdir(parents=True,exist_ok=True);output=folder/(signature.hexdigest()+'.json')
    if output.exists():return json.loads(output.read_bytes())
    poses={};samples={};result={}
    with audit.open(encoding='utf-8-sig') as stream:
        for line in stream:
            packet=json.loads(line)
            for scan in packet['scans']:
                node=scan['node_id'];origin=poses.get(node,np.array(scan['deployment_anchor']))+scan['odometry_delta'];poses[node]=origin
                if scan['pose_sigma_m']>.15:continue
                key=tuple(np.floor(origin/.03).astype(int));weight=1/(1+(scan['pose_sigma_m']/.05)**2)
                value=samples.setdefault(key,[0.,0.,0]);value[0]+=scan['temperature_delta_c']*weight;value[1]+=weight;value[2]+=1
            step=packet['sim_step']
            if step in maps:
                proposals=maps[step]['proposals'];extras=query(samples,[p['position'] for p in proposals])
                result[str(step)]=[p['features']+extra for p,extra in zip(proposals,extras)]
    output.write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8');return result


def extend(directory,step,proposals):
    extra=cached(directory)[str(step)]
    if len(extra)!=len(proposals):raise ValueError('Proposal cache mismatch')
    return [dict(p,features=f) for p,f in zip(proposals,extra)]
