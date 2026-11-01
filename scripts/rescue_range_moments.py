"""Sub-voxel endpoint features reconstructed ONLY from recorded range/odometry packets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from rescue_local_features import SHAPE,MINIMUM,evidence

ROOT=Path(__file__).resolve().parents[1]


def build(audit,full=False,fine=False):
    poses={};count=np.zeros(np.prod(SHAPE));weight=np.zeros_like(count);sums=np.zeros((15 if full else 6,len(count)))
    if fine:
        node_ids={};support=np.zeros(len(count),dtype=np.uint64)
        micro_count=np.zeros((8,len(count)));micro_weight=np.zeros_like(micro_count)
        micro_sums=np.zeros((8,6,len(count)));micro_support=np.zeros((8,len(count)),dtype=np.uint64)
    with audit.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if '"scans"' not in line:continue
            packet=json.loads(line)
            for scan in packet['scans']:
                key=scan['node_id'];origin=poses.get(key,np.asarray(scan['deployment_anchor'],dtype=float))+scan['odometry_delta'];poses[key]=origin
                if fine:
                    if key not in node_ids:node_ids[key]=len(node_ids)
                    if node_ids[key]>=64:raise ValueError('Observer bitset budget')
                    bit=np.uint64(1)<<np.uint64(node_ids[key])
                if scan['pose_sigma_m']>.15:continue
                rays=[r for r in scan['rays'] if r['hit']]
                if not rays:continue
                directions=np.asarray([r['direction'] for r in rays]);distance=np.asarray([r['distance_m'] for r in rays])
                points=origin+directions*distance[:,None];keys=np.floor(points/.05).astype(np.int32);indices=keys-MINIMUM
                keep=np.all((indices>=0)&(indices<SHAPE),axis=1)
                if not np.any(keep):continue
                indices=indices[keep];flat=np.ravel_multi_index(tuple(indices.T),SHAPE)
                reliability=1/(1+(scan['pose_sigma_m']/.05)**2)
                values=np.column_stack((points[keep]/.05-keys[keep]-.5,directions[keep]))
                if full:
                    off=values[:,:3];direction=values[:,3:]
                    second=np.column_stack((off*off,off[:,0]*off[:,1],off[:,0]*off[:,2],off[:,1]*off[:,2],off*direction))
                    values=np.column_stack((values,second))
                np.add.at(count,flat,1);np.add.at(weight,flat,reliability)
                for axis in range(sums.shape[0]):np.add.at(sums[axis],flat,values[:,axis]*reliability)
                if fine:
                    fine_keys=np.floor(points[keep]/.025).astype(np.int32)
                    bits=fine_keys-keys[keep]*2;octant=bits[:,0]*4+bits[:,1]*2+bits[:,2]
                    index=(octant,flat);local=np.column_stack((points[keep]/.025-fine_keys-.5,directions[keep]))
                    np.add.at(micro_count,index,1);np.add.at(micro_weight,index,reliability)
                    np.bitwise_or.at(support,flat,bit);np.bitwise_or.at(micro_support,index,bit)
                    for axis in range(6):np.add.at(micro_sums,(octant,np.full(len(flat),axis),flat),local[:,axis]*reliability)
    means=np.divide(sums,weight[None,:],out=np.zeros_like(sums),where=weight[None,:]>0)
    if full:
        off=means[:3];direction=means[3:6]
        means[6:9]=np.maximum(0,means[6:9]-off*off)
        means[9:12]-=np.array([off[0]*off[1],off[0]*off[2],off[1]*off[2]])
        means[12:15]-=off*direction
    base=np.concatenate((np.minimum(1,np.log1p(count)/np.log(1025))[None,:],means),axis=0)
    if fine:
        parts=[base,np.minimum(1,np.log1p(np.bitwise_count(support).astype(np.float64))/np.log(49))[None,:]]
        for octant in range(8):
            mean=np.divide(micro_sums[octant],micro_weight[octant][None,:],out=np.zeros_like(micro_sums[octant]),where=micro_weight[octant][None,:]>0)
            parts.append(np.concatenate((np.minimum(1,np.log1p(micro_count[octant])/np.log(1025))[None,:],mean,
                           np.minimum(1,np.log1p(np.bitwise_count(micro_support[octant]).astype(np.float64))/np.log(49))[None,:]),axis=0))
        base=np.concatenate(parts,axis=0)
    return base.astype(np.float32).reshape((72 if fine else 16 if full else 7,)+SHAPE)


def audit_path(row):
    path=row['map']
    if (path.parent/'packets.jsonl').is_file():return path.parent/'packets.jsonl'
    return path.parent.parent/(row['truth']['run_id']+'.jsonl')


def cached(row,full=False,fine=False):
    audit=audit_path(row);signature=hashlib.sha256()
    signature.update(row['map'].read_bytes())
    # Complete records have immutable run IDs. Also include file size and mtime to reject stale partial audits.
    signature.update(str(audit.resolve()).encode());signature.update(str((audit.stat().st_size,audit.stat().st_mtime_ns)).encode())
    cache=ROOT/('tmp/generic-range-fine' if fine else 'tmp/generic-range-covariance' if full else 'tmp/generic-range-moments');cache.mkdir(parents=True,exist_ok=True);target=cache/(signature.hexdigest()+'.npz')
    if not target.exists():
        extra=build(audit,full,fine);np.savez_compressed(target,features=extra)
    result=np.load(target)['features']
    if fine:
        # Observer counts have discrete levels; canonicalize early cache files whose
        # uint8 log1p was evaluated at reduced precision by NumPy's scalar promotion.
        for i in [7]+[15+8*octant for octant in range(8)]:
            observers=np.rint(np.expm1(result[i].astype(np.float64)*np.log(49)))
            result[i]=np.minimum(1,np.log1p(observers)/np.log(49)).astype(np.float32)
    return result


def input_grid(row,channels=9):
    data=json.loads(row['map'].read_bytes())
    native=row['map'].parent/'range-features.npz'
    if native.exists():
        if channels not in (9,74,90):raise ValueError('Registered tensor supports coarse9 or fine74/90')
        extra=np.load(native)['features'][:channels-2]
        if len(extra)!=channels-2:raise ValueError('Native tensor has insufficient channels')
    else:
        if channels==90:raise ValueError('Fine free-space evidence requires native sensor preparation')
        extra=cached(row,channels==18,channels==74)
    return np.concatenate((evidence(data['cells']).transpose(3,0,1,2),extra),axis=0)
