"""Unlabelled range-plane diagnostic; no simulator shapes or person positions."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np


def fit_plane(scan,rng):
    points=np.array([np.array(r['direction'])*r['distance_m'] for r in scan['rays'] if r['hit'] and r['direction'][1]<-.03 and r['distance_m']>.05])
    if len(points)<12:return None
    chosen=np.array([rng.choice(len(points),3,replace=False) for _ in range(64)])
    triples=points[chosen];matrix=np.concatenate((triples[:,:,[0,2]],np.ones((len(triples),3,1))),axis=2)
    keep=np.abs(np.linalg.det(matrix))>1e-5
    if not keep.any():return None
    parameters=np.linalg.solve(matrix[keep],triples[keep,:,1,None])[...,0]
    parameters=parameters[np.linalg.norm(parameters[:,:2],axis=1)<np.tan(np.deg2rad(6))]
    if not len(parameters):return None
    design=np.column_stack((points[:,0],points[:,2],np.ones(len(points))))
    residual=np.abs(design@parameters.T-points[:,1,None]);support=(residual<.004).sum(0)
    best=int(np.argmax(support));mask=residual[:,best]<.004
    if mask.sum()<12:return None
    planar=points[mask];spread=np.linalg.eigvalsh(np.cov(planar[:,[0,2]].T))
    if spread[0]<.0016 or spread[1]<.01:return None
    fit=np.linalg.lstsq(design[mask],points[mask,1],rcond=None)[0]
    errors=design[mask]@fit-points[mask,1]
    return dict(inliers=int(mask.sum()),tilt_degrees=float(np.degrees(np.arctan(np.linalg.norm(fit[:2])))),
                raw_height_rms_m=float(np.std(points[mask,1])),plane_residual_rms_m=float(np.sqrt(np.mean(errors**2))),spread_eigenvalues_m2=spread.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--max-scans',type=int,default=1000);a=p.parse_args()
    root=ROOT/a.root
    if root.name!='train':raise ValueError('TRAIN diagnostics only')
    records=[];scanned=0;rng=np.random.default_rng(20261031)
    cases=sorted(root.glob('*/complete.json'))
    for complete in cases:
        metadata=json.loads(complete.read_bytes())
        if metadata['split']!='train':raise ValueError('TRAIN role required')
        with (complete.parent/'packets.jsonl').open(encoding='utf-8-sig') as stream:
            for line in stream:
                packet=json.loads(line)
                if 'scans' not in packet:continue
                for scan in packet['scans']:
                    if scanned>=a.max_scans:break
                    scanned+=1;fit=fit_plane(scan,rng)
                    if fit:records.append(dict(case=complete.parent.name,step=packet['sim_step'],node=scan['node_id'],**fit))
                if scanned>=a.max_scans:break
        if scanned>=a.max_scans:break
    summary={k:float(np.median([r[k] for r in records])) for k in ['tilt_degrees','raw_height_rms_m','plane_residual_rms_m']} if records else {}
    result=dict(purpose='Unlabelled TRAIN plane-fitting diagnostic; not accuracy or correction validation',truth_opened=False,test_opened=False,scans=scanned,accepted=len(records),median=summary,rows=records)
    path=ROOT/a.output
    if path.exists():raise ValueError('Use a new output')
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
