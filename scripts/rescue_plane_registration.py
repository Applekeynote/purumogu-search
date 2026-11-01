"""Experimental gravity-plane consensus from noisy range/odometry alone.

This prototype does not use floor heights, scene identifiers, or true poses. Plane
heights are estimated from multiple independent nodes' downward range endpoints.
Raw positions remain the shared reference to avoid feeding a correction back into
the evidence used to estimate the plane. It is not yet a production SLAM system.
"""
from collections import defaultdict
import numpy as np


class PlaneConsensus:
    def __init__(self):
        self.poses={};self.ids={};self.hist=defaultdict(lambda:[0.,0.,0]);self.offset={}
        self.planes=[]

    def packet(self,packet):
        rows=[]
        for scan in packet['scans']:
            node=scan['node_id'];origin=np.array(self.poses.get(node,scan['deployment_anchor']))+scan['odometry_delta'];self.poses[node]=origin
            if node not in self.ids:self.ids[node]=len(self.ids)
            down=[r for r in scan['rays'] if r['hit'] and r['direction'][1]<-.35 and r['distance_m']>.004]
            points=np.array([origin+np.array(r['direction'])*r['distance_m'] for r in down]).reshape(-1,3)
            if scan['pose_sigma_m']<=.15:
                for point in points:
                    v=self.hist[int(np.floor(point[1]/.003))];v[0]+=1;v[1]+=point[1];v[2]|=1<<self.ids[node]
            rows.append((scan,origin,points))
        # Support is counted across nodes, not fabricated by repeating one stationary scan.
        peaks=[]
        for key in self.hist:
            counts=total=0.;nodes=0
            for offset in range(-2,3):
                value=self.hist.get(key+offset)
                if value is None:continue
                counts+=value[0];total+=value[1];nodes|=value[2]
            if counts>=80 and nodes.bit_count()>=3:peaks.append((counts,total/counts))
        self.planes=[]
        for _,height in sorted(peaks,reverse=True):
            if all(abs(height-other)>.055 for other in self.planes):self.planes.append(height)
            if len(self.planes)==16:break
        result=[]
        for scan,origin,points in rows:
            residuals=[]
            if len(points)>=4 and self.planes:
                for point in points:
                    plane=min(self.planes,key=lambda p:abs(p-point[1]))
                    if abs(plane-point[1])<.035:residuals.append(plane-point[1])
            previous=self.offset.get(scan['node_id'],0.)
            if len(residuals)>=4:
                value=float(np.median(residuals));mad=float(np.median(np.abs(np.array(residuals)-value)))
                if mad<.007:
                    limit=min(.04,max(.004,scan['pose_sigma_m']*2))
                    previous=.7*previous+.3*np.clip(value,-limit,limit)
            self.offset[scan['node_id']]=previous
            corrected=origin.copy();corrected[1]+=previous
            result.append((scan,origin,corrected))
        return result


class AxisConsensus:
    """Experimental independent signed surface-plane consensus; not Manhattan snapping."""
    def __init__(self):
        self.vertical=PlaneConsensus()
        self.sides={(axis,sign):PlaneConsensus() for axis in (0,2) for sign in (-1,1)}

    def packet(self,packet):
        rows=self.vertical.packet(packet)
        sides={}
        for (axis,sign),mapper in self.sides.items():
            order=[1,0,2] if axis==0 else [0,2,1]
            def transform(value):
                v=np.asarray(value)[order].copy();v[1]*=-sign;return v.tolist()
            scans=[]
            for scan in packet['scans']:
                rays=[dict(ray,direction=transform(ray['direction'])) for ray in scan['rays']
                      if ray['hit'] and .01<ray['direction'][1]<.45 and ray['distance_m']>.04 and ray['direction'][axis]*sign>.55]
                scans.append(dict(scan,deployment_anchor=transform(scan['deployment_anchor']),odometry_delta=transform(scan['odometry_delta']),rays=rays))
            sides[(axis,sign)]={s['node_id']:-sign*(corrected[1]-raw[1]) for s,raw,corrected in mapper.packet(dict(scans=scans))}
        result=[]
        for scan,raw,corrected in rows:
            for axis in (0,2):
                offsets=[sides[(axis,sign)][scan['node_id']] for sign in (-1,1)]
                active=[offset for offset in offsets if abs(offset)>1e-8]
                if active:corrected[axis]+=np.mean(active)
            result.append((scan,raw,corrected))
        return result
