"""Candidate context from measured occupancy only; no inferred cells or truth."""
import math

VERSION='thermal-spectral-map40-v4'
FEATURES=['known_volume_3x3x3','solid_volume_3x3x3','known_at_candidate','measured_occupancy_at_candidate','near_support_patch','deeper_support_patch','nearest_support_depth_over_35cm','ceiling_patch']
MINIMUM=(-16,-3,-6);MAXIMUM=(16,18,64)


class MapContext:
    def __init__(self,cells):
        self.observed={}
        for cell in cells:
            if cell.get('observations',0)==0:continue
            key=tuple(cell['key'])
            if any(k<lo or k>=hi for k,lo,hi in zip(key,MINIMUM,MAXIMUM)):continue
            value=cell.get('sensor_occupancy')
            if value is None or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Measured sensor occupancy required; model-filled probabilities cannot substitute')
            self.observed[key]=float(value)

    def features(self,point):
        if len(point)!=3 or not all(math.isfinite(p) for p in point):raise ValueError('Finite inferred position required')
        x,y,z=(math.floor(p/.05) for p in point)
        values=self.observed
        known=solid=0
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    value=values.get((x+dx,y+dy,z+dz))
                    known+=value is not None;solid+=value is not None and value>.65
        def patch(dy):
            return sum(values.get((x+dx,y+dy,z+dz),.5)>.65 for dx in [-1,0,1] for dz in [-1,0,1])/9
        below=[patch(-d) for d in range(8)]
        nearest=next((d/7 for d,value in enumerate(below) if value>=5/9),1.)
        return [known/27,solid/27,float((x,y,z) in values),values.get((x,y,z),.5),max(below[:3]),max(below[3:]),nearest,max(patch(d) for d in range(1,5))]


def extend(data):
    context=MapContext(data['cells']);result=[]
    for proposal in data['proposals']:
        if len(proposal['features'])!=32:raise ValueError('Registered 32-feature proposal prefix required')
        result.append(dict(proposal,features=proposal['features']+context.features(proposal['position'])))
    return result
