"""Causal association of stationary source hypotheses; no labels or map templates."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


class TemporalCues:
    def __init__(self,radius=.2,retention_steps=18000):
        if not math.isfinite(radius) or not .05<=radius<=.3 or retention_steps<=0:raise ValueError('Bounded temporal association required')
        self.radius=radius;self.retention=retention_steps;self.step=-1;self.tracks=[]

    def update(self,step,cues):
        if step<=self.step:raise ValueError('Strictly increasing causal checkpoints required')
        incoming=[]
        for cue in cues:
            position=list(cue['position']);score=float(cue['score'])
            if len(position)!=3 or not all(math.isfinite(v) for v in position) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Finite source hypotheses required')
            incoming.append(dict(position=position,score=score))
        incoming.sort(key=lambda q:(*q['position'],q['score']))
        self.step=step;self.tracks=[t for t in self.tracks if step-t['last']<=self.retention]
        matches={}
        if incoming and self.tracks:
            count=len(incoming);missing=2*(count+1)
            distance=np.array([[math.dist(c['position'],t['position']) for t in self.tracks] for c in incoming])
            cost=np.full((count,len(self.tracks)+count),float(missing));cost[:,:len(self.tracks)]=np.where(distance<=self.radius,distance/self.radius,missing*2)
            a,b=linear_sum_assignment(cost);matches={int(i):int(j) for i,j in zip(a,b) if j<len(self.tracks) and distance[i,j]<=self.radius}
        for index,cue in enumerate(incoming):
            weight=max(1e-9,cue['score'])**2
            if index in matches:
                track=self.tracks[matches[index]];total=track['weight']+weight
                track['position']=[(a*track['weight']+b*weight)/total for a,b in zip(track['position'],cue['position'])]
                track.update(weight=total,score_sum=track['score_sum']+cue['score'],support=track['support']+1,last=step,last_score=cue['score'])
            else:self.tracks.append(dict(position=list(cue['position']),weight=weight,score_sum=cue['score'],support=1,first=step,last=step,last_score=cue['score']))

    def confirmed(self,minimum_support=2,threshold=.4,nms=.25,support_rank=False,include_current=False):
        if minimum_support<1 or not 0<=threshold<=1 or nms<=0:raise ValueError('Bounded confirmation policy required')
        candidates=[]
        for track in self.tracks:
            score=track['score_sum']/track['support']
            confirmed=track['support']>=minimum_support and score>=threshold
            current=include_current and track['last']==self.step
            if confirmed or current:
                score=max(score if confirmed else 0,track['last_score'] if current else 0)
                candidates.append(dict(position=list(track['position']),score=score,support=track['support'],rank=score+(.1*math.log1p(track['support']) if support_rank and confirmed else 0)))
        candidates.sort(key=lambda q:(-q['rank'],*q['position']));result=[]
        for cue in candidates:
            if all(math.dist(cue['position'],c['position'])>nms for c in result):result.append(cue)
            if len(result)==3:break
        return result
