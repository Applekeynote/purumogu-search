"""Binary IoU surrogate for TRAIN occupancy labels.

Lovasz hinge formulation: Berman, Triki and Blaschko, CVPR 2018,
https://arxiv.org/abs/1705.08790 . Independent implementation of the binary
margin/sorted-Jaccard construction; the decision boundary remains p=0.6.
"""
import math
import torch


def area_sampling_mass(areas,probabilities):
    if len(areas)!=len(probabilities) or not areas or not math.isclose(sum(probabilities),1,abs_tol=1e-9):raise ValueError('Normalized TRAIN sampling distribution required')
    result={}
    for area,p in zip(areas,probabilities):
        if not math.isfinite(p) or p<=0:raise ValueError('Positive sampling mass required')
        result[area]=result.get(area,0)+float(p)
    return result


def area_risk_weights(chosen,mass,risks):
    groups=sorted(mass)
    if set(mass)!=set(risks) or any(not math.isfinite(risks[g]) for g in groups):raise ValueError('TRAIN risks must match sampled areas')
    peak=max(risks.values());raw={g:math.exp(max(-20,(risks[g]-peak)/.08)) for g in groups};total=sum(raw.values())
    q={g:.5*raw[g]/total+.5/len(groups) for g in groups}
    return [q[g]/mass[g] for g in chosen]


def occupancy_lovasz(logits,labels,threshold=.6):
    if not 0<threshold<1:raise ValueError('Interior probability threshold required')
    scores=logits.float().flatten(1)-math.log(threshold/(1-threshold))
    target=labels.float().flatten(1)
    margin=1-scores*(2*target-1)
    errors,order=margin.sort(dim=1,descending=True)
    ranked=target.gather(1,order)
    total=ranked.sum(1,keepdim=True)
    intersection=total-ranked.cumsum(1)
    union=total+(1-ranked).cumsum(1)
    jaccard=1-intersection/union.clamp_min(1)
    gradient=torch.cat((jaccard[:,:1],jaccard[:,1:]-jaccard[:,:-1]),1)
    return (errors.relu()*gradient).sum(1)
