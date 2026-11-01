"""Binary anchor plus query-local, NDCG-weighted positive/negative pairs.

The pairwise gradient follows the RankNet/LambdaRank principle; normalization,
the binary anchor and the bounded hard-negative set are specific to this project.
https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
"""
import math
import numpy as np
from scipy.stats import rankdata


def derivatives(logits,target,weights,groups,strength=1.,top_negative=32):
    scores=np.asarray(logits,dtype=float);y=np.asarray(target);w=np.asarray(weights,dtype=float)
    if scores.ndim!=1 or y.shape!=scores.shape or w.shape!=scores.shape or not np.isfinite(scores).all() or not np.isfinite(w).all() or np.any(w<0) or not np.isin(y,[0,1]).all():raise ValueError('Finite weighted binary TRAIN rows required')
    if any(not isinstance(n,(int,np.integer)) or n<0 for n in groups) or sum(groups)!=len(scores):raise ValueError('Exact causal snapshot group sizes required')
    if not math.isfinite(strength) or not 0<=strength<=2 or top_negative<1:raise ValueError('Bounded pairwise strength required')
    probability=1/(1+np.exp(-np.clip(scores,-40,40)))
    gradient=w*(probability-y);hessian=w*np.maximum(probability*(1-probability),.001)
    if not strength:return gradient,hessian
    start=0
    for count in groups:
        s=scores[start:start+count];active=w[start:start+count]>0;labels=y[start:start+count]
        positives=np.flatnonzero(active&(labels==1));negatives=np.flatnonzero(active&(labels==0))
        if len(positives) and len(negatives):
            if len(negatives)>top_negative:
                cutoff=np.partition(s[negatives],len(negatives)-top_negative)[-top_negative]
                negatives=negatives[s[negatives]>=cutoff]  # Retain ties without order-dependent selection.
            discount=np.zeros(count);discount[active]=1/np.log2(rankdata(-s[active],method='average')+1)
            ideal=sum(1/np.log2(np.arange(2,len(positives)+2)))
            importance=np.maximum(np.abs(discount[positives,None]-discount[None,negatives])/ideal,1e-6)
            importance*=strength*w[start:start+count].sum()/importance.sum()
            pair_probability=1/(1+np.exp(np.clip(s[positives,None]-s[None,negatives],-40,40)))
            push=importance*pair_probability;curvature=importance*np.maximum(pair_probability*(1-pair_probability),.001)
            gradient[start+positives]-=push.sum(1);gradient[start+negatives]+=push.sum(0)
            hessian[start+positives]+=curvature.sum(1);hessian[start+negatives]+=curvature.sum(0)
        start+=count
    return gradient,hessian
