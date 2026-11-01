"""Convex parameter interpolation with exact nonfloating-buffer contracts."""
import math
import torch


def interpolate(left,right,weight):
    if not math.isfinite(weight) or not 0<=weight<=1 or left.keys()!=right.keys():raise ValueError('Matching state keys and a bounded convex weight required')
    result={}
    for key,l in left.items():
        r=right[key]
        if l.shape!=r.shape or l.dtype!=r.dtype or l.device!=r.device:raise ValueError('State tensor contract differs: '+key)
        if l.is_floating_point():
            if not torch.isfinite(l).all() or not torch.isfinite(r).all():raise ValueError('Finite checkpoint tensors required')
            result[key]=l.clone() if weight==0 else r.clone() if weight==1 else (1-weight)*l+weight*r
        else:
            if not torch.equal(l,r):raise ValueError('Nonfloating buffers differ: '+key)
            result[key]=l.clone()
    return result
