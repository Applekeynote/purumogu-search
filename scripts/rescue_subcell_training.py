"""Load a coordinate view and its matching labels as one audited training unit."""
import hashlib
import json
import random
from pathlib import Path
import numpy as np


def choose_training_views(chosen,subcell,temporal,share,device,fine_y=None,fine_available=None,rng=None):
    rng=random if rng is None else rng;expanded=[]
    if fine_y is not None and fine_available is None:raise ValueError('Fine-target availability mask required')
    for index,(row,x,y) in enumerate(chosen):
        if subcell.get(row['id']) and rng.random()<share:
            sx,sy,sfine=rng.choice(subcell[row['id']]);expanded.append((row,sx.to(device),sy.to(device)))
            if fine_y is not None:
                if sfine is None:raise ValueError('Shifted fine target missing')
                fine_y[index]=sfine.to(device,dtype=fine_y.dtype);fine_available[index]=1
        elif temporal.get(row['id']) and rng.random()<.5:
            sx,sy=rng.choice(temporal[row['id']]);expanded.append((row,sx.to(device),sy.to(device)))
        else:expanded.append((row,x,y))
    return expanded


def views(directory,record,require_fine=False):
    directory=Path(directory).resolve();fine_cache={}
    for view in record['views']:
        step=view.get('evaluation_step',6000)
        if step not in [6000,18000]:raise ValueError('Registered coordinate-view time required')
        target=(directory/view.get('directory',f"view-{view['index']}")).resolve()
        if not target.is_relative_to(directory):raise ValueError('Coordinate-view path outside case')
        for name,key in [(f'map-{step}.json','map_sha256'),('range-features.npz','features_sha256'),(f'truth-{step}.json','truth_sha256')]:
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=view[key]:raise ValueError('Coordinate-view artifact changed: '+name)
        fine=None
        if view.get('fine_labels'):
            path=(directory/view['fine_labels']).resolve()
            if not path.is_relative_to(directory):raise ValueError('Fine-label path outside coordinate case')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=view['fine_labels_sha256']:raise ValueError('Coordinate fine labels changed')
            if digest not in fine_cache:
                with np.load(path) as data:fine_cache[digest]=data['y']
                label=fine_cache[digest]
                if label.shape!=(1,64,48,144) or label.dtype!=np.uint8 or np.any(label>1):raise ValueError('Invalid coordinate fine target')
            fine=fine_cache[digest]
        elif require_fine:raise ValueError('Shifted input requires its own matching fine target')
        audit=dict(index=view['index'],step=step,translation_m=view['translation_m'],map_sha256=view['map_sha256'],fine_labels_sha256=view.get('fine_labels_sha256'))
        yield dict(map=target/f'map-{step}.json',truth=json.loads((target/f'truth-{step}.json').read_bytes()),evaluation_step=step,fine=fine,audit=audit)
