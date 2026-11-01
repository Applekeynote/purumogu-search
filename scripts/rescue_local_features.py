"""Translation-invariant local evidence. Order is a versioned Rust/Python contract.

No world/scene identity, seed, absolute position, source position or truth is a feature.
"""
from itertools import product
import numpy as np

SHAPE = (32, 21, 70)
MINIMUM = np.array([-16, -3, -6])
OFFSETS = list(product(range(-1, 2), repeat=3))
for radius in (2, 3, 4):
    for axis in range(3):
        for sign in (-1, 1):
            offset = [0, 0, 0]
            offset[axis] = radius * sign
            OFFSETS.append(tuple(offset))
FEATURES = len(OFFSETS) * 2
VERSION = 'relative-rays-90-v1'


def configure(version):
    global OFFSETS, FEATURES, VERSION
    if version == 'relative-rays-322-v2':
        OFFSETS = list(product(range(-2, 3), repeat=3))
        for radius in (3, 4, 6, 8, 12, 16):
            for axis in range(3):
                for sign in (-1, 1):
                    offset = [0, 0, 0];offset[axis] = radius * sign
                    OFFSETS.append(tuple(offset))
    elif version == 'relative-rays-90-v1':
        OFFSETS = list(product(range(-1, 2), repeat=3))
        for radius in (2, 3, 4):
            for axis in range(3):
                for sign in (-1, 1):
                    offset = [0, 0, 0];offset[axis] = radius * sign
                    OFFSETS.append(tuple(offset))
    else:
        raise ValueError(version)
    FEATURES = len(OFFSETS)*2;VERSION = version


def evidence(cells):
    grid = np.zeros(SHAPE + (2,), dtype=np.float32)
    measured = [c for c in cells if c['observations'] > 0 and c.get('sensor_occupancy') is not None]
    if not measured:return grid
    keys=np.asarray([c['key'] for c in measured],dtype=np.int32)-MINIMUM
    keep=np.all((keys>=0)&(keys<SHAPE),axis=1)
    q=np.clip(np.asarray([c['sensor_occupancy'] for c in measured]),1e-6,1-1e-6)[keep]
    counts=np.asarray([c['observations'] for c in measured])[keep];keys=keys[keep]
    grid[tuple(keys.T)] = np.column_stack((np.clip(np.log(q/(1-q))/4,-1,1),
                                          np.minimum(1,np.log1p(counts)/np.log(1025))))
    return grid


def features(grid):
    radius = max(max(abs(x) for x in offset) for offset in OFFSETS)
    padded = np.pad(grid, ((radius, radius), (radius, radius), (radius, radius), (0, 0)))
    return np.concatenate([padded[radius+x:radius+x+32, radius+y:radius+y+21, radius+z:radius+z+70]
                           for x, y, z in OFFSETS], axis=3).reshape(-1, FEATURES)


def truth_mask(keys):
    truth = np.zeros(SHAPE, dtype=np.uint8)
    if len(keys):
        indices=np.asarray(keys,dtype=np.int32)-MINIMUM
        indices=indices[np.all((indices>=0)&(indices<SHAPE),axis=1)]
        truth[tuple(indices.T)] = 1
    return truth.reshape(-1)


def rotate_features(x, turns):
    order = {offset: i for i, offset in enumerate(OFFSETS)}
    indices = []
    for dx, dy, dz in OFFSETS:
        for _ in range(turns % 4):
            dx, dz = -dz, dx
        source = order[(dx, dy, dz)]
        indices.extend([source*2, source*2+1])
    return x[:, indices]


def iou(probabilities, truth, threshold=.6):
    prediction = probabilities > threshold
    intersection = int(np.sum(prediction & (truth == 1)))
    union = int(np.sum(prediction | (truth == 1)))
    return dict(iou=intersection/max(1, union), intersection=intersection, union=union,
                predicted=int(prediction.sum()), occupied=int(truth.sum()))
