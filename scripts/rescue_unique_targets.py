"""TRAIN-only one-to-one source supervision, with unchanged inference inputs."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def targets(row, scores, total, radius=.25, positive_mass=.5, ignore_radius=.08):
    if row['domain'] != 'train' or row.get('held', False):
        raise ValueError('Only fitting TRAIN may define unique source targets')
    if not all(math.isfinite(v) for v in [total, radius, positive_mass, ignore_radius]) or total <= 0 or radius <= 0 or not 0 < positive_mass < 1 or not 0 <= ignore_radius < radius:
        raise ValueError('Finite bounded matching and weight budgets required')
    proposals = row['proposals']; people = row['victim_positions']; n = len(proposals)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (n,) or not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError('One frozen TRAIN teacher probability per proposal required')
    y = np.zeros(n, dtype=np.int32); weights = np.zeros(n)
    if not n:
        return y, weights, dict(represented_people=0, people=len(people), ignored=0)
    selected = []
    if people:
        distances = np.array([[math.dist(person, p['position']) for p in proposals] for person in people])
        if not np.all(np.isfinite(distances)):
            raise ValueError('Finite positions required for TRAIN target assignment')
        # Dummy columns permit unobservable sources. Their cost exceeds the
        # total of all valid edges, so cardinality dominates teacher preference.
        unmatched = 2 * (len(people) + 1)
        cost = np.full((len(people), n + len(people)), float(unmatched))
        cost[:, :n] = np.where(distances < radius, 1 - scores[None, :] + .1 * distances / radius, unmatched * 2)
        person_indices, proposal_indices = linear_sum_assignment(cost)
        selected = [int(p) for v, p in zip(person_indices, proposal_indices) if p < n and distances[v, p] < radius]
        y[selected] = 1
    ignored = np.zeros(n, dtype=bool)
    for index, proposal in enumerate(proposals):
        if not y[index] and any(math.dist(proposal['position'], proposals[j]['position']) < ignore_radius for j in selected):
            ignored[index] = True
    negatives = (y == 0) & ~ignored
    if selected:
        weights[y == 1] = total * (positive_mass if negatives.any() else 1.) / len(selected)
    if negatives.any():
        weights[negatives] = total * (1 - positive_mass if selected else 1.) / negatives.sum()
    return y, weights, dict(represented_people=len(selected), people=len(people), ignored=int(ignored.sum()))
