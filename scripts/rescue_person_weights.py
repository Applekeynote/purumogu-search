"""TRAIN-only equal-person weighting; labels never become inference features."""
from collections import Counter
import math


def snapshot_weights(row,total,radius,positive_mass):
    if row['domain']!='train' or row.get('held',False):raise ValueError('Only fitting TRAIN may define source weights')
    if not math.isfinite(total) or total<=0 or not math.isfinite(radius) or radius<=0 or not 0<positive_mass<1:raise ValueError('Finite bounded source-weight budgets required')
    assignments=[]
    for proposal in row['proposals']:
        distances=[math.dist(proposal['position'],person) for person in row['victim_positions']]
        nearest=min(range(len(distances)),key=distances.__getitem__) if distances else -1
        assignments.append(nearest if nearest>=0 and distances[nearest]<radius else -1)
    counts=Counter(a for a in assignments if a>=0);negatives=assignments.count(-1)
    positive_budget=positive_mass if negatives else 1.
    negative_budget=1-positive_mass if counts else 1.
    return [total*positive_budget/(len(counts)*counts[a]) if a>=0 else total*negative_budget/negatives for a in assignments]
