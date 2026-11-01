"""Bounded Bernoulli Newton updates to a frozen portable proposal forest."""
import copy
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from train_rescue_generic import export_tree


def forest_logits(model,x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite proposal feature matrix required')
    result=np.full(len(x),float(model['track_bias']))
    for tree in model['track_trees']:
        nodes=tree['nodes'];active=[(0,np.arange(len(x)))]
        while active:
            index,rows=active.pop()
            if not len(rows):continue
            node=nodes[index]
            if node['feature'] is None:result[rows]+=node['value'];continue
            feature=node['feature']
            if feature>=x.shape[1]:raise ValueError('Proposal feature contract differs')
            left=x[rows,feature]<node['threshold']
            active.extend([(node['left'],rows[left]),(node['right'],rows[~left])])
    return result


def teacher_features(base,x):
    """Frozen TRAIN forest confidence, computed before any corrective trees."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=32 or not base.get('track_trees') or any(n['feature'] is not None and n['feature']>=32 for t in base['track_trees'] for n in t['nodes']):raise ValueError('Independent frozen spectral32 teacher required')
    return np.column_stack((x,np.clip(forest_logits(base,x),-40,40)/10))


class ResidualHead:
    def __init__(self,base,rounds,leaves,learning_rate=.08):
        if not base.get('track_trees') or len(base['track_trees'])+rounds>256:raise ValueError('Portable proposal forest exceeds 256 trees')
        if not 1<=rounds<=56 or not 2<=leaves<=32:raise ValueError('Bounded residual capacity required')
        self.base=copy.deepcopy(base);self.rounds=rounds;self.leaves=leaves;self.learning_rate=learning_rate

    def fit(self,x,target,weights,ranking_groups=None,ranking_strength=0):
        x=np.asarray(x,dtype=np.float64);target=np.asarray(target,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
        if not len(target) or len(target)!=len(x) or len(weights)!=len(x) or not np.isin(target,[0,1]).all() or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('Binary TRAIN targets and nonnegative weights required')
        if ranking_strength and ranking_groups is None:raise ValueError('Ranking requires causal TRAIN snapshot groups')
        self.trees=copy.deepcopy(self.base['track_trees']);self.bias=float(self.base['track_bias'])
        logits=forest_logits(self.base,x)
        for step in range(self.rounds):
            if ranking_strength:
                from rescue_ranking_loss import derivatives
                gradient,hessian=derivatives(logits,target,weights,ranking_groups,ranking_strength)
                response=np.clip(-gradient/np.maximum(hessian,1e-12),-6,6)
            else:
                probability=1/(1+np.exp(-np.clip(logits,-40,40)))
                curvature=np.maximum(probability*(1-probability),.001)
                response=np.clip((target-probability)/curvature,-6,6)
                hessian=weights*curvature
            hessian*=len(x)/hessian.sum()
            fit=HistGradientBoostingRegressor(max_iter=1,max_leaf_nodes=self.leaves,max_bins=63,learning_rate=self.learning_rate,min_samples_leaf=40,l2_regularization=2,early_stopping=False,random_state=20261101+step)
            fit.fit(x,response,sample_weight=hessian)
            # sklearn's intercept is unshrunk. Shrink both intercept and tree
            # contribution, then export the exact sum using the existing native
            # forest contract. No extra runtime component or Python is needed.
            intercept=float(fit._baseline_prediction[0,0])
            self.bias+=self.learning_rate*intercept
            self.trees.append(export_tree(fit._predictors[0][0]))
            logits+=fit.predict(x)-(1-self.learning_rate)*intercept
        self.training_logits=logits
        return self

    def predict_proba(self,x):
        logits=forest_logits(dict(track_trees=self.trees,track_bias=self.bias),x)
        probability=1/(1+np.exp(-np.clip(logits,-40,40)))
        return np.column_stack((1-probability,probability))
