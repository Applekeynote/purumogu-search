import math
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_rescue_generic
import numpy as np
from threadpoolctl import threadpool_limits
from rescue_residual_head import ResidualHead,forest_logits,teacher_features
from train_rescue_tracks import score_model
from select_rescue_residual_scale import scaled_model


def fixture():
    return dict(track_bias=-.2,track_trees=[dict(nodes=[dict(feature=0,threshold=.5,left=1,right=2,value=0),dict(feature=None,value=-1),dict(feature=None,value=1)])])


class ResidualHeadTests(unittest.TestCase):
    def test_teacher_confidence_is_independent_of_corrective_trees(self):
        base=fixture();x=np.zeros((3,32));x[:,0]=[.1,.5,.9]
        result=teacher_features(base,x)
        self.assertEqual(result.shape,(3,33));np.testing.assert_array_equal(result[:,:32],x)
        np.testing.assert_allclose(result[:,32],[-.12,.08,.08],atol=1e-14)
        self.assertEqual(teacher_features(base,np.empty((0,32))).shape,(0,33))
        cyclic=copy.deepcopy(base);cyclic['track_trees'][0]['nodes'][0]['feature']=32
        with self.assertRaises(ValueError):teacher_features(cyclic,x)
        with self.assertRaises(ValueError):teacher_features(base,x[:,:31])

    def test_global_strength_preserves_the_base_and_exact_portable_logits(self):
        base=fixture();model=copy.deepcopy(base);model['track_bias']+=.3
        model['track_trees'].append(dict(nodes=[dict(feature=None,value=.7)]))
        x=np.array([[.1],[.5],[.9]]);before=copy.deepcopy(model)
        for strength in [0,.25,1,2]:
            actual=scaled_model(base,model,strength)
            expected=forest_logits(base,x)+strength*(forest_logits(model,x)-forest_logits(base,x))
            np.testing.assert_allclose(forest_logits(actual,x),expected,rtol=1e-14,atol=1e-14)
        self.assertEqual(model,before)
        with self.assertRaises(ValueError):scaled_model(base,model,3)
        changed=copy.deepcopy(model);changed['track_trees'][0]['nodes'][0]['threshold']=.6
        with self.assertRaises(ValueError):scaled_model(base,changed,.5)

    def test_vector_forest_matches_portable_scalar_contract_at_threshold(self):
        model=fixture();x=np.array([[.1],[.5],[.9]])
        score=1/(1+np.exp(-forest_logits(model,x)))
        np.testing.assert_allclose(score,[score_model(model,row) for row in x],rtol=1e-14)

    def test_fitted_updates_export_without_changing_frozen_forest(self):
        rng=np.random.default_rng(1);x=rng.uniform(0,1,(240,2));y=(x[:,1]>.5).astype(int);base=fixture()
        before=forest_logits(base,x)
        with threadpool_limits(limits=1):head=ResidualHead(base,6,4).fit(x,y,np.ones(len(x)))
        exported=dict(track_bias=head.bias,track_trees=head.trees)
        np.testing.assert_allclose(head.training_logits,forest_logits(exported,x),rtol=1e-10,atol=1e-10)
        self.assertEqual(exported['track_trees'][0],base['track_trees'][0])
        old=np.logaddexp(0,before)-y*before;new=np.logaddexp(0,head.training_logits)-y*head.training_logits
        self.assertLess(new.mean(),old.mean())
        np.testing.assert_allclose(head.predict_proba(x[:5])[:,1],[score_model(exported,row) for row in x[:5]],rtol=1e-12)

    def test_native_capacity_and_label_contract_are_enforced(self):
        base=fixture();base['track_trees']*=201
        with self.assertRaises(ValueError):ResidualHead(base,56,7)
        with self.assertRaises(ValueError):ResidualHead(fixture(),2,4).fit([[0],[1]],[0,.5],[1,1])

    def test_pairwise_updates_keep_the_same_portable_forest_contract(self):
        rng=np.random.default_rng(7);x=rng.uniform(0,1,(240,2));y=(x[:,1]>.5).astype(int);base=fixture()
        with threadpool_limits(limits=1):head=ResidualHead(base,6,4).fit(x,y,np.ones(len(x)),ranking_groups=[60]*4,ranking_strength=.5)
        exported=dict(track_bias=head.bias,track_trees=head.trees)
        self.assertEqual(exported['track_trees'][0],base['track_trees'][0])
        np.testing.assert_allclose(head.training_logits,forest_logits(exported,x),rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(head.predict_proba(x[:8])[:,1],[score_model(exported,row) for row in x[:8]],rtol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
