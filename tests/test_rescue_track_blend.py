import copy
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'tmp/learning-runtime')]
import numpy as np
from rescue_residual_head import forest_logits
from select_rescue_track_blend import blend


def tree(value):return dict(nodes=[dict(feature=None,value=value)])


class TrackBlendTests(unittest.TestCase):
    def sources(self):
        base=dict(track_trees=[tree(.2)],track_bias=-.3)
        common=dict(track_feature_map='thermal-spectral-teacher33-v5',track_teacher_trees=1,track_teacher_bias=-.3,train_maps=['train'],valid_maps=['valid'])
        left=dict(common,track_trees=base['track_trees']+[tree(.7)],track_bias=-.1)
        right=dict(common,track_trees=base['track_trees']+[tree(-.6)],track_bias=.2)
        return base,left,right

    def test_shared_teacher_counted_once_and_exact_convex_logits(self):
        base,left,right=self.sources();before=copy.deepcopy((base,left,right));x=np.zeros((3,33))
        for weight in [0,.25,.5,.75,1]:
            model=blend(base,left,right,weight)
            np.testing.assert_allclose(forest_logits(model,x),weight*forest_logits(left,x)+(1-weight)*forest_logits(right,x),atol=1e-15)
            self.assertEqual(model['track_trees'][0],base['track_trees'][0])
        self.assertEqual((base,left,right),before)

    def test_mismatched_teacher_and_nonconvex_weights_rejected(self):
        base,left,right=self.sources();right['track_teacher_bias']=0
        with self.assertRaises(ValueError):blend(base,left,right,.5)
        base,left,right=self.sources();left['track_trees'][0]=tree(.5)
        with self.assertRaises(ValueError):blend(base,left,right,.5)
        for weight in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):blend(*self.sources(),weight)


if __name__=='__main__':unittest.main(verbosity=2)
