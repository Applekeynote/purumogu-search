import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/learning-runtime')]
import numpy as np
from rescue_neighbor_features import extend,extend_teacher


def base():return dict(track_bias=-.2,track_trees=[dict(nodes=[dict(feature=0,threshold=.5,left=1,right=2,value=0),dict(feature=None,value=-1),dict(feature=None,value=1)])])
def proposal(tag,point,signal,heat=.4,voice=.5,age=.2):
    features=[0.]*32;features[0]=signal;features[16]=heat;features[27]=voice;features[6]=age
    return dict(tag=tag,position=point,features=features,first_seen_step=0,last_seen_step=100)


class NeighborFeatureTests(unittest.TestCase):
    def test_relative_context_keeps_prefix_and_is_translation_yaw_and_order_invariant(self):
        rows=[proposal('a',[0,.1,0],.2),proposal('b',[.2,.1,0],.9,.8,.7),proposal('c',[.4,.3,.1],.7,.6,.9),proposal('d',[-.8,.2,0],.4)]
        before=copy.deepcopy(rows);expected=extend(base(),rows,100)
        self.assertTrue(all(len(q['features'])==45 for q in expected));self.assertEqual(rows,before)
        for old,new in zip(rows,expected):self.assertEqual(old['features'],new['features'][:32])
        shifted=[dict(q,position=[v+d for v,d in zip(q['position'],[.33,.1,-.7])]) for q in rows]
        turned=[dict(q,position=[-q['position'][2],q['position'][1],q['position'][0]]) for q in rows]
        for changed in [shifted,turned,list(reversed(rows))]:
            actual={q['tag']:q['features'] for q in extend(base(),changed,100)}
            for q in expected:np.testing.assert_allclose(actual[q['tag']],q['features'],atol=1e-12,rtol=0)

    def test_isolated_and_competing_hypotheses_have_explicit_finite_defaults(self):
        one=proposal('a',[0,0,0],.2,.4,.5,.2);result=extend(base(),[one],100)[0]['features'][33:]
        np.testing.assert_allclose(result,[-.12,-.12,0,.5,.4,-.12,1,0,0,0,0,.2],atol=1e-14)
        rival=proposal('b',[.4,0,0],.9,.8,.9)
        result=extend(base(),[one,rival],100)[0]['features'][33:]
        self.assertAlmostEqual(result[5],.08);self.assertAlmostEqual(result[6],.4/.65)
        self.assertAlmostEqual(result[8],-.2);self.assertAlmostEqual(result[9],-.4);self.assertAlmostEqual(result[10],-.4)

    def test_future_or_wrong_prefix_is_not_silently_accepted(self):
        self.assertEqual(extend(base(),[],100),[])
        q=proposal('a',[0,0,0],.2);q['last_seen_step']=125
        with self.assertRaises(ValueError):extend(base(),[q],100)
        q=proposal('a',[0,0,0],.2)
        with self.assertRaises(ValueError):extend_teacher([q],100)


if __name__=='__main__':unittest.main(verbosity=2)
