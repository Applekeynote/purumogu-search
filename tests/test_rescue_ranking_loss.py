from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'tmp/learning-runtime')]
import numpy as np
from rescue_ranking_loss import derivatives


class RankingLossTests(unittest.TestCase):
    def test_pairs_push_positive_above_alias_without_changing_query_intercept(self):
        z=np.array([-.5,2.,-2.]);y=np.array([1,0,0]);w=np.array([.5,.25,.25])
        base,h0=derivatives(z,y,w,[3],0);gradient,hessian=derivatives(z,y,w,[3],1)
        self.assertLess(gradient[0],base[0]);self.assertGreater(gradient[1],base[1]);self.assertAlmostEqual(gradient.sum(),base.sum())
        self.assertTrue(np.all(hessian>=h0));self.assertGreater(gradient[1]-base[1],gradient[2]-base[2])

    def test_queries_are_independent_and_ignored_alias_gets_no_pair_gradient(self):
        left=derivatives([0,1],[1,0],[.5,.5],[2])
        joined=derivatives([0,1,-10,20],[1,0,1,0],[.5,.5,100,100],[2,2])
        np.testing.assert_allclose(left[0],joined[0][:2]);np.testing.assert_allclose(left[1],joined[1][:2])
        gradient,hessian=derivatives([0,1,2],[1,0,0],[.5,.5,0],[3])
        self.assertEqual(gradient[2],0);self.assertEqual(hessian[2],0)
        negative=derivatives([0,1],[0,0],[.5,.5],[2])
        anchor=derivatives([0,1],[0,0],[.5,.5],[2],0)
        np.testing.assert_allclose(negative,anchor)

    def test_ties_and_permutations_do_not_change_training_pairs(self):
        z=np.array([0.,1.,1.,-.5]);y=np.array([1,0,0,0]);w=np.array([.5,.2,.2,.1]);order=np.array([2,0,3,1])
        expected=derivatives(z,y,w,[4],top_negative=1);actual=derivatives(z[order],y[order],w[order],[4],top_negative=1)
        for a,b in zip(expected,actual):np.testing.assert_allclose(a[order],b)
        with self.assertRaises(ValueError):derivatives(z,y,w,[3])


if __name__=='__main__':unittest.main(verbosity=2)
