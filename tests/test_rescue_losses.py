"""Check the IoU surrogate's boundary, extremes and useful gradient."""
import math
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/deep-runtime')]
try:
    import torch
    from rescue_losses import occupancy_lovasz,area_sampling_mass,area_risk_weights
except ImportError:torch=None

@unittest.skipIf(torch is None,'Optional volume-training dependencies required')
class OccupancyLossTests(unittest.TestCase):
    def test_area_risk_corrects_the_actual_mixture_not_raw_case_counts(self):
        areas=['common','common','rare'];sampling=[.1,.1,.8]
        mass=area_sampling_mass(areas,sampling)
        weights=area_risk_weights(areas,mass,dict(common=.1,rare=.1))
        self.assertAlmostEqual(sum(p*w for p,w in zip(sampling,weights)),1)
        self.assertAlmostEqual(sum(p*w for a,p,w in zip(areas,sampling,weights) if a=='common'),.5)
        self.assertAlmostEqual(sum(p*w for a,p,w in zip(areas,sampling,weights) if a=='rare'),.5)
        focused=area_risk_weights(areas,mass,dict(common=.1,rare=.3))
        self.assertGreater(sampling[-1]*focused[-1],.5)
        self.assertAlmostEqual(sum(p*w for p,w in zip(sampling,focused)),1)
    def test_fixed_point_six_boundary_and_extreme_two_cell_cases(self):
        offset=math.log(.6/.4);labels=torch.tensor([[1.,0.]])
        self.assertEqual(occupancy_lovasz(torch.tensor([[2.+offset,-2.+offset]]),labels).item(),0.)
        self.assertEqual(occupancy_lovasz(torch.tensor([[-2.+offset,2.+offset]]),labels).item(),3.)
        self.assertEqual(occupancy_lovasz(torch.full((1,2),offset),labels).item(),1.)

    def test_gradient_improves_wrong_predictions_and_empty_world_is_finite(self):
        scores=torch.tensor([[0.,1.]],requires_grad=True);labels=torch.tensor([[1.,0.]])
        loss=occupancy_lovasz(scores,labels).sum();loss.backward()
        self.assertLess(scores.grad[0,0].item(),0.)
        self.assertGreater(scores.grad[0,1].item(),0.)
        improved=occupancy_lovasz(scores.detach()-.1*scores.grad,labels).sum()
        self.assertLess(improved.item(),loss.item())
        empty=occupancy_lovasz(torch.zeros(2,5),torch.zeros(2,5))
        self.assertTrue(torch.isfinite(empty).all())

if __name__=='__main__':unittest.main(verbosity=2)
