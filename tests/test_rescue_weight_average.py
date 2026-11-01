from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'tmp/deep-runtime')]
import torch
from rescue_weight_average import interpolate


class WeightAverageTests(unittest.TestCase):
    def test_convex_weights_preserve_endpoints_and_original_tensors(self):
        left=dict(weight=torch.tensor([1.,3.]),fixed=torch.tensor([2]));right=dict(weight=torch.tensor([5.,7.]),fixed=torch.tensor([2]))
        for weight in [0,.25,.5,.75,1]:
            result=interpolate(left,right,weight)
            torch.testing.assert_close(result['weight'],torch.tensor([1+4*weight,3+4*weight],dtype=torch.float32));result['weight'][0]=99
            self.assertEqual(left['weight'][0],1);self.assertEqual(right['weight'][0],5)

    def test_mismatched_shapes_buffers_and_nonfinite_parameters_rejected(self):
        with self.assertRaises(ValueError):interpolate(dict(x=torch.zeros(2)),dict(x=torch.zeros(3)),.5)
        with self.assertRaises(ValueError):interpolate(dict(x=torch.tensor([1])),dict(x=torch.tensor([2])),.5)
        with self.assertRaises(ValueError):interpolate(dict(x=torch.tensor([float('nan')])),dict(x=torch.zeros(1)),.5)
        for weight in [-.1,1.1,float('nan')]:
            with self.assertRaises(ValueError):interpolate({}, {}, weight)


if __name__=='__main__':unittest.main(verbosity=2)
