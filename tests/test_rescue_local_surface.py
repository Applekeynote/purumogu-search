"""A local shape model must not silently depend on the whole building."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/deep-runtime')]
try:
    import torch
    from rescue_volume_network import build_network
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Install the optional volume-training dependencies')
class LocalSurfaceTests(unittest.TestCase):
    def test_only_the_declared_local_context_can_change_a_voxel(self):
        torch.set_num_threads(2);torch.manual_seed(20261031)
        model=build_network(dict(architecture='local-surface-v1',width=8,residual=True,channels=74)).eval()
        x=torch.randn(1,74,16,16,16)*.1;far=x.clone();far[:,:,0,0,0]+=3
        near=x.clone();near[:,:,8,8,9]+=3
        with torch.no_grad():a=model(x);b=model(far);c=model(near)
        self.assertEqual(tuple(a.shape),(1,1,16,16,16))
        self.assertEqual(a[0,0,8,8,8].item(),b[0,0,8,8,8].item())
        self.assertGreater(abs(a[0,0,8,8,8].item()-c[0,0,8,8,8].item()),1e-7)


if __name__=='__main__':unittest.main(verbosity=2)
