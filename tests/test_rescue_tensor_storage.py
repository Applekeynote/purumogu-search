"""Streaming must preserve every observation and protect cached inputs."""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/deep-runtime'),str(ROOT/'tmp/learning-runtime')]
try:
    import numpy as np
    import torch
    from rescue_tensor_storage import disk_pair
except ImportError:torch=None

@unittest.skipIf(torch is None,'Optional training dependencies required')
class TensorStorageTests(unittest.TestCase):
    def test_on_demand_tensor_is_exact_and_updates_cannot_change_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            x=np.random.default_rng(42).normal(size=(3,4,5)).astype(np.float32);y=(x[0]>0).astype(np.uint8)
            a,b=disk_pair(directory,'a'*64,x,y)
            tx=a.to('cpu');ty=b.to('cpu')
            np.testing.assert_array_equal(tx.numpy(),x);np.testing.assert_array_equal(ty.numpy(),y)
            tx.zero_();ty.zero_()
            np.testing.assert_array_equal(a.to('cpu').numpy(),x)
            np.testing.assert_array_equal(b.to('cpu').numpy(),y)

if __name__=='__main__':unittest.main(verbosity=2)
