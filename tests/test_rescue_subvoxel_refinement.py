"""Protect subcell spatial order and preserve the selected frozen backbone."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/deep-runtime')]
try:
    import torch
    from rescue_volume_network import lift_octants,fold_octants,build_network,extend_correction_inputs
except ImportError:torch=None

@unittest.skipIf(torch is None,'Optional volume-training dependencies required')
class SubvoxelRefinementTests(unittest.TestCase):
    def test_fine_phases_round_trip_without_losing_opposite_corners(self):
        fine=torch.arange(2*3*8*10*12).reshape(2,3,8,10,12).float()
        phases=fold_octants(fine)
        self.assertEqual(tuple(phases.shape),(2,24,4,5,6))
        self.assertTrue(torch.equal(lift_octants(phases,3),fine))
        fine.requires_grad_();fold_octants(fine).sum().backward()
        self.assertTrue(torch.equal(fine.grad,torch.ones_like(fine)))
        with self.assertRaises(ValueError):fold_octants(torch.zeros(1,1,7,8,8))

    def test_phase_extension_preserves_checkpoint_and_learns_without_truth_at_inference(self):
        torch.set_num_threads(2);torch.manual_seed(61)
        config=dict(width=4,channels=90,backbone_channels=90,profiles=False,fine_width=4,fine_supervision=True,freeze_backbone=False)
        old=build_network(dict(config,architecture='subvoxel-unet-refinement-v3')).eval()
        with torch.no_grad():old.correction[-1].weight.normal_(0,.1)
        model=build_network(dict(config,architecture='subvoxel-phase-refinement-v4')).eval()
        model.load_state_dict(extend_correction_inputs(old.state_dict(),model,8))
        # Counts and confidence weights are nonnegative in the sensor contract.
        x=torch.rand(1,90,8,8,8)*.05
        with torch.no_grad():self.assertLess((old(x)-model(x)).abs().max().item(),1e-6)
        self.assertIsNone(model.fine_logits)
        with torch.no_grad():model.correction[0].weight[:,-8:].normal_(0,.01)
        model.train();model(x).square().mean().backward()
        self.assertGreater(model.fine_out.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.fine_backbone.e0[0].weight.grad.abs().sum().item(),0)
        clone=build_network(dict(config,architecture='subvoxel-phase-refinement-v4')).eval()
        clone.load_state_dict(model.state_dict());model.eval()
        with torch.no_grad():self.assertTrue(torch.equal(model(x),clone(x)))
        with self.assertRaises(ValueError):build_network(dict(config,architecture='subvoxel-phase-refinement-v4',fine_supervision=False))

    def test_octants_are_spatial_neighbors_in_xyz_order(self):
        x=torch.zeros(1,16,2,3,4)
        for bx in range(2):
            for by in range(2):
                for bz in range(2):
                    octant=bx*4+by*2+bz
                    x[0,octant*2,1,2,3]=octant+1
                    x[0,octant*2+1,1,2,3]=100+octant
        fine=lift_octants(x,2)
        self.assertEqual(tuple(fine.shape),(1,2,4,6,8))
        for bx in range(2):
            for by in range(2):
                for bz in range(2):
                    octant=bx*4+by*2+bz
                    self.assertEqual(fine[0,0,2+bx,4+by,6+bz].item(),octant+1)
                    self.assertEqual(fine[0,1,2+bx,4+by,6+bz].item(),100+octant)

    def test_initial_prediction_and_backbone_are_preserved_while_correction_can_learn(self):
        torch.set_num_threads(2);torch.manual_seed(20261031)
        model=build_network(dict(architecture='subvoxel-refinement-v1',width=4,channels=90,backbone_channels=74,profiles=False,fine_width=4,normalization='batch')).eval()
        x=torch.randn(1,90,8,8,8)*.05
        with torch.no_grad():expected=model.backbone(x[:,:74]).clone();actual=model(x)
        self.assertTrue(torch.equal(expected,actual))
        before={k:v.clone() for k,v in model.backbone.state_dict().items()}
        model.train();self.assertFalse(model.backbone.training)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
        loss=model(x).square().mean();loss.backward();optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.backbone.state_dict().items()))
        with torch.no_grad():updated=model(x)
        self.assertGreater((updated-actual).abs().max().item(),1e-7)
        self.assertTrue(torch.isfinite(updated).all())

    def test_fine_plane_extension_preserves_a_nonzero_local_correction(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        config=dict(width=4,channels=90,backbone_channels=74,profiles=False,fine_width=4)
        old=build_network(dict(config,architecture='subvoxel-refinement-v1')).eval()
        with torch.no_grad():old.correction[-1].weight.normal_(0,.1)
        new=build_network(dict(config,architecture='subvoxel-plane-refinement-v2')).eval()
        previous=old.state_dict();extended=torch.zeros_like(new.state_dict()['correction.0.weight'])
        extended[:,:previous['correction.0.weight'].shape[1]]=previous['correction.0.weight']
        previous['correction.0.weight']=extended
        missing=new.load_state_dict(previous,strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all(k.startswith('fine_plane_profile.') for k in missing.missing_keys))
        x=torch.randn(1,90,8,8,8)*.05
        with torch.no_grad():a=old(x);b=new(x)
        self.assertLess((a-b).abs().max().item(),1e-6)

    def test_dense_auxiliary_loss_trains_shared_geometry_without_entering_inference(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=build_network(dict(architecture='subvoxel-plane-refinement-v2',width=4,channels=90,backbone_channels=74,profiles=False,fine_width=4,fine_supervision=True)).train()
        x=torch.randn(1,90,8,8,8)*.05;coarse=model(x)
        self.assertEqual(tuple(coarse.shape),(1,1,8,8,8))
        self.assertEqual(tuple(model.fine_logits.shape),(1,1,16,16,16))
        target=torch.zeros_like(model.fine_logits);target[:,:,:,7:9,:]=1
        torch.nn.functional.binary_cross_entropy_with_logits(model.fine_logits.float(),target.float()).backward()
        self.assertGreater(model.fine_out.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.fine_input.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.eval()
        with torch.no_grad():model(x)
        self.assertIsNone(model.fine_logits)

    def test_fine_unet_starts_at_the_backbone_and_can_learn_dense_geometry(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=build_network(dict(architecture='subvoxel-unet-refinement-v3',width=4,channels=90,backbone_channels=74,profiles=False,fine_width=4,fine_supervision=True)).train()
        x=torch.randn(1,90,8,8,8)*.05
        coarse=model(x)
        with torch.no_grad():expected=model.backbone(x[:,:74])
        self.assertTrue(torch.equal(coarse,expected))
        self.assertEqual(tuple(model.fine_logits.shape),(1,1,16,16,16))
        model.fine_logits.square().mean().backward()
        self.assertGreater(model.fine_backbone.e0[0].weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_explicit_backbone_adaptation_trains_and_survives_checkpoint_factory(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        config=dict(architecture='subvoxel-unet-refinement-v3',width=4,channels=90,backbone_channels=74,profiles=False,fine_width=4,freeze_backbone=False)
        model=build_network(config).train();x=torch.randn(1,90,8,8,8)*.05
        self.assertTrue(model.backbone.training)
        model(x).square().mean().backward()
        self.assertGreater(model.backbone.e0[0].weight.grad.abs().sum().item(),0)
        clone=build_network(config).eval();clone.load_state_dict(model.state_dict())
        model.eval()
        with torch.no_grad():self.assertTrue(torch.equal(model(x),clone(x)))

if __name__=='__main__':unittest.main(verbosity=2)
