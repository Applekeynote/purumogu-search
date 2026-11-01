"""World grouping must prevent paired node counts from leaking across splits."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rescue_dataset import assert_disjoint_worlds,complete_jsonl,control_signature,final_acquisition_reply,node_count,validate_paired_rows,world_group,scene_index


class RescueDatasetTests(unittest.TestCase):
    def test_frozen_exploration_signature_is_independent_of_map_decoder_updates(self):
        base=dict(learned_navigation_v1=True,local_surface_completion=True,geometry_trees=[],geometry_feature_map='',geometry_bias=0,
            volume_sha256='frozen',volume_channels=74,volume_logit_bias=.1,range_tsdf_v1=False)
        changed=dict(base,volume_sha256='updated-map',volume_channels=90,volume_logit_bias=.2,range_tsdf_v1=True,
            navigation_file='navigation.onnx',navigation_sha256='frozen',navigation_channels=74,navigation_logit_bias=.1)
        self.assertEqual(control_signature(base),control_signature(changed))
        changed['volume_sha256']='another-map'
        self.assertEqual(control_signature(base),control_signature(changed))
        changed['navigation_sha256']='changed-explorer'
        self.assertNotEqual(control_signature(base),control_signature(changed))

    def test_collapsed_layers_and_final_supplement_keep_paired_world_identity(self):
        a='rescue-3.5-C03-123032288-N24-D123735432'
        b='rescue-3.5-C03-123032288-N48-D123735433'
        self.assertEqual(world_group(a),world_group(b))
        a='g60-test-supplement-S32-W126032288-N24'
        b='g61-train-S32-W126032288-N48'
        self.assertEqual(world_group(a),world_group(b))
        with self.assertRaises(ValueError):assert_disjoint_worlds([a],[b])
        paired=[dict(map_id=f'g60-test-supplement-S32-W126032288-N{n}') for n in [24,36,48]]
        self.assertEqual(len(validate_paired_rows(paired)),1)
        self.assertEqual([scene_index(c) for c in ['E01','M01','H01','E06','M06','H10','C01','C06']],[0,5,10,15,20,29,30,35])
        with self.assertRaises(ValueError):scene_index('C07')
    def test_same_scene_seed_is_one_group_across_counts_and_generations(self):
        a='rescue-3.0-E02-20800000-N24-D30800000'
        b='rescue-3.0-E02-20800000-N48-D30890000'
        self.assertEqual(world_group(a),world_group(b))
        with self.assertRaises(ValueError):assert_disjoint_worlds([a],[b])
        with self.assertRaises(ValueError):assert_disjoint_worlds([a],['rescue-2.5-E02-20800000'])
        assert_disjoint_worlds([a],['rescue-3.0-E02-20900000-N24-D30900000'])

    def test_scene_identity_and_legacy_fixtures_are_preserved(self):
        assert_disjoint_worlds(['rescue-3.0-E01-20800000-N24-D30800000'],['rescue-3.0-H01-20800000-N48-D30900000'])
        self.assertEqual(world_group('fixture-train'),'fixture-train')
        with self.assertRaises(ValueError):world_group('rescue-3.0-E02-broken')

    def test_physical_generation_prefix_cannot_hide_the_same_world(self):
        a='g38-train-S25-W51025225-N24'
        b='valid-known-S25-W51025225-N48'
        self.assertEqual(world_group(a),world_group(b))
        with self.assertRaises(ValueError):assert_disjoint_worlds([a],[b])
        assert_disjoint_worlds([a],['g38-valid-known-S25-W52025225-N24'])

    def test_live_reader_waits_for_newline_and_rejects_complete_corruption(self):
        root=Path(__file__).resolve().parents[1]/'tmp';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root.resolve()))
            path=Path(folder)/'growing.jsonl'
            path.write_bytes(b'{"n":24}\n{"n":4')
            self.assertEqual(complete_jsonl(path),[{'n':24}])
            with path.open('ab') as stream:stream.write(b'8}\n')
            self.assertEqual(complete_jsonl(path),[{'n':24},{'n':48}])
            with path.open('ab') as stream:stream.write(b'corrupt\n')
            with self.assertRaises(ValueError):complete_jsonl(path)

    def test_bootstrap_and_fitted_distributed_model_keep_the_same_control_contract(self):
        base={'distributed_swarm':True,'acoustic_probe':True,'track_weights':[]}
        fitted={**base,'track_weights':[8.]*13,'fusion_weights':[1.,2.,3.,4.]}
        self.assertEqual(control_signature(base),control_signature(fitted))
        changed={**fitted,'acoustic_probe':False}
        self.assertNotEqual(control_signature(base),control_signature(changed))
        changed={**fitted,'footprint_safe_routes_v1':True}
        self.assertNotEqual(control_signature(base),control_signature(changed))
        self.assertEqual(node_count({'map_id':'rescue-3.0-E02-20800000-N48-D30800000'}),48)

    def test_acceptance_cannot_silently_omit_or_duplicate_a_node_count(self):
        rows=[{'map_id':f'rescue-3.0-E02-20800000-N{n}-D30800000'} for n in [24,36,48]]
        self.assertEqual(len(validate_paired_rows(rows)),1)
        with self.assertRaises(ValueError):validate_paired_rows(rows[:2])
        with self.assertRaises(ValueError):validate_paired_rows(rows+[rows[0]])
        rows[0]['world_group_id']='rescue-world-E02-20900000'
        with self.assertRaises(ValueError):validate_paired_rows(rows)

    def test_final_reply_requires_a_complete_final_map(self):
        root=Path(__file__).resolve().parents[1]/'tmp';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root.resolve()))
            path=Path(folder)/'audit.jsonl';path.write_bytes(b'{"protocol_version":"2.0","scans":[]}\n')
            self.assertIsNone(final_acquisition_reply(path))
            with path.open('ab') as stream:stream.write(b'{"protocol_version":"2.0","sim_step":18000,"cells":[{}]}\n')
            self.assertEqual(final_acquisition_reply(path,18000)['sim_step'],18000)
            self.assertIsNone(final_acquisition_reply(path,1000))


if __name__=='__main__':unittest.main(verbosity=2)
