import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tmp/learning-runtime'),str(ROOT/'tmp/deep-runtime')]
import numpy as np
from prepare_rescue_joint_subcell import translate_packets,fine_tensor,simulator_precision_plan
from rescue_subcell_training import views,choose_training_views


class JointSubcellTests(unittest.TestCase):
    def test_json_round_trip_only_normalizes_coordinate_precision_not_conditions(self):
        a=dict(cases=[dict(seed=1,views=[dict(translation_m=[.0077459122591791996,0,0])])],fine_supervision=True)
        b=copy.deepcopy(a);b['cases'][0]['views'][0]['translation_m'][0]=.0077459122591792
        self.assertEqual(simulator_precision_plan(a),simulator_precision_plan(b))
        b['cases'][0]['seed']=2
        self.assertNotEqual(simulator_precision_plan(a),simulator_precision_plan(b))
        b=copy.deepcopy(a);b['cases'][0]['views'][0]['translation_m'][0]+=.00001
        self.assertNotEqual(simulator_precision_plan(a),simulator_precision_plan(b))

    def test_batch_replaces_only_the_selected_views_matching_fine_target(self):
        import torch
        class First:
            def random(self):return 0.
            def choice(self,values):return values[0]
        zero=torch.zeros(1,2,2,2);one=torch.ones_like(zero)
        chosen=[(dict(id='shift'),zero,zero),(dict(id='time'),zero,zero)]
        fine=torch.stack([zero.clone(),one.clone()*3]);available=torch.tensor([0.,1.])
        result=choose_training_views(chosen,{'shift':[(one,one,one*2)]},{'time':[(one*4,one*5)]},1,'cpu',fine,available,First())
        self.assertTrue(torch.equal(result[0][1],one));self.assertTrue(torch.equal(result[0][2],one))
        self.assertTrue(torch.equal(fine[0],one*2));self.assertTrue(torch.equal(fine[1],one*3))
        self.assertTrue(torch.equal(result[1][1],one*4));self.assertTrue(torch.equal(result[1][2],one*5))
        self.assertEqual(available.tolist(),[1,1]);self.assertTrue(torch.equal(chosen[0][1],zero))
        with self.assertRaises(ValueError):choose_training_views(chosen,{'shift':[(one,one,None)]},{},1,'cpu',fine,available,First())

    def test_coordinate_change_preserves_every_sensor_and_odometry_value(self):
        root=ROOT/'tmp';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            source=Path(folder)/'input.jsonl';target=Path(folder)/'translated.jsonl'
            scans=[dict(node_id='N00',deployment_anchor=[.1,.2,.3],odometry_delta=[.01,-.02,.03],pose_sigma_m=.027,
                rays=[dict(direction=[0,0,1],distance_m=.4,hit=True)],bearing=[0,1,0],temperature_delta_c=.123,voice_band_ratio=.7,acoustic_snr_db=-2)]
            packets=[dict(sim_step=step,scans=copy.deepcopy(scans),seq_no=i) for i,step in enumerate([25,50,75])]
            source.write_text(''.join(json.dumps(p)+'\n' for p in packets));translate_packets(source,target,[.012,-.018,.022],50)
            actual=[json.loads(line) for line in target.read_text().splitlines()]
            expected=copy.deepcopy(packets[:2])
            for p in expected:p['scans'][0]['deployment_anchor']=[v+d for v,d in zip([.1,.2,.3],[.012,-.018,.022])]
            self.assertEqual(actual,expected)
            self.assertEqual([json.loads(line) for line in source.read_text().splitlines()],packets)
            with self.assertRaises(ValueError):translate_packets(source,target,[.025,0,0],50)

    def fixture(self,directory,fine=True):
        target=directory/'view-0';target.mkdir()
        (target/'map-18000.json').write_text('{"sim_step":18000}')
        (target/'truth-18000.json').write_text('{"occupied_keys":[[0,0,0]]}')
        np.savez_compressed(target/'range-features.npz',features=np.zeros((88,1,1,1),dtype=np.float32))
        digest=lambda name:hashlib.sha256((target/name).read_bytes()).hexdigest()
        view=dict(index=0,translation_m=[.01,0,0],directory='view-0',evaluation_step=18000,
            map_sha256=digest('map-18000.json'),truth_sha256=digest('truth-18000.json'),features_sha256=digest('range-features.npz'))
        if fine:
            y=np.zeros((1,64,48,144),dtype=np.uint8);y[0,2,3,4]=1;np.savez_compressed(target/'fine-truth.npz',y=y)
            view.update(fine_labels='view-0/fine-truth.npz',fine_labels_sha256=digest('fine-truth.npz'))
        return dict(views=[view])

    def test_shifted_coarse_and_fine_targets_are_kept_together_and_audited(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            directory=Path(folder);record=self.fixture(directory)
            chosen=list(views(directory,record,True))[0]
            self.assertEqual(chosen['evaluation_step'],18000);self.assertEqual(chosen['truth']['occupied_keys'],[[0,0,0]])
            self.assertEqual(chosen['fine'].sum(),1);self.assertEqual(chosen['fine'][0,2,3,4],1)
            (directory/'view-0/truth-18000.json').write_text('{"occupied_keys":[]}')
            with self.assertRaises(ValueError):list(views(directory,record,True))

    def test_original_fine_labels_cannot_substitute_for_a_translated_view(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            directory=Path(folder);record=self.fixture(directory,False)
            self.assertIsNone(list(views(directory,record,False))[0]['fine'])
            with self.assertRaises(ValueError):list(views(directory,record,True))
            record['views'][0]['directory']='../outside'
            with self.assertRaises(ValueError):list(views(directory,record,False))

    def test_dense_truth_retains_the_declared_metric_grid(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            source=Path(folder)/'fine.json';data=dict(shape=[64,42,140],minimum_key=[-32,-6,-12],cell_m=.025,occupied_keys=[[-32,-6,-12],[31,35,127]])
            source.write_text(json.dumps(data));y=fine_tensor(source)
            self.assertEqual(y.shape,(1,64,48,144));self.assertEqual(y.sum(),2);self.assertEqual(y[0,0,0,0],1)
            data['occupied_keys'].append([32,0,0]);source.write_text(json.dumps(data))
            with self.assertRaises(ValueError):fine_tensor(source)


if __name__=='__main__':unittest.main(verbosity=2)
