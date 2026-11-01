"""Protect TRAIN/VALID isolation and the time-specific 120/360-second scores."""
import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_rescue_generic import registered_extra_cases
from evaluate_rescue_closed_loop import evaluate
import train_rescue_generic
from train_rescue_generic_tracks import snapshot_weights


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data),encoding='utf-8')


class LearningProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def collection(self,name,identifier,split='train'):
        root=self.root/name
        write(root/'collection.json',dict(status='complete',split=split,plan=dict(cases=[dict(id=identifier)])))
        write(root/identifier/'complete.json',dict(id=identifier,split=split))
        return root

    def test_only_newest_registered_training_cohort_gets_onpolicy_weight(self):
        a=self.collection('old','g38-train-S25-W51025225-N24')
        b=self.collection('new','g40-train-S25-W71025225-N36')
        rows=list(registered_extra_cases([a,b]))
        self.assertEqual([row[1]['id'] for row in rows],['g38-train-S25-W51025225-N24','g40-train-S25-W71025225-N36'])
        self.assertEqual([row[3] for row in rows],[False,True])

    def test_validation_or_duplicate_trajectory_cannot_enter_training(self):
        a=self.collection('train','g40-train-S25-W71025225-N24')
        b=self.collection('duplicate','g40-train-S25-W71025225-N24')
        with self.assertRaises(ValueError):list(registered_extra_cases([a,b]))
        valid=self.collection('valid','g41-valid-known-S25-W82025225-N24','valid-known')
        with self.assertRaises(ValueError):list(registered_extra_cases(valid))
        write(a/'collection.json',dict(status='complete',split='train',plan=dict(cases=[dict(id='missing')])))
        with self.assertRaises(ValueError):list(registered_extra_cases(a))

    def test_long_training_world_uses_matching_final_input_and_labels(self):
        identifier='g44-train-S25-W91025225-N24';root=self.collection('long',identifier)
        write(root/'collection.json',dict(status='complete',split='train',steps=18000,plan=dict(cases=[dict(id=identifier)])))
        write(root/identifier/'complete.json',dict(id=identifier,split='train',scene='H06'))
        write(root/identifier/'truth-6000.json',dict(checkpoint='early'))
        write(root/identifier/'truth-18000.json',dict(checkpoint='late'))
        legacy=self.root/'experiments/raw/rescue/swarm-v31/grids-train.jsonl';legacy.parent.mkdir(parents=True);legacy.write_text('')
        with patch.object(train_rescue_generic,'ROOT',self.root):rows=list(train_rescue_generic.cases([root]))
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]['map'].name,rows[0]['evaluation_step'],rows[0]['truth']['checkpoint']),('map-18000.json',18000,'late'))

    def test_more_snapshots_do_not_count_as_more_independent_worlds(self):
        records=[dict(group='short',domain='train',final_step=6000) for _ in range(4)]+[dict(group='long',domain='train',final_step=18000) for _ in range(8)]
        weights=snapshot_weights(records)
        self.assertAlmostEqual(weights['short']*4,.5)
        self.assertAlmostEqual(weights['long']*8,.5)
        weights=snapshot_weights(records,.25)
        self.assertAlmostEqual(weights['short']*4,.75)
        self.assertAlmostEqual(weights['long']*8,.25)
        with self.assertRaises(ValueError):snapshot_weights([dict(group='validation',domain='valid-known',final_step=18000)])

    def test_legacy_filename_alias_does_not_override_actual_sensor_time(self):
        raw=self.root/'experiments/raw/rescue/swarm-v31';record=dict(run_id='legacy-run',map_id='legacy-train',world_group_id='legacy-world',scene='E01',occupied_keys=[])
        write(raw/'sensor-maps/legacy-run.json',dict(sim_step=18000,cells=[]))
        (raw/'grids-train.jsonl').write_text(json.dumps(record)+'\n')
        with patch.object(train_rescue_generic,'ROOT',self.root):rows=list(train_rescue_generic.cases())
        self.assertEqual((rows[0]['evaluation_step'],rows[0]['legacy_prepared_alias_step']),(18000,6000))

    def test_early_report_uses_early_geometry_people_and_node_labels(self):
        root=self.root/'mission';case=dict(id='g41-valid-known-S00-W82000000-N24',scene=0,nodes=24)
        folder=root/case['id']
        write(root/'collection.json',dict(status='complete',split='valid-known',steps=18000,plan=dict(cases=[case])))
        write(folder/'complete.json',dict(case=case,truth_to_server=False,scene='E01'))
        for step,key,position in [(6000,[0,0,0],[0,0,0]),(18000,[1,0,0],[1,0,0])]:
            write(folder/f'map-{step}.json',dict(cells=[dict(key=key,occupancy=.99)],cues=[dict(position=position)]))
            write(folder/f'truth-{step}.json',dict(occupied_keys=[key],victim_positions=[position]))
        (folder/'labels.jsonl').write_text('\n'.join(json.dumps(dict(step=s,robots=[dict(released=True,lost=s==18000,position=[0,0,0])])) for s in [6000,18000])+'\n')
        early=evaluate(root,6000);final=evaluate(root)
        self.assertEqual((early['seconds'],early['summary']['macro_iou'],early['summary']['matched'],early['rows'][0]['lost_nodes']),(120.,1.,1,0))
        self.assertEqual((final['seconds'],final['summary']['macro_iou'],final['summary']['matched'],final['rows'][0]['lost_nodes']),(360.,1.,1,1))
        self.assertFalse(early['acceptance_deadline']);self.assertTrue(final['acceptance_deadline'])
        for step in [120,19500]:
            with self.assertRaises(ValueError):evaluate(root,step)
        write(root/'collection.json',dict(status='complete',split='test',steps=18000))
        with self.assertRaises(ValueError):evaluate(root)

    def test_native_preparation_deadline_does_not_consume_later_sensor_frames(self):
        binary=Path(os.environ.get('PURUMOGU_GEOMETRY_PREPARER',ROOT/'rust/target/release'/('prepare_geometry.exe' if os.name=='nt' else 'prepare_geometry')))
        if not binary.exists():self.skipTest('Build native prepare_geometry or set PURUMOGU_GEOMETRY_PREPARER')
        model=self.root/'model.json'
        write(model,dict(version='rescue-trained-v1',cell_m=.05,acoustic_weights=[0,1,0,0,0],occupancy_weights=[0,1,0],fusion_weights=[1,1,1,1],prototypes=[],data_sha256='',train_maps=['fixture-train'],valid_maps=['fixture-valid'],volume_channels=74,local_surface_completion=True))
        packets=[]
        for seq,step,distance in [(0,25,.2),(1,50,.7)]:
            packets.append(dict(protocol_version='2.0',run_id='causal-preparation',seq_no=seq,sim_step=step,cell_m=.05,scans=[dict(node_id='N00',deployment_anchor=[0,.025,0],odometry_delta=[0,0,0],pose_sigma_m=.001,rays=[dict(direction=[0,0,1],distance_m=distance,hit=True)],bearing=[0,0,1],bearing_sigma_deg=10,acoustic_snr_db=5,voice_band_ratio=.6,modulation=.6,temperature_delta_c=.2)]))
        audit=self.root/'packets.jsonl';audit.write_text('\n'.join(json.dumps(p) for p in packets)+'\n')
        full=self.root/'full';early=self.root/'early';truncated=self.root/'truncated'
        first=self.root/'first.jsonl';first.write_text(json.dumps(packets[0])+'\n')
        for source,target,snapshots,deadline in [(audit,full,'25','50'),(audit,early,'-','25'),(first,truncated,'-','25')]:
            result=subprocess.run([str(binary),str(model),str(source),str(target),snapshots,deadline],capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
        expected=(full/'step-25/moments-final.f32').read_bytes()
        self.assertEqual(expected,(early/'moments-final.f32').read_bytes())
        self.assertEqual(expected,(truncated/'moments-final.f32').read_bytes())
        self.assertEqual(json.loads((early/'map-25.json').read_bytes())['sim_step'],25)
        self.assertFalse((early/'map-50.json').exists())
        self.assertNotEqual(expected,(full/'moments-final.f32').read_bytes())
        result=subprocess.run([str(binary),str(model),str(audit),str(self.root/'missing'),'-','75'],capture_output=True)
        self.assertNotEqual(result.returncode,0)


if __name__=='__main__':unittest.main(verbosity=2)
