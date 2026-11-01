import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rescue_validation_data import validation_cases


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class ValidationDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def cohort(self,name):
        root=self.root/name
        for split,scene in [('valid-known',30),('valid-unknown',34)]:
            case=dict(id=f'{name}-{split}-S{scene}-W1-N24',scene=scene,seed=1,nodes=24)
            write(root/split/'collection.json',dict(status='complete',split=split,steps=18000,plan=dict(cases=[case])))
            folder=root/split/case['id']
            write(folder/'complete.json',dict(id=case['id'],split=split,case=case,truth_to_server=False))
            for step in [6000,18000]:
                write(folder/f'map-{step}.json',dict(sim_step=step));write(folder/f'truth-{step}.json',dict(occupied_keys=[]))
        return root

    def test_multiple_complete_valid_cohorts_keep_time_and_roles(self):
        a=self.cohort('old');b=self.cohort('new');rows=list(validation_cases([a,b],6000))
        self.assertEqual(len(rows),4);self.assertEqual({r['source'] for r in rows},{'valid-known','valid-unknown'})
        self.assertTrue(all(r['map'].name=='map-6000.json' for r in rows))

    def test_incomplete_mislabelled_and_duplicate_cohorts_are_rejected(self):
        root=self.cohort('valid');identifier=next((root/'valid-known').glob('*/complete.json')).parent.name
        with self.assertRaises(ValueError):list(validation_cases([root],18000,[identifier]))
        with self.assertRaises(ValueError):list(validation_cases([root,root],18000))
        path=root/'valid-known/collection.json';data=json.loads(path.read_bytes());data['status']='partial';write(path,data)
        with self.assertRaises(ValueError):list(validation_cases([root],18000))
        data['status']='complete';data['split']='train';write(path,data)
        with self.assertRaises(ValueError):list(validation_cases([root],18000))


if __name__=='__main__':unittest.main(verbosity=2)
