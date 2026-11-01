import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rescue_person_weights import snapshot_weights


class PersonWeightTests(unittest.TestCase):
    def row(self):return dict(domain='train',held=False,victim_positions=[[0,0,0],[1,0,0]],proposals=[dict(position=[x,0,0]) for x in [.01,.02,.99,2]])

    def test_dense_strong_source_cannot_dilute_another_person(self):
        row=self.row();before=copy.deepcopy(row);weights=snapshot_weights(row,.2,.25,.5)
        self.assertAlmostEqual(sum(weights),.2);self.assertAlmostEqual(sum(weights[:2]),.05)
        self.assertAlmostEqual(weights[2],.05);self.assertAlmostEqual(weights[3],.1)
        expanded=copy.deepcopy(row);expanded['proposals']=row['proposals'][:2]*10+row['proposals'][2:]
        more=snapshot_weights(expanded,.2,.25,.5)
        self.assertAlmostEqual(sum(more[:20]),.05);self.assertAlmostEqual(more[-2],.05);self.assertAlmostEqual(more[-1],.1)
        self.assertEqual(row,before)

    def test_negative_and_positive_only_snapshots_keep_the_world_budget(self):
        row=self.row();row['victim_positions']=[]
        self.assertEqual(snapshot_weights(row,1,.25,.5),[.25]*4)
        row=self.row();row['proposals']=row['proposals'][:3]
        self.assertEqual(snapshot_weights(row,1,.25,.5),[.25,.25,.5])
        row['proposals']=[];self.assertEqual(snapshot_weights(row,1,.25,.5),[])

    def test_evaluation_worlds_and_invalid_weight_budgets_are_rejected(self):
        for role in ['valid-known','valid-unknown','test']:
            row=self.row();row['domain']=role
            with self.assertRaises(ValueError):snapshot_weights(row,1,.25,.5)
        row=self.row();row['held']=True
        with self.assertRaises(ValueError):snapshot_weights(row,1,.25,.5)
        for mass in [0,1,float('nan')]:
            with self.assertRaises(ValueError):snapshot_weights(self.row(),1,.25,mass)


if __name__=='__main__':unittest.main(verbosity=2)
