from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'tmp/deep-runtime'),str(Path(__file__).resolve().parents[1]/'tmp/learning-runtime')]
from calibrate_rescue_volume_train import best_train_bias


class TrainInterceptTests(unittest.TestCase):
    def row(self,identifier,group,values):return dict(id=identifier,group=group,held=False,source='physical-3.5-train',ious=values)

    def test_duplicate_snapshots_cannot_change_world_balance(self):
        rows=[self.row('g47-train-S25-W1-N24','a',[.9,.8]),self.row('g47-train-S25-W2-N24','b',[.7,.9])]
        chosen,_=best_train_bias(rows,[0,.1]);more=[rows[0]]*10+[rows[1]]
        self.assertEqual(best_train_bias(more,[0,.1])[0],chosen);self.assertEqual(chosen['bias'],.1)

    def test_validation_labels_cannot_fit_bias_and_zero_preferred_on_tie(self):
        row=self.row('g47-train-S25-W1-N24','a',[.9,.9,.9])
        self.assertEqual(best_train_bias([row],[-.1,0,.1])[0]['bias'],0)
        for role in ['valid-known','valid-unknown','test-known']:
            row=dict(row,source=role)
            with self.assertRaises(ValueError):best_train_bias([row],[-.1,0,.1])
        row=dict(row,source='train',held=True)
        with self.assertRaises(ValueError):best_train_bias([row],[-.1,0,.1])


if __name__=='__main__':unittest.main(verbosity=2)
