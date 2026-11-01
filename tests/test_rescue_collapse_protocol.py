"""Guard meaningful voids and held-composition isolation for partial collapse."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_rescue_generic import registered_extra_cases
from plan_rescue_collapse import TRAIN,VALID_COMPOSITIONS


class CollapseProtocolTests(unittest.TestCase):
    def test_held_collapse_cannot_be_relabelled_as_training(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            root=Path(temp);identifier='g58-train-S34-W123034306-N24'
            case=dict(id=identifier,scene=34)
            (root/identifier).mkdir()
            (root/'collection.json').write_text(json.dumps(dict(status='complete',split='train',plan=dict(cases=[case]))))
            (root/identifier/'complete.json').write_text(json.dumps(dict(id=identifier,split='train',case=case)))
            with self.assertRaisesRegex(ValueError,'Held geometry'):list(registered_extra_cases(root))

    def test_train_and_new_compositions_are_disjoint(self):
        self.assertFalse(set(TRAIN)&set(VALID_COMPOSITIONS))
        self.assertEqual(set(TRAIN)|set(VALID_COMPOSITIONS),set(range(30,36)))


if __name__=='__main__':unittest.main(verbosity=2)
