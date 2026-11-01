from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from collect_rescue_layered_pilot import pilot_report


class PilotScopeTests(unittest.TestCase):
    def test_good_partial_strata_cannot_be_labelled_full_acceptance(self):
        original=dict(all_95=True,summary=dict(macro_iou=1))
        result=pilot_report(original)
        self.assertNotIn('all_95',result);self.assertFalse(result['full_acceptance']);self.assertTrue(result['observed_groups_all_95'])
        self.assertIn('all_95',original)


if __name__=='__main__':unittest.main(verbosity=2)
