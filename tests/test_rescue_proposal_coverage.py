import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_rescue_proposal_coverage import oracle_match,audit


class ProposalCoverageTests(unittest.TestCase):
    def test_exact_matching_does_not_assign_one_proposal_twice_or_use_greedy(self):
        people=[[0,0,0],[.3,0,0]]
        self.assertEqual(oracle_match([dict(position=[.15,0,0])],people)[0],1)
        self.assertEqual(oracle_match([dict(position=[.15,0,0]),dict(position=[-.1,0,0])],people)[0],2)
        self.assertEqual(oracle_match([dict(position=[.25,0,0])],[[0,0,0]])[0],0)

    def test_large_duplicate_pool_stays_exact_and_test_is_rejected(self):
        proposals=[dict(position=[0,0,0]) for _ in range(500)]+[dict(position=[1,0,0])]
        self.assertEqual(oracle_match(proposals,[[0,0,0],[1,0,0]])[0],2)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temp:
            path=Path(temp)/'plan.json';path.write_text(json.dumps(dict(split='test-unknown')))
            with self.assertRaises(ValueError):audit(path)


if __name__=='__main__':unittest.main(verbosity=2)
