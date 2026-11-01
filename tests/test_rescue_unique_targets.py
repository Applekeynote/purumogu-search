import copy
from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'), str(Path(__file__).resolve().parents[1] / 'tmp/learning-runtime')]
from rescue_unique_targets import targets


class UniqueTargetsTests(unittest.TestCase):
    def row(self, positions, people):
        return dict(domain='train', held=False, proposals=[dict(position=[x, 0, 0]) for x in positions], victim_positions=[[x, 0, 0] for x in people])

    def test_global_matching_preserves_two_nearby_sources(self):
        # The first candidate fits both people; the second fits only person 0.
        # A greedy highest-score assignment would miss person 1.
        row = self.row([.1, -.1, 2], [0, .3]); before = copy.deepcopy(row)
        y, w, audit = targets(row, [.99, .8, .9], .2)
        self.assertEqual(y.tolist(), [1, 1, 0]); self.assertAlmostEqual(sum(w), .2)
        self.assertEqual(audit['represented_people'], 2); self.assertEqual(row, before)

    def test_aliases_ignored_only_next_to_representative(self):
        row = self.row([.01, .02, .2, 2], [0])
        y, w, audit = targets(row, [.9, .8, .7, .99], 1)
        self.assertEqual(y.tolist(), [1, 0, 0, 0]); self.assertEqual(w.tolist(), [.5, 0, .25, .25])
        self.assertEqual(audit['ignored'], 1)
        row['victim_positions'] = []
        y, w, audit = targets(row, [.9, .8, .7, .99], 1)
        self.assertEqual(y.sum(), 0); self.assertEqual(w.tolist(), [.25] * 4)
        row['proposals'] = []
        self.assertEqual(len(targets(row, [], 1)[0]), 0)

    def test_held_out_truth_cannot_supply_training_labels(self):
        for role in ['valid-known', 'valid-unknown', 'test']:
            row = self.row([0], [0]); row['domain'] = role
            with self.assertRaises(ValueError): targets(row, [.5], 1)
        row = self.row([0], [0]); row['held'] = True
        with self.assertRaises(ValueError): targets(row, [.5], 1)
        for scores in [[float('nan')], [1.1], []]:
            with self.assertRaises(ValueError): targets(self.row([0], [0]), scores, 1)


if __name__ == '__main__': unittest.main(verbosity=2)
