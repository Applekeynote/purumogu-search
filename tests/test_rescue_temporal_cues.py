import copy
from pathlib import Path
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts'),str(Path(__file__).resolve().parents[1]/'tmp/learning-runtime')]
from rescue_temporal_cues import TemporalCues


def cue(x,p=.9):return dict(position=[x,0,0],score=p)


class TemporalCuesTests(unittest.TestCase):
    def test_two_nearby_sources_keep_separate_support_per_snapshot(self):
        tracker=TemporalCues(.2);first=[cue(0),cue(.3)];before=copy.deepcopy(first)
        tracker.update(1500,first);tracker.update(3000,[cue(.12),cue(-.08)])
        self.assertEqual(sorted(t['support'] for t in tracker.tracks),[2,2]);self.assertEqual(first,before)
        self.assertEqual(len(tracker.confirmed(nms=.1)),2)

    def test_old_hypotheses_expire_and_future_or_duplicate_steps_rejected(self):
        tracker=TemporalCues(.2,1500);tracker.update(1500,[cue(0)])
        self.assertEqual(tracker.confirmed(),[])
        with self.assertRaises(ValueError):tracker.update(1500,[cue(0)])
        tracker.update(3000,[cue(.01)]);self.assertEqual(tracker.confirmed()[0]['support'],2)
        tracker.update(5000,[]);self.assertEqual(tracker.confirmed(),[])

    def test_same_snapshot_order_cannot_change_association(self):
        a=TemporalCues();b=TemporalCues()
        for step,cues in [(1500,[cue(0),cue(.5)]),(3000,[cue(.03),cue(.48)]),(4500,[cue(1),cue(.51),cue(.02)])]:
            a.update(step,cues);b.update(step,list(reversed(cues)))
        self.assertEqual(a.confirmed(),b.confirmed())

    def test_current_observation_can_appear_without_waiting_for_confirmation(self):
        tracker=TemporalCues()
        for step in [1500,3000,4500]:tracker.update(step,[cue(0)])
        tracker.update(6000,[cue(1,.8)])
        self.assertEqual(len(tracker.confirmed(3,.6)),1)
        combined=tracker.confirmed(3,.6,include_current=True)
        self.assertEqual(len(combined),2);self.assertEqual(sorted(c['support'] for c in combined),[1,3])


if __name__=='__main__':unittest.main(verbosity=2)
