from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rescue_map_context import MapContext


class MapContextTests(unittest.TestCase):
    def floor(self):return [dict(key=[x,-1,z],observations=4,sensor_occupancy=.99,occupancy=.01) for x in [-1,0,1] for z in [-1,0,1]]

    def test_uses_measured_occupancy_and_ignores_filled_cells(self):
        cells=self.floor();point=[.025,.075,.025];expected=MapContext(cells).features(point)
        self.assertEqual(expected[4],1);self.assertAlmostEqual(expected[6],2/7)
        extra=[dict(key=[x,1,z],observations=0,occupancy=1,sensor_occupancy=1) for x in [-1,0,1] for z in [-1,0,1]]
        self.assertEqual(MapContext(cells+extra).features(point),expected)
        self.assertEqual(MapContext(cells).features([.025,.425,.025])[6],1)

    def test_voxel_translation_and_gravity_preserving_yaw_keep_context(self):
        cells=self.floor();point=[.025,.075,.025];expected=MapContext(cells).features(point)
        moved=[dict(c,key=[c['key'][i]+[4,2,3][i] for i in range(3)]) for c in cells]
        self.assertEqual(MapContext(moved).features([point[i]+[.2,.1,.15][i] for i in range(3)]),expected)
        turned=[dict(c,key=[-c['key'][2]-1,c['key'][1],c['key'][0]]) for c in cells]
        self.assertEqual(MapContext(turned).features([-point[2],point[1],point[0]]),expected)

    def test_missing_measured_channel_is_rejected(self):
        with self.assertRaises(ValueError):MapContext([dict(key=[0,0,0],observations=1,occupancy=.9)])


if __name__=='__main__':unittest.main(verbosity=2)
