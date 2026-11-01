"""The optimized suffix path must match Mapper and obey causal deadlines."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np

class TsdfPreparationTests(unittest.TestCase):
    def test_suffix_matches_full_mapper_and_cannot_consume_future_frames(self):
        folder=Path(os.environ.get('PURUMOGU_TSDF_BINARIES',ROOT/'tmp/generic-tsdf-rust-target/release'))
        suffix='.exe' if os.name=='nt' else ''
        full=folder/('prepare_geometry'+suffix);band=folder/('prepare_tsdf_band'+suffix)
        if not full.exists() or not band.exists():self.skipTest('Build native preparation binaries')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            root=Path(temporary);model=root/'model.json'
            # This is a sensor-feature contract test, not a trained-model test.
            # A clean checkout must not depend on an unpublished research model.
            model.write_text(json.dumps(dict(version='rescue-trained-v1',cell_m=.05,
                acoustic_weights=[0,1,0,0,0],occupancy_weights=[0,1,0],fusion_weights=[1,1,1,1],
                prototypes=[],data_sha256='',train_maps=['fixture-train'],valid_maps=['fixture-valid'],
                volume_channels=90,range_tsdf_v1=True,plane_registration_v1=True,axis_plane_registration_v2=True,
                local_surface_completion=True)))
            packets=[]
            for seq,step,distance in [(0,25,.2),(1,50,.7)]:
                packets.append(dict(protocol_version='2.0',run_id='tsdf-causal',seq_no=seq,sim_step=step,cell_m=.05,
                    scans=[dict(node_id='N00',deployment_anchor=[0,.025,0],odometry_delta=[0,0,0],pose_sigma_m=.001,
                    rays=[dict(direction=[0,0,1],distance_m=distance,hit=True)],bearing=[0,0,1],bearing_sigma_deg=10,
                    acoustic_snr_db=5,voice_band_ratio=.6,modulation=.6,temperature_delta_c=.2)]))
            audit=root/'packets.jsonl';audit.write_text('\n'.join(json.dumps(p) for p in packets)+'\n')
            for binary,target,shots,end in [(full,'full','25','50'),(band,'band','25','50'),(band,'early','-','25')]:
                result=subprocess.run([str(binary),str(model),str(audit),str(root/target),shots,end],capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
            for prefix in ['', 'step-25/']:
                expected=np.fromfile(root/'full'/prefix/'moments-final.f32',dtype='<f4').reshape(-1,88)[:,72:]
                actual=np.fromfile(root/'band'/prefix/'tsdf.f32',dtype='<f4').reshape(-1,16)
                np.testing.assert_array_equal(actual,expected)
            self.assertEqual((root/'early/tsdf.f32').read_bytes(),(root/'band/step-25/tsdf.f32').read_bytes())
            self.assertNotEqual((root/'early/tsdf.f32').read_bytes(),(root/'band/tsdf.f32').read_bytes())
            result=subprocess.run([str(band),str(model),str(audit),str(root/'missing'),'-','75'],capture_output=True)
            self.assertNotEqual(result.returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
