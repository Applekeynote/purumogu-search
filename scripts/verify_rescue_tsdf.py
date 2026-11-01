"""Compare the optimized corpus suffix with the actual full Mapper path."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/learning-runtime'))
import numpy as np

def main():
    corpus=ROOT/'experiments/raw/rescue/generalization-v46-tsdf/registered-mixed'
    selected=[next((corpus/'train').glob('g40-train-S26-*/complete.json')),
              corpus/'valid-unknown/g41-valid-unknown-S29-W83029261-N24/complete.json']
    rows=[]
    for source in selected:
        record=json.loads(source.read_bytes());step=record['evaluation_step'];target=source.parent
        snapshots=[r['step'] for r in record['intermediate']]
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='tsdf-full-parity-') as temporary:
            full=Path(temporary)
            subprocess.run([str(ROOT/'tmp/generic-tsdf-rust-target/release/prepare_geometry.exe'),
                str(ROOT/'models/generic-v46-tsdf-preparation/model.json'),str(ROOT/record['source_audit']),str(full),
                ','.join(map(str,snapshots)) if snapshots else '-',str(step)],check=True,capture_output=True)
            maximum=0.;cell_equal=True
            for sub,end in [('',step)]+[(f'step-{s}',s) for s in snapshots]:
                expected=np.fromfile(full/sub/'moments-final.f32',dtype='<f4').reshape(32,21,70,88).transpose(3,0,1,2)
                with np.load(target/sub/'range-features.npz') as data:actual=data['features']
                maximum=max(maximum,float(np.max(np.abs(expected-actual))))
                np.testing.assert_array_equal(expected,actual)
                lhs=json.loads((full/sub/f'map-{end}.json').read_bytes())['cells']
                rhs=json.loads((target/sub/f'map-{end}.json').read_bytes())['cells']
                cell_equal &= lhs==rhs
            if not cell_equal:raise AssertionError('Original occupancy changed')
            rows.append(dict(id=record['id'],checkpoints=len(snapshots)+1,maximum_feature_difference=maximum,original_cells_equal=cell_equal,
                             sensor_sha256=record['source_sha256']))
    result=dict(status='pass',purpose='Optimized TSDF suffix equals full Mapper input; not accuracy acceptance',truth_to_native=False,test_opened=False,rows=rows)
    out=ROOT/'reports/rescue/generic-v35/tsdf-native-composition-parity.json'
    if out.exists():raise ValueError('Immutable parity report already exists')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
