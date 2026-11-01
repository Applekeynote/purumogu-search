import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import merge_rescue_validation_geometry as union


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


class PreparedUnionTests(unittest.TestCase):
    def test_union_excludes_training_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as temporary:
            root=Path(temporary);profile=root/'profile.json';write(profile,dict(volume_channels=90))
            raw=root/'experiments/raw/rescue';sources=[]
            for label in ['old','new']:
                pool=raw/label;sources.append(pool)
                identifiers=[]
                for role in ['train','valid-known','valid-unknown']:
                    identifier=label+'-'+role;identifiers.append(identifier)
                    write(pool/role/identifier/'complete.json',dict(id=identifier));(pool/role/identifier/'features.bin').write_bytes(b'unchanged-sensor-features')
                write(pool/'manifest.json',dict(status='complete',test_opened=False,case_ids=identifiers,lineage=dict(model=str(profile),model_sha256=union.sha(profile))))
            with patch.object(union,'ROOT',root):
                output=raw/'merged';union.merge(sources,output,4)
                result=json.loads((output/'manifest.json').read_bytes())
                self.assertEqual(result['cases'],4);self.assertFalse((output/'train').exists())
                self.assertEqual((output/'valid-known/old-valid-known/features.bin').read_bytes(),b'unchanged-sensor-features')
                with self.assertRaises(ValueError):union.merge([sources[0],sources[0]],raw/'duplicate',4)
                record=json.loads((sources[1]/'manifest.json').read_bytes());record['test_opened']=True;write(sources[1]/'manifest.json',record)
                with self.assertRaises(ValueError):union.merge(sources,raw/'test-pool',4)


if __name__=='__main__':unittest.main(verbosity=2)
