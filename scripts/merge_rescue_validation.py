"""Join complete VALID collections, retaining all source provenance and audit hashes."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',action='append',required=True);parser.add_argument('--name',required=True);parser.add_argument('--model',default='models/rescue-v12-geometry/model.json');parser.add_argument('--output-model',default='rescue-v14-geometry-source');args=parser.parse_args()
    destination=ROOT/'experiments/raw/rescue'/args.name;report=ROOT/'reports/rescue'/args.name/'valid';destination.mkdir(parents=True,exist_ok=False);report.mkdir(parents=True,exist_ok=False);(destination/'sensor-maps').mkdir()
    ground=[];outcomes=[];provenance=[];sensor_rows=[];ids=set()
    for name in args.source:
        source=ROOT/'experiments/raw/rescue'/name;record=ROOT/'reports/rescue'/name/'valid';manifest=json.loads((record/'manifest.json').read_text(encoding='utf-8-sig'));assert manifest['status']=='complete' and manifest['steps']==18000 and manifest['split']=='valid'
        source_ground=[json.loads(line) for line in (source/'grids-valid.jsonl').read_text(encoding='utf-8-sig').splitlines()];assert len(source_ground)==manifest['expected_maps']
        sensor_manifest=json.loads((source/'sensor-maps/manifest-valid.json').read_bytes());assert sensor_manifest['maps']==len(source_ground) and sensor_manifest['model_applied'] is False
        for row in source_ground:
            assert row['map_id'] not in ids;ids.add(row['map_id']);ground.append(row)
            audit=row['run_id']+'.jsonl';shutil.copyfile(source/audit,destination/audit)
            for suffix in ['.json','.meta.json']:shutil.copyfile(source/'sensor-maps'/(row['run_id']+suffix),destination/'sensor-maps'/(row['run_id']+suffix))
        outcomes.extend(json.loads((record/'outcomes.json').read_text(encoding='utf-8-sig')));sensor_rows.extend(sensor_manifest['rows'])
        provenance.append(dict(experiment=name,manifest=manifest,manifest_sha256=hashlib.sha256((record/'manifest.json').read_bytes()).hexdigest(),grids_sha256=hashlib.sha256((source/'grids-valid.jsonl').read_bytes()).hexdigest(),rollout=json.loads((record/'rollout-provenance.json').read_text(encoding='utf-8-sig'))))
    assert {r['map_id'] for r in outcomes}==ids
    ground.sort(key=lambda r:r['map_id']);outcomes.sort(key=lambda r:r['map_id'])
    (destination/'grids-valid.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in ground),encoding='utf-8',newline='\n')
    (destination/'sensor-maps/manifest-valid.json').write_text(json.dumps(dict(split='valid',maps=len(ground),model_applied=False,rows=sensor_rows),indent=2)+'\n',encoding='utf-8',newline='\n')
    manifest=dict(status='complete',generation='rescue-2.5',split='valid',steps=18000,seconds=360,nodes=12,controller_revision='progress-3cm-v2',expected_maps=len(ground),source_collections=[r['experiment'] for r in provenance])
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n');(report/'outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n',encoding='utf-8',newline='\n');(report/'source-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8',newline='\n')
    base=ROOT/args.model;model=json.loads(base.read_bytes());assert not set(model['train_maps'])&ids;model['valid_maps']=sorted(ids)
    out=ROOT/'models'/args.output_model;out.mkdir(exist_ok=False);data=json.dumps(model,separators=(',',':')).encode();(out/'model.json').write_bytes(data);(out/'SHA256SUMS').write_text(hashlib.sha256(data).hexdigest()+'  model.json\n',encoding='ascii')
    (out/'selection-report.json').write_text(json.dumps(dict(note='VALID ID metadata extended only; all fitted coefficients unchanged.',parent_model_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(data).hexdigest(),valid_maps=len(ids),test_accessed_for_fitting=False),indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Merged',len(ground),'VALID maps; TRAIN weights unchanged')

if __name__=='__main__':main()
