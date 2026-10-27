"""Join complete TRAIN proposal collections; retain the fitted geometry unchanged."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',action='append',required=True);parser.add_argument('--name',required=True);parser.add_argument('--base',required=True);parser.add_argument('--output-model',required=True);args=parser.parse_args()
    base=ROOT/args.base;model=json.loads(base.read_bytes());original_train=list(model['train_maps'])
    destination=ROOT/'experiments/raw/rescue'/args.name/'track-proposals';destination.mkdir(parents=True,exist_ok=False)
    ids=set();rows=[];sources=[];proposal_hash=None
    for name in args.source:
        source=ROOT/'experiments/raw/rescue'/name/'track-proposals';record=ROOT/'reports/rescue'/name/'train'
        collection=json.loads((record/'manifest.json').read_text(encoding='utf-8-sig'));manifest=json.loads((source/'manifest-train.json').read_bytes())
        assert collection['status']=='complete' and collection['steps']==18000 and collection['split']=='train'
        assert manifest['split']=='train' and manifest['maps']==collection['expected_maps']==len(manifest['rows'])
        if proposal_hash is None:proposal_hash=manifest['model_sha256']
        assert manifest['model_sha256']==proposal_hash,'Candidate-generating model must be identical'
        for entry in manifest['rows']:
            path=source/(entry['run_id']+'.json');row=json.loads(path.read_bytes())
            assert row['map_id']==entry['map_id'] and row['split']=='train' and row['audit_sha256']==entry['audit_sha256']
            assert row['map_id'] not in ids and row['map_id'] not in model['valid_maps'];ids.add(row['map_id'])
            shutil.copyfile(path,destination/path.name);rows.append(dict(**entry,proposal_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        sources.append(dict(experiment=name,manifest_sha256=hashlib.sha256((source/'manifest-train.json').read_bytes()).hexdigest(),rollout=json.loads((record/'rollout-provenance.json').read_text(encoding='utf-8-sig'))))
    assert set(original_train)<=ids
    rows.sort(key=lambda r:r['map_id'])
    manifest=dict(split='train',maps=len(ids),model_sha256=proposal_hash,rows=rows,source_collections=sources)
    (destination/'manifest-train.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    model['train_maps']=sorted(ids);output=ROOT/'models'/args.output_model;output.mkdir(exist_ok=False)
    data=json.dumps(model,separators=(',',':')).encode();(output/'model.json').write_bytes(data)
    (output/'SHA256SUMS').write_text(hashlib.sha256(data).hexdigest()+'  model.json\n',encoding='ascii')
    report=dict(note='All sensor/geometry coefficients unchanged. Only TRAIN ID metadata extended before fitting the candidate classifier.',parent_model_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(data).hexdigest(),candidate_training_maps=len(ids),sensor_and_geometry_training_maps=len(original_train),sensor_and_geometry_train_map_ids=original_train,valid_maps=len(model['valid_maps']),test_accessed=False,source_collections=sources)
    (output/'training-subsets.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Merged',len(ids),'TRAIN proposal maps; sensor and geometry coefficients unchanged')

if __name__=='__main__':main()
