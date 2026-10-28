"""Export the Rust proposals already recorded at acquisition, without a second rollout."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from rescue_dataset import complete_jsonl,final_acquisition_reply,world_group

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True);parser.add_argument('--report',required=True)
    parser.add_argument('--split',choices=['train','valid'],required=True)
    parser.add_argument('--watch-expected',type=int,default=0)
    args=parser.parse_args();raw=ROOT/args.raw;report=ROOT/args.report
    provenance=json.loads((report/'rollout-provenance.json').read_text(encoding='utf-8-sig'))
    assert provenance['diagnostics'] and provenance['generation'] in ['rescue-3.0','rescue-3.1'] and provenance['split']==args.split
    model_path=ROOT/provenance['model'];model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert model_hash==provenance['model_sha256']
    out=raw/'track-proposals';out.mkdir(exist_ok=True);completed={}
    ground_path=raw/f'grids-{args.split}.jsonl'
    while True:
        ground=complete_jsonl(ground_path)
        assert len({r['map_id'] for r in ground})==len(ground)
        for row in ground:
            if row['run_id'] in completed:continue
            assert row['split']==args.split and row['world_group_id']==world_group(row['map_id'])
            audit=raw/f"{row['run_id']}.jsonl"
            reply=final_acquisition_reply(audit,provenance['steps'])
            if reply is None:continue
            assert 'proposals' in reply and reply['distributed_swarm'] and reply['run_id']==row['run_id']
            proposals=reply['proposals']
            for item in proposals:
                distance=min((math.dist(item['position'],person) for person in row['victim_positions']),default=None)
                item['distance_to_person_m']=distance;item['label']=int(distance is not None and distance<.25)
            result=dict(map_id=row['map_id'],world_group_id=row['world_group_id'],nodes=row['nodes'],deployment_seed=row['deployment_seed'],run_id=row['run_id'],scene=row['scene'],split=args.split,victim_positions=row['victim_positions'],proposals=proposals,audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),model_sha256=model_hash,server_sha256=provenance['server_sha256'])
            (out/f"{row['run_id']}.json").write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
            completed[row['run_id']]=result
            print('ACQUIRED_PROPOSALS',row['map_id'],len(proposals),sum(p['label'] for p in proposals),flush=True)
        manifest=json.loads((report/'manifest.json').read_text(encoding='utf-8-sig'))
        expected=args.watch_expected or manifest['expected_maps']
        if len(completed)==expected and manifest['status']=='complete':break
        if not args.watch_expected:raise RuntimeError('Acquisition has not completed')
        time.sleep(2)
    rows=[completed[r['run_id']] for r in ground]
    metadata=dict(split=args.split,maps=len(rows),truth_sha256=hashlib.sha256(ground_path.read_bytes()).hexdigest(),model_sha256=model_hash,server_sha256=provenance['server_sha256'],method='Final diagnostic proposals from the actual Rust acquisition replies; no replay and no new physics',rows=[dict(map_id=r['map_id'],run_id=r['run_id'],audit_sha256=r['audit_sha256']) for r in rows])
    (out/f'manifest-{args.split}.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
