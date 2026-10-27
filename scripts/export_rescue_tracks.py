"""Extract completed diagnostic rollouts for offline TRAIN/VALID track fitting."""
from pathlib import Path
import argparse,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw',required=True);parser.add_argument('--report',required=True);parser.add_argument('--split',choices=['train','valid'],required=True);args=parser.parse_args()
    raw=ROOT/args.raw;report=ROOT/args.report;manifest=json.loads((report/'manifest.json').read_text(encoding='utf-8-sig'));provenance=json.loads((report/'rollout-provenance.json').read_text(encoding='utf-8-sig'))
    assert manifest['status']=='complete' and provenance['diagnostics'] and manifest['split']==args.split
    truth_path=raw/f'grids-{args.split}.jsonl';ground=[json.loads(line) for line in truth_path.read_text(encoding='utf-8-sig').splitlines()]
    assert len(ground)==manifest['expected_maps']
    out=raw/'track-proposals';out.mkdir(exist_ok=True);rows=[]
    for row in ground:
        assert row['split']==args.split
        audit=raw/f"{row['run_id']}.jsonl"
        with audit.open('rb') as stream:
            stream.seek(max(0,audit.stat().st_size-8_388_609));data=stream.read().splitlines()[-1]
        reply=json.loads(data);assert reply['sim_step']==manifest['steps'] and reply['cells'] and 'proposals' in reply
        proposals=reply['proposals']
        for p in proposals:
            distance=min((math.dist(p['position'],v) for v in row['victim_positions']),default=None)
            p['distance_to_person_m']=distance;p['label']=int(distance is not None and distance<.25)
        result=dict(map_id=row['map_id'],run_id=row['run_id'],scene=row['scene'],split=args.split,victim_positions=row['victim_positions'],proposals=proposals,
                    audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),model_sha256=provenance['model_sha256'],server_sha256=provenance['server_sha256'])
        (out/f"{row['run_id']}.json").write_text(json.dumps(result,separators=(',',':'))+'\n',encoding='utf-8',newline='\n');rows.append(dict(map_id=row['map_id'],run_id=row['run_id'],audit_sha256=result['audit_sha256']))
    result=dict(split=args.split,maps=len(rows),truth_sha256=hashlib.sha256(truth_path.read_bytes()).hexdigest(),model_sha256=provenance['model_sha256'],server_sha256=provenance['server_sha256'],rows=rows)
    (out/f'manifest-{args.split}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print('EXPORTED',args.split,len(rows))

if __name__=='__main__':main()
