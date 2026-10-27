"""VALID-only diagnostic. Oracle labels here are NEVER used by the product/model."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    model=json.loads((ROOT/'models/rescue-v4/model.json').read_text(encoding='utf-8'))
    prototypes={p['family']:p for p in model['prototypes']};rows=[]
    for line in (ROOT/'experiments/raw/rescue/grids-valid.jsonl').open(encoding='utf-8'):
        truth=json.loads(line);path=ROOT/f"experiments/raw/rescue/{truth['run_id']}.jsonl"
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-8388608));reply=json.loads(stream.read().splitlines()[-1])
        actual={tuple(k) for k in truth['occupied_keys']};seen={tuple(c['key']) for c in reply['cells']}
        prior=prototypes[truth['scene']]['occupancy'];prediction=set()
        for x in range(-16,16):
            for y in range(-3,18):
                for z in range(-6,64):
                    k=(x,y,z);i=((x+16)*21+y+3)*70+z+6
                    # Explicit oracle: perfect family + perfect observed labels only.
                    if (k in seen and k in actual) or (k not in seen and prior[i]>.6):prediction.add(k)
        intersection=len(prediction&actual);union=len(prediction|actual)
        rows.append({'scene':truth['scene'],'oracle_family_and_observed_labels_iou':intersection/max(1,union),'unobserved_true_occupied':len(actual-seen),'truth_occupied':len(actual),'seen':len(seen)})
    result={'diagnostic_only':True,'not_a_model_score':True,'oracle_labels_never_sent_to_server':True,'scope':'Perfect family identity and observed-cell labels, TRAIN prior for unseen cells; measures remaining completion uncertainty, not a universal upper bound.','mean_oracle_iou':sum(r['oracle_family_and_observed_labels_iou'] for r in rows)/len(rows),'rows':rows}
    (ROOT/'reports/rescue/diagnostic-oracle-valid.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
