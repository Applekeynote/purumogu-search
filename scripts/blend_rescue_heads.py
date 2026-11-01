"""Blend fitted head logits without changing the native feature or control contract."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from rescue_dataset import acoustic_feature_signature,assert_disjoint_worlds

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--new',required=True)
    p.add_argument('--new-weight',type=float,required=True);p.add_argument('--name',required=True)
    p.add_argument('--research-tree-budget',type=int,default=256);a=p.parse_args()
    if not 1<=a.research_tree_budget<=512:raise ValueError('Bounded research tree count required')
    if not 0<a.new_weight<1:raise ValueError('Use a proper mixture')
    paths=[ROOT/a.base,ROOT/a.new];models=[json.loads(p.read_bytes()) for p in paths]
    if acoustic_feature_signature(models[0])!=acoustic_feature_signature(models[1]):raise ValueError('Feature extraction must be identical')
    if any(m['track_feature_map']!=models[0]['track_feature_map'] or not m['track_trees'] for m in models):raise ValueError('Compatible fitted tree heads required')
    model=deepcopy(models[0]);model['track_trees']=[];model['track_bias']=0;model['track_weights']=[]
    for source,weight in zip(models,[1-a.new_weight,a.new_weight]):
        model['track_bias']+=source['track_bias']*weight
        for tree in deepcopy(source['track_trees']):
            for node in tree['nodes']:node['value']*=weight
            model['track_trees'].append(tree)
    if len(model['track_trees'])>a.research_tree_budget:raise ValueError('Head tree budget exceeded')
    model['train_maps']=sorted(set(models[0]['train_maps'])|set(models[1]['train_maps']))
    validation=set(models[0]['valid_maps'])|set(models[1]['valid_maps']);overlap=validation&set(model['train_maps'])
    if any('-valid-' in i or '-test-' in i or i.startswith(('valid-','test-')) for i in overlap):raise ValueError('Formal evaluation ID appears in fitting lineage')
    model['valid_maps']=sorted(validation-set(model['train_maps']))
    assert_disjoint_worlds(model['train_maps'],model['valid_maps'])
    model['data_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in paths)+str(a.new_weight).encode()).hexdigest()
    out=ROOT/'models'/a.name
    if out.exists():raise ValueError('Use an immutable model name')
    out.mkdir(parents=True);(out/'model.json').write_text(json.dumps(model,separators=(',',':'))+'\n')
    (out/'development.json').write_text(json.dumps(dict(purpose='Logit blend of TRAIN-fitted heads; no new label fitting',test_opened=False,
        source_models=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],new_weight=a.new_weight,tree_count=len(model['track_trees']),native_256_tree_compatible=len(model['track_trees'])<=256),indent=2)+'\n')


if __name__=='__main__':main()
