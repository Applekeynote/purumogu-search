"""Select an illustrative run with a formation change, independent of correctness."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
truth=[json.loads(line) for line in (root/'datasets/generated/v1/comparison-truth.jsonl').read_text().splitlines()]
truth={t['run_id']:t for t in truth if t['policy']=='ghost_flow'}
for path in sorted((root/'experiments/raw').glob('session-*.jsonl')):
    with path.open() as f:first=f.readline()
    if not first:continue
    run=json.loads(first)['message']['run_id']
    if run not in truth:continue
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    actions=[r['message'] for r in rows if r['direction']=='out' and r['message']['type']=='action']
    if any(a['formation_mode']=='CHAIN_INSERT' for a in actions):
        result={'selection_rule':'First measured GHOST-FLOW run containing a chain command; prediction correctness not considered. Demo is separate from evaluation.', 'truth':truth[run], 'actions':actions}
        (root/'media/demo-selection.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2));break
else:raise RuntimeError('No measured chain action found')
