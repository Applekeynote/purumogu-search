"""Offline split identity and complete-record reads; never sent to inference."""
import json
import re
from pathlib import Path

_WORLD=re.compile(r'^rescue-\d+(?:\.\d+)?-([EMH]\d{2})-(\d+)(?:-N(?:24|36|48)-D\d+)?$')


def world_group(map_id):
    match=_WORLD.fullmatch(map_id)
    if match:return f'rescue-world-{match[1]}-{match[2]}'
    if map_id.startswith('rescue-3.'):
        raise ValueError(f'Invalid generation-3 map ID: {map_id}')
    return map_id  # Historical fixtures without a world seed.


def assert_disjoint_worlds(left,right):
    overlap={world_group(i) for i in left}&{world_group(i) for i in right}
    if overlap:raise ValueError(f'World-group leakage across splits: {sorted(overlap)}')


def validate_model_splits(model):
    fitted=set(model['train_maps'])|set(model.get('calibration_train_maps',[]))
    assert_disjoint_worlds(fitted,model['valid_maps'])


def node_count(row):
    if 'nodes' in row:return int(row['nodes'])
    match=re.search(r'-N(24|36|48)-D',row['map_id'])
    return int(match[1]) if match else 12


def validate_paired_rows(rows):
    ids=[row['map_id'] for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate map IDs')
    groups={}
    for row in rows:
        group=world_group(row['map_id'])
        if row.get('world_group_id',group)!=group:raise ValueError('World group does not match the map ID')
        groups.setdefault(group,[]).append(node_count(row))
    if not groups or any(sorted(counts)!=[24,36,48] for counts in groups.values()):
        raise ValueError('Each world must have exactly one 24-, 36- and 48-node case')
    return groups


def control_signature(model,diagnostics=False):
    fields=['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma']
    signature={key:model.get(key) for key in fields}
    signature['acoustic_probe']=model.get('acoustic_probe',False)
    signature['distributed_swarm']=model.get('distributed_swarm',False)
    signature['proposal_mode']=bool(model.get('track_weights') or model.get('track_trees') or model.get('distributed_swarm') or diagnostics)
    return signature


def complete_jsonl(path):
    """Ignore only an in-progress, unterminated final record; reject corrupt lines."""
    path=Path(path)
    if not path.exists():return []
    with path.open('rb') as stream:
        return [json.loads(line) for line in stream if line.endswith(b'\n')]


def final_acquisition_reply(path,steps=None):
    """A flushed final reply also proves that the audit writer finished its case."""
    path=Path(path)
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-8_388_609));tail=stream.read()
    if not tail.endswith(b'\n'):return None
    reply=json.loads(tail.splitlines()[-1])
    if reply.get('protocol_version')!='2.0' or 'scans' in reply or not reply.get('cells'):return None
    if steps is not None and reply['sim_step']!=steps:return None
    return reply
