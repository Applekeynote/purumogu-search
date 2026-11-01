"""Offline split identity and complete-record reads; never sent to inference."""
import json
import hashlib
import re
from pathlib import Path

_WORLD=re.compile(r'^rescue-\d+(?:\.\d+)?-([EMHC]\d{2})-(\d+)(?:-N(?:24|36|48)-D\d+)?$')
_GENERIC_WORLD=re.compile(r'^(?:g\d+-)?(?:train|valid-known|valid-unknown|test-known|test-unknown|test-validation-families|test-supplement)-S(\d+)-W(\d+)-N(?:24|36|48)$')


def scene_index(code):
    if not re.fullmatch(r'[EMHC]\d{2}',code):raise ValueError('Unregistered scene code')
    number=int(code[1:])
    if code[0]=='C':
        if not 1<=number<=6:raise ValueError('Unregistered collapse scene')
        return 29+number
    if not 1<=number<=10:raise ValueError('Unregistered scene code')
    group={'E':0,'M':1,'H':2}[code[0]]
    return group*5+number-1 if number<=5 else 15+group*5+number-6


def world_group(map_id):
    generic=_GENERIC_WORLD.fullmatch(map_id)
    if generic:return f'rescue-generic-world-S{int(generic[1])}-W{generic[2]}'
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
    match=re.search(r'-N(24|36|48)(?:-D\d+)?$',row['map_id'])
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
    fields=['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma','bearing_likelihood']
    signature={key:model.get(key) for key in fields}
    signature['acoustic_probe']=model.get('acoustic_probe',False)
    signature['acoustic_peaks_v2']=model.get('acoustic_peaks_v2',False)
    signature['joint_acoustic_fusion']=model.get('joint_acoustic_fusion',False)
    signature['plane_registration_v1']=model.get('plane_registration_v1',False)
    signature['axis_plane_registration_v2']=model.get('axis_plane_registration_v2',False)
    signature['learned_navigation_v1']=model.get('learned_navigation_v1',False)
    signature['scout_inferred_routes_v1']=model.get('scout_inferred_routes_v1',False)
    signature['footprint_safe_routes_v1']=model.get('footprint_safe_routes_v1',False)
    signature['distributed_swarm']=model.get('distributed_swarm',False)
    signature['proposal_mode']=bool(model.get('track_weights') or model.get('track_trees') or model.get('distributed_swarm') or diagnostics)
    if model.get('learned_navigation_v1'):
        independent=bool(model.get('navigation_file'))
        signature['navigation_geometry']=dict(volume_sha256=model.get('navigation_sha256','') if independent else model.get('volume_sha256',''),channels=model.get('navigation_channels') if independent else model.get('volume_channels',2),bias=model.get('navigation_logit_bias',0) if independent else model.get('volume_logit_bias',0),range_tsdf_v1=False if independent else model.get('range_tsdf_v1',False),
            local_geometry_sha256=hashlib.sha256(json.dumps({k:model.get(k) for k in ['geometry_trees','geometry_feature_map','geometry_bias','local_surface_completion']},sort_keys=True).encode()).hexdigest())
    return signature


def acoustic_feature_signature(model):
    """A fixed sensor trajectory can compare heads without claiming changed-policy acceptance."""
    fields=['acoustic_weights','acoustic_feature_map','acoustic_window_frames','log_range_weights','log_range_sigma','bearing_likelihood','acoustic_peaks_v2','joint_acoustic_fusion','distributed_swarm','plane_registration_v1','axis_plane_registration_v2','local_surface_completion']
    return {key:model.get(key) for key in fields}


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
