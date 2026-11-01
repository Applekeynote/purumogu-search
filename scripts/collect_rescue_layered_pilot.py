"""Six-case closed-loop development pilot; explicitly not full acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from collect_rescue_registered import hidden,write_once
from rescue_dataset import assert_disjoint_worlds

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cases(prefix):
    result={'valid-known':[],'valid-unknown':[]};sources=[]
    for parent,scene in [('generalization-v41-mission360/valid-known.json',25),('generalization-v41-mission360/valid-unknown.json',29),('generalization-v60-partial-collapse/valid-known.json',33)]:
        path=ROOT/'experiments'/parent;plan=json.loads(path.read_bytes());split=plan['split']
        selected=[case for case in plan['cases'] if case['scene']==scene and case['nodes'] in [24,48]]
        if len(selected)!=2:raise ValueError('Expected paired 24/48-node development worlds')
        sources.append(dict(plan=str(path.relative_to(ROOT)),sha256=sha(path)))
        for case in selected:result[split].append(dict(case,id=f"{prefix}-{split}-S{case['scene']:02d}-W{case['seed']}-N{case['nodes']}"))
    return result,sources


def pilot_report(report):
    report=dict(report)
    report['observed_groups_all_95']=report.pop('all_95')
    report.update(purpose='Targeted six-case closed-loop pilot, excluding 36 nodes; not full protocol acceptance',full_acceptance=False)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--id-prefix',required=True)
    p.add_argument('--server',default='tmp/generic-tsdf-rust-target/release/rescue_server.exe');a=p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',a.name) or not re.fullmatch(r'g\d+',a.id_prefix):raise ValueError('Plain experiment identifiers required')
    model=ROOT/a.model;descriptor=json.loads(model.read_bytes());cases,sources=select_cases(a.id_prefix)
    assert_disjoint_worlds([case['id'] for rows in cases.values() for case in rows],descriptor['train_maps'])
    folder=ROOT/'experiments'/a.name
    geometry={path.name:sha(path) for path in (ROOT/'unity/Assets/Scripts').glob('Rescue*.cs')}
    acquisition={path.relative_to(ROOT).as_posix():sha(path) for path in [ROOT/'unity/Assets/Editor/RescueGeneralization.cs',ROOT/'unity/Assets/Editor/ProjectTools.cs',ROOT/'scripts/collect_rescue_registered.py',Path(__file__).resolve()]}
    common=dict(version=2,steps=18000,inference_map_interval=100,geometry_revision='layered-partial-collapse-variable-storeys-v2',geometry_sources=geometry,acquisition_sources=acquisition,
                model=a.model,server=a.server,model_sha256=sha(model),server_sha256=sha(ROOT/a.server),sensor_source_sha256=sha(ROOT/'unity/Assets/Scripts/RescueRobot.cs'),deployment_mode='RoofAndFacadeDrop',test_opened=False)
    write_once(folder/'protocol.json',dict(**common,purpose='Six-case targeted development pilot of difficult old and layered-collapse worlds',full_acceptance=False,excluded_node_count=36,source_plans=sources))
    for split,rows in cases.items():write_once(folder/f'{split}.json',dict(**common,split=split,output=str(ROOT/'experiments/raw/rescue'/a.name/split),cases=rows))
    subprocess.run([sys.executable,'scripts/freeze_rescue_sources.py','--output',str(folder.relative_to(ROOT)/'source-snapshot')],cwd=ROOT,check=True,**hidden())
    def collect(split):
        command=[sys.executable,'scripts/collect_rescue_registered.py','--plan',str((folder/f'{split}.json').relative_to(ROOT)),'--model',a.model,'--server',a.server,'--workers','2' if split=='valid-known' else '1','--project-prefix',a.id_prefix+'-layered-'+split]
        subprocess.run(command,cwd=ROOT,check=True,**hidden())
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(collect,cases))
    from evaluate_rescue_closed_loop import evaluate
    reports=ROOT/'reports/rescue/generic-v35';reports.mkdir(parents=True,exist_ok=True)
    for split in cases:
        for step in [6000,18000]:
            report=pilot_report(evaluate(ROOT/'experiments/raw/rescue'/a.name/split,step))
            write_once(reports/f'{a.id_prefix}-pilot-{split}-{step}.json',report)
            print(json.dumps(dict(split=split,step=step,summary=report['summary'],full_acceptance=False)),flush=True)


if __name__=='__main__':main()
