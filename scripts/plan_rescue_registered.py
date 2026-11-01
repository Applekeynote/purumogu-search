"""Freeze new physical TRAIN/VALID acquisition without opening final TEST."""
import argparse
import hashlib
import json
from pathlib import Path
from collect_rescue_registered import write_once

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--server',default='tmp/generic-rust-target/release/rescue_server.exe')
    parser.add_argument('--name',default='generalization-v38-overhead')
    parser.add_argument('--deployment',choices=['overhead','mixed'],default='overhead')
    parser.add_argument('--seed-block',type=int,default=51);parser.add_argument('--id-prefix',default='g38')
    parser.add_argument('--all-train-families',action='store_true');parser.add_argument('--train-repeats',type=int,default=3)
    parser.add_argument('--splits',default='train,valid-known,valid-unknown')
    parser.add_argument('--geometry-revision',default='')
    parser.add_argument('--steps',type=int,choices=[6000,18000],default=6000)
    args=parser.parse_args()
    if not args.name.replace('-','').isalnum():raise ValueError('Plain generation name required')
    if not 1<=args.seed_block<=100 or not args.id_prefix.startswith('g') or not args.id_prefix[1:].isdigit():raise ValueError('Registered seed block and generation prefix required')
    if not 1<=args.train_repeats<=12 or not set(args.splits.split(','))<={'train','valid-known','valid-unknown'}:raise ValueError('Development TRAIN/VALID only')
    folder=ROOT/'experiments'/args.name
    folder.mkdir(parents=True,exist_ok=True)
    geometry={p.name:sha(p) for p in (ROOT/'unity/Assets/Scripts').glob('Rescue*.cs')}
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
        ROOT/'unity/Assets/Editor/RescueGeneralization.cs',ROOT/'unity/Assets/Editor/ProjectTools.cs',
        ROOT/'scripts/collect_rescue_registered.py',Path(__file__).resolve()]}
    common=dict(version=2,steps=args.steps,inference_map_interval=100,
        geometry_revision=args.geometry_revision or ('ramp-clearance-and-exterior-openings-v2' if args.deployment=='mixed' else 'ramp-clearance-v1'),geometry_sources=geometry,acquisition_sources=sources,
        model_sha256=sha(ROOT/args.model),server_sha256=sha(ROOT/args.server),
        sensor_source_sha256=sha(ROOT/'unity/Assets/Scripts/RescueRobot.cs'),
        model=args.model,server=args.server,
        deployment_mode='RoofAndFacadeDrop' if args.deployment=='mixed' else 'WholeSceneDrop',
        deployment='One third overhead; two thirds released at several visible facade thresholds on all levels' if args.deployment=='mixed' else 'WholeSceneDrop: original overhead scattering; no facade deployment',
        test_opened=False)
    write_once(folder/'protocol.json',dict(**common,
        target=dict(global_occupancy_iou=.95,human_precision=.95,human_recall=.95),
        cell_m=.05,truth_box_extent_fraction=.42,occupancy_threshold=.6,match_radius_m=.25,
        sensor_errors='Unchanged RescueRobot.cs; no external position references',
        learning='Additional independent TRAIN on corrected stairs and current inferred-route scout policy',
        comparison='New geometry revision; report separately from old geometry and old fixed-trajectory results',
        held_final_families=dict(Y=[17,27],dogleg=[23]),seen_family_held_composition=[28],
        acceptance='Fresh closed-loop evaluation required after each navigation-weight change; every domain, count and area must meet all targets'))
    train_scenes=[25,26,13,20,21,24,15,16,18]+[s for s in list(range(15))+[15,16,18,20,21,24,25,26] if s not in [25,26,13,20,21,24,15,16,18]] if args.all_train_families else [25,26,15,16,18,20,21,24]
    for split,scenes,repeats,base in [
        ('train',train_scenes,args.train_repeats,args.seed_block*1000000),
        ('valid-known',[0,3,8,13,15,18,20,25,26],1,(args.seed_block+1)*1000000),
        ('valid-unknown',[19,22,29],1,(args.seed_block+2)*1000000)]:
        if split not in args.splits.split(','):continue
        cases=[]
        for repeat in range(repeats):
            for i,scene in enumerate(scenes):
                counts=[[24,36,48][(repeat+i)%3]] if split=='train' else [24,36,48]
                seed=base+scene*1009+repeat*101
                for nodes in counts:
                    cases.append(dict(id=f'{args.id_prefix}-{split}-S{scene:02d}-W{seed}-N{nodes}',scene=scene,seed=seed,
                        nodes=nodes,deployment_seed=seed+nodes*131+700000,source_seed=seed+900000))
        write_once(folder/f'{split}.json',dict(**common,split=split,
            output=str(ROOT/'experiments/raw/rescue'/args.name/split),cases=cases))
    print(json.dumps(dict(generation=args.name,train_worlds=len(train_scenes)*args.train_repeats,validation_worlds=12 if args.splits!='train' else 0,final_test_opened=False)))


if __name__=='__main__':main()
