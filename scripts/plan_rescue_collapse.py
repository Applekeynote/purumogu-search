"""Register layered partial-collapse acquisition before any new labels exist."""
import argparse
import hashlib
from pathlib import Path
from collect_rescue_registered import write_once

ROOT=Path(__file__).resolve().parents[1]
TRAIN=[30,31,32,33]
VALID_COMPOSITIONS=[34,35]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/generic-v47-acquisition-r28-long/model.json')
    p.add_argument('--server',default='tmp/generic-tsdf-rust-target/release/rescue_server.exe')
    p.add_argument('--name',default='generalization-v60-partial-collapse')
    a=p.parse_args()
    if not a.name.replace('-','').isalnum():raise ValueError('Plain experiment name required')
    folder=ROOT/'experiments'/a.name
    geometry={path.name:sha(path) for path in (ROOT/'unity/Assets/Scripts').glob('Rescue*.cs')}
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in [ROOT/'unity/Assets/Editor/RescueGeneralization.cs',ROOT/'unity/Assets/Editor/ProjectTools.cs',ROOT/'scripts/collect_rescue_registered.py',Path(__file__).resolve()]}
    common=dict(version=2,steps=18000,inference_map_interval=100,geometry_revision='layered-partial-collapse-variable-storeys-v2',
                geometry_sources=geometry,acquisition_sources=sources,model=a.model,server=a.server,
                model_sha256=sha(ROOT/a.model),server_sha256=sha(ROOT/a.server),sensor_source_sha256=sha(ROOT/'unity/Assets/Scripts/RescueRobot.cs'),
                deployment_mode='RoofAndFacadeDrop',deployment='One third roof, two thirds real visible exterior openings on all storeys',test_opened=False)
    protocol=dict(**common,train_scenes=TRAIN,validation_new_compositions=VALID_COMPOSITIONS,node_counts=[24,36,48],
                  targets=dict(global_occupancy_iou=.95,human_precision=.95,human_recall=.95),
                  cell_m=.05,truth_box_extent_fraction=.42,occupancy_threshold=.6,match_radius_m=.25,
                  deadlines_seconds=[120,360],acceptance_seconds=360,sensors_and_errors='Unchanged RescueRobot.cs; no external pose references',
                  scope='Post-collapse retained floors, dropped slabs, masonry bearings and real inter-layer air spaces; not a structural failure solver',
                  storey_height_m=[.25,.32],storey_height_randomization='Independent geometry seeds; not fixed .28m templates; no floor heights enter inference',
                  inference_inputs='Sensor packets only; no collapse kind, floor masks, scene ID, truth or source positions',
                  void_policy='Actual collapse openings remain empty in truth; filling every hole is an error',
                  final_test='Original TEST90 remains unopened. Supplement18 uses new seeds; freeze model before collection; never fit failed TEST labels',
                  limitation='Unseen collapse compositions use known footprint families; this does not establish arbitrary real-world generalization',
                  primary_references=['https://www.fema.gov/pdf/emergency/usr/module1c2.pdf','https://ndrf.gov.in/sites/default/files/CSSR%20Instructor%27s%20Guide%20-%20Lesson%203%20-%20Construction%20Materials%2C%20Structures%20and%20Damage%20Types.pdf'])
    write_once(folder/'protocol.json',protocol)
    for split,scenes,repeats,seed_base in [('train',TRAIN,6,123000000),('valid-known',TRAIN,1,124000000),('valid-unknown',VALID_COMPOSITIONS,1,125000000),('test-supplement',TRAIN+VALID_COMPOSITIONS,1,126000000)]:
        cases=[]
        for repeat in range(repeats):
            for i,scene in enumerate(scenes):
                seed=seed_base+scene*1009+repeat*101
                for nodes in ([[24,36,48][(repeat+i)%3]] if split=='train' else [24,36,48]):
                    cases.append(dict(id=f'g60-{split}-S{scene:02d}-W{seed}-N{nodes}',scene=scene,seed=seed,nodes=nodes,deployment_seed=seed+nodes*131+700000,source_seed=seed+900000))
        write_once(folder/f'{split}.json',dict(**common,split=split,output=str(ROOT/'experiments/raw/rescue'/a.name/split),cases=cases))
    print(dict(train_worlds=24,valid_cases=18,final_supplement_cases=18,final_test_opened=False))


if __name__=='__main__':main()
