"""Pre-register world/family splits before any new labels are collected."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCENES = list(range(15)) + [15, 16, 18, 20, 21, 24, 25, 26]
VALID_UNKNOWN = [19, 22, 29]  # courtyard and ring/chord families
TEST_UNKNOWN = [17, 23, 27, 28]  # Y branches and dogleg families, never model selection


def main():
    folder = ROOT / 'experiments/generalization-v35'
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(version=1, targets=dict(global_occupancy_iou=.95, human_precision=.95, human_recall=.95),
                    cell_m=.05, match_radius_m=.25, occupancy_threshold=.6,
                    train_scenes=TRAIN_SCENES, valid_unknown_scenes=VALID_UNKNOWN,
                    test_unknown_scenes=TEST_UNKNOWN, node_counts=[24, 36, 48],
                    inference_excludes=['scene ID', 'world seed', 'absolute-coordinate shape prototypes',
                                        'true robot positions', 'true occupied voxels', 'true victim positions'],
                    aggregation='Existing and unknown domains reported separately, including every node-count stratum and per-area rows.',
                    test_policy='No TEST collection or label access until model selection is frozen. A failed TEST cannot be reused for selection.',
                    acquisition='rescue-3.4 physical sensors and control, 120 seconds; no sensor or truth definition changes')
    write_once(folder / 'protocol.json', protocol)
    write_once(folder/'family-audit.json',dict(status='Amendment registered before any TEST collection or labels',
               reason='Scene 28 is a new three-storey U composition, but U is a trained footprint family. Do not call it an unseen footprint family.',
               truly_unseen_final_families=dict(Y=[17,27],dogleg=[23]),
               seen_family_new_composition=[28],validation_only_families=dict(courtyard=[19,29],ring_chord=[22]),
               complete_final_scenes=sorted(TRAIN_SCENES+VALID_UNKNOWN+TEST_UNKNOWN)))
    # First four TRAIN cases expose single-floor and three-floor failure modes quickly.
    order = [15, 18, 26, 0] + [s for s in TRAIN_SCENES if s not in [15, 18, 26, 0]]
    for split, scenes, repeats, seed_base in [
        ('train', order, 3, 41000000), ('valid-known', [0, 3, 8, 13, 15, 18, 20, 25, 26], 1, 42000000),
        ('valid-unknown', VALID_UNKNOWN, 1, 43000000),
        ('test-known', TRAIN_SCENES, 1, 44000000), ('test-unknown', TEST_UNKNOWN, 1, 45000000),
        ('test-validation-families', VALID_UNKNOWN, 1, 46000000)]:
        cases = []
        for repeat in range(repeats):
            for scene in scenes:
                # TRAIN alternates counts on independent worlds; validation/test pair counts within worlds.
                for nodes in ([36, 24, 48][repeat:repeat+1] if split == 'train' else [24, 36, 48]):
                    seed = seed_base + scene * 1009 + repeat * 101
                    cases.append(dict(id=f'{split}-S{scene:02d}-W{seed}-N{nodes}', scene=scene, seed=seed,
                                      nodes=nodes, deployment_seed=seed + nodes * 131 + 700000,
                                      source_seed=seed + 900000))
        plan = dict(split=split, steps=6000, output=str(ROOT / 'experiments/raw/rescue/generalization-v35' / split), cases=cases)
        write_once(folder / f'{split}.json', plan)
    print(json.dumps(dict(protocol_sha256=hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest(),
                          train_worlds=69, valid_known_worlds=9, valid_unknown_worlds=3,
                          test_labels_opened=False)))


def write_once(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError(f'Immutable plan differs: {path}')
    path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
