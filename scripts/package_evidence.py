"""Package measured evidence without publishing local paths or raw truth to inference."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/generated/v1"
MODEL = ROOT / "models/m-v1"

def read_lines(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            yield json.loads(line)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    model = json.loads((MODEL / "model.json").read_text())
    truth = list(read_lines(DATA / "truth.jsonl"))
    partitions = {}
    for row in truth:
        assert partitions.setdefault(row["map_id"], row["split"]) == row["split"]
    comparison = list(read_lines(DATA / "comparison-truth.jsonl"))
    test_maps = set(row["map_id"] for row in comparison)
    train_maps = set(model["training_maps"])
    valid_maps = set(model["validation_maps"])
    assert not (train_maps & valid_maps or train_maps & test_maps or valid_maps & test_maps)
    save(MODEL / "feature_schema.json", {"schema":model["schema"],"features":model["features"],"presence_masks":True,"channels":32,"context":"entry and formation index the likelihood table; prior deployed count is a feature"})
    save(MODEL / "labels.json", model["labels"])
    save(MODEL / "training_config.json", {"k":16,"alpha":1,"kmeans_iterations":25,"temperature":model["temperature"],"temperature_selection":"minimum episode-level VALID log loss","abstain_threshold":model["abstain_threshold"],"support_radius_sq":model["support_radius_sq"],"test_used_for_tuning":False})
    save(MODEL / "train_manifest.json", {"training_maps":model["training_maps"],"validation_maps":model["validation_maps"],"training_episodes":sum(t["split"]=="train" for t in truth),"validation_episodes":sum(t["split"]=="valid" for t in truth),"batches_per_episode":3,"class_action_counts":model["counts"],"note":"Balanced collection uses 50 TRAIN, 15 VALID and 25 held-out maps; an additional 25 held-out maps extend the paired test to 50 maps. This intentionally differs from the design's suggested 70/15/15 ratio."})
    hashes = {}
    for split, ids in [("train",train_maps),("valid",valid_maps),("paired_test",test_maps)]:
        hashes[split]={"maps":sorted(ids),"sha256":hashlib.sha256(("\n".join(sorted(ids))+"\n").encode()).hexdigest()}
    save(MODEL / "data_split_hashes.json", hashes)
    files = ["observations.jsonl","truth.jsonl","comparison.jsonl","comparison-truth.jsonl","fault.jsonl","fault-truth.jsonl"]
    save(ROOT / "datasets/manifest.json", {"version":"unity-events-v1","unity":"6000.4.1f1","files":{f:{"bytes":(DATA/f).stat().st_size,"sha256":digest(DATA/f)} for f in files},"collection":json.loads((DATA/"collection.json").read_text()),"partition_maps":dict(Counter(partitions.values())),"comparison":{"episodes":len(comparison),"maps":len(test_maps),"policies":["ghost_flow","random","fixed","passive"],"sensor_repeats":2,"node_budget":15,"reserve":5},"policy_seed":"9001 + map_index*37 + repeat*101","packet_loss":"correlated receiver scan loss; no unreceived-peer identities are exposed"})
    target=next(t for t in comparison if t["label"]==4 and t["policy"]=="ghost_flow" and t["map_id"]=="map-4-13")
    for path in (ROOT / "experiments/raw").glob("session-*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            first=stream.readline()
        if not first:continue
        first=json.loads(first)
        if first.get("message",{}).get("run_id")==target["run_id"]:
            shutil.copyfile(path,ROOT/"datasets/sample.jsonl")
            decisions=[r["message"] for r in read_lines(path) if r["direction"]=="out" and r["message"]["type"]=="inference"]
            assert len(decisions)==3
            expected=[{k:d[k] for k in ["run_id","batch","beliefs","abstain","entropy_bits"]} for d in decisions]
            save(ROOT/"datasets/sample-expected.json",expected)
            break
    else:raise RuntimeError("Measured sample audit not found")
    sums="".join(f"{digest(p)}  {p.name}\n" for p in sorted(MODEL.glob("*.json")))
    (MODEL/"SHA256SUMS").write_text(sums,encoding="ascii")
    save(ROOT/"reports/build-manifest.json",{"project_version":"0.1.0","protocol":"1.0","feature_schema":model["schema"],"unity":"6000.4.1f1","rustc":"1.94.1 (e408947bf 2026-03-25)","model_sha256":digest(MODEL/"model.json"),"cargo_lock_sha256":digest(ROOT/"rust/Cargo.lock"),"reproducibility":"Exact replay of the observations; PhysX reruns are not guaranteed bitwise identical."})
    print("Evidence packaged; train/valid/test maps are disjoint; measured replay has three batches.")

if __name__=="__main__":main()
