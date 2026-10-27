"""Summarize measured Unity runs; bootstrap paired differences at map level."""
from pathlib import Path
import json
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
local = ROOT / "tmp/media-runtime"
if local.exists():
    sys.path.insert(0, str(local))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["STRAIGHT", "T_BRANCH", "DEAD_END", "REJOIN", "NARROWING", "OTHER"]
POLICIES = ["ghost_flow", "fixed", "random", "passive"]
NAMES = ["GHOST-FLOW", "Fixed cycle", "Random", "Passive"]

def paired(outcomes, a, b, field):
    maps = sorted(set(o["map_id"] for o in outcomes))
    differences = []
    for map_id in maps:
        values = {}
        for policy in [a, b]:
            rows = [o for o in outcomes if o["map_id"] == map_id and o["policy"] == policy]
            assert len(rows) == 2, "Expected exactly two paired sensor repeats per map/policy"
            values[policy] = np.mean([float(o["predicted"] == o["label"]) if field == "accuracy" else o[field] for o in rows])
        differences.append(values[a] - values[b])
    rng = np.random.default_rng(20261027)
    boot = np.mean(rng.choice(differences, size=(10000, len(maps)), replace=True), axis=1)
    return {"mean_difference": float(np.mean(differences)), "ci95": np.quantile(boot, [.025, .975]).tolist(), "maps": len(maps), "bootstrap_seed": 20261027}

def main():
    report = ROOT / "reports/comparison"
    outcomes = json.loads((report / "outcomes.json").read_text())
    metrics = json.loads((report / "metrics.json").read_text())
    result = {
        "ghost_minus_fixed_accuracy": paired(outcomes, "ghost_flow", "fixed", "accuracy"),
        "ghost_minus_random_accuracy": paired(outcomes, "ghost_flow", "random", "accuracy"),
        "active_cluster_minus_passive_depth_m": paired(outcomes, "fixed", "passive", "max_depth_m"),
        "active_cluster_minus_passive_passed_nodes": paired(outcomes, "fixed", "passive", "passed_nodes"),
    }
    physical = {}
    for policy in POLICIES:
        rows = [o for o in outcomes if o["policy"] == policy]
        physical[policy] = {"mean_depth_m": float(np.mean([o["max_depth_m"] for o in rows])), "mean_passed_nodes": float(np.mean([o["passed_nodes"] for o in rows])), "selected_episodes": sum(not o["abstain"] for o in rows)}
    result["physical"] = physical
    fault = json.loads((ROOT / "reports/fault/outcomes.json").read_text())
    result["faults"] = []
    for loss in [0., .1, .2, .4]:
        rows = [o for o in fault if abs(o["packet_loss"] - loss) < 1e-8]
        result["faults"].append({"packet_loss": loss, "episodes": len(rows), "accuracy": float(np.mean([o["predicted"] == o["label"] for o in rows])), "abstention_rate": float(np.mean([o["abstain"] for o in rows]))})
    ood = json.loads((ROOT / "reports/ood/outcomes.json").read_text())
    result["ood"] = {"episodes": len(ood), "maps": len(set(o["map_id"] for o in ood)), "abstained": sum(o["abstain"] for o in ood), "confidently_wrong": sum(not o["abstain"] and o["predicted"] != o["label"] for o in ood)}
    result["model_sha256"] = hashlib.sha256((ROOT / "models/m-v1/model.json").read_bytes()).hexdigest()
    (ROOT / "reports/analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#f7f9f6", "axes.facecolor": "#f7f9f6", "text.color": "#17303a", "axes.labelcolor": "#17303a"})
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for i, policy in enumerate(POLICIES):
        m = metrics[policy]; x = m["accuracy"] * 100; lo, hi = np.array(m["accuracy_ci95"]) * 100
        ax.errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", markersize=10, capsize=5, color="#159b80" if i == 0 else "#5f7580", linewidth=2)
        ax.text(hi+2, i, f"{x:.0f}%", va="center", fontsize=13)
    ax.set_yticks(range(4), NAMES); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Top-1 accuracy (%) / 95% map bootstrap interval")
    ax.grid(axis="x", alpha=.15); ax.set_axisbelow(True)
    fig.savefig(ROOT / "reports/accuracy.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.6, 3), layout="constrained")
    x = np.arange(4); width = .36
    ax.bar(x-width/2, [metrics[p]["accuracy"]*100 for p in POLICIES], width, label="All episodes: correct top candidate", color="#159b80")
    ax.bar(x+width/2, [metrics[p]["abstention_rate"]*100 for p in POLICIES], width, label="Abstained", color="#d69a4b")
    ax.set_xticks(x, NAMES); ax.set_ylim(0, 105); ax.set_ylabel("Episodes (%)"); ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0,1.25)); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.savefig(ROOT / "reports/coverage.png", dpi=200); plt.close(fig)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
