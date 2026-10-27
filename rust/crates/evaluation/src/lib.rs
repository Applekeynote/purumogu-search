//! Offline-only truth reader. The inference/planner crates cannot depend on this crate.
use anyhow::{Context, Result, ensure};
use event_core::EventStore;
use likelihood::{Model, Sample};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Truth {
    pub run_id: String,
    pub map_id: String,
    pub label: usize,
    pub split: String,
    pub policy: String,
    pub map_seed: i32,
    pub physics_seed: i32,
    pub sensor_seed: i32,
    #[serde(default)]
    pub max_depth_m: f64,
    #[serde(default)]
    pub passed_nodes: usize,
    #[serde(default)]
    pub packet_loss: f64,
}
#[derive(Clone)]
pub struct Row {
    pub run_id: String,
    pub truth: Truth,
    pub sample: Sample,
}
pub fn load(events: &Path, truth: &Path, ablation: &str) -> Result<Vec<Row>> {
    let mut truths = HashMap::new();
    let mut partitions = HashMap::new();
    for line in BufReader::new(File::open(truth)?).lines() {
        let t: Truth = serde_json::from_str(&line?)?;
        ensure!(
            t.label < 6 && matches!(t.split.as_str(), "train" | "valid" | "test" | "ood"),
            "invalid truth label/split"
        );
        if let Some(split) = partitions.insert(t.map_id.clone(), t.split.clone()) {
            ensure!(split == t.split, "map leaked across partitions");
        }
        ensure!(
            truths.insert(t.run_id.clone(), t).is_none(),
            "duplicate run truth"
        );
    }
    let mut stores: HashMap<String, EventStore> = HashMap::new();
    let mut prior = HashMap::new();
    let mut guards: HashMap<String, event_core::SequenceGuard> = HashMap::new();
    let mut rows = Vec::new();
    for line in BufReader::new(File::open(events)?).lines() {
        let line = line?;
        let p = protocol::parse(line.as_bytes()).context("invalid observation file")?;
        if p.message_type != "observation_batch" {
            continue;
        }
        let guard = guards.entry(p.run_id.clone()).or_default();
        match guard.accept(p.seq_no) {
            event_core::Sequence::New => {}
            event_core::Sequence::Duplicate => continue,
            event_core::Sequence::Stale => anyhow::bail!("out-of-order packet in dataset"),
        };
        let store = stores.entry(p.run_id.clone()).or_default();
        store.append(&p)?;
        if p.batch_end {
            let t = truths
                .get(&p.run_id)
                .context("missing separate truth record")?;
            let deployed = prior.entry(p.run_id.clone()).or_insert(0);
            let features = graph::summarize(&store.finish(), *deployed, ablation);
            *deployed += p.node_count;
            rows.push(Row {
                run_id: p.run_id.clone(),
                truth: t.clone(),
                sample: Sample {
                    features,
                    label: t.label,
                    action: p.entry_id * 2 + usize::from(p.formation_mode == "CHAIN_INSERT"),
                },
            });
        }
    }
    ensure!(
        stores.values().all(|s| s.events.is_empty()),
        "unfinished observation batch"
    );
    Ok(rows)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub run_id: String,
    pub map_id: String,
    pub policy: String,
    pub label: usize,
    pub predicted: usize,
    pub probabilities: [f64; 6],
    pub abstain: bool,
    pub reason: String,
    pub entropy: f64,
    pub log_loss: f64,
    pub brier: f64,
    pub nodes: usize,
    pub information_per_node: f64,
    pub max_depth_m: f64,
    pub passed_nodes: usize,
    pub packet_loss: f64,
}
pub fn evaluate(rows: &[Row], m: &Model, split: &str) -> Vec<Outcome> {
    let mut state: HashMap<&str, (inference::Belief, usize)> = HashMap::new();
    let mut out = BTreeMap::new();
    for row in rows.iter().filter(|r| r.truth.split == split) {
        let (p, n) = state.entry(&row.run_id).or_insert((inference::prior(), 0));
        let (z, d) = m.encode(&row.sample.features);
        if row.sample.features.observed > 0 {
            *p = inference::update(p, z, row.sample.action, m);
        }
        *n += 5;
        let (abstain, reason) = inference::abstain(p, &row.sample.features, d, m);
        let predicted = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let label = row.truth.label;
        out.insert(
            &row.run_id,
            Outcome {
                run_id: row.run_id.clone(),
                map_id: row.truth.map_id.clone(),
                policy: row.truth.policy.clone(),
                label,
                predicted,
                probabilities: *p,
                abstain,
                reason: reason.into(),
                entropy: inference::entropy(p),
                log_loss: -p[label].max(1e-15).ln(),
                brier: p
                    .iter()
                    .enumerate()
                    .map(|(g, v)| (v - f64::from(g == label)).powi(2))
                    .sum(),
                nodes: *n,
                information_per_node: (inference::entropy(&inference::prior())
                    - inference::entropy(p))
                    / *n as f64,
                max_depth_m: row.truth.max_depth_m,
                passed_nodes: row.truth.passed_nodes,
                packet_loss: row.truth.packet_loss,
            },
        );
    }
    out.into_values().collect()
}
pub fn calibrate(rows: &[Row], m: &mut Model) -> Result<()> {
    let train: BTreeSet<_> = m.training_maps.iter().collect();
    let valid: BTreeSet<_> = rows
        .iter()
        .filter(|r| r.truth.split == "valid")
        .map(|r| r.truth.map_id.clone())
        .collect();
    ensure!(
        !valid.is_empty() && valid.iter().all(|id| !train.contains(id)),
        "missing/disjoint validation maps required"
    );
    let mut best = (f64::INFINITY, 1.);
    for temp in [0.5, 1., 1.5, 2., 3., 4.] {
        m.temperature = temp;
        let outcomes = evaluate(rows, m, "valid");
        let loss = outcomes.iter().map(|o| o.log_loss).sum::<f64>() / outcomes.len() as f64;
        if loss < best.0 {
            best = (loss, temp);
        }
    }
    m.temperature = best.1;
    m.validation_maps = valid.into_iter().collect();
    Ok(())
}
#[derive(Serialize)]
pub struct Metrics {
    pub episodes: usize,
    pub maps: usize,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
    pub macro_f1: f64,
    pub log_loss: f64,
    pub brier: f64,
    pub ece: f64,
    pub abstention_rate: f64,
    pub selective_accuracy: Option<f64>,
    pub final_entropy: f64,
    pub information_per_node: f64,
    pub accuracy_ci95: [f64; 2],
    pub confusion: Vec<Vec<usize>>,
}
pub fn metrics(v: &[Outcome]) -> Metrics {
    let n = v.len().max(1) as f64;
    let mut confusion = vec![vec![0; 6]; 6];
    for o in v {
        confusion[o.label][o.predicted] += 1;
    }
    let present: Vec<_> = (0..6)
        .filter(|&g| confusion[g].iter().sum::<usize>() > 0)
        .collect();
    let mut balanced = 0.;
    let mut f1 = 0.;
    for &g in &present {
        let tp = confusion[g][g] as f64;
        let actual = confusion[g].iter().sum::<usize>() as f64;
        let predicted = (0..6).map(|i| confusion[i][g]).sum::<usize>() as f64;
        balanced += tp / actual;
        f1 += if actual + predicted > 0. {
            2. * tp / (actual + predicted)
        } else {
            0.
        };
    }
    let selected: Vec<_> = v.iter().filter(|o| !o.abstain).collect();
    let mut ece = 0.;
    for bin in 0..10 {
        let b: Vec<_> = v
            .iter()
            .filter(|o| ((o.probabilities[o.predicted] * 10.) as usize).min(9) == bin)
            .collect();
        if b.is_empty() {
            continue;
        }
        let confidence =
            b.iter().map(|o| o.probabilities[o.predicted]).sum::<f64>() / b.len() as f64;
        let accuracy = b.iter().filter(|o| o.predicted == o.label).count() as f64 / b.len() as f64;
        ece += b.len() as f64 / n * (confidence - accuracy).abs();
    }
    let mut maps: BTreeMap<&str, Vec<&Outcome>> = BTreeMap::new();
    for o in v {
        maps.entry(&o.map_id).or_default().push(o);
    }
    let values: Vec<_> = maps.values().collect();
    let mut ci = Vec::new();
    let mut seed = 83492791u64;
    if !values.is_empty() {
        for _ in 0..2000 {
            let mut correct = 0;
            let mut total = 0;
            for _ in 0..values.len() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let group = values[((seed >> 32) as usize) % values.len()];
                correct += group.iter().filter(|o| o.label == o.predicted).count();
                total += group.len();
            }
            ci.push(correct as f64 / total as f64);
        }
        ci.sort_by(f64::total_cmp);
    }
    Metrics {
        episodes: v.len(),
        maps: maps.len(),
        accuracy: v.iter().filter(|o| o.label == o.predicted).count() as f64 / n,
        balanced_accuracy: balanced / present.len().max(1) as f64,
        macro_f1: f1 / present.len().max(1) as f64,
        log_loss: v.iter().map(|o| o.log_loss).sum::<f64>() / n,
        brier: v.iter().map(|o| o.brier).sum::<f64>() / n,
        ece,
        abstention_rate: v.iter().filter(|o| o.abstain).count() as f64 / n,
        selective_accuracy: if selected.is_empty() {
            None
        } else {
            Some(
                selected.iter().filter(|o| o.label == o.predicted).count() as f64
                    / selected.len() as f64,
            )
        },
        final_entropy: v.iter().map(|o| o.entropy).sum::<f64>() / n,
        information_per_node: v.iter().map(|o| o.information_per_node).sum::<f64>() / n,
        accuracy_ci95: if ci.is_empty() {
            [0., 0.]
        } else {
            [ci[50], ci[1950]]
        },
        confusion,
    }
}
pub fn report(outcomes: &[Outcome], dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(
        dir.join("outcomes.json"),
        serde_json::to_vec_pretty(outcomes)?,
    )?;
    let mut csv="run_id,map_id,policy,label,predicted,abstain,entropy,log_loss,brier,nodes,information_per_node,max_depth_m,passed_nodes,packet_loss\n".to_string();
    for o in outcomes {
        csv += &format!(
            "{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6},{},{}\n",
            o.run_id,
            o.map_id,
            o.policy,
            o.label,
            o.predicted,
            o.abstain,
            o.entropy,
            o.log_loss,
            o.brier,
            o.nodes,
            o.information_per_node,
            o.max_depth_m,
            o.passed_nodes,
            o.packet_loss
        );
    }
    fs::write(dir.join("outcomes.csv"), csv)?;
    let policies: BTreeSet<_> = outcomes.iter().map(|o| o.policy.as_str()).collect();
    let mut grouped = BTreeMap::new();
    for policy in policies {
        let v: Vec<_> = outcomes
            .iter()
            .filter(|o| o.policy == policy)
            .cloned()
            .collect();
        grouped.insert(policy, metrics(&v));
    }
    fs::write(
        dir.join("metrics.json"),
        serde_json::to_vec_pretty(&grouped)?,
    )?;
    Ok(())
}
pub fn train_file(events: &Path, truth: &Path, output: &Path, ablation: &str) -> Result<()> {
    let rows = load(events, truth, ablation)?;
    let maps: BTreeSet<_> = rows
        .iter()
        .filter(|r| r.truth.split == "train")
        .map(|r| r.truth.map_id.clone())
        .collect();
    let samples: Vec<_> = rows
        .iter()
        .filter(|r| r.truth.split == "train")
        .map(|r| r.sample.clone())
        .collect();
    let mut model = likelihood::train(
        &samples,
        maps.into_iter().collect(),
        format!("ghost-flow-v1-{ablation}"),
    )?;
    calibrate(&rows, &mut model)?;
    model.save(output)?;
    report(
        &evaluate(&rows, &model, "valid"),
        &output
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("validation-{ablation}")),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_empty_is_finite() {
        let m = metrics(&[]);
        assert!(m.brier.is_finite());
        assert_eq!(m.selective_accuracy, None);
    }
}
