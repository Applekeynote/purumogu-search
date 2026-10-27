use anyhow::{Context, Result, ensure};
use graph::{DIM, Summary};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub const K: usize = 16;
pub const ACTIONS: usize = 6;
pub const CLASSES: usize = 6;

#[derive(Clone)]
pub struct Sample {
    pub features: Summary,
    pub label: usize,
    pub action: usize,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub version: String,
    pub schema: String,
    pub labels: Vec<String>,
    pub features: Vec<String>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub centers: Vec<Vec<f64>>,
    pub likelihood: Vec<Vec<Vec<f64>>>,
    pub counts: Vec<Vec<usize>>,
    pub temperature: f64,
    pub support_radius_sq: f64,
    pub abstain_threshold: f64,
    pub training_maps: Vec<String>,
    pub validation_maps: Vec<String>,
}
fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}
impl Model {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == protocol::SCHEMA && self.features == graph::FEATURES.map(String::from),
            "feature schema mismatch"
        );
        ensure!(
            self.labels
                == protocol::LABELS
                    .iter()
                    .copied()
                    .chain(["OTHER"])
                    .map(String::from)
                    .collect::<Vec<_>>(),
            "label mismatch"
        );
        ensure!(
            self.mean.len() == DIM * 2
                && self.scale.len() == DIM * 2
                && self.mean.iter().all(|x| x.is_finite())
                && self.scale.iter().all(|x| x.is_finite() && *x > 0.),
            "invalid scaler"
        );
        ensure!(
            self.centers.len() == K
                && self
                    .centers
                    .iter()
                    .all(|c| c.len() == DIM * 2 && c.iter().all(|x| x.is_finite())),
            "invalid codebook"
        );
        ensure!(
            self.likelihood.len() == CLASSES
                && self.counts.len() == CLASSES
                && self.counts.iter().all(|a| a.len() == ACTIONS),
            "invalid class/action counts"
        );
        for g in &self.likelihood {
            ensure!(g.len() == ACTIONS, "action dimensions");
            for a in g {
                ensure!(
                    a.len() == K
                        && a.iter().all(|p| p.is_finite() && *p > 0.)
                        && (a.iter().sum::<f64>() - 1.).abs() < 1e-6,
                    "invalid likelihood"
                );
            }
        }
        ensure!(
            self.temperature.is_finite()
                && self.temperature > 0.
                && self.support_radius_sq.is_finite()
                && self.support_radius_sq > 0.
                && (0.5..=1.).contains(&self.abstain_threshold),
            "invalid calibration"
        );
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let m: Self = serde_json::from_slice(&fs::read(path)?)?;
        m.validate()?;
        Ok(m)
    }
    /// Publish a new artifact beside the old one. Serving processes keep their immutable snapshot.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        if let Some(p) = path.parent() {
            fs::create_dir_all(p)?;
        }
        let temp = path.with_extension("json.tmp");
        fs::write(&temp, serde_json::to_vec_pretty(self)?)?;
        ensure!(
            !path.exists(),
            "model path already exists; choose a new model version"
        );
        fs::rename(temp, path)?;
        Ok(())
    }
    pub fn normalized(&self, s: &Summary) -> Vec<f64> {
        s.vector()
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, m), sd)| (x - m) / sd)
            .collect()
    }
    pub fn encode(&self, s: &Summary) -> (usize, f64) {
        let v = self.normalized(s);
        self.centers
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance(&v, c)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
    }
    pub fn probabilities(&self, g: usize, a: usize) -> Vec<f64> {
        let mut p: Vec<_> = self.likelihood[g][a]
            .iter()
            .map(|p| p.powf(1. / self.temperature))
            .collect();
        let sum = p.iter().sum::<f64>();
        p.iter_mut().for_each(|x| *x /= sum);
        p
    }
}

pub fn train(samples: &[Sample], maps: Vec<String>, version: String) -> Result<Model> {
    ensure!(samples.len() >= K, "need at least {K} training batches");
    let data: Vec<_> = samples.iter().map(|s| s.features.vector()).collect();
    let d = DIM * 2;
    let mean: Vec<f64> = (0..d)
        .map(|j| data.iter().map(|v| v[j]).sum::<f64>() / data.len() as f64)
        .collect();
    let scale: Vec<f64> = (0..d)
        .map(|j| {
            (data.iter().map(|v| (v[j] - mean[j]).powi(2)).sum::<f64>() / data.len() as f64)
                .sqrt()
                .max(0.1)
        })
        .collect();
    let normalized: Vec<Vec<f64>> = data
        .iter()
        .map(|v| {
            v.iter()
                .zip(&mean)
                .zip(&scale)
                .map(|((x, m), sd)| (x - m) / sd)
                .collect()
        })
        .collect();
    let mut centers = vec![normalized[0].clone()];
    while centers.len() < K {
        let next = normalized
            .iter()
            .max_by(|a, b| {
                let da = centers
                    .iter()
                    .map(|c| distance(a, c))
                    .fold(f64::INFINITY, f64::min);
                let db = centers
                    .iter()
                    .map(|c| distance(b, c))
                    .fold(f64::INFINITY, f64::min);
                da.total_cmp(&db)
            })
            .context("empty training data")?;
        centers.push(next.clone());
    }
    for _ in 0..25 {
        let mut sums = vec![vec![0.; d]; K];
        let mut counts = [0usize; K];
        for v in &normalized {
            let k = centers
                .iter()
                .enumerate()
                .min_by(|a, b| distance(v, a.1).total_cmp(&distance(v, b.1)))
                .unwrap()
                .0;
            counts[k] += 1;
            for (sum, x) in sums[k].iter_mut().zip(v) {
                *sum += x;
            }
        }
        for k in 0..K {
            if counts[k] > 0 {
                centers[k] = sums[k].iter().map(|x| x / counts[k] as f64).collect();
            }
        }
    }
    let mut m = Model {
        version,
        schema: protocol::SCHEMA.into(),
        labels: protocol::LABELS
            .iter()
            .copied()
            .chain(["OTHER"])
            .map(String::from)
            .collect(),
        features: graph::FEATURES.map(String::from).to_vec(),
        mean,
        scale,
        centers,
        likelihood: vec![vec![vec![1.; K]; ACTIONS]; CLASSES],
        counts: vec![vec![0; ACTIONS]; CLASSES],
        temperature: 1.,
        support_radius_sq: 1.,
        abstain_threshold: 0.75,
        training_maps: maps,
        validation_maps: Vec::new(),
    };
    let mut distances = Vec::new();
    for s in samples {
        ensure!(
            s.label < 5 && s.action < ACTIONS,
            "invalid training label/action"
        );
        let (k, dist) = m.encode(&s.features);
        m.likelihood[s.label][s.action][k] += 1.;
        m.counts[s.label][s.action] += 1;
        distances.push(dist);
    }
    for g in 0..5 {
        for a in 0..ACTIONS {
            ensure!(
                m.counts[g][a] > 0,
                "missing training coverage for class {g} action {a}"
            );
        }
    }
    for g in &mut m.likelihood {
        for a in g {
            let sum = a.iter().sum::<f64>();
            a.iter_mut().for_each(|v| *v /= sum);
        }
    }
    distances.sort_by(f64::total_cmp);
    m.support_radius_sq = (distances[distances.len() * 99 / 100] * 2.).max(4.);
    m.validate()?;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_action_coverage_fails() {
        let s = graph::summarize(&[], 0, "full");
        let v: Vec<_> = (0..20)
            .map(|_| Sample {
                features: s.clone(),
                label: 0,
                action: 0,
            })
            .collect();
        assert!(train(&v, vec![], "x".into()).is_err());
    }
}
