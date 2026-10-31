use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const GRID_CELLS: usize = 32 * 21 * 70;
pub fn index(k: [i32; 3]) -> Option<usize> {
    if !(-16..16).contains(&k[0]) || !(-3..18).contains(&k[1]) || !(-6..64).contains(&k[2]) {
        return None;
    }
    Some((((k[0] + 16) * 21 + (k[1] + 3)) * 70 + k[2] + 6) as usize)
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Prototype {
    pub family: String,
    pub occupancy: Vec<f64>,
    #[serde(default)]
    pub sensor_occupancy: Vec<f64>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TreeNode {
    pub feature: Option<usize>,
    pub threshold: f64,
    pub left: usize,
    pub right: usize,
    pub value: f64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionTree {
    pub nodes: Vec<TreeNode>,
}
impl DecisionTree {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.nodes.is_empty() && self.nodes.len() <= 31,
            "tree size budget"
        );
        for (i, node) in self.nodes.iter().enumerate() {
            ensure!(
                node.threshold.is_finite() && node.value.is_finite() && node.value.abs() <= 16.0,
                "non-finite tree"
            );
            if let Some(feature) = node.feature {
                ensure!(
                    feature < 12
                        && node.left > i
                        && node.right > i
                        && node.left < self.nodes.len()
                        && node.right < self.nodes.len(),
                    "invalid tree edge"
                );
            }
        }
        Ok(())
    }
    pub fn score(&self, x: &[f64; 12]) -> f64 {
        let mut index = 0;
        loop {
            let node = &self.nodes[index];
            if let Some(feature) = node.feature {
                index = if x[feature] < node.threshold {
                    node.left
                } else {
                    node.right
                };
            } else {
                return node.value;
            }
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearnedModel {
    pub version: String,
    pub cell_m: f64,
    pub acoustic_weights: Vec<f64>,
    #[serde(default = "default_acoustic_features")]
    pub acoustic_feature_map: String,
    pub occupancy_weights: Vec<f64>,
    #[serde(default)]
    pub fusion_weights: Vec<f64>,
    #[serde(default = "default_window")]
    pub acoustic_window_frames: usize,
    #[serde(default)]
    pub log_range_weights: Vec<f64>,
    #[serde(default = "default_range_sigma")]
    pub log_range_sigma: f64,
    #[serde(default = "default_sensor_mix")]
    pub sensor_likelihood_mix: f64,
    #[serde(default = "default_geometry_scale")]
    pub geometry_log_likelihood_scale: f64,
    #[serde(default)]
    pub signed_geometry_prior: bool,
    #[serde(default)]
    pub track_weights: Vec<f64>,
    #[serde(default)]
    pub track_trees: Vec<DecisionTree>,
    #[serde(default)]
    pub track_bias: f64,
    #[serde(default = "default_track_threshold")]
    pub track_threshold: f64,
    #[serde(default = "default_track_nms")]
    pub track_nms_m: f64,
    /// Experimental fallback: approach sensor-derived acoustic peaks when coverage has no frontier.
    #[serde(default)]
    pub acoustic_probe: bool,
    #[serde(default)]
    pub distributed_swarm: bool,
    #[serde(default)]
    pub local_surface_completion: bool,
    pub prototypes: Vec<Prototype>,
    pub train_maps: Vec<String>,
    #[serde(default)]
    pub calibration_train_maps: Vec<String>,
    pub valid_maps: Vec<String>,
    pub data_sha256: String,
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-40.0, 40.0)).exp())
}
fn default_window() -> usize {
    1
}
fn default_acoustic_features() -> String {
    "linear-v1".into()
}
fn default_range_sigma() -> f64 {
    0.7
}
fn default_sensor_mix() -> f64 {
    1.0
}
fn default_geometry_scale() -> f64 {
    0.04
}
fn default_track_threshold() -> f64 {
    0.5
}
fn default_track_nms() -> f64 {
    0.35
}
impl LearnedModel {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == "rescue-trained-v1" && (self.cell_m - 0.05).abs() < 1e-9,
            "model/schema mismatch"
        );
        ensure!(
            ((self.acoustic_feature_map == "linear-v1" && self.acoustic_weights.len() == 5)
                || (self.acoustic_feature_map == "quadratic-v1"
                    && self.acoustic_weights.len() == 15))
                && self.occupancy_weights.len() == 3,
            "model feature mismatch"
        );
        ensure!(
            (self.fusion_weights.is_empty() || self.fusion_weights.len() == 4)
                && self.fusion_weights.iter().all(|w| w.is_finite())
                && (1..=64).contains(&self.acoustic_window_frames),
            "fusion feature mismatch"
        );
        ensure!(
            self.acoustic_weights
                .iter()
                .chain(&self.occupancy_weights)
                .all(|x| x.is_finite()),
            "invalid coefficients"
        );
        ensure!(
            (self.local_surface_completion && self.prototypes.is_empty())
                || self.prototypes.len() == 15
                    && self
                        .prototypes
                        .iter()
                        .all(|p| p.occupancy.len() == GRID_CELLS
                            && p.occupancy
                                .iter()
                                .all(|x| x.is_finite() && (0.0..=1.0).contains(x))),
            "invalid geometry priors"
        );
        ensure!(
            !self.train_maps.is_empty()
                && self
                    .train_maps
                    .iter()
                    .all(|id| !self.valid_maps.contains(id)),
            "split leakage"
        );
        ensure!(
            self.calibration_train_maps
                .iter()
                .all(|id| !self.valid_maps.contains(id)),
            "calibration/validation split leakage"
        );
        ensure!(
            self.prototypes.iter().all(|p| p.sensor_occupancy.is_empty()
                || (p.sensor_occupancy.len() == GRID_CELLS
                    && p.sensor_occupancy
                        .iter()
                        .all(|x| x.is_finite() && (0.0..=1.0).contains(x)))),
            "invalid sensor likelihood model"
        );
        ensure!(
            (self.log_range_weights.is_empty() || matches!(self.log_range_weights.len(), 5 | 6))
                && self.log_range_weights.iter().all(|x| x.is_finite())
                && self.log_range_sigma.is_finite()
                && (0.05..=2.0).contains(&self.log_range_sigma)
                && self.sensor_likelihood_mix.is_finite()
                && (0.0..=1.0).contains(&self.sensor_likelihood_mix),
            "invalid acoustic range calibration"
        );
        ensure!(
            self.geometry_log_likelihood_scale.is_finite()
                && (0.001..=10.0).contains(&self.geometry_log_likelihood_scale),
            "invalid geometry likelihood scale"
        );
        ensure!(
            (self.track_weights.is_empty() || self.track_weights.len() == 13)
                && self.track_weights.iter().all(|w| w.is_finite())
                && self.track_threshold.is_finite()
                && (0.0..=1.0).contains(&self.track_threshold)
                && self.track_nms_m.is_finite()
                && (0.15..=0.6).contains(&self.track_nms_m),
            "invalid track classifier"
        );
        ensure!(
            self.track_trees.len() <= 256
                && self.track_bias.is_finite()
                && self.track_bias.abs() <= 40.0,
            "invalid tree ensemble"
        );
        for tree in &self.track_trees {
            tree.validate()?;
        }
        Ok(())
    }
    pub fn human_score(&self, x: [f64; 4]) -> f64 {
        let mut logit = self.acoustic_weights[0]
            + x.iter()
                .zip(&self.acoustic_weights[1..5])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if self.acoustic_feature_map == "quadratic-v1" {
            let mut index = 5;
            for i in 0..4 {
                for j in i..4 {
                    logit += self.acoustic_weights[index] * x[i] * x[j];
                    index += 1;
                }
            }
        }
        sigmoid(logit)
    }
    pub fn track_score(&self, x: [f64; 12]) -> Option<f64> {
        if !self.track_trees.is_empty() {
            return Some(sigmoid(
                self.track_bias + self.track_trees.iter().map(|t| t.score(&x)).sum::<f64>(),
            ));
        }
        if self.track_weights.is_empty() {
            return None;
        }
        Some(sigmoid(
            self.track_weights[0]
                + x.iter()
                    .zip(&self.track_weights[1..])
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
        ))
    }
    pub fn has_track_model(&self) -> bool {
        !self.track_weights.is_empty() || !self.track_trees.is_empty()
    }
    pub fn acoustic_range(&self, x: [f64; 4], bearing_sigma_deg: f64) -> Option<(f64, f64)> {
        if self.log_range_weights.is_empty() {
            return None;
        }
        let mut log = self.log_range_weights[0]
            + x.iter()
                .zip(&self.log_range_weights[1..])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if self.log_range_weights.len() == 6 {
            log += self.log_range_weights[5] * bearing_sigma_deg / 90.0;
        }
        Some(((log.exp() - 0.04).clamp(0.03, 3.5), self.log_range_sigma))
    }
    pub fn occupancy_score(&self, log_odds: f64, count: u32) -> f64 {
        sigmoid(
            self.occupancy_weights[0]
                + self.occupancy_weights[1] * log_odds
                + self.occupancy_weights[2] * (1.0 + f64::from(count)).ln(),
        )
    }
    pub fn fused_score(&self, log_odds: f64, count: u32, prior: f64) -> f64 {
        if self.fusion_weights.is_empty() {
            return self.occupancy_score(log_odds, count);
        }
        let prior = prior.clamp(0.000001, 0.999999);
        let prior_log_odds = (prior / (1.0 - prior)).ln();
        let x = [
            log_odds,
            (1.0 + f64::from(count)).ln(),
            if self.signed_geometry_prior {
                prior_log_odds
            } else {
                prior_log_odds.max(0.0)
            },
        ];
        sigmoid(
            self.fusion_weights[0]
                + x.iter()
                    .zip(&self.fusion_weights[1..])
                    .map(|(a, b)| a * b)
                    .sum::<f64>(),
        )
    }
}
