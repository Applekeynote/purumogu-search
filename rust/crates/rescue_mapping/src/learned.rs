use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const GRID_CELLS: usize = 32 * 21 * 70;
pub const TEACHER_FEATURE_MAP: &str = "thermal-spectral-teacher33-v5";
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
        self.validate_features(12, 31)
    }
    pub fn validate_features(&self, features: usize, nodes: usize) -> Result<()> {
        ensure!(
            !self.nodes.is_empty() && self.nodes.len() <= nodes,
            "tree size budget"
        );
        for (i, node) in self.nodes.iter().enumerate() {
            ensure!(
                node.threshold.is_finite() && node.value.is_finite() && node.value.abs() <= 16.0,
                "non-finite tree"
            );
            if let Some(feature) = node.feature {
                ensure!(
                    feature < features
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
        self.score_features(x)
    }
    pub fn score_features(&self, x: &[f64]) -> f64 {
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
    #[serde(default)]
    pub model_id: String,
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
    #[serde(default)]
    pub bearing_likelihood: Vec<f64>,
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
    #[serde(default)]
    pub track_feature_map: String,
    #[serde(default)]
    pub track_teacher_trees: usize,
    #[serde(default)]
    pub track_teacher_bias: f64,
    #[serde(default = "default_track_threshold")]
    pub track_threshold: f64,
    #[serde(default = "default_track_nms")]
    pub track_nms_m: f64,
    #[serde(default = "default_height_scale")]
    pub track_height_scale: f64,
    #[serde(default)]
    pub track_refine_radius_m: f64,
    #[serde(default = "default_height_scale")]
    pub track_refine_score_power: f64,
    /// Experimental fallback: approach sensor-derived acoustic peaks when coverage has no frontier.
    #[serde(default)]
    pub acoustic_probe: bool,
    #[serde(default)]
    pub acoustic_peaks_v2: bool,
    #[serde(default)]
    pub joint_acoustic_fusion: bool,
    #[serde(default)]
    pub plane_registration_v1: bool,
    #[serde(default)]
    pub axis_plane_registration_v2: bool,
    #[serde(default)]
    pub learned_navigation_v1: bool,
    #[serde(default)]
    pub scout_inferred_routes_v1: bool,
    #[serde(default)]
    pub footprint_safe_routes_v1: bool,
    #[serde(default)]
    pub range_tsdf_v1: bool,
    #[serde(default)]
    pub distributed_swarm: bool,
    #[serde(default)]
    pub local_surface_completion: bool,
    #[serde(default)]
    pub geometry_feature_map: String,
    #[serde(default)]
    pub geometry_trees: Vec<DecisionTree>,
    #[serde(default)]
    pub geometry_bias: f64,
    #[serde(default)]
    pub volume_file: String,
    #[serde(default)]
    pub volume_sha256: String,
    #[serde(default)]
    pub volume_logit_bias: f64,
    #[serde(default = "default_volume_channels")]
    pub volume_channels: usize,
    #[serde(skip)]
    pub volume_runtime: Option<std::sync::Arc<crate::volume::VolumeRuntime>>,
    /// Optional frozen sensor-only exploration model, independent of map decoder updates.
    #[serde(default)]
    pub navigation_file: String,
    #[serde(default)]
    pub navigation_sha256: String,
    #[serde(default)]
    pub navigation_logit_bias: f64,
    #[serde(default)]
    pub navigation_channels: usize,
    #[serde(skip)]
    pub navigation_runtime: Option<std::sync::Arc<crate::volume::VolumeRuntime>>,
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
fn default_volume_channels() -> usize {
    2
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
fn default_height_scale() -> f64 {
    1.0
}
impl LearnedModel {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.version == "rescue-trained-v1" && (self.cell_m - 0.05).abs() < 1e-9,
            "model/schema mismatch"
        );
        ensure!(
            self.model_id.len() <= 96
                && self
                    .model_id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
            "invalid model identifier"
        );
        ensure!(
            !self.axis_plane_registration_v2 || self.plane_registration_v1,
            "wall registration requires shared plane registration"
        );
        ensure!(
            !self.learned_navigation_v1 || self.local_surface_completion,
            "learned navigation requires a completed spatial map"
        );
        ensure!(
            !self.scout_inferred_routes_v1 || self.learned_navigation_v1,
            "scout routing requires learned navigation"
        );
        ensure!(
            !self.footprint_safe_routes_v1 || self.learned_navigation_v1,
            "footprint support requires learned navigation"
        );
        ensure!(
            !self.range_tsdf_v1 || self.volume_channels == 90,
            "Signed distance features require 90 channels"
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
            self.bearing_likelihood.is_empty()
                || (self.bearing_likelihood.len() == 4
                    && self.bearing_likelihood.iter().all(|v| v.is_finite())
                    && (0.05..=2.0).contains(&self.bearing_likelihood[0])
                    && (0.05..=2.0).contains(&self.bearing_likelihood[1])
                    && (0.0..=0.9).contains(&self.bearing_likelihood[2])
                    && (0.05..=0.95).contains(&self.bearing_likelihood[3])),
            "invalid bearing calibration"
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
            self.track_height_scale.is_finite() && (1.0..=3.0).contains(&self.track_height_scale),
            "invalid track height scale"
        );
        ensure!(
            self.track_refine_radius_m.is_finite()
                && (0.0..=0.6).contains(&self.track_refine_radius_m)
                && self.track_refine_score_power.is_finite()
                && (0.5..=2.0).contains(&self.track_refine_score_power),
            "invalid candidate fusion"
        );
        ensure!(
            self.track_trees.len() <= 256
                && self.track_bias.is_finite()
                && self.track_bias.abs() <= 40.0,
            "invalid tree ensemble"
        );
        for tree in &self.track_trees {
            tree.validate_features(
                if self.track_feature_map == TEACHER_FEATURE_MAP {
                    33
                } else if self.track_feature_map == crate::map_context::FEATURE_MAP {
                    40
                } else if self.track_feature_map == "thermal-spectral32-v3" {
                    32
                } else if self.track_feature_map == "thermal24-v2" {
                    24
                } else if self.track_feature_map == "thermal16-v1" {
                    16
                } else {
                    12
                },
                if !self.track_feature_map.is_empty() {
                    63
                } else {
                    31
                },
            )?;
        }
        if self.track_feature_map == TEACHER_FEATURE_MAP {
            ensure!(
                (1..=200).contains(&self.track_teacher_trees)
                    && self.track_teacher_trees <= self.track_trees.len()
                    && self.track_teacher_bias.is_finite()
                    && self.track_teacher_bias.abs() <= 40.0,
                "invalid frozen teacher prefix"
            );
            for tree in &self.track_trees[..self.track_teacher_trees] {
                tree.validate_features(32, 63)?;
            }
        } else {
            ensure!(
                self.track_teacher_trees == 0 && self.track_teacher_bias == 0.0,
                "teacher metadata requires teacher feature map"
            );
        }
        ensure!(
            matches!(
                self.track_feature_map.as_str(),
                "" | "thermal16-v1"
                    | "thermal24-v2"
                    | "thermal-spectral32-v3"
                    | crate::map_context::FEATURE_MAP
                    | TEACHER_FEATURE_MAP
            ),
            "unknown track feature map"
        );
        ensure!(
            self.geometry_trees.is_empty()
                || (self.local_surface_completion
                    && self.prototypes.is_empty()
                    && matches!(
                        self.geometry_feature_map.as_str(),
                        crate::generic::FEATURE_MAP | crate::generic::WIDE_FEATURE_MAP
                    )
                    && self.geometry_trees.len() <= 1024
                    && self.geometry_bias.is_finite()
                    && self.geometry_bias.abs() <= 40.0),
            "invalid generic geometry model"
        );
        for tree in &self.geometry_trees {
            tree.validate_features(
                if self.geometry_feature_map == crate::generic::FEATURE_MAP {
                    90
                } else {
                    crate::generic::FEATURES
                },
                127,
            )?;
        }
        ensure!(
            self.volume_file.is_empty()
                || (self.volume_file == "volume.onnx"
                    && self.local_surface_completion
                    && self.prototypes.is_empty()
                    && self.geometry_trees.is_empty()
                    && self.volume_sha256.len() == 64
                    && self.volume_sha256.bytes().all(|c| c.is_ascii_hexdigit())
                    && self.volume_logit_bias.is_finite()
                    && self.volume_logit_bias.abs() <= 16.0
                    && matches!(self.volume_channels, 2 | 9 | 18 | 74 | 90)),
            "invalid 3D model descriptor"
        );
        ensure!(
            if self.navigation_file.is_empty() {
                self.navigation_sha256.is_empty()
                    && self.navigation_channels == 0
                    && self.navigation_logit_bias == 0.0
            } else {
                self.navigation_file == "navigation.onnx"
                    && self.learned_navigation_v1
                    && self.volume_file == "volume.onnx"
                    && self.volume_channels >= 74
                    && self.navigation_channels == 74
                    && self.navigation_sha256.len() == 64
                    && self
                        .navigation_sha256
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit())
                    && self.navigation_logit_bias.is_finite()
                    && self.navigation_logit_bias.abs() <= 16.0
            },
            "invalid independent navigation descriptor"
        );
        Ok(())
    }
    pub fn load_volume(&mut self, directory: &std::path::Path) -> Result<()> {
        self.validate()?;
        if !self.volume_file.is_empty() {
            self.volume_runtime = Some(std::sync::Arc::new(crate::volume::VolumeRuntime::load(
                directory,
                &self.volume_sha256,
                self.volume_channels,
            )?));
        }
        if !self.navigation_file.is_empty() {
            self.navigation_runtime = Some(std::sync::Arc::new(
                crate::volume::VolumeRuntime::load_named(
                    directory,
                    &self.navigation_file,
                    &self.navigation_sha256,
                    self.navigation_channels,
                )?,
            ));
        }
        Ok(())
    }
    pub fn geometry_score(&self, x: &[f64; crate::generic::FEATURES]) -> f64 {
        sigmoid(
            self.geometry_bias
                + self
                    .geometry_trees
                    .iter()
                    .map(|t| t.score_features(x))
                    .sum::<f64>(),
        )
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
    pub fn track_teacher_feature(&self, x: &[f64]) -> f64 {
        (self.track_teacher_bias
            + self.track_trees[..self.track_teacher_trees]
                .iter()
                .map(|t| t.score_features(x))
                .sum::<f64>())
        .clamp(-40.0, 40.0)
            / 10.0
    }
    pub fn track_score(&self, x: impl AsRef<[f64]>) -> Option<f64> {
        let x = x.as_ref();
        if !self.track_trees.is_empty() {
            return Some(sigmoid(
                self.track_bias
                    + self
                        .track_trees
                        .iter()
                        .map(|t| t.score_features(x))
                        .sum::<f64>(),
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
