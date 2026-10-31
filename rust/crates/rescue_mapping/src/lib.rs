//! Sensor-only occupancy and bearing fusion. No simulator geometry or victim labels.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
pub mod completion;
pub mod hazards;
pub mod learned;
mod navigation;
pub mod surface;
use learned::LearnedModel;

pub const VERSION: &str = "2.0";
fn default_true() -> bool {
    true
}
pub const MAX_LINE: usize = 4_194_304;
pub const MAX_REPLY: usize = 8_388_608;
pub const MAX_NODES: usize = 48;
type V3 = [f64; 3];
fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|i| a[i] - b[i])
}
fn mul(a: V3, s: f64) -> V3 {
    a.map(|x| x * s)
}
fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn finite(a: V3) -> bool {
    a.iter().all(|v| v.is_finite())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Ray {
    pub direction: V3,
    pub distance_m: f64,
    pub hit: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scan {
    pub node_id: String,
    /// Known deployment fixture, accepted only on a node's first scan.
    pub deployment_anchor: V3,
    /// Noisy odometry increment, never an absolute world pose.
    pub odometry_delta: V3,
    pub pose_sigma_m: f64,
    pub rays: Vec<Ray>,
    pub bearing: V3,
    pub bearing_sigma_deg: f64,
    pub acoustic_snr_db: f64,
    pub voice_band_ratio: f64,
    pub modulation: f64,
    pub temperature_delta_c: f64,
    /// Relative simulated gas reaction, not a calibrated ppm or %LEL reading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gas_level: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub water_depth_m: Option<f64>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub protocol_version: String,
    pub run_id: String,
    pub seq_no: u64,
    pub sim_step: u64,
    pub cell_m: f64,
    pub scans: Vec<Scan>,
    #[serde(default = "default_true")]
    pub request_map: bool,
    #[serde(default)]
    pub request_diagnostics: bool,
    #[serde(default)]
    pub request_surface: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct Cell {
    pub key: [i32; 3],
    pub occupancy: f64,
    pub observations: u32,
    pub source: &'static str,
    /// Uncalibrated ray evidence, before learned completion; absent for unobserved cells.
    pub sensor_occupancy: Option<f64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Node {
    pub node_id: String,
    pub estimated_position: V3,
    pub sigma_m: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Goal {
    pub node_id: String,
    pub position: V3,
    pub reason: &'static str,
    pub hold_until_step: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Cue {
    pub position: V3,
    /// Ranking score, NOT a calibrated probability that a person is present.
    pub score: f64,
    pub radius_m: f64,
    pub observers: usize,
    pub status: &'static str,
    pub last_seen_step: u64,
    evidence: CueEvidence,
}
#[derive(Clone, Debug, Default, Serialize)]
struct CueEvidence {
    near: f64,
    human: f64,
    heat: f64,
    range_error: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct Proposal {
    pub position: V3,
    pub features: [f64; 12],
    pub observers: usize,
    pub first_seen_step: u64,
    pub last_seen_step: u64,
}
#[derive(Clone)]
struct Track {
    position: V3,
    score: f64,
    radius: f64,
    observers: usize,
    hits: u32,
    last_seen: u64,
    spread: f64,
    first_seen: u64,
    peak_score: f64,
    evidence: CueEvidence,
}
impl Track {
    fn proposal(&self, step: u64, observer_denominator: f64) -> Proposal {
        Proposal {
            position: self.position,
            observers: self.observers,
            first_seen_step: self.first_seen,
            last_seen_step: self.last_seen,
            features: [
                self.score,
                self.peak_score,
                (1.0 + f64::from(self.hits)).ln() / 5.0,
                self.observers as f64 / observer_denominator,
                self.spread.sqrt() / 0.25,
                self.radius,
                (step - self.last_seen) as f64 / 9000.0,
                (self.last_seen - self.first_seen) as f64 / 9000.0,
                self.evidence.near / 2.0,
                self.evidence.human,
                self.evidence.heat / 2.0,
                self.evidence.range_error / 3.0,
            ],
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct MapReply {
    pub protocol_version: &'static str,
    pub run_id: String,
    pub seq_no: u64,
    pub sim_step: u64,
    pub cell_m: f64,
    pub cells: Vec<Cell>,
    pub nodes: Vec<Node>,
    pub cues: Vec<Cue>,
    pub goals: Vec<Goal>,
    pub observed_cells: usize,
    pub rejected_scans: usize,
    pub mean_entropy_bits: f64,
    pub note: &'static str,
    pub model_version: String,
    pub person_classifier_ready: bool,
    pub distributed_swarm: bool,
    pub completion_family: String,
    pub completion_confidence: f64,
    pub proposals: Vec<Proposal>,
    pub hazards: Vec<hazards::Hazard>,
    pub gas_observed_nodes: usize,
    pub water_observed_nodes: usize,
    pub surface_resolution_m: Option<f64>,
    pub surface_points: Vec<[f64; 6]>,
}
#[derive(Default, Clone)]
struct Voxel {
    log_odds: f64,
    count: u32,
}
#[derive(Clone)]
struct Pose {
    position: V3,
    sigma: f64,
    anchor: V3,
}
#[derive(Clone)]
struct Bearing {
    id: String,
    origin: V3,
    direction: V3,
    sigma: f64,
    weight: f64,
    step: u64,
    pose_sigma: f64,
    radial: Option<(f64, f64)>,
    human: f64,
    heat: f64,
}
type NodeEvidence = (f64, V3, f64, usize, f64, f64, f64, f64);
pub struct Mapper {
    run: String,
    cell: f64,
    last_seq: Option<u64>,
    last_step: u64,
    voxels: BTreeMap<[i32; 3], Voxel>,
    poses: BTreeMap<String, Pose>,
    bearings: Vec<Bearing>,
    rejected: usize,
    cached: Option<(Vec<u8>, MapReply)>,
    model: Option<Arc<LearnedModel>>,
    acoustic_history: BTreeMap<String, VecDeque<[f64; 4]>>,
    tracks: Vec<Track>,
    last_cue_step: u64,
    verify_until: BTreeMap<String, u64>,
    verify_after: BTreeMap<String, u64>,
    archived_tracks: Vec<Track>,
    proposal_mode: bool,
    probe_cues: Vec<Cue>,
    environment: hazards::HazardMap,
    fine_surface: surface::SurfaceMap,
    recent_visits: BTreeMap<[i32; 2], u64>,
    recent_visits3: BTreeMap<[i32; 3], u64>,
}
impl Mapper {
    pub fn new(run: &str, cell: f64) -> Result<Self> {
        ensure!(
            !run.is_empty() && run.len() <= 96 && cell.is_finite() && (0.01..=0.2).contains(&cell),
            "invalid session"
        );
        Ok(Self {
            run: run.into(),
            cell,
            last_seq: None,
            last_step: 0,
            voxels: BTreeMap::new(),
            poses: BTreeMap::new(),
            bearings: vec![],
            rejected: 0,
            cached: None,
            model: None,
            acoustic_history: BTreeMap::new(),
            tracks: vec![],
            last_cue_step: 0,
            verify_until: BTreeMap::new(),
            verify_after: BTreeMap::new(),
            archived_tracks: vec![],
            proposal_mode: false,
            probe_cues: vec![],
            environment: hazards::HazardMap::default(),
            fine_surface: surface::SurfaceMap::default(),
            recent_visits: BTreeMap::new(),
            recent_visits3: BTreeMap::new(),
        })
    }
    pub fn with_model(run: &str, cell: f64, model: Option<Arc<LearnedModel>>) -> Result<Self> {
        let mut mapper = Self::new(run, cell)?;
        if let Some(m) = &model {
            m.validate()?;
            ensure!((m.cell_m - cell).abs() < 1e-9, "model cell size mismatch");
            mapper.proposal_mode = m.has_track_model() || m.distributed_swarm;
        }
        mapper.model = model;
        Ok(mapper)
    }
    fn key(&self, p: V3) -> [i32; 3] {
        p.map(|v| (v / self.cell).floor() as i32)
    }
    fn center(&self, k: [i32; 3]) -> V3 {
        k.map(|v| (f64::from(v) + 0.5) * self.cell)
    }
    fn in_bounds(k: [i32; 3]) -> bool {
        (-16..16).contains(&k[0]) && (-3..18).contains(&k[1]) && (-6..64).contains(&k[2])
    }
    fn update(&mut self, k: [i32; 3], occupied: bool, reliability: f64) {
        if !Self::in_bounds(k) {
            return;
        }
        let v = self.voxels.entry(k).or_default();
        v.log_odds =
            (v.log_odds + if occupied { 0.85 } else { -0.42 } * reliability).clamp(-4.0, 4.0);
        v.count = v.count.saturating_add(1);
    }
    pub fn apply(&mut self, p: &Packet) -> Result<MapReply> {
        ensure!(
            p.protocol_version == VERSION
                && p.run_id == self.run
                && (p.cell_m - self.cell).abs() < 1e-9,
            "session contract mismatch"
        );
        let encoded = serde_json::to_vec(p)?;
        ensure!(
            encoded.len() <= MAX_LINE && p.scans.len() <= MAX_NODES,
            "packet budget exceeded"
        );
        if self.last_seq == Some(p.seq_no) {
            let (previous, reply) = self.cached.as_ref().expect("cached response");
            ensure!(*previous == encoded, "sequence reused with changed payload");
            return Ok(reply.clone());
        }
        ensure!(
            self.last_seq
                .map_or(p.seq_no == 0, |s| s.checked_add(1) == Some(p.seq_no)),
            "non-contiguous sequence"
        );
        ensure!(
            p.sim_step >= self.last_step && p.sim_step - self.last_step <= 500,
            "invalid clock step"
        );
        let mut ids = BTreeSet::new();
        ensure!(
            self.poses.len()
                + p.scans
                    .iter()
                    .filter(|s| !self.poses.contains_key(&s.node_id))
                    .count()
                <= MAX_NODES,
            "session node budget exceeded"
        );
        // Validate the complete packet before mutating state.
        for s in &p.scans {
            ensure!(
                !s.node_id.is_empty() && s.node_id.len() <= 48 && ids.insert(&s.node_id),
                "duplicate/invalid node"
            );
            ensure!(
                finite(s.deployment_anchor)
                    && finite(s.odometry_delta)
                    && norm(s.odometry_delta)
                        < if s.deployment_anchor[1] > 0.25 {
                            1.5
                        } else {
                            self.cell * 12.0
                        },
                "invalid odometry"
            );
            ensure!(
                s.deployment_anchor[0].abs() <= self.cell * 14.0
                    && (-0.15..=1.2).contains(&s.deployment_anchor[1])
                    && (-0.3..=3.1).contains(&s.deployment_anchor[2]),
                "anchor is outside deployment zone"
            );
            ensure!(
                s.pose_sigma_m.is_finite()
                    && (0.0..=2.0).contains(&s.pose_sigma_m)
                    && s.rays.len() <= 64,
                "invalid scan"
            );
            ensure!(
                finite(s.bearing) && (0.0..=1.1).contains(&norm(s.bearing)),
                "invalid bearing"
            );
            ensure!(
                s.bearing_sigma_deg.is_finite() && (3.0..=180.0).contains(&s.bearing_sigma_deg),
                "invalid acoustic uncertainty"
            );
            for v in [
                s.acoustic_snr_db,
                s.voice_band_ratio,
                s.modulation,
                s.temperature_delta_c,
            ] {
                ensure!(v.is_finite(), "non-finite cue");
            }
            ensure!(
                (-60.0..=80.0).contains(&s.acoustic_snr_db)
                    && (0.0..=1.0).contains(&s.voice_band_ratio)
                    && (0.0..=1.0).contains(&s.modulation)
                    && (-20.0..=40.0).contains(&s.temperature_delta_c),
                "cue outside range"
            );
            ensure!(
                s.gas_level
                    .is_none_or(|v| v.is_finite() && (0.0..=1.0).contains(&v))
                    && s.water_depth_m
                        .is_none_or(|v| v.is_finite() && (0.0..=0.15).contains(&v)),
                "invalid environmental observation"
            );
            for r in &s.rays {
                ensure!(
                    finite(r.direction)
                        && (0.95..=1.05).contains(&norm(r.direction))
                        && r.distance_m.is_finite()
                        && (0.0..=self.cell * 40.0).contains(&r.distance_m),
                    "invalid range ray"
                );
            }
            if let Some(pose) = self.poses.get(&s.node_id) {
                ensure!(
                    pose.anchor == s.deployment_anchor,
                    "deployment anchor changed"
                );
            }
        }
        self.proposal_mode |= p.request_diagnostics;
        for s in &p.scans {
            let pose = self.poses.entry(s.node_id.clone()).or_insert(Pose {
                position: s.deployment_anchor,
                sigma: 0.0,
                anchor: s.deployment_anchor,
            });
            pose.position = add(pose.position, s.odometry_delta);
            pose.sigma = s.pose_sigma_m;
            let origin = pose.position;
            let visit = [
                (origin[0] / self.cell).floor() as i32,
                (origin[2] / self.cell).floor() as i32,
            ];
            self.recent_visits.insert(visit, p.sim_step);
            self.recent_visits3.insert(
                pose.position.map(|v| (v / self.cell).floor() as i32),
                p.sim_step,
            );
            // High pose uncertainty cannot silently paint precise walls.
            if s.pose_sigma_m > self.cell * 3.0 {
                self.rejected += 1;
                continue;
            }
            if let Some(value) = s.gas_level {
                self.environment.observe(
                    "gas",
                    origin,
                    value,
                    s.pose_sigma_m,
                    &s.node_id,
                    p.sim_step,
                );
            }
            if let Some(value) = s.water_depth_m {
                self.environment.observe(
                    "water",
                    origin,
                    value,
                    s.pose_sigma_m,
                    &s.node_id,
                    p.sim_step,
                );
            }
            let reliability = 1.0 / (1.0 + (s.pose_sigma_m / self.cell).powi(2));
            let mut observed = BTreeMap::new();
            for ray in &s.rays {
                let n = (ray.distance_m / (self.cell * 0.45)).ceil() as usize;
                for i in 0..n {
                    observed
                        .entry(
                            self.key(add(origin, mul(ray.direction, i as f64 * self.cell * 0.45))),
                        )
                        .or_insert(false);
                }
                if ray.hit {
                    if p.request_surface {
                        self.fine_surface.observe(
                            add(origin, mul(ray.direction, ray.distance_m)),
                            reliability,
                            p.sim_step,
                        );
                    }
                    observed.insert(
                        self.key(add(origin, mul(ray.direction, ray.distance_m))),
                        true,
                    );
                }
            }
            for (k, occupied) in observed {
                self.update(k, occupied, reliability);
            }
            let features = [
                s.acoustic_snr_db / 20.0,
                s.voice_band_ratio,
                s.modulation,
                s.temperature_delta_c / 2.0,
            ];
            let history = self.acoustic_history.entry(s.node_id.clone()).or_default();
            history.push_back(features);
            let window = self.model.as_ref().map_or(1, |m| m.acoustic_window_frames);
            while history.len() > window {
                history.pop_front();
            }
            let mean: [f64; 4] = std::array::from_fn(|i| {
                history.iter().map(|x| x[i]).sum::<f64>() / history.len() as f64
            });
            if let Some(model) = &self.model
                && model.human_score(features) > 0.8
                && s.temperature_delta_c > 0.45
                && p.sim_step >= *self.verify_after.get(&s.node_id).unwrap_or(&0)
            {
                let duration = (model.acoustic_window_frames as u64 * 25 + 400).min(2000);
                self.verify_until
                    .insert(s.node_id.clone(), p.sim_step + duration);
                self.verify_after
                    .insert(s.node_id.clone(), p.sim_step + duration + 1500);
            }
            if s.acoustic_snr_db > if self.model.is_some() { -60.0 } else { 2.0 }
                && norm(s.bearing) > 0.9
            {
                let weight = if let Some(model) = &self.model {
                    let human = model.human_score(mean);
                    if human < if self.proposal_mode { 0.35 } else { 0.55 } {
                        0.0
                    } else {
                        // SNR is already a classifier feature and increases bearing sigma.
                        // Weight by angular precision instead of rejecting weak multi-node evidence.
                        human / (1.0 + (s.bearing_sigma_deg / 45.0).powi(2))
                    }
                } else {
                    ((s.acoustic_snr_db - 2.0) / 20.0).clamp(0.0, 1.0)
                        * s.voice_band_ratio
                        * s.modulation
                };
                if weight > 0.05 {
                    self.bearings.push(Bearing {
                        id: s.node_id.clone(),
                        origin,
                        direction: s.bearing,
                        sigma: s.bearing_sigma_deg.to_radians(),
                        weight: weight * (1.0 + s.temperature_delta_c.clamp(0.0, 3.0) * 0.08),
                        step: p.sim_step,
                        pose_sigma: s.pose_sigma_m,
                        radial: self.model.as_ref().and_then(|m| {
                            m.acoustic_range(
                                if m.log_range_weights.len() == 6 {
                                    features
                                } else {
                                    mean
                                },
                                s.bearing_sigma_deg,
                            )
                        }),
                        human: self
                            .model
                            .as_ref()
                            .map_or(s.voice_band_ratio * s.modulation, |m| m.human_score(mean)),
                        heat: s.temperature_delta_c,
                    });
                }
            }
        }
        self.bearings.retain(|b| p.sim_step - b.step < 1500);
        let bearing_budget = if self.model.as_ref().is_some_and(|m| m.distributed_swarm) {
            self.poses.len().max(12) * 20
        } else {
            240
        };
        if self.bearings.len() > bearing_budget {
            self.bearings.drain(0..self.bearings.len() - bearing_budget);
        }
        self.last_seq = Some(p.seq_no);
        self.last_step = p.sim_step;
        let mut cells: Vec<_> = self
            .voxels
            .iter()
            .filter(|_| p.request_map)
            .map(|(k, v)| Cell {
                key: *k,
                occupancy: self.model.as_ref().map_or_else(
                    || 1.0 / (1.0 + (-v.log_odds).exp()),
                    |m| {
                        if m.local_surface_completion {
                            1.0 / (1.0 + (-v.log_odds).exp())
                        } else {
                            m.occupancy_score(v.log_odds, v.count)
                        }
                    },
                ),
                observations: v.count,
                source: "sensor",
                sensor_occupancy: Some(1.0 / (1.0 + (-v.log_odds).exp())),
            })
            .collect();
        let mut completion_family = String::new();
        let mut completion_confidence = 0.0;
        if p.request_map
            && let Some(model) = &self.model
            && !model.local_surface_completion
            && self.voxels.len() >= 200
        {
            let mut scores = vec![0.0; model.prototypes.len()];
            for (k, v) in &self.voxels {
                if let Some(index) = learned::index(*k) {
                    let observed = 1.0 / (1.0 + (-v.log_odds).exp());
                    let certainty = (observed * 2.0 - 1.0).abs();
                    for (score, prototype) in scores.iter_mut().zip(&model.prototypes) {
                        let q = if prototype.sensor_occupancy.is_empty() {
                            prototype.occupancy[index]
                        } else {
                            (1.0 - model.sensor_likelihood_mix) * prototype.occupancy[index]
                                + model.sensor_likelihood_mix * prototype.sensor_occupancy[index]
                        }
                        .clamp(0.05, 0.95);
                        *score += model.geometry_log_likelihood_scale
                            * certainty
                            * (observed * q.ln() + (1.0 - observed) * (1.0 - q).ln());
                    }
                }
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut weights: Vec<_> = scores.iter().map(|s| (s - maximum).exp()).collect();
            let total: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= total;
            }
            let (best, confidence) = weights
                .iter()
                .copied()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("prototypes");
            completion_family = model.prototypes[best].family.clone();
            completion_confidence = confidence;
            for cell in &mut cells {
                let i = learned::index(cell.key).expect("bounded");
                let prior = weights
                    .iter()
                    .zip(&model.prototypes)
                    .map(|(w, p)| w * p.occupancy[i])
                    .sum::<f64>();
                let observed = self.voxels.get(&cell.key).expect("sensor cell");
                cell.occupancy = model.fused_score(observed.log_odds, observed.count, prior);
                if !model.fusion_weights.is_empty() && prior > 0.65 {
                    cell.source = "sensor+learned_prior";
                }
            }
            for x in -16..16 {
                for y in -3..18 {
                    for z in -6..64 {
                        let k = [x, y, z];
                        if self.voxels.contains_key(&k) {
                            continue;
                        }
                        let i = learned::index(k).expect("bounded");
                        let prior = weights
                            .iter()
                            .zip(&model.prototypes)
                            .map(|(w, p)| w * p.occupancy[i])
                            .sum::<f64>();
                        if prior > 0.65 {
                            cells.push(Cell {
                                key: k,
                                occupancy: prior,
                                observations: 0,
                                source: "learned_prior",
                                sensor_occupancy: None,
                            });
                        }
                    }
                }
            }
        }
        if p.request_map
            && self
                .model
                .as_ref()
                .is_some_and(|m| m.local_surface_completion)
        {
            let mut inferred = completion::infer(&cells);
            for c in &mut cells {
                if let Some(q) = inferred.remove(&c.key) {
                    c.occupancy = q;
                    c.source = "local_surface_completion";
                }
            }
            cells.extend(inferred.into_iter().map(|(key, occupancy)| Cell {
                key,
                occupancy,
                observations: 0,
                source: "local_surface_completion",
                sensor_occupancy: None,
            }));
            completion_family = "local-plane-v1".into();
        }
        let entropy = cells
            .iter()
            .map(|c| {
                let q = c.occupancy.clamp(1e-9, 1.0 - 1e-9);
                -q * q.log2() - (1.0 - q) * (1.0 - q).log2()
            })
            .sum::<f64>()
            / cells.len().max(1) as f64;
        let cues = if p.sim_step >= self.last_cue_step + 100 {
            let instantaneous = self.cues();
            if self.model.as_ref().is_some_and(|m| m.acoustic_probe) {
                self.probe_cues.clone_from(&instantaneous);
            }
            self.last_cue_step = p.sim_step;
            self.track_cues(&instantaneous, p.sim_step)
        } else {
            self.current_tracks(p.sim_step)
        };
        let mut reply = MapReply {
            protocol_version: VERSION,
            run_id: self.run.clone(),
            seq_no: p.seq_no,
            sim_step: p.sim_step,
            cell_m: self.cell,
            cells,
            nodes: self
                .poses
                .iter()
                .map(|(id, p)| Node {
                    node_id: id.clone(),
                    estimated_position: p.position,
                    sigma_m: p.sigma,
                })
                .collect(),
            cues,
            goals: self.frontiers(),
            observed_cells: self.voxels.len(),
            rejected_scans: self.rejected,
            hazards: self.environment.current(p.sim_step),
            gas_observed_nodes: self.environment.gas_nodes.len(),
            water_observed_nodes: self.environment.water_nodes.len(),
            surface_resolution_m: p.request_surface.then_some(0.01),
            surface_points: if p.request_surface {
                self.fine_surface.current(p.sim_step)
            } else {
                vec![]
            },
            mean_entropy_bits: entropy,
            note: "Sensor observations and learned geometry completion are distinguished. Cue scores are uncalibrated search priorities, not confirmed people.",
            model_version: self
                .model
                .as_ref()
                .map_or_else(|| "sensor-baseline".into(), |m| m.version.clone()),
            person_classifier_ready: self.model.as_ref().is_some_and(|m| m.has_track_model()),
            distributed_swarm: self.model.as_ref().is_some_and(|m| m.distributed_swarm),
            completion_family,
            completion_confidence,
            proposals: if p.request_map && p.request_diagnostics {
                self.tracks
                    .iter()
                    .chain(&self.archived_tracks)
                    .map(|t| t.proposal(p.sim_step, self.observer_denominator()))
                    .collect()
            } else {
                vec![]
            },
        };
        if !p.request_map {
            reply.cells.clear();
        }
        if !reply.surface_points.is_empty() {
            let cell_bytes = serde_json::to_vec(&reply.cells)?.len();
            let limit = MAX_REPLY.saturating_sub(cell_bytes + 262_144) / 96;
            if limit == 0 {
                reply.surface_points.clear();
            } else if reply.surface_points.len() > limit {
                let stride = reply.surface_points.len().div_ceil(limit);
                reply.surface_points = reply
                    .surface_points
                    .into_iter()
                    .step_by(stride)
                    .take(limit)
                    .collect();
            }
        }
        self.cached = Some((encoded, reply.clone()));
        Ok(reply)
    }
    fn cues(&self) -> Vec<Cue> {
        let mut candidates = vec![];
        // Acoustic evidence can nominate unobserved space, but never known solid cells.
        for x in (-14..15).step_by(2) {
            for y in (0..if self
                .model
                .as_ref()
                .is_some_and(|m| m.local_surface_completion)
            {
                18
            } else {
                12
            })
                .step_by(2)
            {
                for z in (0..60).step_by(2) {
                    let k = [x, y, z];
                    if self.voxels.get(&k).is_some_and(|v| v.log_odds > 1.5) {
                        continue;
                    }
                    let p = self.center(k);
                    let mut by_node: BTreeMap<&str, NodeEvidence> = BTreeMap::new();
                    for b in &self.bearings {
                        let delta = sub(p, b.origin);
                        let d = norm(delta);
                        if d < self.cell * 2.0 || d > self.cell * 45.0 {
                            continue;
                        }
                        let angle = (dot(delta, b.direction) / d).clamp(-1.0, 1.0).acos();
                        let uncertainty = b.sigma + (b.pose_sigma / d).atan();
                        let radial = b.radial.map_or(1.0, |(range, sigma)| {
                            (-0.5
                                * (((d + 0.04) / (range + 0.04)).ln()
                                    / (sigma + b.pose_sigma / (d + 0.04)))
                                    .powi(2))
                            .exp()
                        });
                        let score =
                            b.weight * (-0.5 * (angle / uncertainty).powi(2)).exp() * radial;
                        let old = by_node
                            .entry(&b.id)
                            .or_insert((0.0, b.origin, 0.0, 0, 0.0, 0.0, 0.0, 0.0));
                        old.0 += score;
                        old.2 += uncertainty;
                        old.3 += 1;
                        old.4 += b.human;
                        old.5 += b.heat;
                        old.6 += d;
                        old.7 += b.radial.map_or(0.0, |(range, sigma)| {
                            (((d + 0.04) / (range + 0.04)).ln()
                                / (sigma + b.pose_sigma / (d + 0.04)))
                                .abs()
                        });
                    }
                    for item in by_node.values_mut() {
                        item.0 /= item.3 as f64;
                        item.2 /= item.3 as f64;
                        item.4 /= item.3 as f64;
                        item.5 /= item.3 as f64;
                        item.6 /= item.3 as f64;
                        item.7 /= item.3 as f64;
                    }
                    let support: Vec<_> = by_node
                        .values()
                        .filter(|v| v.0 > if self.proposal_mode { 0.05 } else { 0.16 })
                        .collect();
                    if support.len() < 3 {
                        continue;
                    }
                    let mut baseline: f64 = 0.0;
                    for a in &support {
                        for b in &support {
                            baseline = baseline.max(norm(sub(a.1, b.1)));
                        }
                    }
                    if baseline < self.cell * 2.0 {
                        continue;
                    }
                    let score = support.iter().map(|s| s.0).sum::<f64>()
                        / if self
                            .model
                            .as_ref()
                            .is_some_and(|m| m.local_surface_completion)
                        {
                            support.len().max(4)
                        } else {
                            by_node.len().max(1)
                        } as f64;
                    if score > if self.proposal_mode { 0.06 } else { 0.18 } {
                        let angular =
                            support.iter().map(|s| s.2).sum::<f64>() / support.len() as f64;
                        candidates.push(Cue {
                            position: p,
                            score: score.min(0.99),
                            radius_m: (self.cell * 2.0 + angular * self.cell * 12.0)
                                .max(self.cell * 2.0),
                            observers: support.len(),
                            status: "UNCONFIRMED",
                            last_seen_step: self.last_step,
                            evidence: CueEvidence {
                                near: support.iter().map(|s| s.6).fold(f64::INFINITY, f64::min),
                                human: support.iter().map(|s| s.4).sum::<f64>()
                                    / support.len() as f64,
                                heat: support.iter().map(|s| s.5).sum::<f64>()
                                    / support.len() as f64,
                                range_error: support.iter().map(|s| s.7).sum::<f64>()
                                    / support.len() as f64,
                            },
                        });
                    }
                }
            }
        }
        // A broad likelihood hill is one location hypothesis, not many people.
        let field: BTreeMap<_, _> = candidates
            .iter()
            .map(|c| (self.key(c.position), c.score))
            .collect();
        candidates.retain(|c| {
            let k = self.key(c.position);
            for dx in [-2, 0, 2] {
                for dy in [-2, 0, 2] {
                    for dz in [-2, 0, 2] {
                        let neighbor = [k[0] + dx, k[1] + dy, k[2] + dz];
                        if neighbor == k {
                            continue;
                        }
                        if let Some(&score) = field.get(&neighbor)
                            && (score > c.score + 1e-9
                                || ((score - c.score).abs() < 1e-9 && neighbor < k))
                        {
                            return false;
                        }
                    }
                }
            }
            true
        });
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
        if let Some(best) = candidates.first().map(|c| c.score)
            && !self.proposal_mode
        {
            candidates.retain(|c| c.score >= best * 0.70);
        }
        let mut result: Vec<Cue> = vec![];
        for c in candidates {
            if result
                .iter()
                .all(|r| norm(sub(r.position, c.position)) > r.radius_m.max(self.cell * 4.0))
            {
                result.push(c);
            }
            if result.len() == if self.proposal_mode { 6 } else { 3 } {
                break;
            }
        }
        result
    }
    fn track_cues(&mut self, cues: &[Cue], step: u64) -> Vec<Cue> {
        let mut active = vec![];
        for track in self.tracks.drain(..) {
            if step - track.last_seen <= if track.hits < 4 { 500 } else { 6000 } {
                active.push(track);
            } else if self.proposal_mode {
                self.archived_tracks.push(track);
            }
        }
        self.tracks = active;
        if self.archived_tracks.len() > 512 {
            self.archived_tracks
                .drain(..self.archived_tracks.len() - 512);
        }
        let mut used = BTreeSet::new();
        for cue in cues {
            let nearest = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(i, t)| !used.contains(i) && norm(sub(t.position, cue.position)) < 0.20)
                .min_by(|a, b| {
                    norm(sub(a.1.position, cue.position))
                        .total_cmp(&norm(sub(b.1.position, cue.position)))
                })
                .map(|(i, _)| i);
            if let Some(i) = nearest {
                let track = &mut self.tracks[i];
                let delta = sub(cue.position, track.position);
                let alpha = 1.0 / f64::from((track.hits + 1).min(12));
                track.position = add(track.position, mul(delta, alpha));
                track.spread = (1.0 - alpha) * track.spread + alpha * dot(delta, delta);
                track.score = (1.0 - alpha) * track.score + alpha * cue.score;
                track.peak_score = track.peak_score.max(cue.score);
                track.evidence = CueEvidence {
                    near: (1.0 - alpha) * track.evidence.near + alpha * cue.evidence.near,
                    human: (1.0 - alpha) * track.evidence.human + alpha * cue.evidence.human,
                    heat: (1.0 - alpha) * track.evidence.heat + alpha * cue.evidence.heat,
                    range_error: (1.0 - alpha) * track.evidence.range_error
                        + alpha * cue.evidence.range_error,
                };
                track.radius = cue.radius_m;
                track.observers = track.observers.max(cue.observers);
                track.hits += 1;
                track.last_seen = step;
                used.insert(i);
            } else if self.tracks.len() < if self.proposal_mode { 64 } else { 24 } {
                self.tracks.push(Track {
                    position: cue.position,
                    score: cue.score,
                    radius: cue.radius_m,
                    observers: cue.observers,
                    hits: 1,
                    last_seen: step,
                    spread: 0.0,
                    first_seen: step,
                    peak_score: cue.score,
                    evidence: cue.evidence.clone(),
                });
                used.insert(self.tracks.len() - 1);
            }
        }
        self.tracks.retain(|t| step - t.last_seen <= 6000);
        self.current_tracks(step)
    }
    fn current_tracks(&self, step: u64) -> Vec<Cue> {
        if let Some(model) = &self.model
            && model.has_track_model()
        {
            let mut candidates: Vec<_> = self
                .tracks
                .iter()
                .chain(&self.archived_tracks)
                .filter_map(|t| {
                    let score = model
                        .track_score(t.proposal(step, self.observer_denominator()).features)?;
                    (score >= model.track_threshold).then(|| Cue {
                        position: t.position,
                        score,
                        radius_m: t.radius.max(t.spread.sqrt() * 2.0),
                        observers: t.observers,
                        status: "UNCONFIRMED_LEARNED_TRACK",
                        last_seen_step: t.last_seen,
                        evidence: t.evidence.clone(),
                    })
                })
                .collect();
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            let mut result: Vec<Cue> = vec![];
            for cue in candidates {
                if result
                    .iter()
                    .all(|c| norm(sub(c.position, cue.position)) > model.track_nms_m)
                {
                    result.push(cue);
                }
                if result.len() == 3 {
                    break;
                }
            }
            return result;
        }
        let mut cues: Vec<_> = self
            .tracks
            .iter()
            .filter(|t| t.hits >= 4 && t.observers >= 4 && t.spread < 0.04)
            .map(|t| Cue {
                position: t.position,
                score: t.score / (1.0 + (step - t.last_seen) as f64 / 6000.0),
                radius_m: t.radius.max(t.spread.sqrt() * 2.0),
                observers: t.observers,
                status: "UNCONFIRMED_TRACK",
                last_seen_step: t.last_seen,
                evidence: t.evidence.clone(),
            })
            .filter(|c| c.score >= 0.18)
            .collect();
        cues.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut result: Vec<Cue> = vec![];
        for cue in cues {
            if result
                .iter()
                .all(|c| norm(sub(c.position, cue.position)) > 0.25)
            {
                result.push(cue);
            }
            if result.len() == 3 {
                break;
            }
        }
        result
    }
    fn observer_denominator(&self) -> f64 {
        if self.model.as_ref().is_some_and(|m| m.distributed_swarm) {
            self.poses.len().max(1) as f64
        } else {
            12.0
        }
    }
    fn frontiers(&self) -> Vec<Goal> {
        let distributed = self.model.as_ref().is_some_and(|m| m.distributed_swarm);
        let verification_budget = if distributed {
            self.poses.len().div_ceil(8).clamp(3, 6)
        } else {
            usize::MAX
        };
        let mut verification_roster = BTreeSet::new();
        if distributed {
            // A cue never recruits the whole swarm when frontier routing is temporarily unavailable.
            // Reserve at most three nearby observers per raw cue; all other nodes keep mapping.
            for cue in &self.probe_cues {
                let mut nearby: Vec<_> = self
                    .poses
                    .iter()
                    .filter(|(id, _)| !verification_roster.contains(*id))
                    .collect();
                nearby.sort_by(|(aid, a), (bid, b)| {
                    norm(sub(a.position, cue.position))
                        .total_cmp(&norm(sub(b.position, cue.position)))
                        .then_with(|| aid.cmp(bid))
                });
                for (id, _) in nearby.into_iter().take(3) {
                    if verification_roster.len() >= verification_budget {
                        break;
                    }
                    verification_roster.insert(id.clone());
                }
                if verification_roster.len() >= verification_budget {
                    break;
                }
            }
            for (id, &until) in &self.verify_until {
                if until > self.last_step && verification_roster.len() < verification_budget {
                    verification_roster.insert(id.clone());
                }
            }
        }
        if self
            .model
            .as_ref()
            .is_some_and(|m| m.local_surface_completion)
        {
            return navigation::goals(self, &verification_roster);
        }
        let mut free = BTreeSet::new();
        let mut blocked = BTreeSet::new();
        for (k, v) in &self.voxels {
            if k[1] == 0 {
                if v.log_odds < -0.5 {
                    free.insert([k[0], k[2]]);
                }
                if v.log_odds > 1.2 {
                    blocked.insert([k[0], k[2]]);
                }
            }
        }
        let neighbors = |k: [i32; 2]| {
            [
                [k[0] - 1, k[1]],
                [k[0] + 1, k[1]],
                [k[0], k[1] - 1],
                [k[0], k[1] + 1],
            ]
        };
        let mut assigned: Vec<[i32; 2]> = vec![];
        let mut goals = vec![];
        for (ordinal, (id, pose)) in self.poses.iter().enumerate() {
            // Scatter landing heights vary. A single global y=0 slice incorrectly leaves
            // nodes standing on a slab without any mapped route, and used to recruit them as probes.
            if distributed {
                free.clear();
                blocked.clear();
                for (k, v) in &self.voxels {
                    let y = (f64::from(k[1]) + 0.5) * self.cell;
                    if y < pose.position[1] - 0.025 || y > pose.position[1] + 0.085 {
                        continue;
                    }
                    if v.log_odds < -0.5 {
                        free.insert([k[0], k[2]]);
                    }
                    if v.log_odds > 1.2 && y > pose.position[1] + 0.012 {
                        blocked.insert([k[0], k[2]]);
                    }
                }
            }
            if let Some(&until) = self.verify_until.get(id)
                && until > self.last_step
                && (!distributed || verification_roster.contains(id))
            {
                goals.push(Goal {
                    node_id: id.clone(),
                    position: pose.position,
                    reason: "acoustic and thermal verification dwell",
                    hold_until_step: until,
                });
                continue;
            }
            let start3 = self.key(pose.position);
            let start = [start3[0], start3[2]];
            let mut queue = VecDeque::from([start]);
            let mut previous = BTreeMap::from([(start, start)]);
            let mut distance = BTreeMap::from([(start, 0u32)]);
            while let Some(k) = queue.pop_front() {
                if previous.len() > 2000 {
                    break;
                }
                for n in neighbors(k) {
                    if (-12..12).contains(&n[0])
                        && (if distributed {
                            (-6..64).contains(&n[1])
                        } else {
                            (-4..54).contains(&n[1])
                        })
                        && free.contains(&n)
                        && !blocked.contains(&n)
                        && !previous.contains_key(&n)
                    {
                        previous.insert(n, k);
                        distance.insert(n, distance[&k] + 1);
                        queue.push_back(n);
                    }
                }
            }
            let mut best = None;
            let mut best_score = f64::NEG_INFINITY;
            for (&k, &d) in &distance {
                if d < 2
                    || assigned
                        .iter()
                        .any(|a| (a[0] - k[0]).abs() + (a[1] - k[1]).abs() < 4)
                {
                    continue;
                }
                let unknown = neighbors(k)
                    .iter()
                    .filter(|n| {
                        (!distributed || ((-12..12).contains(&n[0]) && (-6..64).contains(&n[1])))
                            && !free.contains(*n)
                            && !blocked.contains(*n)
                    })
                    .count();
                if unknown == 0 && !distributed {
                    continue;
                }
                let score = if distributed {
                    let age = self.recent_visits.get(&k).map_or(1.0, |step| {
                        self.last_step.saturating_sub(*step).min(3000) as f64 / 3000.0
                    });
                    let spacing = self
                        .poses
                        .iter()
                        .filter(|(other, _)| *other != id)
                        .map(|(_, p)| {
                            norm(sub(
                                [
                                    (f64::from(k[0]) + 0.5) * self.cell,
                                    pose.position[1],
                                    (f64::from(k[1]) + 0.5) * self.cell,
                                ],
                                p.position,
                            ))
                        })
                        .fold(1.0, f64::min);
                    unknown as f64 * 0.9 + age * 0.7 + spacing * 0.8 - f64::from(d) * 0.008
                } else {
                    unknown as f64 * 0.4 + k[1] as f64 * 0.06 - d as f64 * 0.005
                };
                if score > best_score {
                    best_score = score;
                    best = Some(k);
                }
            }
            let probe_nodes = if self.model.as_ref().is_some_and(|m| m.distributed_swarm) {
                self.poses.len().div_ceil(4).max(3)
            } else {
                3
            };
            let should_probe = self.model.as_ref().is_some_and(|m| m.acoustic_probe)
                && !self.probe_cues.is_empty()
                && if distributed {
                    verification_roster.contains(id)
                } else {
                    ordinal < probe_nodes || best.is_none()
                };
            if let Some(target) = best.filter(|_| !should_probe) {
                assigned.push(target);
                let mut waypoint = target;
                while previous[&waypoint] != start {
                    waypoint = previous[&waypoint];
                }
                let mut route = vec![target];
                let mut cursor = target;
                while previous[&cursor] != start {
                    cursor = previous[&cursor];
                    route.push(cursor);
                }
                waypoint = *route.iter().rev().nth(2).unwrap_or(&waypoint);
                goals.push(Goal {
                    node_id: id.clone(),
                    position: [
                        (f64::from(waypoint[0]) + 0.5) * self.cell,
                        pose.position[1],
                        (f64::from(waypoint[1]) + 0.5) * self.cell,
                    ],
                    reason: "observed-free route toward an unobserved frontier",
                    hold_until_step: 0,
                });
            } else if should_probe {
                // Coverage can finish before a weak sound is close enough to classify.
                // Assign peaks to separate nodes using only raw bearing evidence and
                // observed-free routes. Neither retained classifier scores nor truth enter.
                let cue = &self.probe_cues[ordinal % self.probe_cues.len()];
                let point = |k: [i32; 2]| {
                    [
                        (f64::from(k[0]) + 0.5) * self.cell,
                        pose.position[1],
                        (f64::from(k[1]) + 0.5) * self.cell,
                    ]
                };
                if norm(sub(pose.position, cue.position)) <= 0.15 {
                    goals.push(Goal {
                        node_id: id.clone(),
                        position: pose.position,
                        reason: "acoustic probe sampling at a sensor-derived peak",
                        hold_until_step: self.last_step + 100,
                    });
                    continue;
                }
                let target = distance
                    .iter()
                    .filter(|(k, d)| {
                        **d >= 2
                            && norm(sub(point(**k), cue.position)) + self.cell
                                < norm(sub(pose.position, cue.position))
                    })
                    .min_by(|(a, da), (b, db)| {
                        (norm(sub(point(**a), cue.position)) + f64::from(**da) * 0.002).total_cmp(
                            &(norm(sub(point(**b), cue.position)) + f64::from(**db) * 0.002),
                        )
                    })
                    .map(|(k, _)| *k);
                if let Some(target) = target {
                    let mut route = vec![target];
                    let mut cursor = target;
                    while previous[&cursor] != start {
                        cursor = previous[&cursor];
                        route.push(cursor);
                    }
                    let waypoint = *route.last().expect("nonempty acoustic route");
                    goals.push(Goal {
                        node_id: id.clone(),
                        position: point(waypoint),
                        reason: "observed-free acoustic probe route",
                        hold_until_step: 0,
                    });
                }
            }
        }
        goals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scattered_swarm_reserves_mapping_nodes_even_when_all_nodes_hear_a_person() {
        let mut model = learned_fixture();
        model.distributed_swarm = true;
        model.acoustic_probe = true;
        let mut mapper = Mapper::with_model("roles", 0.05, Some(Arc::new(model))).unwrap();
        mapper.last_step = 25;
        for x in -12..12 {
            for z in -6..64 {
                mapper.voxels.insert(
                    [x, 4, z],
                    Voxel {
                        log_odds: -2.0,
                        count: 4,
                    },
                );
            }
        }
        for i in 0..48 {
            let position = [
                -0.4 + (i % 8) as f64 * 0.1,
                0.225,
                0.2 + (i / 8) as f64 * 0.35,
            ];
            let id = format!("N{i:02}");
            mapper.poses.insert(
                id.clone(),
                Pose {
                    position,
                    sigma: 0.001,
                    anchor: position,
                },
            );
            mapper.verify_until.insert(id, 1000);
        }
        mapper.probe_cues.push(Cue {
            position: [0.0, 0.225, 1.2],
            score: 0.8,
            radius_m: 0.3,
            observers: 48,
            status: "UNCONFIRMED",
            last_seen_step: 25,
            evidence: CueEvidence::default(),
        });
        let goals = mapper.frontiers();
        assert_eq!(goals.len(), 48);
        assert_eq!(
            goals
                .iter()
                .filter(|g| g.hold_until_step > 0 || g.reason.contains("probe"))
                .count(),
            6
        );
        assert_eq!(
            goals
                .iter()
                .filter(|g| g.reason.contains("frontier"))
                .count(),
            42
        );
        assert!(
            goals
                .iter()
                .all(|g| mapper.voxels[&mapper.key(g.position)].log_odds < -0.5)
        );
    }
    #[test]
    fn absent_frontiers_do_not_recruit_the_rest_of_the_scattered_swarm_as_probes() {
        let mut model = learned_fixture();
        model.distributed_swarm = true;
        model.acoustic_probe = true;
        let mut mapper = Mapper::with_model("quota", 0.05, Some(Arc::new(model))).unwrap();
        for i in 0..24 {
            let position = [i as f64 * 0.01, 0.025, 0.3];
            mapper.poses.insert(
                format!("N{i:02}"),
                Pose {
                    position,
                    sigma: 0.001,
                    anchor: position,
                },
            );
        }
        mapper.probe_cues.push(Cue {
            position: [0.12, 0.025, 0.3],
            score: 0.8,
            radius_m: 0.3,
            observers: 24,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        });
        let goals = mapper.frontiers();
        assert_eq!(goals.len(), 3);
        assert!(goals.iter().all(|g| g.reason.contains("probe")));
    }
    #[test]
    fn distributed_release_accepts_forty_eight_nodes_and_normalizes_observers() {
        let mut model = learned_fixture();
        model.distributed_swarm = true;
        model.acoustic_probe = true;
        let mut mapper = Mapper::with_model("test", 0.05, Some(Arc::new(model))).unwrap();
        let mut p = packet();
        let template = p.scans[0].clone();
        p.scans.clear();
        p.sim_step = 100;
        p.request_diagnostics = true;
        for i in 0..48 {
            let mut scan = template.clone();
            scan.node_id = format!("N{i:02}");
            scan.deployment_anchor = [
                -0.5 + (i % 8) as f64 / 7.0,
                1.05,
                0.1 + (i / 8) as f64 * 0.2,
            ];
            scan.odometry_delta = [0.0, -0.3, 0.0];
            scan.pose_sigma_m = 0.04;
            scan.rays.clear();
            let origin = add(scan.deployment_anchor, scan.odometry_delta);
            let delta = sub([0.025, 0.025, 0.425], origin);
            scan.bearing = mul(delta, 1.0 / norm(delta));
            scan.bearing_sigma_deg = 10.0;
            scan.temperature_delta_c = 0.0;
            p.scans.push(scan);
        }
        let reply = mapper.apply(&p).unwrap();
        assert_eq!(reply.nodes.len(), 48);
        assert!(!reply.proposals.is_empty());
        assert!(
            reply
                .proposals
                .iter()
                .all(|p| (0.0..=1.0).contains(&p.features[3]))
        );
        assert!(reply.proposals.iter().any(|p| p.observers >= 24));
        p.seq_no = 1;
        p.sim_step = 125;
        for scan in &mut p.scans {
            scan.odometry_delta = [0.0, -0.7, 0.0];
        }
        assert_eq!(
            mapper.apply(&p).unwrap().nodes.len(),
            48,
            "a drop can span two sensor frames"
        );
        p.seq_no = 2;
        p.sim_step = 150;
        let mut extra = p.scans[0].clone();
        extra.node_id = "N48".into();
        p.scans = vec![extra];
        assert!(
            mapper.apply(&p).is_err(),
            "a forty-ninth node must not bypass the cumulative budget"
        );
        assert_eq!(mapper.last_seq, Some(1));
    }
    #[test]
    fn geometry_calibration_and_retained_classification_do_not_change_probe_control() {
        for (distributed, bootstrap) in [(false, false), (true, false), (true, true)] {
            check_classifier_independent_control(distributed, bootstrap);
        }
    }
    fn check_classifier_independent_control(distributed: bool, bootstrap: bool) {
        let mut accept = learned_fixture();
        accept.acoustic_probe = true;
        accept.distributed_swarm = distributed;
        accept.track_weights = vec![0.0; 13];
        accept.track_weights[0] = 8.0;
        accept.occupancy_weights[0] = 2.0;
        let mut reject = accept.clone();
        reject.track_weights[0] = -8.0;
        if bootstrap {
            reject.track_weights.clear();
        }
        reject.occupancy_weights[0] = -2.0;
        let mut a = Mapper::with_model("test", 0.05, Some(Arc::new(accept))).unwrap();
        let mut b = Mapper::with_model("test", 0.05, Some(Arc::new(reject))).unwrap();
        let mut p = packet();
        let template = p.scans[0].clone();
        p.scans.clear();
        for (i, x) in [-0.2, 0.0, 0.2].into_iter().enumerate() {
            let mut scan = template.clone();
            scan.node_id = format!("N{i}");
            scan.deployment_anchor = [x, 0.025, 0.0];
            scan.odometry_delta = [0.0; 3];
            scan.rays = vec![Ray {
                direction: [0.0, 0.0, 1.0],
                distance_m: 0.8,
                hit: false,
            }];
            let direction = sub([0.025, 0.025, 0.425], scan.deployment_anchor);
            scan.bearing = mul(direction, 1.0 / norm(direction));
            scan.bearing_sigma_deg = 10.0;
            scan.temperature_delta_c = 0.0;
            p.scans.push(scan);
        }
        let mut different_cues = false;
        let mut different_maps = false;
        let mut had_goal = false;
        for sequence in 0..20 {
            p.seq_no = sequence;
            p.sim_step = (sequence + 1) * 25;
            let left = a.apply(&p).unwrap();
            let right = b.apply(&p).unwrap();
            assert_eq!(
                serde_json::to_value(&left.goals).unwrap(),
                serde_json::to_value(&right.goals).unwrap()
            );
            had_goal |= !left.goals.is_empty();
            different_cues |= serde_json::to_value(&left.cues).unwrap()
                != serde_json::to_value(&right.cues).unwrap();
            different_maps |= left
                .cells
                .iter()
                .zip(&right.cells)
                .any(|(x, y)| x.occupancy != y.occupancy);
        }
        assert!(
            had_goal && different_cues && different_maps,
            "exercise actual independent control, classifier, and map outputs"
        );
    }
    #[test]
    fn acoustic_probe_requires_evidence_and_routes_around_an_observed_barrier() {
        let mut model = learned_fixture();
        model.acoustic_probe = true;
        let mut mapper = Mapper::with_model("probe", 0.05, Some(Arc::new(model))).unwrap();
        let position = [0.025, 0.025, 0.025];
        mapper.poses.insert(
            "N00".into(),
            Pose {
                position,
                sigma: 0.001,
                anchor: position,
            },
        );
        for x in -16..16 {
            for z in -6..64 {
                mapper.voxels.insert(
                    [x, 0, z],
                    Voxel {
                        log_odds: -2.0,
                        count: 4,
                    },
                );
            }
        }
        assert!(
            mapper.frontiers().is_empty(),
            "an enabled probe must not invent sound evidence"
        );
        mapper.voxels.insert(
            [1, 0, 0],
            Voxel {
                log_odds: 2.0,
                count: 4,
            },
        );
        mapper.probe_cues.push(Cue {
            position: [0.425, 0.025, 0.025],
            score: 0.2,
            radius_m: 0.3,
            observers: 3,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        });
        let goals = mapper.frontiers();
        assert_eq!(goals.len(), 1);
        let next = mapper.key(goals[0].position);
        assert_ne!(next, [1, 0, 0], "never aim through the observed barrier");
        assert!(mapper.voxels[&next].log_odds < -0.5);
        assert!(
            (norm(sub(goals[0].position, position)) - 0.05).abs() < 1e-9,
            "take the next free cell, without cutting a route corner"
        );
        let mut disabled = learned_fixture();
        disabled.acoustic_probe = false;
        mapper.model = Some(Arc::new(disabled));
        assert!(
            mapper.frontiers().is_empty(),
            "legacy policy remains unchanged when disabled"
        );
    }
    #[test]
    fn unrelated_floor_observers_do_not_dilute_a_local_person_hypothesis() {
        let mut model = learned_fixture();
        model.local_surface_completion = true;
        model.distributed_swarm = true;
        let mut map = Mapper::with_model("floor-audio", 0.05, Some(Arc::new(model))).unwrap();
        let target = [-0.075, 0.125, 0.825];
        for (i, origin) in [
            [-0.4, 0.075, 0.5],
            [0.35, 0.075, 0.55],
            [-0.25, 0.075, 1.1],
            [0.3, 0.075, 1.15],
        ]
        .into_iter()
        .enumerate()
        {
            let delta = sub(target, origin);
            map.bearings.push(Bearing {
                id: format!("speaker-observer{i}"),
                origin,
                direction: mul(delta, 1.0 / norm(delta)),
                sigma: 0.08,
                weight: 0.9,
                step: 0,
                pose_sigma: 0.003,
                radial: None,
                human: 0.95,
                heat: 1.0,
            });
        }
        assert!(
            map.cues()
                .iter()
                .any(|c| norm(sub(c.position, target)) < 0.15)
        );
        for i in 0..24 {
            map.bearings.push(Bearing {
                id: format!("other-floor{i}"),
                origin: [0.4, 0.65, 2.0],
                direction: [0.0, 0.0, -1.0],
                sigma: 0.08,
                weight: 0.01,
                step: 0,
                pose_sigma: 0.003,
                radial: None,
                human: 0.1,
                heat: 0.0,
            });
        }
        assert!(
            map.cues()
                .iter()
                .any(|c| norm(sub(c.position, target)) < 0.15)
        );
    }
    fn learned_fixture() -> LearnedModel {
        serde_json::from_value(serde_json::json!({
            "version":"rescue-trained-v1", "cell_m":0.05,
            "acoustic_weights":[3.0,0.0,0.0,0.0,0.0], "occupancy_weights":[0.0,1.0,0.0],
            "acoustic_window_frames":32,
            "prototypes":(0..15).map(|i|serde_json::json!({"family":format!("fixture{i}"),"occupancy":vec![0.5;learned::GRID_CELLS]})).collect::<Vec<_>>(),
            "train_maps":["fixture-train"],"valid_maps":["fixture-valid"],"data_sha256":"fixture"
        })).unwrap()
    }
    #[test]
    fn learned_uncertain_bearing_is_not_discarded_by_a_hard_snr_gate() {
        let model = Arc::new(learned_fixture());
        let mut mapper = Mapper::with_model("test", 0.05, Some(model)).unwrap();
        let mut p = packet();
        p.request_map = false;
        p.scans[0].bearing = [0.0, 0.0, 1.0];
        p.scans[0].bearing_sigma_deg = 45.0;
        p.scans[0].acoustic_snr_db = -15.0;
        let reply = mapper.apply(&p).unwrap();
        assert_eq!(mapper.bearings.len(), 1);
        assert!(reply.cues.is_empty());
    }
    #[test]
    fn verification_dwell_uses_observed_heat_and_audio_and_expires() {
        let model = Arc::new(learned_fixture());
        let mut mapper = Mapper::with_model("test", 0.05, Some(model)).unwrap();
        let mut p = packet();
        p.request_map = false;
        p.scans[0].temperature_delta_c = 0.7;
        let first = mapper.apply(&p).unwrap();
        assert_eq!(first.goals[0].hold_until_step, 1225);
        assert_eq!(first.goals[0].position, p.scans[0].deployment_anchor);
        p.scans[0].temperature_delta_c = 0.0;
        for step in [525, 1025, 1525] {
            p.seq_no += 1;
            p.sim_step = step;
            mapper.apply(&p).unwrap();
        }
        assert!(mapper.frontiers().iter().all(|g| g.hold_until_step <= 1525));
    }
    #[test]
    fn learned_track_retention_uses_evidence_and_preserves_old_hypotheses() {
        let mut model = learned_fixture();
        model.track_weights = vec![0.0; 13];
        model.track_weights[0] = -3.0;
        model.track_weights[10] = 5.0;
        let mut mapper = Mapper::with_model("test", 0.05, Some(Arc::new(model))).unwrap();
        let mut cue = Cue {
            position: [0.1, 0.05, 1.0],
            score: 0.4,
            radius_m: 0.2,
            observers: 5,
            status: "UNCONFIRMED",
            last_seen_step: 100,
            evidence: CueEvidence {
                near: 0.4,
                human: 0.95,
                heat: 0.8,
                range_error: 0.2,
            },
        };
        assert_eq!(mapper.track_cues(&[cue.clone()], 100).len(), 1);
        assert_eq!(mapper.track_cues(&[], 1000).len(), 1);
        assert!(mapper.tracks.is_empty());
        assert_eq!(mapper.archived_tracks.len(), 1);
        cue.position = [-0.3, 0.05, 2.0];
        cue.evidence.human = 0.1;
        assert_eq!(mapper.track_cues(&[cue], 1100).len(), 1);
        assert_eq!(mapper.current_tracks(9000)[0].position, [0.1, 0.05, 1.0]);
    }
    #[test]
    fn diagnostic_track_archive_is_bounded_and_one_node_stays_unconfirmed() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let mut p = packet();
        p.request_diagnostics = true;
        p.scans[0].bearing = [0.0, 0.0, 1.0];
        p.scans[0].acoustic_snr_db = 25.0;
        p.scans[0].voice_band_ratio = 1.0;
        p.scans[0].modulation = 1.0;
        let reply = mapper.apply(&p).unwrap();
        assert!(reply.cues.is_empty() && reply.proposals.is_empty());
        let cue = Cue {
            position: [0.1, 0.05, 1.0],
            score: 0.4,
            radius_m: 0.2,
            observers: 5,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        };
        for i in 1..520 {
            mapper.track_cues(std::slice::from_ref(&cue), i * 1000);
        }
        assert_eq!(mapper.archived_tracks.len(), 512);
        assert_eq!(mapper.tracks.len(), 1);
    }
    #[test]
    fn quadratic_audio_feature_order_matches_the_training_contract() {
        let mut model = learned_fixture();
        model.acoustic_feature_map = "quadratic-v1".into();
        model.acoustic_weights = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ];
        model.validate().unwrap();
        let expected = 1.0 / (1.0 + (-1.202f64).exp());
        assert!((model.human_score([0.5, 0.2, 0.3, 0.4]) - expected).abs() < 1e-12);
    }
    #[test]
    fn bounded_tree_score_uses_strict_split_and_rejects_cycles_or_unknown_features() {
        let mut model = learned_fixture();
        model.track_bias = 0.3;
        model.track_trees = vec![learned::DecisionTree {
            nodes: vec![
                learned::TreeNode {
                    feature: Some(0),
                    threshold: 0.5,
                    left: 1,
                    right: 2,
                    value: 0.0,
                },
                learned::TreeNode {
                    feature: None,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    value: -1.0,
                },
                learned::TreeNode {
                    feature: None,
                    threshold: 0.0,
                    left: 0,
                    right: 0,
                    value: 1.0,
                },
            ],
        }];
        model.validate().unwrap();
        assert!(model.has_track_model());
        let mut x = [0.0; 12];
        x[0] = 0.499;
        assert!((model.track_score(x).unwrap() - 1.0 / (1.0 + 0.7f64.exp())).abs() < 1e-12);
        x[0] = 0.5;
        assert!((model.track_score(x).unwrap() - 1.0 / (1.0 + (-1.3f64).exp())).abs() < 1e-12);
        model.track_trees[0].nodes[0].left = 0;
        assert!(model.validate().is_err());
        model.track_trees[0].nodes[0].left = 1;
        model.track_trees[0].nodes[0].feature = Some(12);
        assert!(model.validate().is_err());
        model.track_trees[0].nodes[0].feature = Some(0);
        model.track_trees[0].nodes[2].value = f64::NAN;
        assert!(model.validate().is_err());
    }
    fn packet() -> Packet {
        Packet {
            protocol_version: VERSION.into(),
            run_id: "test".into(),
            seq_no: 0,
            sim_step: 25,
            cell_m: 0.05,
            scans: vec![Scan {
                node_id: "N0".into(),
                deployment_anchor: [0.0, 0.05, 0.0],
                odometry_delta: [0.0; 3],
                pose_sigma_m: 0.001,
                rays: vec![Ray {
                    direction: [0.0, 0.0, 1.0],
                    distance_m: 0.5,
                    hit: true,
                }],
                bearing: [0.0; 3],
                bearing_sigma_deg: 90.0,
                acoustic_snr_db: 0.0,
                voice_band_ratio: 0.0,
                modulation: 0.0,
                temperature_delta_c: 0.0,
                gas_level: None,
                water_depth_m: None,
            }],
            request_map: true,
            request_diagnostics: false,
            request_surface: false,
        }
    }
    #[test]
    fn ray_carves_free_and_endpoint_occupied() {
        let p = packet();
        let r = Mapper::new("test", 0.05).unwrap().apply(&p).unwrap();
        assert!(r.cells.iter().any(|c| c.key[2] == 10 && c.occupancy > 0.5));
        assert!(r.cells.iter().any(|c| c.key[2] == 3 && c.occupancy < 0.5));
        assert!(r.cues.is_empty());
    }
    #[test]
    fn fine_surface_is_visual_only_and_does_not_change_acceptance_voxels_or_control() {
        let mut a = Mapper::new("test", 0.05).unwrap();
        let mut b = Mapper::new("test", 0.05).unwrap();
        let mut p = packet();
        for i in 0..4 {
            p.seq_no = i;
            p.sim_step = (i + 1) * 25;
            let plain = a.apply(&p).unwrap();
            let mut fine = p.clone();
            fine.request_surface = true;
            let shown = b.apply(&fine).unwrap();
            assert_eq!(
                serde_json::to_value(&plain.cells).unwrap(),
                serde_json::to_value(&shown.cells).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&plain.goals).unwrap(),
                serde_json::to_value(&shown.goals).unwrap()
            );
            assert_eq!(
                serde_json::to_value(&plain.cues).unwrap(),
                serde_json::to_value(&shown.cues).unwrap()
            );
            assert!(serde_json::to_vec(&shown).unwrap().len() <= MAX_REPLY);
            if i > 0 {
                assert!(!shown.surface_points.is_empty());
            }
        }
    }
    #[test]
    fn environmental_reactions_use_only_estimated_poses_and_validate_before_mutation() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let mut p = packet();
        p.scans[0].gas_level = Some(0.8);
        p.scans[0].water_depth_m = Some(0.025);
        let mut second = p.scans[0].clone();
        second.node_id = "N1".into();
        second.deployment_anchor[0] = 0.03;
        p.scans.push(second);
        let mut invalid = p.clone();
        invalid.scans[1].water_depth_m = Some(-0.1);
        assert!(mapper.apply(&invalid).is_err());
        assert!(mapper.poses.is_empty());
        assert!(mapper.environment.current(25).is_empty());
        let reply = mapper.apply(&p).unwrap();
        assert_eq!(reply.gas_observed_nodes, 2);
        assert_eq!(reply.water_observed_nodes, 2);
        assert_eq!(reply.hazards.len(), 2);
        assert!(
            reply
                .hazards
                .iter()
                .all(|h| h.position[0] >= 0.0 && h.position[0] <= 0.03 && h.observers == 2)
        );
    }
    #[test]
    fn duplicates_are_idempotent_and_changed_payload_rejected() {
        let p = packet();
        let mut m = Mapper::new("test", 0.05).unwrap();
        let a = m.apply(&p).unwrap();
        let b = m.apply(&p).unwrap();
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        let mut changed = p;
        changed.scans.clear();
        assert!(m.apply(&changed).is_err());
    }
    #[test]
    fn unknown_truth_fields_rejected() {
        let mut v = serde_json::to_value(packet()).unwrap();
        v["victims"] = serde_json::json!([[1, 2, 3]]);
        assert!(serde_json::from_value::<Packet>(v).is_err());
    }
    #[test]
    fn invalid_late_scan_does_not_mutate_state() {
        let mut p = packet();
        let mut bad = p.scans[0].clone();
        bad.node_id = "N1".into();
        bad.odometry_delta = [f64::NAN, 0., 0.];
        p.scans.push(bad);
        let mut m = Mapper::new("test", 0.05).unwrap();
        assert!(m.apply(&p).is_err());
        assert!(m.voxels.is_empty());
        assert!(m.poses.is_empty());
    }
    #[test]
    fn uncertain_odometry_does_not_paint_walls() {
        let mut p = packet();
        p.scans[0].pose_sigma_m = 0.4;
        let r = Mapper::new("test", 0.05).unwrap().apply(&p).unwrap();
        assert!(r.cells.is_empty());
        assert_eq!(r.rejected_scans, 1);
    }
    #[test]
    fn one_robot_cannot_create_a_confirmed_target() {
        let mut p = packet();
        p.scans[0].bearing = [0., 0., 1.];
        p.scans[0].acoustic_snr_db = 25.;
        p.scans[0].voice_band_ratio = 1.;
        p.scans[0].modulation = 1.;
        let r = Mapper::new("test", 0.05).unwrap().apply(&p).unwrap();
        assert!(r.cues.is_empty());
    }
    #[test]
    fn compact_batches_preserve_the_full_map_for_later() {
        let mut p = packet();
        p.request_map = false;
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let a = mapper.apply(&p).unwrap();
        assert!(a.cells.is_empty());
        assert!(a.observed_cells > 0);
        p.seq_no = 1;
        p.sim_step = 50;
        p.request_map = true;
        let b = mapper.apply(&p).unwrap();
        assert!(!b.cells.is_empty());
        let mut reference = Mapper::new("test", 0.05).unwrap();
        let mut first = packet();
        first.request_map = true;
        reference.apply(&first).unwrap();
        let expected = reference.apply(&p).unwrap();
        assert_eq!(
            serde_json::to_value(b).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
    #[test]
    fn first_sequence_cannot_start_at_an_arbitrary_or_overflowing_value() {
        let mut p = packet();
        p.seq_no = u64::MAX;
        assert!(Mapper::new("test", 0.05).unwrap().apply(&p).is_err());
    }
    #[test]
    fn nested_truth_is_rejected() {
        let mut value = serde_json::to_value(packet()).unwrap();
        value["scans"][0]["rays"][0]["true_position"] = serde_json::json!([0, 0, 0]);
        assert!(serde_json::from_value::<Packet>(value).is_err());
    }
    #[test]
    fn persistent_targets_survive_after_a_robot_moves_away() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let cue = Cue {
            position: [0.1, 0.05, 1.0],
            score: 0.6,
            radius_m: 0.2,
            observers: 8,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        };
        for i in 1..4 {
            assert!(
                mapper
                    .track_cues(std::slice::from_ref(&cue), i * 100)
                    .is_empty()
            );
        }
        assert_eq!(mapper.track_cues(&[cue], 400).len(), 1);
        assert_eq!(mapper.current_tracks(3000).len(), 1);
    }
    #[test]
    fn transient_audio_does_not_become_a_persistent_person_candidate() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let cue = Cue {
            position: [0.1, 0.05, 1.0],
            score: 0.8,
            radius_m: 0.2,
            observers: 10,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        };
        mapper.track_cues(&[cue], 100);
        assert!(mapper.track_cues(&[], 1000).is_empty());
        assert!(mapper.tracks.is_empty());
    }
    #[test]
    fn stale_weak_track_does_not_remain_as_a_permanent_alias() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let cue = Cue {
            position: [0.1, 0.05, 1.0],
            score: 0.19,
            radius_m: 0.2,
            observers: 6,
            status: "UNCONFIRMED",
            last_seen_step: 0,
            evidence: CueEvidence::default(),
        };
        for step in [100, 200, 300, 400] {
            mapper.track_cues(std::slice::from_ref(&cue), step);
        }
        assert_eq!(mapper.current_tracks(400).len(), 1);
        assert!(mapper.track_cues(&[], 3000).is_empty());
        assert!(mapper.track_cues(&[], 6500).is_empty());
        assert!(mapper.tracks.is_empty());
    }
    #[test]
    fn independent_bearings_and_attenuation_localize_a_source_without_a_map() {
        let mut mapper = Mapper::new("test", 0.05).unwrap();
        let target = [0.0, 0.05, 0.9];
        for i in 0..5 {
            let origin = [-0.2 + i as f64 * 0.1, 0.05, 0.0];
            let delta = sub(target, origin);
            let distance = norm(delta);
            mapper.bearings.push(Bearing {
                id: format!("N{i}"),
                origin,
                direction: mul(delta, 1.0 / distance),
                sigma: 8f64.to_radians(),
                weight: 1.0,
                step: 100,
                pose_sigma: 0.001,
                radial: Some((distance, 0.3)),
                human: 1.0,
                heat: 1.0,
            });
        }
        let cues = mapper.cues();
        assert!(!cues.is_empty());
        assert!(norm(sub(cues[0].position, target)) < 0.15);
        assert!(cues.iter().all(|c| c.observers >= 3));
    }
}
