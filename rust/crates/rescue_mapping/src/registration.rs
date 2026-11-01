//! Experimental gravity-plane consensus from existing noisy range/odometry only.
//! Plane heights are measured across nodes, never supplied by a scene or evaluator.
use crate::{Scan, V3};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct PlaneConsensus {
    raw_poses: BTreeMap<String, V3>,
    ids: BTreeMap<String, usize>,
    histogram: BTreeMap<i32, (u64, f64, u64)>,
    offsets: BTreeMap<String, f64>,
}
#[derive(Default)]
pub struct AxisConsensus {
    vertical: PlaneConsensus,
    sides: [PlaneConsensus; 4],
}
impl AxisConsensus {
    /// Signed, measured axis-aligned wall planes are optional. Oblique walls are
    /// not supplied as templates; insufficient shared support leaves a pose alone.
    pub fn packet(&mut self, scans: &[Scan], axes: bool) -> BTreeMap<String, V3> {
        let mut result = self.vertical.packet(scans);
        if !axes {
            return result;
        }
        let mut corrections: BTreeMap<String, [[f64; 2]; 2]> = BTreeMap::new();
        for (index, (axis, sign)) in [(0usize, -1.), (0, 1.), (2, -1.), (2, 1.)]
            .into_iter()
            .enumerate()
        {
            let transform = |value: V3| -> V3 {
                if axis == 0 {
                    [value[1], -sign * value[0], value[2]]
                } else {
                    [value[0], -sign * value[2], value[1]]
                }
            };
            let transformed: Vec<_> = scans
                .iter()
                .cloned()
                .map(|mut scan| {
                    scan.deployment_anchor = transform(scan.deployment_anchor);
                    scan.odometry_delta = transform(scan.odometry_delta);
                    scan.rays.retain(|r| {
                        r.hit
                            && r.direction[1] > 0.01
                            && r.direction[1] < 0.45
                            && r.distance_m > 0.04
                            && r.direction[axis] * sign > 0.55
                    });
                    for ray in &mut scan.rays {
                        ray.direction = transform(ray.direction);
                    }
                    scan
                })
                .collect();
            self.sides[index].packet(&transformed);
            for scan in scans {
                let offset = -sign
                    * self.sides[index]
                        .offsets
                        .get(&scan.node_id)
                        .copied()
                        .unwrap_or(0.);
                if offset.abs() > 1e-8 {
                    let value = corrections.entry(scan.node_id.clone()).or_default();
                    let channel = axis / 2;
                    value[channel][0] += offset;
                    value[channel][1] += 1.;
                }
            }
        }
        for (id, point) in &mut result {
            if let Some(c) = corrections.get(id) {
                for (i, axis) in [0, 2].into_iter().enumerate() {
                    if c[i][1] > 0. {
                        point[axis] += c[i][0] / c[i][1];
                    }
                }
            }
        }
        result
    }
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}
impl PlaneConsensus {
    pub fn packet(&mut self, scans: &[Scan]) -> BTreeMap<String, V3> {
        let mut rows = vec![];
        for scan in scans {
            let origin = self
                .raw_poses
                .entry(scan.node_id.clone())
                .or_insert(scan.deployment_anchor);
            for (value, delta) in origin.iter_mut().zip(scan.odometry_delta) {
                *value += delta;
            }
            let origin = *origin;
            let next = self.ids.len();
            let observer = *self.ids.entry(scan.node_id.clone()).or_insert(next);
            let points: Vec<V3> = scan
                .rays
                .iter()
                .filter(|r| r.hit && r.direction[1] < -0.35 && r.distance_m > 0.004)
                .map(|r| std::array::from_fn(|i| origin[i] + r.direction[i] * r.distance_m))
                .collect();
            if scan.pose_sigma_m <= 0.15 {
                for point in &points {
                    let key = (point[1] / 0.003).floor() as i32;
                    if self.histogram.len() >= 4096 && !self.histogram.contains_key(&key) {
                        continue;
                    }
                    let value = self.histogram.entry(key).or_default();
                    value.0 = value.0.saturating_add(1);
                    value.1 += point[1];
                    value.2 |= 1u64 << observer.min(63);
                }
            }
            rows.push((scan, origin, points));
        }
        let mut peaks = vec![];
        for &key in self.histogram.keys() {
            let (mut count, mut sum, mut nodes) = (0u64, 0., 0u64);
            for offset in -2..=2 {
                if let Some(value) = self.histogram.get(&key.saturating_add(offset)) {
                    count += value.0;
                    sum += value.1;
                    nodes |= value.2;
                }
            }
            if count >= 80 && nodes.count_ones() >= 3 {
                peaks.push((count, sum / count as f64));
            }
        }
        peaks.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.total_cmp(&a.1)));
        let mut planes = vec![];
        for (_, height) in peaks {
            if planes
                .iter()
                .all(|other: &f64| (height - other).abs() > 0.055)
            {
                planes.push(height);
            }
            if planes.len() == 16 {
                break;
            }
        }
        let mut result = BTreeMap::new();
        for (scan, mut origin, points) in rows {
            let mut residuals = vec![];
            if points.len() >= 4 && !planes.is_empty() {
                for point in points {
                    let plane = planes
                        .iter()
                        .min_by(|a, b| (*a - point[1]).abs().total_cmp(&(*b - point[1]).abs()))
                        .unwrap();
                    if (plane - point[1]).abs() < 0.035 {
                        residuals.push(plane - point[1]);
                    }
                }
            }
            let offset = self.offsets.entry(scan.node_id.clone()).or_default();
            if residuals.len() >= 4 {
                let value = median(&mut residuals);
                let mut deviations: Vec<_> = residuals.iter().map(|v| (v - value).abs()).collect();
                if median(&mut deviations) < 0.007 {
                    let limit = (scan.pose_sigma_m * 2.).clamp(0.004, 0.04);
                    *offset = 0.7 * *offset + 0.3 * value.clamp(-limit, limit);
                }
            }
            origin[1] += *offset;
            result.insert(scan.node_id.clone(), origin);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ray;
    fn scan(id: usize, floor: f64, bias: f64, rays: usize) -> Scan {
        Scan {
            node_id: format!("N{id}"),
            deployment_anchor: [id as f64 * 0.08, floor + 0.05 + bias, 0.4],
            odometry_delta: [0.; 3],
            pose_sigma_m: 0.02,
            rays: (0..rays)
                .map(|_| Ray {
                    direction: [0., -1., 0.],
                    distance_m: 0.05,
                    hit: true,
                })
                .collect(),
            bearing: [0.; 3],
            bearing_sigma_deg: 30.,
            acoustic_snr_db: 0.,
            voice_band_ratio: 0.,
            modulation: 0.,
            temperature_delta_c: 0.,
            gas_level: None,
            water_depth_m: None,
        }
    }
    #[test]
    fn one_node_cannot_establish_a_shared_plane() {
        let mut map = PlaneConsensus::default();
        let scan = scan(0, 0.017, 0.012, 32);
        for _ in 0..20 {
            assert_eq!(
                map.packet(std::slice::from_ref(&scan))["N0"],
                scan.deployment_anchor
            );
        }
    }
    #[test]
    fn consensus_reduces_drift_at_measured_heights_and_is_translation_equivariant() {
        for floor in [0.017, 0.154] {
            let scans: Vec<_> = (0..6)
                .map(|id| {
                    scan(
                        id,
                        floor,
                        if id < 3 { 0. } else { 0.012 },
                        if id < 3 { 32 } else { 8 },
                    )
                })
                .collect();
            let mut map = PlaneConsensus::default();
            let mut positions = BTreeMap::new();
            for _ in 0..20 {
                positions = map.packet(&scans);
            }
            for id in 3..6 {
                assert!((positions[&format!("N{id}")][1] - (floor + 0.05)).abs() < 0.001);
            }
        }
    }
}
