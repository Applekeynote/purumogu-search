//! Sub-voxel hit statistics, accumulated from accepted range rays only.
use crate::{V3, learned};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MomentRow {
    pub key: [i32; 3],
    pub features: Vec<f32>,
}
#[derive(Default)]
struct Micro {
    count: u32,
    weight: f64,
    sum: [f64; 6],
    observers: u64,
    evidence_log_odds: f64,
    evidence_count: u32,
    signed_sum: f64,
    signed_weight: f64,
}
#[derive(Default)]
struct Moment {
    count: u32,
    weight: f64,
    sum: [f64; 15],
    observers: u64,
    micro: [Micro; 8],
}
#[derive(Default)]
pub struct RangeMoments {
    cells: BTreeMap<[i32; 3], Moment>,
    signed_distance: bool,
}
impl RangeMoments {
    /// Sparse projective TSDF band around measured returns. Positive values are
    /// in front of the return, negative values behind it. This is a learned-map
    /// feature, not an occupied observation or proof of a hidden solid volume.
    pub fn integrate_surface_distance(
        &mut self,
        origin: V3,
        rays: &[crate::Ray],
        reliability: f64,
    ) {
        self.signed_distance = true;
        for ray in rays.iter().filter(|r| r.hit) {
            let length = ray.direction.iter().map(|v| v * v).sum::<f64>().sqrt();
            if length < 1e-9 {
                continue;
            }
            let direction = ray.direction.map(|v| v / length);
            let mut visited = Vec::with_capacity(9);
            for step in -4..=4 {
                let distance = ray.distance_m + f64::from(step) * 0.0125;
                if distance <= 0. {
                    continue;
                }
                let fine: [i32; 3] = std::array::from_fn(|i| {
                    ((origin[i] + direction[i] * distance) / 0.025).floor() as i32
                });
                if visited.contains(&fine) {
                    continue;
                }
                visited.push(fine);
                let key = fine.map(|v| v.div_euclid(2));
                if learned::index(key).is_none() {
                    continue;
                }
                let projected: f64 = (0..3)
                    .map(|i| ((f64::from(fine[i]) + 0.5) * 0.025 - origin[i]) * direction[i])
                    .sum();
                let signed = ((ray.distance_m - projected) / 0.05).clamp(-1., 1.);
                let bits = fine.map(|v| v.rem_euclid(2));
                let octant = (bits[0] * 4 + bits[1] * 2 + bits[2]) as usize;
                let micro = &mut self.cells.entry(key).or_default().micro[octant];
                micro.signed_sum += reliability * signed;
                micro.signed_weight += reliability;
            }
        }
    }
    pub fn integrate_free_space(&mut self, origin: V3, rays: &[crate::Ray], reliability: f64) {
        let mut observed = BTreeMap::new();
        for ray in rays {
            let steps = (ray.distance_m / (0.025 * 0.45)).ceil() as usize;
            for step in 0..steps {
                let key: [i32; 3] = std::array::from_fn(|i| {
                    ((origin[i] + ray.direction[i] * step as f64 * 0.025 * 0.45) / 0.025).floor()
                        as i32
                });
                observed.entry(key).or_insert(false);
            }
            if ray.hit {
                let key = std::array::from_fn(|i| {
                    ((origin[i] + ray.direction[i] * ray.distance_m) / 0.025).floor() as i32
                });
                observed.insert(key, true);
            }
        }
        for (fine_key, occupied) in observed {
            let key = fine_key.map(|v| v.div_euclid(2));
            if learned::index(key).is_none() {
                continue;
            }
            let bits = fine_key.map(|v| v.rem_euclid(2));
            let octant = (bits[0] * 4 + bits[1] * 2 + bits[2]) as usize;
            let micro = &mut self.cells.entry(key).or_default().micro[octant];
            micro.evidence_log_odds = (micro.evidence_log_odds
                + (if occupied { 0.85 } else { -0.42 }) * reliability)
                .clamp(-4., 4.);
            micro.evidence_count = micro.evidence_count.saturating_add(1);
        }
    }
    pub fn observe(&mut self, origin: V3, direction: V3, distance: f64, reliability: f64) {
        self.observe_for_node(origin, direction, distance, reliability, 0, false);
    }
    pub fn observe_for_node(
        &mut self,
        origin: V3,
        direction: V3,
        distance: f64,
        reliability: f64,
        observer: usize,
        fine: bool,
    ) {
        let point: V3 = std::array::from_fn(|i| origin[i] + direction[i] * distance);
        let key = point.map(|v| (v / 0.05).floor() as i32);
        if learned::index(key).is_none() {
            return;
        }
        let value = self.cells.entry(key).or_default();
        value.count = value.count.saturating_add(1);
        value.weight += reliability;
        value.observers |= 1u64 << observer.min(63);
        for i in 0..3 {
            value.sum[i] += (point[i] / 0.05 - f64::from(key[i]) - 0.5) * reliability;
            value.sum[i + 3] += direction[i] * reliability;
        }
        let offset: V3 = std::array::from_fn(|i| point[i] / 0.05 - f64::from(key[i]) - 0.5);
        for i in 0..3 {
            value.sum[6 + i] += offset[i] * offset[i] * reliability;
            value.sum[12 + i] += offset[i] * direction[i] * reliability;
        }
        for (i, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            value.sum[9 + i] += offset[a] * offset[b] * reliability;
        }
        if fine {
            let fine_key = point.map(|v| (v / 0.025).floor() as i32);
            let bits: [i32; 3] = std::array::from_fn(|i| fine_key[i] - key[i] * 2);
            let octant = (bits[0] * 4 + bits[1] * 2 + bits[2]) as usize;
            let micro = &mut value.micro[octant];
            micro.count = micro.count.saturating_add(1);
            micro.weight += reliability;
            micro.observers |= 1u64 << observer.min(63);
            for i in 0..3 {
                micro.sum[i] += (point[i] / 0.025 - f64::from(fine_key[i]) - 0.5) * reliability;
                micro.sum[i + 3] += direction[i] * reliability;
            }
        }
    }
    pub fn rows(&self) -> Vec<MomentRow> {
        self.rows_for(18)
    }
    pub fn rows_for(&self, channels: usize) -> Vec<MomentRow> {
        self.cells
            .iter()
            .map(|(key, m)| {
                let mut features = vec![
                    0f32;
                    if channels == 90 {
                        88
                    } else if channels == 74 {
                        72
                    } else {
                        16
                    }
                ];
                features[0] = ((1.0 + f64::from(m.count)).ln() / 1025f64.ln()).min(1.0) as f32;
                let means = m.sum.map(|v| if m.weight > 0. { v / m.weight } else { 0. });
                for i in 0..6 {
                    features[i + 1] = means[i] as f32;
                }
                if matches!(channels, 74 | 90) {
                    features[7] = support(m.observers);
                    for (octant, micro) in m.micro.iter().enumerate() {
                        let start = 8 + octant * 8;
                        features[start] =
                            ((1.0 + f64::from(micro.count)).ln() / 1025f64.ln()).min(1.0) as f32;
                        if micro.weight > 0. {
                            for i in 0..6 {
                                features[start + 1 + i] = (micro.sum[i] / micro.weight) as f32;
                            }
                        }
                        features[start + 7] = support(micro.observers);
                        if channels == 90 {
                            features[72 + octant * 2] = if self.signed_distance {
                                if micro.signed_weight > 0. {
                                    (micro.signed_sum / micro.signed_weight) as f32
                                } else {
                                    0.
                                }
                            } else {
                                (micro.evidence_log_odds / 4.) as f32
                            };
                            let count = if self.signed_distance {
                                micro.signed_weight
                            } else {
                                f64::from(micro.evidence_count)
                            };
                            features[73 + octant * 2] =
                                ((1. + count).ln() / 1025f64.ln()).min(1.) as f32;
                        }
                    }
                } else {
                    for i in 0..3 {
                        features[7 + i] = (means[6 + i] - means[i] * means[i]).max(0.0) as f32;
                        features[13 + i] = (means[12 + i] - means[i] * means[3 + i]) as f32;
                    }
                    for (i, (a, b)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
                        features[10 + i] = (means[9 + i] - means[a] * means[b]) as f32;
                    }
                }
                MomentRow {
                    key: *key,
                    features,
                }
            })
            .collect()
    }
}
fn support(observers: u64) -> f32 {
    ((1. + f64::from(observers.count_ones())).ln() / 49f64.ln()).min(1.) as f32
}
pub fn grid(rows: &[MomentRow], channels: usize) -> anyhow::Result<Vec<[f32; 88]>> {
    anyhow::ensure!(
        matches!(channels, 9 | 18 | 74 | 90),
        "Invalid moment channel count"
    );
    let mut result = vec![[0f32; 88]; learned::GRID_CELLS];
    for row in rows {
        let index =
            learned::index(row.key).ok_or_else(|| anyhow::anyhow!("Moment cell outside map"))?;
        anyhow::ensure!(
            matches!(row.features.len(), 7 | 16 | 72 | 88)
                && row.features.len() >= channels - 2
                && row.features.iter().all(|v| v.is_finite())
                && (0.0..=1.0).contains(&row.features[0])
                && row.features[1..4].iter().all(|v| v.abs() <= 0.500001)
                && row.features[4..7].iter().all(|v| v.abs() <= 1.050001)
                && row.features[7..].iter().all(|v| v.abs() <= 1.100001),
            "Invalid range moments"
        );
        result[index][..row.features.len()].copy_from_slice(&row.features);
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn signed_distance_keeps_front_surface_and_back_separate_without_fake_hits() {
        let mut map = RangeMoments::default();
        map.integrate_surface_distance(
            [0.0125, 0.0125, 0.],
            &[crate::Ray {
                direction: [0., 0., 1.],
                distance_m: 0.2125,
                hit: true,
            }],
            1.,
        );
        let rows = map.rows_for(90);
        let data = grid(&rows, 90).unwrap();
        assert!((data[learned::index([0, 0, 3]).unwrap()][74] - 0.5).abs() < 1e-6);
        assert!(data[learned::index([0, 0, 4]).unwrap()][72].abs() < 1e-6);
        assert!((data[learned::index([0, 0, 4]).unwrap()][74] + 0.5).abs() < 1e-6);
        assert!(
            rows.iter()
                .all(|r| r.features[..72].iter().all(|v| *v == 0.))
        );
    }
    #[test]
    fn signed_distance_averages_return_noise_and_has_no_max_range_surface() {
        let mut map = RangeMoments::default();
        for distance in [0.2085, 0.2165] {
            map.integrate_surface_distance(
                [0.0125, 0.0125, 0.],
                &[crate::Ray {
                    direction: [0., 0., 1.],
                    distance_m: distance,
                    hit: true,
                }],
                1.,
            );
        }
        let data = grid(&map.rows_for(90), 90).unwrap();
        assert!(data[learned::index([0, 0, 4]).unwrap()][72].abs() < 1e-6);
        let mut empty = RangeMoments::default();
        empty.integrate_surface_distance(
            [0.; 3],
            &[crate::Ray {
                direction: [0., 0., 1.],
                distance_m: 2.,
                hit: false,
            }],
            1.,
        );
        assert!(empty.rows_for(90).is_empty());
    }
    #[test]
    fn surface_distance_does_not_change_the_original_seventy_four_channels() {
        let mut a = RangeMoments::default();
        let mut b = RangeMoments::default();
        let ray = crate::Ray {
            direction: [0., 0., 1.],
            distance_m: 0.2125,
            hit: true,
        };
        a.observe_for_node(
            [0.0125, 0.0125, 0.],
            ray.direction,
            ray.distance_m,
            1.,
            3,
            true,
        );
        b.integrate_surface_distance([0.0125, 0.0125, 0.], std::slice::from_ref(&ray), 1.);
        b.observe_for_node(
            [0.0125, 0.0125, 0.],
            ray.direction,
            ray.distance_m,
            1.,
            3,
            true,
        );
        let aa = grid(&a.rows_for(74), 74).unwrap();
        let bb = grid(&b.rows_for(90), 90).unwrap();
        assert!(aa.iter().zip(bb).all(|(a, b)| a[..72] == b[..72]));
    }
    #[test]
    fn fine_ray_preserves_free_space_hits_and_unknown_without_double_counting() {
        let ray = crate::Ray {
            direction: [0., 0., 1.],
            distance_m: 0.1,
            hit: true,
        };
        let mut map = RangeMoments::default();
        map.integrate_free_space([0.01, 0.01, 0.01], &[ray.clone(), ray], 1.);
        let rows = map.rows_for(90);
        let grid = grid(&rows, 90).unwrap();
        let start = learned::index([0, 0, 0]).unwrap();
        let end = learned::index([0, 0, 2]).unwrap();
        assert!((grid[start][72] + 0.42 / 4.).abs() < 1e-6);
        assert!((grid[end][72] - 0.85 / 4.).abs() < 1e-6);
        assert!((grid[start][73] - 2f32.ln() / 1025f32.ln()).abs() < 1e-6);
        assert_eq!(grid[learned::index([4, 4, 4]).unwrap()], [0.; 88]);
        assert!(rows.iter().flat_map(|r| &r.features).all(|v| v.is_finite()));
    }
    #[test]
    fn relative_offsets_preserve_subvoxel_hits_and_translation() {
        let mut a = RangeMoments::default();
        let mut b = RangeMoments::default();
        for x in [0.011, 0.013] {
            a.observe([x, 0.02, 0.02], [0., 0., 1.], 0.1, 0.5);
            b.observe([x + 0.1, 0.07, 0.12], [0., 0., 1.], 0.1, 0.5);
        }
        let aa = a.rows();
        let bb = b.rows();
        assert_eq!(aa.len(), 1);
        assert_eq!(bb.len(), 1);
        for i in 0..7 {
            assert!((aa[0].features[i] - bb[0].features[i]).abs() < 1e-6);
        }
        assert!((aa[0].features[1] + 0.26).abs() < 1e-6);
        assert!((aa[0].features[7] - 0.0004).abs() < 1e-6);
        let dense = grid(&aa, 18).unwrap();
        assert_eq!(dense.iter().filter(|v| v[0] > 0.).count(), 1);
        let mut bad = aa;
        bad[0].features[4] = 2.;
        assert!(grid(&bad, 18).is_err());
    }
    #[test]
    fn octants_keep_separate_surfaces_and_count_distinct_observers() {
        let mut hits = RangeMoments::default();
        for node in [0, 0, 1] {
            hits.observe_for_node([0.01, 0.01, 0.01], [0., 0., 1.], 0., 1., node, true);
        }
        hits.observe_for_node([0.04, 0.04, 0.04], [0., 1., 0.], 0., 1., 2, true);
        let rows = hits.rows_for(74);
        let f = &rows[0].features;
        assert_eq!(rows.len(), 1);
        assert!((f[7] - support(7)).abs() < 1e-7);
        assert!((f[15] - support(3)).abs() < 1e-7);
        assert!((f[71] - support(4)).abs() < 1e-7);
        assert!((f[9] + 0.1).abs() < 1e-6);
        assert!((f[65] - 0.1).abs() < 1e-6);
        assert_eq!(f[14], 1.);
        assert_eq!(f[69], 1.);
        assert!(grid(&rows, 74).is_ok());
        assert!(grid(&hits.rows(), 74).is_err());
    }
}
