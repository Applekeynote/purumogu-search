//! Fine visual surface fusion from noisy range endpoints; independent of acceptance voxels.
use std::collections::HashMap;
const CELL: f64 = 0.01;
const CAPACITY: usize = 120_000;
const OUTPUT_LIMIT: usize = 24_000;
type Key = [i32; 3];
#[derive(Clone)]
struct Sample {
    sum: [f64; 3],
    weight: f64,
    observations: u32,
    first: u64,
    last: u64,
}
impl Sample {
    fn point(&self) -> [f64; 3] {
        self.sum.map(|v| v / self.weight)
    }
}
#[derive(Default)]
pub struct SurfaceMap {
    samples: HashMap<Key, Sample>,
    cleaned: u64,
}
impl SurfaceMap {
    pub fn observe(&mut self, point: [f64; 3], weight: f64, step: u64) {
        if !weight.is_finite()
            || weight <= 0.0
            || !point.iter().all(|p| p.is_finite())
            || !(-0.8..0.8).contains(&point[0])
            || !(-0.15..0.9).contains(&point[1])
            || !(-0.3..3.2).contains(&point[2])
        {
            return;
        }
        if step.saturating_sub(self.cleaned) >= 250 {
            self.samples
                .retain(|_, v| step.saturating_sub(v.last) <= 3000);
            self.cleaned = step;
        }
        let key = point.map(|p| (p / CELL).floor() as i32);
        if self.samples.len() >= CAPACITY && !self.samples.contains_key(&key) {
            return;
        }
        let sample = self.samples.entry(key).or_insert(Sample {
            sum: [0.0; 3],
            weight: 0.0,
            observations: 0,
            first: step,
            last: step,
        });
        let weight = weight.clamp(0.01, 1.0);
        // Bound the effective temporal window, so a moving fragment can replace old evidence.
        if sample.weight > 32.0 {
            sample.sum = sample.sum.map(|v| v * 0.5);
            sample.weight *= 0.5;
        }
        for (sum, p) in sample.sum.iter_mut().zip(point) {
            *sum += p * weight;
        }
        sample.weight += weight;
        sample.observations = sample.observations.saturating_add(1);
        sample.last = step;
    }
    /// Compact rows: x,y,z,strength,first_step,last_step. Strength is display support, not probability.
    pub fn current(&self, step: u64) -> Vec<[f64; 6]> {
        let mut candidates: Vec<_> = self
            .samples
            .iter()
            .filter(|(_, v)| v.observations >= 2 && step.saturating_sub(v.last) <= 1500)
            .collect();
        let priority = |key: &Key| {
            (key[0] as u64).wrapping_mul(0x9e3779b185ebca87)
                ^ (key[1] as u64).wrapping_mul(0xc2b2ae3d27d4eb4f)
                ^ (key[2] as u64).wrapping_mul(0x165667b19e3779f9)
        };
        candidates.sort_unstable_by(|(a, _), (b, _)| {
            priority(a).cmp(&priority(b)).then_with(|| a.cmp(b))
        });
        candidates.truncate(OUTPUT_LIMIT);
        candidates.sort_unstable_by_key(|(k, _)| **k);
        candidates
            .into_iter()
            .map(|(key, sample)| {
                let point = sample.point();
                let mut sum = point.map(|v| v * 4.0);
                let mut total = 4.0;
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            if x == 0 && y == 0 && z == 0 {
                                continue;
                            }
                            let neighbor = [key[0] + x, key[1] + y, key[2] + z];
                            if let Some(value) = self.samples.get(&neighbor).filter(|v| {
                                v.observations >= 2 && step.saturating_sub(v.last) <= 1500
                            }) {
                                let q = value.point();
                                let d: f64 =
                                    q.iter().zip(point).map(|(a, b)| (a - b).powi(2)).sum();
                                if d > 0.0004 {
                                    continue;
                                }
                                let w = (-d / 0.0002).exp();
                                for (i, v) in q.into_iter().enumerate() {
                                    sum[i] += v * w;
                                }
                                total += w;
                            }
                        }
                    }
                }
                let smooth: [f64; 3] = std::array::from_fn(|i| {
                    point[i] + ((sum[i] / total - point[i]) * 0.5).clamp(-0.003, 0.003)
                });
                let round = |v: f64| (v * 100_000.0).round() / 100_000.0;
                let strength = (0.5 + (1.0 + f64::from(sample.observations)).ln() * 0.12).min(1.0);
                [
                    round(smooth[0]),
                    round(smooth[1]),
                    round(smooth[2]),
                    (strength * 1000.0).round() / 1000.0,
                    sample.first as f64,
                    sample.last as f64,
                ]
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fuses_repeated_actual_endpoints_without_inventing_an_unobserved_surface() {
        let mut map = SurfaceMap::default();
        map.observe([0.001, 0.002, 0.501], 1.0, 25);
        assert!(map.current(25).is_empty());
        map.observe([0.003, 0.002, 0.503], 1.0, 50);
        let points = map.current(50);
        assert_eq!(points.len(), 1);
        assert!((points[0][0] - 0.002).abs() < 1e-9);
        assert!((points[0][2] - 0.502).abs() < 1e-9);
        assert!(map.current(1600).is_empty());
    }
    #[test]
    fn distinguishes_features_inside_one_acceptance_voxel_and_bounds_smoothing() {
        let mut map = SurfaceMap::default();
        for step in [25, 50] {
            for x in [0.001, 0.011, 0.021, 0.031] {
                map.observe([x, 0.001, 0.501], 1.0, step);
            }
        }
        let points = map.current(50);
        assert_eq!(points.len(), 4);
        for (p, x) in points.iter().zip([0.001, 0.011, 0.021, 0.031]) {
            assert!((p[0] - x).abs() <= 0.00301);
        }
    }
}
