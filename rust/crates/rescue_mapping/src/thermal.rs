//! Spatially averaged thermal evidence. No source positions or human labels.
use crate::V3;
use std::collections::BTreeMap;
#[derive(Default)]
struct Sample {
    sum: f64,
    weight: f64,
    count: u32,
    spectral_sum: [f64; 4],
}
#[derive(Default)]
pub struct ThermalField {
    samples: BTreeMap<[i32; 3], Sample>,
}
impl ThermalField {
    pub fn observe(&mut self, origin: V3, temperature: f64, pose_sigma: f64) {
        let key = origin.map(|v| (v / 0.03).floor() as i32);
        if self.samples.len() >= 24000 && !self.samples.contains_key(&key) {
            return;
        }
        let weight = 1.0 / (1.0 + (pose_sigma / 0.05).powi(2));
        let sample = self.samples.entry(key).or_default();
        sample.sum += temperature * weight;
        sample.weight += weight;
        sample.count = sample.count.saturating_add(1);
    }
    /// Called immediately after observe(), sharing its spatial bin and pose weight.
    pub fn observe_spectral(
        &mut self,
        origin: V3,
        pose_sigma: f64,
        snr: f64,
        voice: f64,
        modulation: f64,
    ) {
        let key = origin.map(|v| (v / 0.03).floor() as i32);
        if let Some(sample) = self.samples.get_mut(&key) {
            let weight = 1. / (1. + (pose_sigma / 0.05).powi(2));
            for (sum, value) in sample.spectral_sum.iter_mut().zip([
                snr / 20.,
                voice,
                modulation,
                voice * modulation,
            ]) {
                *sum += weight * value;
            }
        }
    }
    pub fn features(&self, point: V3) -> [f64; 4] {
        let mut et = 0.0;
        let mut ee = 0.0;
        let mut tt = 0.0;
        let mut near = 2.0f64;
        let mut support = 0.0;
        for (key, sample) in &self.samples {
            if sample.count < 3 {
                continue;
            }
            let distance = key
                .iter()
                .zip(point)
                .map(|(k, p)| ((f64::from(*k) + 0.5) * 0.03 - p).powi(2))
                .sum::<f64>()
                .sqrt();
            let weight = sample.weight.min(12.0);
            let temperature = sample.sum / sample.weight;
            let kernel = (-7.0 * distance).exp();
            et += weight * kernel * temperature;
            ee += weight * kernel * kernel;
            tt += weight * temperature * temperature;
            if temperature > 0.12 {
                near = near.min(distance);
                support += kernel * weight;
            }
        }
        if ee < 1e-12 || tt < 1e-12 {
            return [0.0; 4];
        }
        [
            (et / (ee * tt).sqrt()).clamp(-1.0, 1.0),
            (et / ee / 4.0).clamp(-4.0, 4.0),
            near.min(2.0) / 2.0,
            (1.0 + support).ln() / 5.0,
        ]
    }
    pub fn local_features(&self, point: V3) -> Vec<f64> {
        let mut result = self.features(point).to_vec();
        for radius in [0.18, 0.4] {
            let (mut et, mut ee, mut tt, mut wt, mut den) = (0., 0., 0., 0., 0.);
            for (key, sample) in &self.samples {
                if sample.count < 3 {
                    continue;
                }
                let distance = key
                    .iter()
                    .zip(point)
                    .map(|(k, p)| ((f64::from(*k) + 0.5) * 0.03 - p).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let weight = (-0.5 * (distance / radius).powi(2)).exp() * sample.weight.min(12.);
                let temperature = sample.sum / sample.weight;
                let kernel = (-7. * distance).exp();
                et += weight * kernel * temperature;
                ee += weight * kernel * kernel;
                tt += weight * temperature * temperature;
                wt += weight * temperature;
                den += weight;
            }
            result.extend([
                (et / (ee * tt).max(1e-24).sqrt()).clamp(-1., 1.),
                (et / ee.max(1e-12) / 4.).clamp(-4., 4.),
                wt / den.max(1e-12) / 4.,
                (1. + den).ln() / 5.,
            ]);
        }
        result
    }
    pub fn spectral_features(&self, point: V3) -> Vec<f64> {
        // Fuse the five spatial passes. Distances and kernels are identical for
        // thermal and spectral moments; calculating them once keeps CPU inference practical.
        let (mut et, mut ee, mut tt, mut near, mut support) = (0., 0., 0., 2f64, 0.);
        let mut local = [[0f64; 9]; 2];
        for (key, sample) in &self.samples {
            if sample.count < 3 {
                continue;
            }
            let distance = key
                .iter()
                .zip(point)
                .map(|(k, p)| ((f64::from(*k) + 0.5) * 0.03 - p).powi(2))
                .sum::<f64>()
                .sqrt();
            let temperature = sample.sum / sample.weight;
            let weight = sample.weight.min(12.);
            let kernel = (-7. * distance).exp();
            et += weight * kernel * temperature;
            ee += weight * kernel * kernel;
            tt += weight * temperature * temperature;
            if temperature > 0.12 {
                near = near.min(distance);
                support += kernel * weight;
            }
            for (i, radius) in [0.18, 0.4].into_iter().enumerate() {
                let w = (-0.5 * (distance / radius).powi(2)).exp() * weight;
                let l = &mut local[i];
                l[0] += w * kernel * temperature;
                l[1] += w * kernel * kernel;
                l[2] += w * temperature * temperature;
                l[3] += w * temperature;
                l[4] += w;
                for (sum, value) in l[5..9].iter_mut().zip(sample.spectral_sum) {
                    *sum += w * value / sample.weight;
                }
            }
        }
        let mut result = if ee < 1e-12 || tt < 1e-12 {
            vec![0.; 4]
        } else {
            vec![
                (et / (ee * tt).sqrt()).clamp(-1., 1.),
                (et / ee / 4.).clamp(-4., 4.),
                near.min(2.) / 2.,
                (1. + support).ln() / 5.,
            ]
        };
        for l in &local {
            result.extend([
                (l[0] / (l[1] * l[2]).max(1e-24).sqrt()).clamp(-1., 1.),
                (l[0] / l[1].max(1e-12) / 4.).clamp(-4., 4.),
                l[3] / l[4].max(1e-12) / 4.,
                (1. + l[4]).ln() / 5.,
            ]);
        }
        for l in &local {
            result.extend(l[5..9].iter().map(|v| v / l[4].max(1e-12)));
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stationary_noise_averaging_keeps_location_information() {
        let target = [0.015, 0.015, 0.615];
        let mut field = ThermalField::default();
        for x in -4..5 {
            for z in 0..12 {
                let origin = [
                    (f64::from(x) + 0.5) * 0.03,
                    0.015,
                    (f64::from(z) + 0.5) * 0.06,
                ];
                let d = origin
                    .iter()
                    .zip(target)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for _ in 0..4 {
                    field.observe(origin, 3.6 * (-7.0 * d).exp(), 0.01);
                    field.observe_spectral(
                        origin,
                        0.01,
                        f64::from(x + z),
                        0.5 + f64::from(x) * 0.01,
                        0.6 + f64::from(z) * 0.01,
                    );
                }
            }
        }
        let actual = field.features(target);
        let wrong = field.features([0.4, 0.015, 0.1]);
        assert!(actual[0] > wrong[0]);
        assert!((actual[1] - 0.9).abs() < 0.1);
        let fused = field.spectral_features(target);
        let mut reference = field.local_features(target);
        for radius in [0.18, 0.4] {
            let mut sums = [0.; 4];
            let mut den = 0.;
            for (key, sample) in &field.samples {
                if sample.count < 3 {
                    continue;
                }
                let d2 = key
                    .iter()
                    .zip(target)
                    .map(|(k, p)| ((f64::from(*k) + 0.5) * 0.03 - p).powi(2))
                    .sum::<f64>();
                let weight = (-0.5 * d2 / (radius * radius)).exp() * sample.weight.min(12.);
                for (sum, value) in sums.iter_mut().zip(sample.spectral_sum) {
                    *sum += weight * value / sample.weight;
                }
                den += weight;
            }
            reference.extend(sums.map(|v| v / den.max(1e-12)));
        }
        assert_eq!(fused.len(), 20);
        assert!(
            fused
                .iter()
                .zip(reference)
                .all(|(a, b)| (a - b).abs() < 1e-12),
            "Fused CPU path must retain the trained feature contract"
        );
    }
}
