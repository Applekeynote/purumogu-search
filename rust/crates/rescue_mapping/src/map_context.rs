//! Relative candidate context from raw measured occupancy, never completed geometry.
pub const FEATURE_MAP: &str = "thermal-spectral-map40-v4";

pub fn features(position: [f64; 3], measured: impl Fn([i32; 3]) -> Option<f64>) -> [f64; 8] {
    let center = position.map(|p| (p / 0.05).floor() as i32);
    let sample = |dx, dy, dz| {
        let key = [center[0] + dx, center[1] + dy, center[2] + dz];
        crate::learned::index(key).and_then(|_| measured(key))
    };
    let (mut known, mut solid) = (0, 0);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if let Some(value) = sample(dx, dy, dz) {
                    known += 1;
                    solid += usize::from(value > 0.65);
                }
            }
        }
    }
    let patch = |dy| {
        let mut count = 0;
        for dx in -1..=1 {
            for dz in -1..=1 {
                count += usize::from(sample(dx, dy, dz).is_some_and(|p| p > 0.65));
            }
        }
        count as f64 / 9.0
    };
    let below: [f64; 8] = std::array::from_fn(|d| patch(-(d as i32)));
    let maximum = |values: &[f64]| values.iter().copied().fold(0.0, f64::max);
    let at = sample(0, 0, 0);
    [
        known as f64 / 27.0,
        solid as f64 / 27.0,
        f64::from(at.is_some()),
        at.unwrap_or(0.5),
        maximum(&below[..3]),
        maximum(&below[3..]),
        below
            .iter()
            .position(|p| *p >= 5.0 / 9.0)
            .map_or(1.0, |d| d as f64 / 7.0),
        maximum(&(1..=4).map(patch).collect::<Vec<_>>()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn measured_support_and_unknown_space_are_distinct() {
        let mut values = BTreeMap::new();
        for x in -1..=1 {
            for z in -1..=1 {
                values.insert([x, -2, z], 0.9);
            }
        }
        values.insert([0, 0, 0], 0.2);
        let actual = features([0.025; 3], |k| values.get(&k).copied());
        assert_eq!(
            actual,
            [1.0 / 27.0, 0.0, 1.0, 0.2, 1.0, 0.0, 2.0 / 7.0, 0.0]
        );
        assert_eq!(
            features([0.025; 3], |_| None),
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn relative_queries_preserve_translation_and_respect_the_map_boundary() {
        let base = features([0.025; 3], |k| Some(if k[1] == -3 { 0.8 } else { 0.1 }));
        let shifted = features([0.175, 0.225, 0.375], |k| {
            Some(if k[1] == 1 { 0.8 } else { 0.1 })
        });
        assert_eq!(base, shifted);
        assert!(features([-0.775, 0.025, 0.025], |_| Some(0.9))[0] < 1.0);
        assert_eq!(features([9.0; 3], |_| Some(0.9))[0], 0.0);
    }
}
