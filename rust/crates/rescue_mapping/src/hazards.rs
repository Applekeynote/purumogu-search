//! Local environmental readings at estimated node poses, never simulator source positions.
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize)]
pub struct Hazard {
    pub kind: &'static str,
    pub position: [f64; 3],
    pub radius_m: f64,
    pub value: f64,
    pub observations: u32,
    pub observers: usize,
    pub last_seen_step: u64,
    pub status: &'static str,
}
struct Reading {
    kind: &'static str,
    position: [f64; 3],
    value: f64,
    sigma: f64,
    count: u32,
    observers: BTreeSet<String>,
    step: u64,
}
#[derive(Default)]
pub struct HazardMap {
    readings: Vec<Reading>,
    pub gas_nodes: BTreeSet<String>,
    pub water_nodes: BTreeSet<String>,
}
impl HazardMap {
    pub fn observe(
        &mut self,
        kind: &'static str,
        position: [f64; 3],
        value: f64,
        sigma: f64,
        node: &str,
        step: u64,
    ) {
        if kind == "gas" {
            self.gas_nodes.insert(node.into());
        } else {
            self.water_nodes.insert(node.into());
        }
        let threshold = if kind == "gas" { 0.25 } else { 0.004 };
        let nearest = self
            .readings
            .iter()
            .enumerate()
            .filter(|(_, r)| r.kind == kind)
            .map(|(i, r)| {
                (
                    i,
                    r.position
                        .iter()
                        .zip(position)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>(),
                )
            })
            .filter(|(_, d)| *d < 0.20_f64.powi(2))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((index, _)) = nearest {
            let r = &mut self.readings[index];
            r.value = r.value * 0.65 + value * 0.35;
            r.sigma = r.sigma.max(sigma);
            r.count = r.count.saturating_add(1);
            r.step = step;
            if value >= threshold {
                r.observers.insert(node.into());
                for (p, q) in r.position.iter_mut().zip(position) {
                    *p = *p * 0.9 + q * 0.1;
                }
            }
        } else if value >= threshold {
            self.readings
                .retain(|r| step.saturating_sub(r.step) <= 3000);
            if self.readings.len() < 64 {
                self.readings.push(Reading {
                    kind,
                    position,
                    value,
                    sigma,
                    count: 1,
                    observers: BTreeSet::from([node.into()]),
                    step,
                });
            }
        }
    }
    pub fn current(&self, step: u64) -> Vec<Hazard> {
        let mut result: Vec<_> = self
            .readings
            .iter()
            .filter(|r| {
                r.count >= 2
                    && step.saturating_sub(r.step) <= 1500
                    && r.value >= if r.kind == "gas" { 0.25 } else { 0.004 }
            })
            .map(|r| Hazard {
                kind: r.kind,
                position: r.position,
                radius_m: 0.15 + r.sigma.min(0.10),
                value: r.value,
                observations: r.count,
                observers: r.observers.len(),
                last_seen_step: r.step,
                status: if r.observers.len() > 1 {
                    "MULTI_NODE_REACTION"
                } else {
                    "SINGLE_NODE_REACTION"
                },
            })
            .collect();
        result.sort_by(|a, b| {
            let severity = |h: &Hazard| {
                if h.kind == "gas" {
                    h.value
                } else {
                    h.value / 0.04
                }
            };
            severity(b).total_cmp(&severity(a))
        });
        // Preserve both hazard kinds when one has many observed patches.
        let (mut gas, mut water) = (0, 0);
        result.retain(|h| {
            let count = if h.kind == "gas" {
                &mut gas
            } else {
                &mut water
            };
            *count += 1;
            *count <= 8
        });
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn requires_repeated_readings_and_ages_out_without_claiming_a_leak_source() {
        let mut map = HazardMap::default();
        let p = [0.0, 0.02, 0.7];
        map.observe("gas", p, 0.8, 0.02, "N00", 25);
        assert!(map.current(25).is_empty());
        map.observe("gas", p, 0.8, 0.02, "N01", 50);
        let h = map.current(50);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].observers, 2);
        assert!(
            h[0].position
                .iter()
                .zip(p)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        assert!(map.current(1600).is_empty());
        for step in (75..500).step_by(25) {
            map.observe("gas", p, 0.0, 0.02, "N01", step);
        }
        assert!(map.current(500).is_empty());
    }
    #[test]
    fn missing_environment_data_produces_no_hazards_and_water_uses_its_own_units() {
        let mut map = HazardMap::default();
        assert!(map.current(25).is_empty());
        for step in [25, 50] {
            map.observe("water", [0.1, 0.01, 1.2], 0.025, 0.01, "N02", step);
        }
        assert_eq!(map.current(50)[0].kind, "water");
        assert!((map.current(50)[0].value - 0.025).abs() < 1e-9);
    }
    #[test]
    fn numerous_water_patches_cannot_hide_a_gas_reaction() {
        let mut map = HazardMap::default();
        for step in [25, 50] {
            map.observe("gas", [-0.4, 0.02, 0.2], 0.30, 0.01, "N00", step);
            for i in 0..24 {
                map.observe(
                    "water",
                    [(i % 6) as f64 * 0.23, 0.01, (i / 6) as f64 * 0.23],
                    0.04,
                    0.01,
                    "N01",
                    step,
                );
            }
        }
        let displayed = map.current(50);
        assert!(displayed.iter().any(|h| h.kind == "gas"));
        assert!(displayed.iter().filter(|h| h.kind == "water").count() <= 8);
    }
}
