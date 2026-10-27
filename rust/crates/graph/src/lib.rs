//! Permutation-invariant temporal/radio graph summary. No geometry or truth imports.
use protocol::{Event, Kind};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
pub const DIM: usize = 16;
pub const FEATURES: [&str; DIM] = [
    "impact_per_node",
    "impact_mean",
    "still_fraction",
    "radio_per_node",
    "radio_rssi_mean",
    "radio_rssi_sd",
    "largest_component_ratio",
    "radio_components_per_node",
    "timeout_per_node",
    "recontact_per_node",
    "first_impact_s",
    "coupling_per_node",
    "quality_mean",
    "missing_ratio",
    "elapsed_s",
    "prior_deployed_fraction",
];

#[derive(Clone, Debug)]
pub struct Summary {
    pub x: [f64; DIM],
    pub mask: [bool; DIM],
    pub observed: usize,
}
impl Summary {
    pub fn vector(&self) -> Vec<f64> {
        self.x
            .iter()
            .copied()
            .chain(self.mask.iter().map(|&x| if x { 1.0 } else { 0.0 }))
            .collect()
    }
}

pub fn summarize(events: &[Event], prior_deployed: usize, ablation: &str) -> Summary {
    let ns = event_core::nodes(events);
    let n = ns.len().max(1) as f64;
    let mut ordered: Vec<_> = events.iter().collect();
    ordered.sort_by_key(|e| e.t_ms);
    let t0 = ordered.first().map_or(0, |e| e.t_ms);
    let t1 = ordered.last().map_or(t0, |e| e.t_ms);
    let duration = ((t1 - t0) as f64 / 1000.0).max(0.02);
    let mut impacts = Vec::new();
    let mut radio = Vec::new();
    let mut still = HashMap::new();
    let mut still_ms = 0u64;
    let mut edges = BTreeSet::new();
    let mut heard = HashSet::new();
    let mut timed_out = HashSet::new();
    let mut recontact = 0.;
    let mut timeout = 0.;
    let mut coupled = 0.;
    let mut missing = 0.;
    let mut first = None;
    for e in &ordered {
        match e.kind {
            Kind::Impact if ablation != "radio" => {
                impacts.push(e.value);
                first.get_or_insert((e.t_ms - t0) as f64 / 1000.0);
            }
            Kind::StillStart if ablation != "radio" => {
                still.entry(e.node_id.as_str()).or_insert(e.t_ms);
            }
            Kind::StillEnd if ablation != "radio" => {
                if let Some(start) = still.remove(e.node_id.as_str()) {
                    still_ms += e.t_ms - start;
                }
            }
            Kind::PeerHeard if ablation != "impact" => {
                radio.push(e.value);
                let key = (e.node_id.as_str(), e.peer_id.as_str());
                if timed_out.remove(&key) {
                    recontact += 1.;
                }
                heard.insert(key);
                let (a, b) = if key.0 < key.1 { key } else { (key.1, key.0) };
                edges.insert((a, b));
            }
            Kind::PeerTimeout if ablation != "impact" => {
                timeout += 1.;
                let key = (e.node_id.as_str(), e.peer_id.as_str());
                if heard.contains(&key) {
                    timed_out.insert(key);
                }
            }
            Kind::CoupleStart => coupled += 1.,
            Kind::ObservationGap => missing += 1.,
            _ => {}
        }
    }
    for start in still.values() {
        still_ms += t1 - *start;
    }
    let mean = |v: &Vec<f64>| {
        if v.is_empty() {
            0.
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let rm = mean(&radio);
    let rs = if radio.is_empty() {
        0.
    } else {
        (radio.iter().map(|v| (v - rm).powi(2)).sum::<f64>() / radio.len() as f64).sqrt()
    };
    let mut adj: BTreeMap<&str, Vec<&str>> = ns.iter().map(|&k| (k, Vec::new())).collect();
    for &(a, b) in &edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    let mut visited = HashSet::new();
    let mut largest = 0;
    let mut components = 0;
    for &node in adj.keys() {
        if visited.contains(node) {
            continue;
        }
        components += 1;
        let mut stack = vec![node];
        let mut size = 0;
        while let Some(v) = stack.pop() {
            if !visited.insert(v) {
                continue;
            }
            size += 1;
            stack.extend(adj.get(v).into_iter().flatten().copied());
        }
        largest = largest.max(size);
    }
    let x = [
        impacts.len() as f64 / n,
        mean(&impacts),
        still_ms as f64 / (duration * 1000. * n),
        radio.len() as f64 / n,
        rm,
        rs,
        largest as f64 / n,
        components as f64 / n,
        timeout / n,
        recontact / n,
        first.unwrap_or(0.),
        coupled / n,
        if events.is_empty() {
            0.
        } else {
            events.iter().map(|e| e.quality).sum::<f64>() / events.len() as f64
        },
        missing / (missing + radio.len() as f64 + 1.),
        duration,
        prior_deployed as f64 / 20.,
    ];
    let mut mask = [true; DIM];
    mask[1] = !impacts.is_empty();
    mask[4] = !radio.is_empty();
    mask[5] = !radio.is_empty();
    mask[10] = first.is_some();
    Summary {
        x,
        mask,
        observed: ns.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn e(id: &str, peer: &str, kind: Kind, t: u64) -> Event {
        Event {
            node_id: id.into(),
            peer_id: peer.into(),
            kind,
            t_ms: t,
            node_time_ms: t as i64,
            value: -60.,
            quality: 1.,
        }
    }
    #[test]
    fn id_and_order_invariant() {
        let a = vec![
            e("A", "B", Kind::PeerHeard, 100),
            e("B", "A", Kind::PeerHeard, 200),
            e("A", "", Kind::StillStart, 300),
        ];
        let mut b = a.clone();
        b.reverse();
        for e in &mut b {
            e.node_id = e.node_id.replace('A', "Z").replace('B', "Y");
            e.peer_id = e.peer_id.replace('A', "Z").replace('B', "Y");
        }
        assert_eq!(
            summarize(&a, 5, "full").vector(),
            summarize(&b, 5, "full").vector()
        );
    }
    #[test]
    fn empty_has_mask() {
        let s = summarize(&[], 0, "full");
        assert!(s.x.iter().all(|x| x.is_finite()));
        assert!(!s.mask[4]);
        assert_eq!(s.observed, 0);
    }
}
