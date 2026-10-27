use inference::{Belief, entropy};
use likelihood::{ACTIONS, CLASSES, K, Model};
use protocol::Action;
pub fn eig(p: &Belief, action: usize, m: &Model) -> f64 {
    let likelihood: Vec<_> = (0..CLASSES).map(|g| m.probabilities(g, action)).collect();
    let mut expected = 0.;
    for (z, _) in likelihood[0].iter().enumerate().take(K) {
        let pz = (0..CLASSES).map(|g| p[g] * likelihood[g][z]).sum::<f64>();
        if pz <= 0. {
            continue;
        }
        let mut posterior = [0.; CLASSES];
        for g in 0..CLASSES {
            posterior[g] = p[g] * likelihood[g][z] / pz;
        }
        expected += pz * entropy(&posterior);
    }
    (entropy(p) - expected).max(0.)
}
pub fn choose(
    p: &Belief,
    m: &Model,
    policy: &str,
    batch: usize,
    random: &mut u64,
) -> (Action, f64) {
    let a = match policy {
        "fixed" | "passive" => (batch % 3) * 2,
        "random" => {
            *random = random
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*random >> 32) as usize) % ACTIONS
        }
        _ => (0..ACTIONS)
            .max_by(|&a, &b| {
                eig(p, a, m)
                    .total_cmp(&eig(p, b, m))
                    .then_with(|| b.cmp(&a))
            })
            .unwrap(),
    };
    (
        Action {
            action_id: format!("a-{batch:03}"),
            entry_id: a / 2,
            node_count: 5,
            formation_mode: protocol::MODES[a % 2].into(),
            delay_steps: 0,
            policy: policy.into(),
        },
        eig(p, a, m),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    pub fn uniform() -> Model {
        Model {
            version: "test".into(),
            schema: protocol::SCHEMA.into(),
            labels: vec![],
            features: vec![],
            mean: vec![0.; 32],
            scale: vec![1.; 32],
            centers: vec![vec![0.; 32]; K],
            likelihood: vec![vec![vec![1. / K as f64; K]; ACTIONS]; CLASSES],
            counts: vec![vec![1; ACTIONS]; CLASSES],
            temperature: 1.,
            support_radius_sq: 10.,
            abstain_threshold: 0.75,
            training_maps: vec![],
            validation_maps: vec![],
        }
    }
    #[test]
    fn no_information_preserves_prior() {
        let m = uniform();
        let p = inference::prior();
        assert!(eig(&p, 0, &m) < 1e-10);
        let q = inference::update(&p, 0, 0, &m);
        for g in 0..6 {
            assert!((q[g] - p[g]).abs() < 1e-10);
        }
    }
    #[test]
    fn informative_evidence_increases_candidate() {
        let mut m = uniform();
        m.likelihood[0][0][0] = 0.8;
        let p = inference::prior();
        let q = inference::update(&p, 0, 0, &m);
        assert!(q[0] > p[0]);
        assert!((q.iter().sum::<f64>() - 1.).abs() < 1e-10);
        assert!(q.iter().all(|p| p.is_finite()));
    }
    #[test]
    fn information_gain_bounded() {
        let m = uniform();
        let p = inference::prior();
        for a in 0..6 {
            assert!(eig(&p, a, &m) <= entropy(&p) + 1e-10);
        }
    }
}
