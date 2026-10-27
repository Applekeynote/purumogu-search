use likelihood::{CLASSES, Model};
pub type Belief = [f64; CLASSES];
pub fn prior() -> Belief {
    [0.19, 0.19, 0.19, 0.19, 0.19, 0.05]
}
pub fn entropy(p: &Belief) -> f64 {
    p.iter().filter(|&&p| p > 0.).map(|p| -p * p.log2()).sum()
}
pub fn update(p: &Belief, z: usize, action: usize, m: &Model) -> Belief {
    let mut log = [0.; CLASSES];
    for g in 0..CLASSES {
        log[g] = p[g].max(1e-300).ln() + m.probabilities(g, action)[z].max(1e-300).ln();
    }
    let max = log.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum = log.iter().map(|l| (l - max).exp()).sum::<f64>();
    let mut out = [0.; CLASSES];
    for g in 0..CLASSES {
        out[g] = (log[g] - max).exp() / sum;
    }
    out
}
pub fn abstain(
    p: &Belief,
    summary: &graph::Summary,
    distance: f64,
    m: &Model,
) -> (bool, &'static str) {
    if summary.observed == 0 {
        return (true, "no observations");
    }
    if distance > m.support_radius_sq {
        return (true, "outside training support");
    }
    if summary.x[13] > 0.6 {
        return (true, "sensor coverage insufficient");
    }
    if p[5] > 0.25 {
        return (
            true,
            "OTHER is plausible; score is not a calibrated novelty probability",
        );
    }
    if p.iter().copied().fold(0., f64::max) < m.abstain_threshold {
        return (true, "candidate families remain ambiguous");
    }
    (
        false,
        "provisional family estimate; geometry is not reconstructed",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn entropy_limits() {
        assert!((entropy(&[1. / 6.; 6]) - 6f64.log2()).abs() < 1e-10);
        assert_eq!(entropy(&[1., 0., 0., 0., 0., 0.]), 0.);
    }
    #[test]
    fn prior_normalized() {
        assert!((prior().iter().sum::<f64>() - 1.).abs() < 1e-10);
    }
}
