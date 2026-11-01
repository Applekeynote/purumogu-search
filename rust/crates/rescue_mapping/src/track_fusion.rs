//! Relative candidate fusion. No scene geometry, person labels or absolute-coordinate priors.
use super::*;

pub fn classify(model: &LearnedModel, proposals: &[Proposal]) -> Vec<Cue> {
    let mut candidates: Vec<_> = proposals
        .iter()
        .filter_map(|p| {
            if !p.position.iter().all(|v| v.is_finite()) {
                return None;
            }
            let score = model.track_score(p.features.clone())?;
            let feature = |i| p.features.get(i).copied().unwrap_or(0.0);
            Some(Cue {
                position: p.position,
                score,
                radius_m: feature(5).max(feature(4) * 0.5),
                observers: p.observers,
                status: "UNCONFIRMED_LEARNED_TRACK",
                last_seen_step: p.last_seen_step,
                evidence: CueEvidence {
                    near: feature(8) * 2.0,
                    human: feature(9),
                    heat: feature(10) * 2.0,
                    range_error: feature(11) * 3.0,
                },
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    select(model, &candidates)
}

fn distance(a: V3, b: V3, height: f64) -> f64 {
    let mut delta = sub(a, b);
    delta[1] *= height;
    norm(delta)
}

fn select(model: &LearnedModel, candidates: &[Cue]) -> Vec<Cue> {
    let mut seeds: Vec<usize> = vec![];
    let mut result = vec![];
    for (i, seed) in candidates.iter().enumerate() {
        if seed.score < model.track_threshold {
            break;
        }
        if seeds.iter().any(|&j| {
            distance(
                seed.position,
                candidates[j].position,
                model.track_height_scale,
            ) <= model.track_nms_m
        }) {
            continue;
        }
        seeds.push(i);
        let mut cue = seed.clone();
        if model.track_refine_radius_m > 0.0 {
            let neighbors: Vec<_> = candidates
                .iter()
                .filter(|c| {
                    distance(seed.position, c.position, model.track_height_scale)
                        <= model.track_refine_radius_m
                })
                .map(|c| (c, c.score.max(1e-9).powf(model.track_refine_score_power)))
                .collect();
            let sum: f64 = neighbors.iter().map(|(_, w)| w).sum();
            if sum > 0.0 {
                cue.position = std::array::from_fn(|axis| {
                    neighbors
                        .iter()
                        .map(|(c, w)| c.position[axis] * w)
                        .sum::<f64>()
                        / sum
                });
                let variance = neighbors
                    .iter()
                    .map(|(c, w)| norm(sub(c.position, cue.position)).powi(2) * w)
                    .sum::<f64>()
                    / sum;
                cue.radius_m = cue.radius_m.max(variance.sqrt());
            }
        }
        // Uncertainty may grow; repeated proposals never inflate the observer count.
        result.push(cue);
        if result.len() == 3 {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cue(x: f64, y: f64, score: f64) -> Cue {
        Cue {
            position: [x, y, 0.0],
            score,
            radius_m: 0.03,
            observers: 3,
            status: "test",
            last_seen_step: 100,
            evidence: CueEvidence::default(),
        }
    }
    fn model() -> LearnedModel {
        serde_json::from_str(r#"{"version":"rescue-trained-v1","cell_m":0.05,"acoustic_weights":[0,1,0,0,0],"occupancy_weights":[0,1,0],"fusion_weights":[1,1,1,1],"prototypes":[],"data_sha256":"","train_maps":[],"valid_maps":[],"track_threshold":0.5,"track_nms_m":0.3}"#).unwrap()
    }
    #[test]
    fn disabled_fusion_preserves_the_selected_maximum() {
        let m = model();
        let result = select(
            &m,
            &[cue(0.0, 0.0, 0.9), cue(0.2, 0.0, 0.8), cue(1.0, 0.0, 0.6)],
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(result[1].position, [1.0, 0.0, 0.0]);
    }
    #[test]
    fn weak_neighbors_refine_a_seed_without_creating_a_new_detection() {
        let mut m = model();
        m.track_refine_radius_m = 0.45;
        let result = select(&m, &[cue(0.0, 0.0, 0.9), cue(0.2, 0.0, 0.1)]);
        assert_eq!(result.len(), 1);
        assert!((result[0].position[0] - 0.02).abs() < 1e-12);
        assert!(result[0].radius_m > 0.03);
        assert_eq!(result[0].observers, 3);
    }
    #[test]
    fn height_scaling_keeps_distinct_storeys_out_of_a_fusion_neighborhood() {
        let mut m = model();
        m.track_refine_radius_m = 0.45;
        m.track_height_scale = 3.0;
        let result = select(&m, &[cue(0.0, 0.0, 0.9), cue(0.0, 0.28, 0.8)]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].position[1], 0.0);
        assert_eq!(result[1].position[1], 0.28);
    }
}
