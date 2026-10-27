//! TRAIN-only penalized logistic fitting for sensor-derived track evidence.
use anyhow::{Result, ensure};
use rescue_mapping::learned::LearnedModel;
use serde_json::Value;
use std::fs;
#[path = "../boosting.rs"]
mod boosting;

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-40.0, 40.0)).exp())
}

fn solve(mut matrix: Vec<Vec<f64>>, mut target: Vec<f64>) -> Result<Vec<f64>> {
    let n = target.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        ensure!(matrix[pivot][column].abs() > 1e-14, "singular Hessian");
        matrix.swap(column, pivot);
        target.swap(column, pivot);
        let scale = matrix[column][column];
        for value in &mut matrix[column][column..] {
            *value /= scale;
        }
        target[column] /= scale;
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            let pivot_row = matrix[column][column..].to_vec();
            for (value, pivot_value) in matrix[row][column..].iter_mut().zip(pivot_row) {
                *value -= factor * pivot_value;
            }
            target[row] -= factor * target[column];
        }
    }
    Ok(target)
}

fn objective(x: &[Vec<f64>], y: &[f64], w: &[f64], ridge: f64) -> f64 {
    let loss = x
        .iter()
        .zip(y)
        .map(|(row, label)| {
            let z = row.iter().zip(w).map(|(a, b)| a * b).sum::<f64>();
            z.max(0.0) - label * z + (-z.abs()).exp().ln_1p()
        })
        .sum::<f64>()
        / y.len() as f64;
    loss + ridge * w[1..].iter().map(|v| v * v).sum::<f64>() / 2.0
}

fn fit(x: &[Vec<f64>], y: &[f64], ridge: f64) -> Result<Vec<f64>> {
    ensure!(
        !x.is_empty() && x.len() == y.len() && ridge.is_finite() && ridge > 0.0,
        "invalid training configuration"
    );
    let size = x[0].len();
    ensure!(
        x.iter()
            .all(|r| r.len() == size && r.iter().all(|v| v.is_finite()))
            && y.contains(&0.0)
            && y.contains(&1.0)
            && y.iter().all(|v| *v == 0.0 || *v == 1.0),
        "invalid rows"
    );
    let mut weights = vec![0.0; size];
    for _ in 0..100 {
        let mut gradient = vec![0.0; size];
        let mut hessian = vec![vec![0.0; size]; size];
        for (row, label) in x.iter().zip(y) {
            let p = sigmoid(row.iter().zip(&weights).map(|(a, b)| a * b).sum());
            for i in 0..size {
                gradient[i] += row[i] * (p - label) / y.len() as f64;
                for j in 0..size {
                    hessian[i][j] += row[i] * row[j] * p * (1.0 - p) / y.len() as f64;
                }
            }
        }
        for i in 0..size {
            if i > 0 {
                gradient[i] += ridge * weights[i];
                hessian[i][i] += ridge;
            }
            hessian[i][i] += 1e-8;
        }
        let delta = solve(hessian, gradient)?;
        if delta.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-7 {
            break;
        }
        let before = objective(x, y, &weights, ridge);
        let mut rate = 1.0;
        loop {
            let trial: Vec<_> = weights
                .iter()
                .zip(&delta)
                .map(|(w, d)| w - rate * d)
                .collect();
            if objective(x, y, &trial, ridge) <= before {
                weights = trial;
                break;
            }
            rate *= 0.5;
            if rate < 1e-8 {
                return Ok(weights);
            }
        }
    }
    Ok(weights)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        matches!(args.len(), 6 | 7 | 9),
        "usage: train_tracks TRAIN.json BASE_MODEL.json OUTPUT.json RIDGE DATA_SHA256 [track|occupancy|fusion|track-boost] [DEPTH ROUNDS]"
    );
    let mode = args.get(6).map_or("track", String::as_str);
    let features = match mode {
        "track" | "track-boost" => 12,
        "occupancy" => 2,
        "fusion" => 3,
        _ => anyhow::bail!("unknown fitting mode"),
    };
    let rows: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut model: LearnedModel = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut x = vec![];
    let mut y = vec![];
    for row in rows
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("expected maps array"))?
    {
        ensure!(row["split"] == "train", "only TRAIN may be fitted");
        let id = row["map_id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("map_id required"))?;
        ensure!(
            model.train_maps.iter().any(|m| m == id) && !model.valid_maps.iter().any(|m| m == id),
            "training map is not exclusively TRAIN"
        );
        for proposal in row
            .get("examples")
            .unwrap_or(&row["proposals"])
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("proposals required"))?
        {
            let feature = proposal["features"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("features required"))?;
            ensure!(feature.len() == features, "feature contract mismatch");
            let mut sample = vec![1.0];
            for value in feature {
                sample.push(
                    value
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("numeric feature required"))?,
                );
            }
            x.push(sample);
            y.push(
                proposal["label"]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("numeric label required"))?,
            );
        }
    }
    if mode == "track-boost" {
        let depth = args.get(7).map_or(Ok(2), |s| s.parse::<usize>())?;
        let rounds = args.get(8).map_or(Ok(60), |s| s.parse::<usize>())?;
        let (bias, trees) = boosting::fit(&x, &y, depth, rounds)?;
        model.track_weights.clear();
        model.track_bias = bias;
        model.track_trees = trees;
    } else {
        let weights = fit(&x, &y, args[4].parse()?)?;
        match mode {
            "track" => model.track_weights = weights,
            "occupancy" => model.occupancy_weights = weights,
            "fusion" => model.fusion_weights = weights,
            _ => unreachable!(),
        }
        if mode == "track" {
            model.track_trees.clear();
            model.track_bias = 0.0;
        }
    }
    model.data_sha256 = args[5].clone();
    model.validate()?;
    fs::write(&args[3], serde_json::to_vec(&model)?)?;
    println!(
        "TRAIN logistic fit: {} proposals, {} positives, {} maps",
        y.len(),
        y.iter().sum::<f64>(),
        rows.as_array().unwrap().len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn irls_separates_heldout_points_and_rejects_degenerate_labels() {
        let x = vec![
            vec![1.0, -3.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
            vec![1.0, 3.0],
        ];
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let w = fit(&x, &y, 0.001).unwrap();
        assert!(sigmoid(w[0] + w[1] * 0.5) > 0.8);
        assert!(sigmoid(w[0] - w[1] * 0.5) < 0.2);
        assert!(fit(&x, &[1.0; 4], 0.001).is_err());
    }
}
