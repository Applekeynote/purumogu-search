//! Bounded Newton tree boosting for Bernoulli loss, fitted on TRAIN only.
use anyhow::{Result, ensure};
use rescue_mapping::learned::{DecisionTree, TreeNode};

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-40.0, 40.0)).exp())
}

struct Builder<'a> {
    x: &'a [[f64; 12]],
    gradient: &'a [f64],
    hessian: &'a [f64],
    cuts: &'a [Vec<f64>],
    nodes: Vec<TreeNode>,
    lambda: f64,
    min_leaf: usize,
}
impl Builder<'_> {
    fn branch(&mut self, rows: Vec<usize>, depth: usize) -> usize {
        let index = self.nodes.len();
        let g = rows.iter().map(|&i| self.gradient[i]).sum::<f64>();
        let h = rows.iter().map(|&i| self.hessian[i]).sum::<f64>();
        self.nodes.push(TreeNode {
            feature: None,
            threshold: 0.0,
            left: 0,
            right: 0,
            value: (g / (h + self.lambda)).clamp(-3.0, 3.0) * 0.1,
        });
        if depth == 0 || rows.len() < self.min_leaf * 2 {
            return index;
        }
        let mut best = None;
        let mut best_gain = 1e-6;
        for feature in 0..12 {
            for &threshold in &self.cuts[feature] {
                let mut left_g = 0.0;
                let mut left_h = 0.0;
                let mut count = 0;
                for &row in &rows {
                    if self.x[row][feature] < threshold {
                        left_g += self.gradient[row];
                        left_h += self.hessian[row];
                        count += 1;
                    }
                }
                if count < self.min_leaf || rows.len() - count < self.min_leaf {
                    continue;
                }
                let gain = left_g.powi(2) / (left_h + self.lambda)
                    + (g - left_g).powi(2) / (h - left_h + self.lambda)
                    - g.powi(2) / (h + self.lambda);
                if gain > best_gain {
                    best_gain = gain;
                    best = Some((feature, threshold));
                }
            }
        }
        if let Some((feature, threshold)) = best {
            let (left, right) = rows
                .into_iter()
                .partition(|&row| self.x[row][feature] < threshold);
            let left = self.branch(left, depth - 1);
            let right = self.branch(right, depth - 1);
            self.nodes[index] = TreeNode {
                feature: Some(feature),
                threshold,
                left,
                right,
                value: 0.0,
            };
        }
        index
    }
}

pub fn fit(
    x: &[Vec<f64>],
    y: &[f64],
    depth: usize,
    rounds: usize,
) -> Result<(f64, Vec<DecisionTree>)> {
    ensure!(
        x.len() == y.len()
            && !x.is_empty()
            && (1..=4).contains(&depth)
            && (1..=256).contains(&rounds),
        "invalid boosting configuration"
    );
    ensure!(
        x.iter()
            .all(|r| r.len() == 13 && r.iter().all(|v| v.is_finite()))
            && y.contains(&0.0)
            && y.contains(&1.0)
            && y.iter().all(|v| *v == 0.0 || *v == 1.0),
        "invalid boosting rows"
    );
    let samples: Vec<[f64; 12]> = x.iter().map(|r| r[1..].try_into().unwrap()).collect();
    let rate = y.iter().sum::<f64>() / y.len() as f64;
    let bias = (rate / (1.0 - rate)).ln();
    let mut cuts = vec![vec![]; 12];
    for feature in 0..12 {
        let mut sorted: Vec<_> = samples.iter().map(|r| r[feature]).collect();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        let bins = sorted.len().min(32);
        for q in 1..bins {
            let j = sorted.len() * q / bins;
            if j > 0 && j < sorted.len() {
                cuts[feature].push((sorted[j - 1] + sorted[j]) / 2.0);
            }
        }
        cuts[feature].dedup();
    }
    let mut predictions = vec![bias; y.len()];
    let mut trees = vec![];
    for _ in 0..rounds {
        let p: Vec<_> = predictions.iter().map(|&z| sigmoid(z)).collect();
        let gradient: Vec<_> = y.iter().zip(&p).map(|(label, p)| label - p).collect();
        let hessian: Vec<_> = p.iter().map(|p| (p * (1.0 - p)).max(1e-6)).collect();
        let mut builder = Builder {
            x: &samples,
            gradient: &gradient,
            hessian: &hessian,
            cuts: &cuts,
            nodes: vec![],
            lambda: 5.0,
            min_leaf: 12,
        };
        builder.branch((0..y.len()).collect(), depth);
        let tree = DecisionTree {
            nodes: builder.nodes,
        };
        tree.validate()?;
        for (prediction, row) in predictions.iter_mut().zip(&samples) {
            *prediction += tree.score(row);
        }
        trees.push(tree);
    }
    Ok((bias, trees))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depth_two_fits_a_bounded_signal_interval_on_heldout_points() {
        let mut x = vec![];
        let mut y = vec![];
        for value in [-2.0f64, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0] {
            for _ in 0..30 {
                let mut row = vec![0.0; 13];
                row[0] = 1.0;
                row[1] = value;
                x.push(row);
                y.push(if value.abs() < 0.5 { 1.0 } else { 0.0 });
            }
        }
        let (bias, trees) = fit(&x, &y, 2, 60).unwrap();
        for (value, label) in [(-1.5, false), (-0.1, true), (0.1, true), (1.5, false)] {
            let mut row = [0.0; 12];
            row[0] = value;
            let p = sigmoid(bias + trees.iter().map(|t| t.score(&row)).sum::<f64>());
            assert_eq!(p > 0.5, label);
            assert!(if label { p > 0.8 } else { p < 0.2 });
        }
    }
}
