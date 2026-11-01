//! Learned local volumetric calibration. Input is relative ray evidence only.
//! No scene identifier, global coordinates, geometry prototype or source truth.
use crate::{
    Cell,
    learned::{self, LearnedModel},
};

pub const FEATURES: usize = 322;
pub const FEATURE_MAP: &str = "relative-rays-90-v1";
pub const WIDE_FEATURE_MAP: &str = "relative-rays-322-v2";

pub fn wide_offsets() -> Vec<[i32; 3]> {
    let mut result = Vec::with_capacity(FEATURES / 2);
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -2..=2 {
                result.push([x, y, z]);
            }
        }
    }
    for radius in [3, 4, 6, 8, 12, 16] {
        for axis in 0..3 {
            for sign in [-1, 1] {
                let mut offset = [0; 3];
                offset[axis] = radius * sign;
                result.push(offset);
            }
        }
    }
    result
}

pub fn offsets() -> Vec<[i32; 3]> {
    let mut result = Vec::with_capacity(FEATURES / 2);
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                result.push([x, y, z]);
            }
        }
    }
    for radius in [2, 3, 4] {
        for axis in 0..3 {
            for sign in [-1, 1] {
                let mut offset = [0; 3];
                offset[axis] = radius * sign;
                result.push(offset);
            }
        }
    }
    result
}

pub fn evidence(cells: &[Cell]) -> Vec<[f64; 2]> {
    let mut grid = vec![[0.0; 2]; learned::GRID_CELLS];
    for cell in cells {
        if cell.observations == 0 {
            continue;
        }
        if let (Some(index), Some(q)) = (learned::index(cell.key), cell.sensor_occupancy) {
            let q = q.clamp(1e-6, 1.0 - 1e-6);
            grid[index] = [
                f64::from((((q / (1.0 - q)).ln() / 4.0).clamp(-1.0, 1.0)) as f32),
                f64::from(
                    (((1.0 + f64::from(cell.observations)).ln() / 1025f64.ln()).min(1.0)) as f32,
                ),
            ];
        }
    }
    grid
}

pub fn features(grid: &[[f64; 2]], key: [i32; 3], offsets: &[[i32; 3]]) -> [f64; FEATURES] {
    let mut result = [0.0; FEATURES];
    for (i, offset) in offsets.iter().enumerate() {
        if let Some(index) = learned::index(std::array::from_fn(|a| key[a] + offset[a])) {
            result[i * 2] = grid[index][0];
            result[i * 2 + 1] = grid[index][1];
        }
    }
    result
}

pub fn apply(cells: &mut Vec<Cell>, model: &LearnedModel) {
    let grid = evidence(cells);
    let offsets = if model.geometry_feature_map == WIDE_FEATURE_MAP {
        wide_offsets()
    } else {
        offsets()
    };
    let mut positions = vec![None; learned::GRID_CELLS];
    for (i, cell) in cells.iter().enumerate() {
        positions[learned::index(cell.key).expect("bounded")] = Some(i);
    }
    for x in -16..16 {
        for y in -3..18 {
            for z in -6..64 {
                let key = [x, y, z];
                let index = learned::index(key).expect("bounded");
                let f = features(&grid, key, &offsets);
                // Entirely unobserved contexts must remain unknown, even if a biased model is loaded.
                if !(0..FEATURES / 2).any(|i| f[i * 2 + 1] > 0.0) {
                    continue;
                }
                let probability = model.geometry_score(&f);
                if let Some(i) = positions[index] {
                    cells[i].occupancy = probability;
                    cells[i].source = "local_surface_completion";
                } else if !(0.2..=0.6).contains(&probability) {
                    cells.push(Cell {
                        key,
                        occupancy: probability,
                        observations: 0,
                        source: "local_surface_completion",
                        sensor_occupancy: None,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relative_features_and_unknown_mask() {
        let cells = vec![Cell {
            key: [1, 2, 3],
            occupancy: 0.01,
            observations: 8,
            source: "sensor",
            sensor_occupancy: Some(0.9),
        }];
        let grid = evidence(&cells);
        let off = offsets();
        assert_eq!(off.len() * 2, 90);
        assert_eq!(wide_offsets().len() * 2, FEATURES);
        let f = features(&grid, [0, 2, 3], &off);
        let i = off.iter().position(|k| *k == [1, 0, 0]).unwrap();
        assert!((f[i * 2] - 9f64.ln() / 4.0).abs() < 1e-7);
        assert!((f[i * 2 + 1] - 9f64.ln() / 1025f64.ln()).abs() < 1e-7);
        assert!(features(&grid, [-15, 0, 0], &off).iter().all(|v| *v == 0.0));
        let shifted = vec![Cell {
            key: [5, 3, 9],
            ..cells[0].clone()
        }];
        assert_eq!(f, features(&evidence(&shifted), [4, 3, 9], &off));
    }
    #[test]
    fn exported_threshold_keeps_its_ulp_and_unknown_space_stays_unknown() {
        let tree: crate::learned::DecisionTree = serde_json::from_str(
            r#"{"nodes":[
            {"feature":0,"threshold":-0.9999999999999999,"left":1,"right":2,"value":0.0},
            {"feature":null,"threshold":0.0,"left":0,"right":0,"value":3.0},
            {"feature":null,"threshold":0.0,"left":0,"right":0,"value":-3.0}]}"#,
        )
        .unwrap();
        tree.validate_features(FEATURES, 127).unwrap();
        let mut f = [0.0; FEATURES];
        f[0] = -1.0;
        assert_eq!(tree.score_features(&f), 3.0);
        assert!(tree.nodes[0].threshold > -1.0);
        let mut model: LearnedModel = serde_json::from_value(serde_json::json!({
            "version":"rescue-trained-v1","cell_m":0.05,"acoustic_weights":[0.0,0.0,0.0,0.0,0.0],
            "occupancy_weights":[0.0,1.0,0.0],"prototypes":[],"local_surface_completion":true,
            "train_maps":["train-only"],"valid_maps":[],"data_sha256":"fixture",
            "geometry_feature_map":FEATURE_MAP,"geometry_bias":40.0,"geometry_trees":[tree]
        }))
        .unwrap();
        model.validate().unwrap();
        let mut cells = vec![];
        apply(&mut cells, &model);
        assert!(cells.is_empty());
        model.geometry_feature_map = "absolute-coordinates".into();
        assert!(model.validate().is_err());
    }
}
