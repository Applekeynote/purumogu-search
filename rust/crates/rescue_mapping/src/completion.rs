//! Conservative, geometry-independent patch inference from measured local planes.
use crate::{Cell, learned};
use std::collections::{BTreeMap, HashMap};

/// No inferred cell is used to support another inference. Confident free space wins.
pub fn infer(cells: &[Cell]) -> BTreeMap<[i32; 3], f64> {
    let measured: HashMap<_, _> = cells
        .iter()
        .filter_map(|c| {
            let q = c.sensor_occupancy?;
            (c.observations >= 2 && !(0.30..0.70).contains(&q)).then_some((c.key, q >= 0.70))
        })
        .collect();
    let protected: std::collections::BTreeSet<_> = cells
        .iter()
        .filter_map(|c| {
            let q = c.sensor_occupancy?;
            (c.observations > 0 && (q <= 0.30 || q >= 0.70)).then_some(c.key)
        })
        .collect();
    let mut result = BTreeMap::new();
    if measured.len() < 16 {
        return result;
    }
    for x in -16..16 {
        for y in -3..18 {
            for z in -6..64 {
                let key = [x, y, z];
                if protected.contains(&key) {
                    continue;
                }
                for normal in 0..3 {
                    let a = (normal + 1) % 3;
                    let b = (normal + 2) % 3;
                    let mut supports = Vec::new();
                    let mut bracketed = true;
                    for axis in [a, b] {
                        for sign in [-1, 1] {
                            let found = (1..=2).find_map(|distance| {
                                let mut k = key;
                                k[axis] += sign * distance;
                                measured.get(&k).map(|&occupied| (k, occupied))
                            });
                            if let Some(support) = found {
                                supports.push(support);
                            } else {
                                bracketed = false;
                            }
                        }
                    }
                    if !bracketed || supports.iter().any(|s| s.1 != supports[0].1) {
                        continue;
                    }
                    let occupied = supports[0].1;
                    let mut votes = 0;
                    let mut contradictions = 0;
                    let mut surface_side = [0, 0];
                    for da in -2..=2 {
                        for db in -2..=2 {
                            if da == 0 && db == 0 {
                                continue;
                            }
                            let mut k = key;
                            k[a] += da;
                            k[b] += db;
                            if let Some(&value) = measured.get(&k) {
                                if value == occupied {
                                    votes += 1;
                                    for (i, sign) in [-1, 1].iter().enumerate() {
                                        let mut adjacent = k;
                                        adjacent[normal] += sign;
                                        if measured.get(&adjacent) == Some(&false) {
                                            surface_side[i] += 1;
                                        }
                                    }
                                } else {
                                    contradictions += 1;
                                }
                            }
                        }
                    }
                    if votes >= 8
                        && contradictions == 0
                        && (!occupied || surface_side.into_iter().max().unwrap_or(0) >= 3)
                        && learned::index(key).is_some()
                    {
                        result.insert(key, if occupied { 0.88 } else { 0.12 });
                        break;
                    }
                }
                if !result.get(&key).is_some_and(|q| *q > 0.5)
                    && (0..3).any(|axis| fitted_occupied_patch(&measured, key, axis))
                {
                    result.insert(key, 0.82);
                }
            }
        }
    }
    result
}

/// A sloped or quantized surface need not occupy one exact voxel layer.
fn fitted_occupied_patch(measured: &HashMap<[i32; 3], bool>, key: [i32; 3], normal: usize) -> bool {
    let a = (normal + 1) % 3;
    let b = (normal + 2) % 3;
    let sample = |da: i32, db: i32| -> Option<([i32; 3], f64)> {
        for dn in [0, -1, 1] {
            let mut k = key;
            k[a] += da;
            k[b] += db;
            k[normal] += dn;
            if measured.get(&k) == Some(&true) {
                return Some((k, f64::from(dn)));
            }
        }
        None
    };
    for axis in [a, b] {
        for sign in [-1, 1] {
            if !(1..=3).any(|d| {
                sample(
                    if axis == a { sign * d } else { 0 },
                    if axis == b { sign * d } else { 0 },
                )
                .is_some()
            }) {
                return false;
            }
        }
    }
    let mut points = vec![];
    let mut free_side = [0, 0];
    for da in -3..=3 {
        for db in -3..=3 {
            if da == 0 && db == 0 {
                continue;
            }
            if let Some((k, h)) = sample(da, db) {
                points.push((f64::from(da), f64::from(db), h));
                for (i, sign) in [-1, 1].iter().enumerate() {
                    let mut n = k;
                    n[normal] += sign;
                    if measured.get(&n) == Some(&false) {
                        free_side[i] += 1;
                    }
                }
            }
        }
    }
    if points.len() < 12 || free_side.into_iter().max().unwrap_or(0) < 4 {
        return false;
    }
    let count = points.len() as f64;
    let (mx, my, mh) = points.iter().fold((0.0, 0.0, 0.0), |(x, y, h), p| {
        (x + p.0 / count, y + p.1 / count, h + p.2 / count)
    });
    let (xx, yy, xy, xh, yh) =
        points
            .iter()
            .fold((0.0, 0.0, 0.0, 0.0, 0.0), |(xx, yy, xy, xh, yh), p| {
                let x = p.0 - mx;
                let y = p.1 - my;
                let h = p.2 - mh;
                (xx + x * x, yy + y * y, xy + x * y, xh + x * h, yh + y * h)
            });
    let det = xx * yy - xy * xy;
    if det < 1e-6 {
        return false;
    }
    let sx = (xh * yy - yh * xy) / det;
    let sy = (yh * xx - xh * xy) / det;
    let intercept = mh - sx * mx - sy * my;
    intercept.abs() <= 0.55
        && sx.abs() <= 0.6
        && sy.abs() <= 0.6
        && points
            .iter()
            .filter(|p| (p.2 - (sx * p.0 + sy * p.1 + intercept)).abs() <= 0.55)
            .count()
            * 5
            >= points.len() * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    fn patch(free_hole: bool) -> Vec<Cell> {
        let mut cells = vec![];
        for x in -3..=3 {
            for z in -3..=3 {
                for y in [0, 1] {
                    if x == 0 && z == 0 && (y == 1 || !free_hole) {
                        continue;
                    }
                    let q = if y == 0 && !(free_hole && x == 0 && z == 0) {
                        0.96
                    } else {
                        0.04
                    };
                    cells.push(Cell {
                        key: [x, y, z],
                        occupancy: q,
                        observations: 4,
                        source: "sensor",
                        sensor_occupancy: Some(q),
                    });
                }
            }
        }
        cells
    }
    #[test]
    fn repairs_a_bounded_plane_gap_without_templates() {
        let result = infer(&patch(false));
        assert_eq!(result.get(&[0, 0, 0]), Some(&0.88));
        assert_eq!(result.get(&[0, 1, 0]), Some(&0.12));
        assert!(!result.contains_key(&[8, 0, 0]));
    }
    #[test]
    fn a_measured_opening_is_never_closed_and_inferences_do_not_expand() {
        let result = infer(&patch(true));
        assert!(!result.contains_key(&[0, 0, 0]));
        assert!(infer(&[]).is_empty());
        let only_priors: Vec<_> = patch(false)
            .into_iter()
            .map(|mut c| {
                c.observations = 0;
                c.sensor_occupancy = None;
                c
            })
            .collect();
        assert!(infer(&only_priors).is_empty());
    }
    #[test]
    fn a_quantized_tilted_plane_can_repair_a_gap_without_overwriting_free_space() {
        let mut cells = vec![];
        for x in -3..=3 {
            for z in -3..=3 {
                let floor = if x < 0 { -1 } else { 0 };
                for y in floor..=floor + 1 {
                    if x == 0 && z == 0 && y == 0 {
                        continue;
                    }
                    let q = if y == floor { 0.95 } else { 0.05 };
                    cells.push(Cell {
                        key: [x, y, z],
                        occupancy: q,
                        observations: 4,
                        source: "sensor",
                        sensor_occupancy: Some(q),
                    });
                }
            }
        }
        assert!(infer(&cells).get(&[0, 0, 0]).is_some_and(|q| *q > 0.6));
        cells.push(Cell {
            key: [0, 0, 0],
            occupancy: 0.05,
            observations: 1,
            source: "sensor",
            sensor_occupancy: Some(0.05),
        });
        assert!(!infer(&cells).contains_key(&[0, 0, 0]));
    }
}
