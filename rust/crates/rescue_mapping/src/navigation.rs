//! Supported 3D routes: levels are inferred from range evidence, never scene IDs.
use super::*;

pub(super) fn goals(map: &Mapper, verification: &BTreeSet<String>) -> Vec<Goal> {
    let walkable: BTreeSet<_> = map
        .voxels
        .iter()
        .filter_map(|(key, v)| {
            (v.log_odds < -0.5
                && (1..=2).any(|down| {
                    let mut below = *key;
                    below[1] -= down;
                    map.voxels.get(&below).is_some_and(|v| v.log_odds > 0.8)
                }))
            .then_some(*key)
        })
        .collect();
    let mut result = vec![];
    let mut assigned: Vec<V3> = vec![];
    for (id, pose) in &map.poses {
        if verification.contains(id)
            && let Some(&until) = map.verify_until.get(id)
            && until > map.last_step
        {
            result.push(Goal {
                node_id: id.clone(),
                position: pose.position,
                reason: "acoustic and thermal verification dwell",
                hold_until_step: until,
            });
            continue;
        }
        let origin = map.key(pose.position);
        let Some(start) = walkable
            .iter()
            .filter(|k| {
                (k[0] - origin[0]).abs() <= 2
                    && (k[1] - origin[1]).abs() <= 2
                    && (k[2] - origin[2]).abs() <= 2
            })
            .min_by_key(|k| {
                (k[0] - origin[0]).pow(2) + (k[1] - origin[1]).pow(2) + (k[2] - origin[2]).pow(2)
            })
            .copied()
        else {
            continue;
        };
        let cue = verification
            .contains(id)
            .then(|| {
                map.probe_cues.iter().min_by(|a, b| {
                    norm(sub(a.position, pose.position))
                        .total_cmp(&norm(sub(b.position, pose.position)))
                })
            })
            .flatten();
        let mut previous = BTreeMap::from([(start, start)]);
        let mut distance = BTreeMap::from([(start, 0usize)]);
        let mut queue = VecDeque::from([start]);
        let mut best = None;
        let mut score = f64::NEG_INFINITY;
        while let Some(key) = queue.pop_front() {
            let steps = distance[&key];
            let position = map.center(key);
            if steps > 0 {
                let unknown = [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ]
                .iter()
                .filter(|d| {
                    let k = std::array::from_fn(|i| key[i] + d[i]);
                    learned::index(k).is_some() && !map.voxels.contains_key(&k)
                })
                .count() as f64;
                let age = map.recent_visits3.get(&key).map_or(1.0, |step| {
                    (map.last_step.saturating_sub(*step) as f64 / 1000.0).min(1.0)
                });
                let separation = assigned
                    .iter()
                    .map(|p| norm(sub(*p, position)))
                    .fold(0.4, f64::min)
                    .min(0.4);
                let value = if let Some(c) = cue {
                    -norm(sub(position, c.position)) * 5.0 - (steps as f64) * 0.005
                } else {
                    unknown * 0.6 + age * 0.8 + separation * 2.0 - (steps as f64).sqrt() * 0.04
                };
                if value > score {
                    score = value;
                    best = Some(key);
                }
            }
            if previous.len() > 6000 {
                break;
            }
            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                for dy in [-1, 0, 1] {
                    let next = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if !walkable.contains(&next) || previous.contains_key(&next) {
                        continue;
                    }
                    previous.insert(next, key);
                    distance.insert(next, steps + 1);
                    queue.push_back(next);
                }
            }
        }
        if let Some(target) = best {
            assigned.push(map.center(target));
            let mut route = vec![target];
            let mut cursor = target;
            while previous[&cursor] != start {
                cursor = previous[&cursor];
                route.push(cursor);
            }
            let waypoint = *route.iter().rev().nth(2).unwrap_or(&route[0]);
            result.push(Goal {
                node_id: id.clone(),
                position: map.center(waypoint),
                reason: if cue.is_some() {
                    "observed-supported 3D acoustic approach"
                } else {
                    "observed-supported 3D frontier"
                },
                hold_until_step: 0,
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separate_storeys_never_become_a_vertical_teleport() {
        let mut map = Mapper::with_model("levels", 0.05, None).unwrap();
        for y in [0, 8] {
            for x in 0..6 {
                map.voxels.insert(
                    [x, y, 0],
                    Voxel {
                        log_odds: -3.0,
                        count: 4,
                    },
                );
                map.voxels.insert(
                    [x, y - 1, 0],
                    Voxel {
                        log_odds: 3.0,
                        count: 4,
                    },
                );
            }
        }
        for (id, y) in [("low", 0), ("high", 8)] {
            map.poses.insert(
                id.into(),
                Pose {
                    position: map.center([0, y, 0]),
                    anchor: map.center([0, y, 0]),
                    sigma: 0.001,
                },
            );
        }
        let targets = goals(&map, &BTreeSet::new());
        assert_eq!(targets.len(), 2);
        for target in targets {
            assert_eq!(
                map.key(target.position)[1],
                if target.node_id == "low" { 0 } else { 8 }
            );
        }
    }
    #[test]
    fn measured_step_routes_change_height_and_free_air_is_not_walkable() {
        let mut map = Mapper::with_model("ramp", 0.05, None).unwrap();
        for x in 0..7 {
            map.voxels.insert(
                [x, x, 0],
                Voxel {
                    log_odds: -3.0,
                    count: 4,
                },
            );
            map.voxels.insert(
                [x, x - 1, 0],
                Voxel {
                    log_odds: 3.0,
                    count: 4,
                },
            );
        }
        for x in 0..7 {
            map.voxels.insert(
                [x, 14, 4],
                Voxel {
                    log_odds: -3.0,
                    count: 4,
                },
            );
        }
        map.poses.insert(
            "node".into(),
            Pose {
                position: map.center([0, 0, 0]),
                anchor: map.center([0, 0, 0]),
                sigma: 0.001,
            },
        );
        let target = goals(&map, &BTreeSet::new());
        assert_eq!(target.len(), 1);
        let key = map.key(target[0].position);
        assert!(key[1] > 0 && key[1] < 7);
        assert_eq!(key[2], 0);
    }
}
