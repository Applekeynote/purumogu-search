//! Supported 3D routes: levels are inferred from range evidence, never scene IDs.
use super::*;

/// Five support columns cover the 7 cm body plus a grid-scale edge margin.
/// This is route risk control, never a source of occupied map evidence.
fn footprint_supported(key: [i32; 3], solid: impl Fn([i32; 3]) -> bool) -> bool {
    (1..=2).any(|down| solid([key[0], key[1] - down, key[2]]))
        && [(-1, 0), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .all(|(dx, dz)| (0..=2).any(|down| solid([key[0] + dx, key[1] - down, key[2] + dz])))
}

pub(super) fn goals(map: &Mapper, verification: &BTreeSet<String>) -> Vec<Goal> {
    let learned_navigation = map.model.as_ref().is_some_and(|m| m.learned_navigation_v1);
    let scout_mode = map
        .model
        .as_ref()
        .is_some_and(|m| m.scout_inferred_routes_v1);
    let footprint_safe = map
        .model
        .as_ref()
        .is_some_and(|m| m.footprint_safe_routes_v1);
    let scouts = map.inferred_scouts();
    let failed_positions: Vec<_> = if scout_mode {
        map.poses
            .values()
            .filter(|p| (251..1500).contains(&map.last_step.saturating_sub(p.last_seen)))
            .map(|p| p.position)
            .collect()
    } else {
        vec![]
    };
    let probability = |key| {
        learned::index(key)
            .and_then(|i| map.planning_probability.get(i))
            .copied()
            .unwrap_or(0.5)
    };
    let mut walkable: BTreeSet<_> = map
        .voxels
        .iter()
        .filter_map(|(key, v)| {
            (v.log_odds < -0.5
                && (1..=2).any(|down| {
                    let mut below = *key;
                    below[1] -= down;
                    map.voxels.get(&below).is_some_and(|v| v.log_odds > 0.8)
                })
                && (!footprint_safe
                    || footprint_supported(*key, |support| {
                        map.voxels.get(&support).is_some_and(|v| v.log_odds > 0.8)
                    })))
            .then_some(*key)
        })
        .collect();
    let observed_walkable = walkable.clone();
    if learned_navigation && map.planning_probability.len() == learned::GRID_CELLS {
        for x in -16..16 {
            for y in -3..18 {
                for z in -6..64 {
                    let key = [x, y, z];
                    if probability(key) < 0.1
                        && !map.voxels.get(&key).is_some_and(|v| v.log_odds > 1.5)
                        && (1..=2).any(|down| probability([x, y - down, z]) > 0.9)
                        && (!footprint_safe
                            || footprint_supported(key, |support| probability(support) > 0.9))
                        && (observed_walkable.contains(&key)
                            || !failed_positions.iter().any(|p| {
                                sub(map.center(key), *p).iter().map(|v| v * v).sum::<f64>() < 0.0225
                            }))
                    {
                        walkable.insert(key);
                    }
                }
            }
        }
    }
    // Discover populated support levels from the inferred spatial map, not scene heights.
    let mut heights = BTreeMap::new();
    for key in &walkable {
        let flat_support = (-1..=1)
            .flat_map(|dx| (-1..=1).map(move |dz| [key[0] + dx, key[1], key[2] + dz]))
            .filter(|neighbor| walkable.contains(neighbor))
            .count();
        if flat_support >= 7 {
            *heights.entry(key[1]).or_insert(0usize) += 1;
        }
    }
    let mut peaks: Vec<_> = heights.into_iter().collect();
    peaks.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut levels = vec![];
    if learned_navigation {
        for (height, count) in peaks {
            if count >= 16 && levels.iter().all(|h: &i32| (h - height).abs() >= 3) {
                levels.push(height);
            }
        }
    }
    levels.sort();
    let band = |y: i32| {
        levels
            .iter()
            .enumerate()
            .min_by_key(|(_, h)| (*h - y).abs())
            .map(|(i, _)| i)
    };
    let mut population = vec![0isize; levels.len()];
    for pose in map
        .poses
        .values()
        .filter(|p| !learned_navigation || map.last_step.saturating_sub(p.last_seen) <= 250)
    {
        if let Some(i) = band(map.key(pose.position)[1]) {
            population[i] += 1;
        }
    }
    let mut result = vec![];
    let mut assigned: Vec<V3> = vec![];
    for (id, pose) in &map.poses {
        if learned_navigation && map.last_step.saturating_sub(pose.last_seen) > 250 {
            continue;
        }
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
        let allowed = if scout_mode && !scouts.contains(id) {
            &observed_walkable
        } else {
            &walkable
        };
        let Some(start) = allowed
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
            if scout_mode {
                result.push(Goal {
                    node_id: id.clone(),
                    position: pose.position,
                    reason: "no supported route; local sensing",
                    hold_until_step: map.last_step.saturating_add(50),
                });
            }
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
        let current_band = band(origin[1]);
        let seek_level = learned_navigation && cue.is_none();
        let mut best_level_priority = 0isize;
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
                let level_priority = if seek_level {
                    current_band.zip(band(key[1])).map_or(0, |(from, to)| {
                        (population[from] - population[to] - 1).max(0)
                    })
                } else {
                    0
                };
                if level_priority > best_level_priority
                    || level_priority == best_level_priority && value > score
                {
                    score = value;
                    best = Some(key);
                    best_level_priority = level_priority;
                }
            }
            if previous.len() > 6000 {
                break;
            }
            for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                for dy in [-1, 0, 1] {
                    let next = [key[0] + dx, key[1] + dy, key[2] + dz];
                    if !allowed.contains(&next) || previous.contains_key(&next) {
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
            if let (Some(from), Some(to)) = (current_band, band(target[1]))
                && from != to
            {
                population[from] -= 1;
                population[to] += 1;
            }
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
                reason: if cue.is_some() && learned_navigation {
                    "inferred-support acoustic approach; unconfirmed"
                } else if cue.is_some() {
                    "observed-supported 3D acoustic approach"
                } else if learned_navigation {
                    "inferred-support route to underexplored space; unconfirmed"
                } else {
                    "observed-supported 3D frontier"
                },
                hold_until_step: 0,
            });
        } else if scout_mode {
            result.push(Goal {
                node_id: id.clone(),
                position: pose.position,
                reason: "isolated support; local sensing",
                hold_until_step: map.last_step.saturating_add(50),
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_supported_center_does_not_make_a_floor_edge_safe() {
        let key = [0, 0, 0];
        assert!(!footprint_supported(key, |k| k == [0, -1, 0]));
        assert!(!footprint_supported(key, |k| k[1] == -1 && k[0] >= 0));
        assert!(footprint_supported(key, |k| k[1] == -1));
    }
    #[test]
    fn footprint_support_accepts_a_shallow_slope_but_not_a_drop() {
        assert!(footprint_supported([0, 1, 0], |k| k[1] == k[0]));
        assert!(!footprint_supported([0, 1, 0], |k| {
            k[1] == if k[0] > 0 { -3 } else { 0 }
        }));
    }
    #[test]
    fn supported_navigation_erodes_inferred_edges_without_painting_evidence() {
        let mut map = inferred_levels(false);
        let mut model = (**map.model.as_ref().unwrap()).clone();
        model.footprint_safe_routes_v1 = true;
        map.model = Some(Arc::new(model));
        let targets = goals(&map, &BTreeSet::new());
        assert_eq!(targets.len(), 2);
        for goal in targets {
            assert!(footprint_supported(map.key(goal.position), |key| {
                learned::index(key).is_some_and(|i| map.planning_probability[i] > 0.9)
            }));
        }
        assert!(map.voxels.is_empty());
    }
    fn inferred_levels(ramp: bool) -> Mapper {
        let model=serde_json::from_value(serde_json::json!({"version":"rescue-trained-v1","cell_m":0.05,"acoustic_weights":[3.,0.,0.,0.,0.],"occupancy_weights":[0.,1.,0.],"prototypes":[],"local_surface_completion":true,"learned_navigation_v1":true,"train_maps":["train"],"valid_maps":["valid"],"data_sha256":"fixture"})).unwrap();
        let mut map = Mapper::with_model("learned-levels", 0.05, Some(Arc::new(model))).unwrap();
        map.last_step = 1000;
        map.planning_probability = vec![0.5; learned::GRID_CELLS];
        let mut keys = vec![];
        for (xs, y) in [(-6..1, 0), (8..15, 8)] {
            for x in xs {
                for z in 0..6 {
                    keys.push([x, y, z]);
                }
            }
        }
        if ramp {
            for x in 0..9 {
                keys.push([x, x, 2]);
            }
        }
        for key in keys {
            map.planning_probability[learned::index(key).unwrap()] = 0.02;
            let mut below = key;
            below[1] -= 1;
            map.planning_probability[learned::index(below).unwrap()] = 0.98;
        }
        for (id, key, last_seen) in [
            ("N0", [10, 8, 2], 1000),
            ("N1", [12, 8, 3], 1000),
            ("stale", [-3, 0, 2], 0),
        ] {
            let position = map.center(key);
            map.poses.insert(
                id.into(),
                Pose {
                    position,
                    anchor: position,
                    sigma: 0.01,
                    last_seen,
                },
            );
        }
        map
    }
    #[test]
    fn inferred_ramp_sends_live_nodes_toward_an_underexplored_level() {
        let map = inferred_levels(true);
        let targets = goals(&map, &BTreeSet::new());
        assert_eq!(
            targets.len(),
            2,
            "A node with stale communication is not an active explorer"
        );
        let first = targets.iter().find(|g| g.node_id == "N0").unwrap();
        assert!(first.position[1] < map.poses["N0"].position[1]);
        assert!(
            norm(sub(first.position, map.poses["N0"].position)) < 0.2,
            "Use nearby route waypoints, not a jump between storeys"
        );
        assert!(first.reason.contains("inferred"));
    }
    #[test]
    fn inferred_disconnected_levels_do_not_create_a_vertical_shortcut() {
        let map = inferred_levels(false);
        let targets = goals(&map, &BTreeSet::new());
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(|g| map.key(g.position)[1] == 8));
    }
    #[test]
    fn only_reserved_scouts_take_an_unobserved_inferred_route() {
        let mut map = inferred_levels(true);
        let mut model = (**map.model.as_ref().unwrap()).clone();
        model.scout_inferred_routes_v1 = true;
        map.model = Some(Arc::new(model));
        let scouts = map.inferred_scouts();
        assert_eq!(scouts.len(), 1);
        let targets = goals(&map, &BTreeSet::new());
        let moving: Vec<_> = targets.iter().filter(|g| g.hold_until_step == 0).collect();
        assert_eq!(moving.len(), 1);
        assert!(scouts.contains(&moving[0].node_id));
        assert!(
            targets
                .iter()
                .any(|g| g.hold_until_step > map.last_step && g.reason.contains("local sensing"))
        );
        assert!(targets.iter().all(|g| g.node_id != "stale"));
        assert!(
            map.voxels.is_empty(),
            "Inferred routes do not become measured voxels"
        );
    }
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
                    last_seen: 0,
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
                last_seen: 0,
            },
        );
        let target = goals(&map, &BTreeSet::new());
        assert_eq!(target.len(), 1);
        let key = map.key(target[0].position);
        assert!(key[1] > 0 && key[1] < 7);
        assert_eq!(key[2], 0);
    }
}
