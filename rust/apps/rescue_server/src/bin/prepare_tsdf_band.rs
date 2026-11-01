//! Sensor-only TSDF suffix extraction, sharing registration and integration with Mapper.
use anyhow::{Context, Result, ensure};
use rescue_mapping::{Packet, learned::LearnedModel, range_moments, registration::AxisConsensus};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

fn save(map: &range_moments::RangeMoments, out: &Path) -> Result<()> {
    fs::create_dir_all(out)?;
    let grid = range_moments::grid(&map.rows_for(90), 90)?;
    let mut file = BufWriter::new(fs::File::create(out.join("tsdf.f32"))?);
    for row in grid {
        for value in &row[72..88] {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 6,
        "Usage: prepare_tsdf_band MODEL SENSOR_PACKETS OUTPUT_DIRECTORY SNAPSHOT_STEPS_OR_DASH FINAL_STEP"
    );
    let model: LearnedModel = serde_json::from_slice(&fs::read(&args[1])?)?;
    ensure!(
        model.volume_channels == 90 && model.range_tsdf_v1 && model.plane_registration_v1,
        "Registered TSDF profile required"
    );
    let deadline: u64 = args[5].parse()?;
    ensure!(
        deadline > 0 && deadline.is_multiple_of(25),
        "Invalid final step"
    );
    let snapshots: BTreeSet<u64> = if args[4] == "-" {
        BTreeSet::new()
    } else {
        args[4]
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()?
    };
    ensure!(
        snapshots
            .iter()
            .all(|s| *s > 0 && *s < deadline && s.is_multiple_of(25)),
        "Invalid snapshot step"
    );
    let out = Path::new(&args[3]);
    let mut registration = AxisConsensus::default();
    let mut map = range_moments::RangeMoments::default();
    let mut saved = BTreeSet::new();
    let mut last: Option<(String, u64, u64)> = None;
    for line in BufReader::new(fs::File::open(&args[2])?).lines() {
        let value: serde_json::Value = serde_json::from_str(&line?)?;
        if value.get("scans").is_none() {
            continue;
        }
        let packet: Packet = serde_json::from_value(value)?;
        if packet.sim_step > deadline {
            break;
        }
        ensure!((packet.cell_m - 0.05).abs() < 1e-9, "Expected 5cm grid");
        if let Some((run, seq, step)) = &last {
            ensure!(
                packet.run_id == *run && packet.seq_no > *seq && packet.sim_step >= *step,
                "Invalid packet sequence"
            );
        }
        let origins = registration.packet(&packet.scans, model.axis_plane_registration_v2);
        for scan in &packet.scans {
            if scan.pose_sigma_m > packet.cell_m * 3. {
                continue;
            }
            let origin = *origins
                .get(&scan.node_id)
                .context("Registered pose missing")?;
            let reliability = 1. / (1. + (scan.pose_sigma_m / packet.cell_m).powi(2));
            map.integrate_surface_distance(origin, &scan.rays, reliability);
        }
        if snapshots.contains(&packet.sim_step) {
            save(&map, &out.join(format!("step-{}", packet.sim_step)))?;
            saved.insert(packet.sim_step);
        }
        last = Some((packet.run_id, packet.seq_no, packet.sim_step));
    }
    ensure!(
        last.context("No sensor packets")?.2 == deadline,
        "Final sensor checkpoint missing"
    );
    ensure!(saved == snapshots, "Requested snapshot missing");
    save(&map, out)
}
