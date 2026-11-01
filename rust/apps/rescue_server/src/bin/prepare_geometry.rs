//! Offline sensor-only input preparation using the actual Mapper integration path.
use anyhow::{Context, Result, ensure};
use rescue_mapping::{Mapper, Packet, learned::LearnedModel, range_moments};
use std::{
    collections::BTreeSet,
    env, fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    sync::Arc,
};
fn save_features(mapper: &Mapper, channels: usize, out: &Path) -> Result<()> {
    let grid = range_moments::grid(&mapper.range_moment_rows(channels), channels)?;
    let mut file = BufWriter::new(fs::File::create(out.join("moments-final.f32"))?);
    for row in grid {
        for value in row.into_iter().take(channels - 2) {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    file.flush().context("Feature export failed")?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        matches!(args.len(), 4..=6),
        "Usage: prepare_geometry MODEL SENSOR_PACKETS OUTPUT_DIRECTORY [SNAPSHOT_STEPS_OR_DASH] [FINAL_STEP]"
    );
    let snapshots: BTreeSet<u64> = args
        .get(4)
        .filter(|value| value.as_str() != "-")
        .map(|value| value.split(',').map(str::parse).collect())
        .transpose()?
        .unwrap_or_default();
    let deadline: Option<u64> = args.get(5).map(|value| value.parse()).transpose()?;
    ensure!(
        deadline.is_none_or(|step| step > 0 && step % 25 == 0),
        "Invalid final step"
    );
    ensure!(
        snapshots.iter().all(|step| *step > 0 && *step % 25 == 0),
        "Snapshot steps must match sensor frames"
    );
    let mut saved = BTreeSet::new();
    let mut model: LearnedModel = serde_json::from_slice(&fs::read(&args[1])?)?;
    ensure!(
        matches!(model.volume_channels, 74 | 90),
        "Fine 74/90-channel extraction profile required"
    );
    model.volume_file.clear();
    model.volume_sha256.clear();
    // Pure feature preparation has no planning runtime or navigation artifact.
    model.navigation_file.clear();
    model.navigation_sha256.clear();
    model.navigation_channels = 0;
    model.navigation_logit_bias = 0.0;
    let out = Path::new(&args[3]);
    fs::create_dir_all(out)?;
    let model = Arc::new(model);
    let mut mapper = None;
    let mut final_packet = None;
    for line in BufReader::new(fs::File::open(&args[2])?).lines() {
        let line = line?;
        let value: serde_json::Value = serde_json::from_str(&line)?;
        if value.get("scans").is_none() {
            continue;
        }
        let mut packet: Packet = serde_json::from_value(value)?;
        if deadline.is_some_and(|step| packet.sim_step > step) {
            break;
        }
        // Candidate extraction is independent research, disabled in this geometry-only profile.
        packet.request_diagnostics = false;
        packet.request_surface = false;
        packet.request_map = snapshots.contains(&packet.sim_step);
        if mapper.is_none() {
            let mut fresh = Mapper::with_model(&packet.run_id, packet.cell_m, Some(model.clone()))?;
            fresh.enable_fine_range_export();
            mapper = Some(fresh);
        }
        let mapper = mapper.as_mut().unwrap();
        let reply = mapper.apply(&packet)?;
        if snapshots.contains(&packet.sim_step) {
            let snapshot = out.join(format!("step-{}", packet.sim_step));
            fs::create_dir_all(&snapshot)?;
            fs::write(
                snapshot.join(format!("map-{}.json", packet.sim_step)),
                serde_json::to_vec(&reply)?,
            )?;
            save_features(mapper, model.volume_channels, &snapshot)?;
            saved.insert(packet.sim_step);
        }
        final_packet = Some(packet);
    }
    let mut mapper = mapper.context("No sensor packets found")?;
    let mut packet = final_packet.context("No sensor packets found")?;
    ensure!(
        deadline.is_none_or(|step| packet.sim_step == step),
        "Final sensor checkpoint missing"
    );
    packet.seq_no = packet.seq_no.checked_add(1).context("Sequence overflow")?;
    packet.scans.clear();
    packet.request_map = true;
    let reply = mapper.apply(&packet)?;
    fs::write(
        out.join(format!("map-{}.json", packet.sim_step)),
        serde_json::to_vec(&reply)?,
    )?;
    save_features(&mapper, model.volume_channels, out)?;
    ensure!(
        saved == snapshots,
        "Requested snapshot step missing from sensor audit"
    );
    Ok(())
}
