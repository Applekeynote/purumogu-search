//! Offline inference/export parity probe. Reads observed map evidence, never ground truth.
use anyhow::{Context, Result, ensure};
use rescue_mapping::{Cell, generic, learned::LearnedModel};
use std::{env, fs, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 4 || args.len() == 5,
        "Usage: complete_map MODEL INPUT_MAP OUTPUT_MAP [PROBE_KEYS_JSON]"
    );
    let mut model: LearnedModel = serde_json::from_slice(&fs::read(&args[1])?)?;
    model.load_volume(
        std::path::Path::new(&args[1])
            .parent()
            .context("Model folder missing")?,
    )?;
    ensure!(
        !model.geometry_trees.is_empty() || model.volume_runtime.is_some(),
        "No learned local geometry model"
    );
    let mut map: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut cells = Vec::new();
    for value in map["cells"].as_array().context("No cells")? {
        let observations = value["observations"]
            .as_u64()
            .context("Missing observations")? as u32;
        if observations == 0 {
            continue;
        }
        let key: [i32; 3] = serde_json::from_value(value["key"].clone())?;
        ensure!(
            rescue_mapping::learned::index(key).is_some(),
            "Out-of-bounds cell"
        );
        let sensor_occupancy = value["sensor_occupancy"].as_f64();
        cells.push(Cell {
            key,
            observations,
            sensor_occupancy,
            occupancy: sensor_occupancy.unwrap_or(0.5),
            source: "sensor",
        });
    }
    if args.len() == 5 && model.volume_runtime.is_none() {
        let keys: Vec<[i32; 3]> = serde_json::from_slice(&fs::read(&args[4])?)?;
        let grid = generic::evidence(&cells);
        let offsets = if model.geometry_feature_map == generic::WIDE_FEATURE_MAP {
            generic::wide_offsets()
        } else {
            generic::offsets()
        };
        let probes:Vec<_>=keys.into_iter().map(|key| {let f=generic::features(&grid,key,&offsets);
            serde_json::json!({"key":key,"features":f.to_vec(),"probability":model.geometry_score(&f)})}).collect();
        map["generic_probes"] = serde_json::to_value(probes)?;
    }
    let watch = Instant::now();
    if let Some(runtime) = &model.volume_runtime {
        let extra = if model.volume_channels > 2 {
            let rows: Vec<rescue_mapping::range_moments::MomentRow> =
                serde_json::from_value(map["range_moments"].clone())
                    .context("Missing sub-voxel evidence")?;
            Some(rescue_mapping::range_moments::grid(
                &rows,
                model.volume_channels,
            )?)
        } else {
            None
        };
        runtime.apply(&mut cells, model.volume_logit_bias, extra.as_deref())?;
    } else {
        generic::apply(&mut cells, &model);
    }
    let seconds = watch.elapsed().as_secs_f64();
    map["cells"] = serde_json::to_value(cells)?;
    map["completion_family"] = if model.volume_runtime.is_some() {
        "volume-unet-v1"
    } else {
        "local-learned-v1"
    }
    .into();
    fs::write(&args[3], serde_json::to_vec(&map)?)?;
    println!("{{\"inference_seconds\":{seconds}}}");
    Ok(())
}
