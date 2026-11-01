//! Classification/export parity with sensor-derived proposals only, no truth inputs.
use anyhow::{Context, Result, ensure};
use rescue_mapping::{Proposal, learned::LearnedModel, track_fusion};
use std::{env, fs};
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    ensure!(
        args.len() == 4,
        "Usage: classify_proposals MODEL INPUT_MAP OUTPUT_JSON"
    );
    let model: LearnedModel = serde_json::from_slice(&fs::read(&args[1])?)?;
    model.validate()?;
    let map: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let proposals: Vec<Proposal> =
        serde_json::from_value(map.get("proposals").context("No proposals")?.clone())?;
    let scores: Vec<_> = proposals
        .iter()
        .map(|p| model.track_score(p.features.clone()))
        .collect();
    fs::write(
        &args[3],
        serde_json::to_vec(
            &serde_json::json!({"cues":track_fusion::classify(&model,&proposals),"proposal_scores":scores}),
        )?,
    )?;
    Ok(())
}
