//! Portable CPU 3D inference. Observations only; training and CUDA are not required.
use crate::{Cell, generic, learned};
use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Mutex};

pub struct VolumeRuntime {
    session: Mutex<Session>,
    channels: usize,
}
impl VolumeRuntime {
    pub fn load(directory: &Path, expected_sha256: &str, channels: usize) -> Result<Self> {
        Self::load_named(directory, "volume.onnx", expected_sha256, channels)
    }
    pub fn load_named(
        directory: &Path,
        name: &str,
        expected_sha256: &str,
        channels: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(name, "volume.onnx" | "navigation.onnx"),
            "Invalid bundled model filename"
        );
        ensure!(
            matches!(channels, 2 | 9 | 18 | 74 | 90),
            "Unsupported 3D input channels"
        );
        let file = directory.join(name);
        let bytes = std::fs::read(&file).context("Bundled 3D model missing")?;
        ensure!(bytes.len() <= 64 * 1024 * 1024, "3D model size budget");
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == expected_sha256,
            "3D model hash mismatch"
        );
        let dll = directory.join(if cfg!(windows) {
            "onnxruntime.dll"
        } else {
            "libonnxruntime.so"
        });
        ensure!(dll.is_file(), "Bundled ONNX Runtime missing");
        ort::init_from(dll.to_string_lossy()).commit()?;
        let session = Session::builder()?
            .with_intra_threads(2)?
            .commit_from_file(file)?;
        Ok(Self {
            session: Mutex::new(session),
            channels,
        })
    }
    pub fn apply(
        &self,
        cells: &mut Vec<Cell>,
        bias: f64,
        extra: Option<&[[f32; 88]]>,
    ) -> Result<()> {
        ensure!(
            self.channels == 2 || extra.is_some_and(|e| e.len() == learned::GRID_CELLS),
            "3D model requires sub-voxel range moments"
        );
        if !cells.iter().any(|c| c.observations > 0) {
            return Ok(());
        }
        let grid = generic::evidence(cells);
        let plane = 32 * 24 * 72;
        let mut input = vec![0f32; self.channels * plane];
        let mut positions = vec![None; learned::GRID_CELLS];
        for (i, c) in cells.iter().enumerate() {
            positions[learned::index(c.key).context("Cell out of volume")?] = Some(i);
        }
        for x in 0..32 {
            for y in 0..21 {
                for z in 0..70 {
                    let source = (x * 21 + y) * 70 + z;
                    let target = (x * 24 + y) * 72 + z;
                    input[target] = grid[source][0] as f32;
                    input[plane + target] = grid[source][1] as f32;
                    if let Some(extra) = extra {
                        for i in 0..self.channels - 2 {
                            input[(i + 2) * plane + target] = extra[source][i];
                        }
                    }
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, self.channels, 32, 24, 72], input))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("3D inference lock poisoned"))?;
        let outputs = session.run(ort::inputs!["observations"=>tensor])?;
        let (shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == [1, 1, 32, 24, 72],
            "3D model output shape mismatch"
        );
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "Non-finite 3D prediction"
        );
        for x in 0..32 {
            for y in 0..21 {
                for z in 0..70 {
                    let key = [x as i32 - 16, y as i32 - 3, z as i32 - 6];
                    let i = (x * 21 + y) * 70 + z;
                    let logit = f64::from(logits[(x * 24 + y) * 72 + z]) + bias;
                    let probability = 1.0 / (1.0 + (-logit.clamp(-40.0, 40.0)).exp());
                    if let Some(index) = positions[i] {
                        cells[index].occupancy = probability;
                        cells[index].source = "local_surface_completion";
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
        Ok(())
    }
}
