use anyhow::Result;
use rescue_mapping::learned::LearnedModel;
use rescue_mapping::{MAX_LINE, MAX_REPLY, Mapper, Packet};
use std::sync::Arc;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    time::{Duration, timeout},
};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let port = args
        .windows(2)
        .find(|w| w[0] == "--port")
        .map(|w| w[1].parse::<u16>())
        .transpose()?
        .unwrap_or(47120);
    let listener = TcpListener::bind(("127.0.0.1", port)).await?;
    let model = if let Some(path) = args.windows(2).find(|w| w[0] == "--model").map(|w| &w[1]) {
        let m: LearnedModel = serde_json::from_slice(&std::fs::read(path)?)?;
        m.validate()?;
        Some(Arc::new(m))
    } else {
        None
    };
    eprintln!("Rescue mapping protocol 2.0: 127.0.0.1:{port}");
    let permits = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    loop {
        let (socket, _) = listener.accept().await?;
        let Ok(permit) = permits.clone().try_acquire_owned() else {
            continue;
        };
        let model = model.clone();
        tokio::spawn(async move {
            let _permit = permit;
            if let Err(e) = session(socket, model).await {
                eprintln!("session ended: {e}");
            }
        });
    }
}
async fn session(socket: TcpStream, model: Option<Arc<LearnedModel>>) -> Result<()> {
    socket.set_nodelay(true)?;
    let (read, mut write) = socket.into_split();
    let mut read = BufReader::new(read);
    let mut mapper = None;
    let mut total = 0usize;
    loop {
        let mut frame = Vec::new();
        loop {
            let available = timeout(Duration::from_secs(30), read.fill_buf()).await??;
            if available.is_empty() {
                return Ok(());
            }
            let count = available
                .iter()
                .position(|b| *b == b'\n')
                .map_or(available.len(), |i| i + 1);
            anyhow::ensure!(frame.len() + count <= MAX_LINE, "frame too large");
            frame.extend_from_slice(&available[..count]);
            read.consume(count);
            if frame.last() == Some(&b'\n') {
                break;
            }
        }
        total += frame.len();
        anyhow::ensure!(total < 128 * 1024 * 1024, "session input budget exceeded");
        let response = (|| -> Result<_> {
            let p: Packet = serde_json::from_slice(&frame)?;
            if mapper.is_none() {
                mapper = Some(Mapper::with_model(&p.run_id, p.cell_m, model.clone())?);
            }
            Ok(serde_json::to_vec(
                &mapper.as_mut().expect("mapper").apply(&p)?,
            )?)
        })();
        match response {
            Ok(mut bytes) => {
                anyhow::ensure!(bytes.len() < MAX_REPLY, "response too large");
                bytes.push(b'\n');
                write.write_all(&bytes).await?;
            }
            Err(e) => {
                let bytes = serde_json::to_vec(
                    &serde_json::json!({"protocol_version":"2.0","type":"error","safe_stop":true,"reason":e.to_string()}),
                )?;
                write.write_all(&bytes).await?;
                write.write_all(b"\n").await?;
                return Ok(());
            }
        }
    }
}
