use anyhow::{Context, Result, bail, ensure};
use event_core::{EventStore, Sequence, SequenceGuard};
use likelihood::Model;
use protocol::{Action, Packet};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    sync::Arc,
    time::Instant,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
    time::{Duration, timeout},
};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<_> = std::env::args().collect();
    a.windows(2)
        .find(|v| v[0] == name)
        .map_or(default.to_string(), |v| v[1].clone())
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let command = args.get(1).map_or("help", String::as_str);
    match command {
        "train" => evaluation::train_file(
            Path::new(&arg("--events", "datasets/generated/observations.jsonl")),
            Path::new(&arg("--truth", "datasets/generated/truth.jsonl")),
            Path::new(&arg("--output", "models/m-v1/model.json")),
            &arg("--ablation", "full"),
        )?,
        "evaluate" => {
            let m = Model::load(arg("--model", "models/m-v1/model.json"))?;
            let rows = evaluation::load(
                Path::new(&arg("--events", "datasets/generated/comparison.jsonl")),
                Path::new(&arg("--truth", "datasets/generated/comparison-truth.jsonl")),
                &arg("--ablation", "full"),
            )?;
            let outcomes = evaluation::evaluate(&rows, &m, &arg("--split", "test"));
            ensure!(!outcomes.is_empty(), "no episodes in requested split");
            evaluation::report(&outcomes, Path::new(&arg("--output", "reports/evaluation")))?;
            println!("{} evaluated episodes", outcomes.len());
        }
        "replay" => replay(
            &Model::load(arg("--model", "models/m-v1/model.json"))?,
            Path::new(&arg("--events", "datasets/sample.jsonl")),
        )?,
        "serve" => {
            let m = Arc::new(Model::load(arg("--model", "models/m-v1/model.json"))?);
            let port = arg("--port", "47110").parse::<u16>()?;
            let listener = TcpListener::bind(("127.0.0.1", port)).await?;
            let logs = arg("--logs", "experiments/raw");
            fs::create_dir_all(&logs)?;
            println!("READY 127.0.0.1:{port} model={}", m.version);
            let limit = Arc::new(tokio::sync::Semaphore::new(4));
            loop {
                let (socket, _) = listener.accept().await?;
                let Ok(permit) = limit.clone().try_acquire_owned() else {
                    drop(socket);
                    continue;
                };
                let m = m.clone();
                let logs = logs.clone();
                tokio::spawn(async move {
                    let _permit = permit;
                    if let Err(e) = session(socket, m, logs).await {
                        eprintln!("session stopped: {e:#}");
                    }
                });
            }
        }
        _ => println!(
            "ghost_flow_server train|serve|evaluate|replay\nSee README.md for complete commands."
        ),
    }
    Ok(())
}

struct Engine {
    model: Arc<Model>,
    run: String,
    policy: String,
    seq: SequenceGuard,
    store: EventStore,
    p: inference::Belief,
    batch: usize,
    deployed: usize,
    last_step: u64,
    seed: u64,
    pending: Option<Action>,
    ack: bool,
    cached: Vec<Value>,
}
impl Engine {
    fn new(model: Arc<Model>) -> Self {
        Self {
            model,
            run: String::new(),
            policy: String::new(),
            seq: SequenceGuard::default(),
            store: EventStore::default(),
            p: inference::prior(),
            batch: 0,
            deployed: 0,
            last_step: 0,
            seed: 20261027,
            pending: None,
            ack: false,
            cached: Vec::new(),
        }
    }
    fn envelope(&self, kind: &str, p: &Packet) -> Value {
        json!({"protocol_version":protocol::VERSION,"type":kind,"run_id":p.run_id,"seq_no":p.seq_no,"timestamp_ms":p.timestamp_ms,"model_version":self.model.version})
    }
    fn action(&mut self, p: &Packet) -> Value {
        let (a, gain) = planner::choose(
            &self.p,
            &self.model,
            &self.policy,
            self.batch,
            &mut self.seed,
        );
        self.pending = Some(a.clone());
        self.ack = false;
        let mut v = self.envelope("action", p);
        let obj = v.as_object_mut().unwrap();
        for (k, val) in serde_json::to_value(a).unwrap().as_object().unwrap() {
            obj.insert(k.clone(), val.clone());
        }
        obj.insert("expected_information_gain_bits".into(), json!(gain));
        v
    }
    fn process(&mut self, p: &Packet) -> Result<Vec<Value>> {
        match self.seq.accept(p.seq_no) {
            Sequence::Duplicate => return Ok(self.cached.clone()),
            Sequence::Stale => bail!("stale sequence"),
            Sequence::New => {}
        }
        ensure!(
            p.sim_step >= self.last_step && p.sim_step - self.last_step <= 600,
            "simulation clock regressed or jumped by more than 12 seconds"
        );
        self.last_step = p.sim_step;
        let mut out = Vec::new();
        if self.run.is_empty() {
            ensure!(p.message_type == "hello" && p.seq_no == 0, "HELLO required");
            self.run = p.run_id.clone();
            self.policy = p.policy.clone();
            self.seed = p.policy_seed;
            out.push(self.envelope("ready", p));
            out.push(self.action(p));
        } else {
            ensure!(self.run == p.run_id, "run cannot change on a connection");
            match p.message_type.as_str() {
                "hello" => bail!("run already initialized"),
                "action_ack" => {
                    let a = self.pending.as_ref().context("no pending action")?;
                    ensure!(
                        p.action_id == a.action_id && p.accepted,
                        "action rejected/mismatched"
                    );
                    if !self.ack {
                        protocol::valid_budget(a, 20 - self.deployed)?;
                        self.deployed += a.node_count;
                        self.ack = true;
                    }
                    out.push(self.envelope("ack", p));
                }
                "observation_batch" => {
                    ensure!(self.ack, "observation before action ACK");
                    let a = self.pending.as_ref().context("no pending action")?;
                    ensure!(
                        p.action_id == a.action_id
                            && p.entry_id == a.entry_id
                            && p.formation_mode == a.formation_mode
                            && p.node_count == a.node_count,
                        "observation/action mismatch"
                    );
                    self.store.append(p)?;
                    if p.batch_end {
                        let started = Instant::now();
                        let s = graph::summarize(&self.store.finish(), self.deployed - 5, "full");
                        let (z, d) = self.model.encode(&s);
                        if s.observed > 0 {
                            self.p = inference::update(&self.p, z, a.index(), &self.model);
                        }
                        let (abstain, reason) = inference::abstain(&self.p, &s, d, &self.model);
                        let mut v = self.envelope("inference", p);
                        v["beliefs"] = json!(
                            self.model
                                .labels
                                .iter()
                                .zip(self.p)
                                .map(|(label, p)| json!({"topology_id":label,"probability":p}))
                                .collect::<Vec<_>>()
                        );
                        v["abstain"] = json!(abstain);
                        v["reason"] = json!(reason);
                        v["entropy_bits"] = json!(inference::entropy(&self.p));
                        v["observed_nodes"] = json!(s.observed);
                        v["sequence_gaps"] = json!(self.seq.missing);
                        v["inference_us"] = json!(started.elapsed().as_micros());
                        v["batch"] = json!(self.batch);
                        out.push(v);
                        self.batch += 1;
                        if self.batch < 3 {
                            out.push(self.action(p));
                        } else {
                            self.ack = false;
                            self.pending = None;
                            out.push(self.envelope("result", p));
                        }
                    } else {
                        out.push(self.envelope("ack", p));
                    }
                }
                "heartbeat" => out.push(self.envelope("heartbeat", p)),
                "run_end" => out.push(self.envelope("stopped", p)),
                _ => bail!("unsupported message"),
            }
        }
        self.cached = out.clone();
        Ok(out)
    }
}

async fn line(reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>) -> Result<Option<Vec<u8>>> {
    let mut out = Vec::new();
    loop {
        let byte = reader.read_u8().await;
        match byte {
            Ok(b'\n') => return Ok(Some(out)),
            Ok(b) => {
                ensure!(out.len() < protocol::MAX_LINE, "line limit exceeded");
                out.push(b);
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                ensure!(out.is_empty(), "truncated frame");
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
    }
}
async fn session(socket: TcpStream, m: Arc<Model>, logs: String) -> Result<()> {
    socket.set_nodelay(true)?;
    let (read, mut write) = socket.into_split();
    let mut reader = BufReader::new(read);
    let mut engine = Engine::new(m);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let path = Path::new(&logs).join(format!("session-{stamp}.jsonl"));
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(128);
    let audit = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
        let mut bytes = 0usize;
        while let Some(record) = rx.blocking_recv() {
            bytes += record.len();
            ensure!(bytes <= 64 * 1024 * 1024, "audit size limit exceeded");
            file.write_all(record.as_bytes())?;
            file.flush()?;
        }
        Ok(())
    });
    let result:Result<()>=async{loop{let frame=timeout(Duration::from_secs(15),line(&mut reader)).await.context("peer timeout; SAFE_STOP")??;let Some(frame)=frame else{break};
        let response=match protocol::parse(&frame).and_then(|p|engine.process(&p)){Ok(out)=>out,Err(e)=>{let error=json!({"protocol_version":protocol::VERSION,"type":"error","run_id":engine.run,"reason":e.to_string(),"safe_stop":true});write.write_all(format!("{error}\n").as_bytes()).await?;bail!("{e}");}};
        tx.send(format!("{}\n",json!({"direction":"in","message":serde_json::from_slice::<Value>(&frame)?}))).await.context("audit unavailable")?;
        for v in response{tx.send(format!("{}\n",json!({"direction":"out","message":v}))).await.context("audit unavailable")?;write.write_all(format!("{v}\n").as_bytes()).await?;}
        if audit.is_finished(){bail!("audit writer stopped; SAFE_STOP");}
    }Ok(())}.await;
    drop(tx);
    audit.await??;
    result
}
fn replay(m: &Model, path: &Path) -> Result<()> {
    use std::io::BufRead;
    let mut engines: HashMap<String, Engine> = HashMap::new();
    let mut stores: HashMap<String, (EventStore, inference::Belief, usize)> = HashMap::new();
    let mut count = 0;
    for line in std::io::BufReader::new(std::fs::File::open(path)?).lines() {
        let line = line?;
        let value: Value = serde_json::from_str(&line)?;
        if value.get("direction").is_some() {
            if value["direction"] != "in" {
                continue;
            }
            let p = protocol::parse(&serde_json::to_vec(&value["message"])?)?;
            let e = engines
                .entry(p.run_id.clone())
                .or_insert_with(|| Engine::new(Arc::new(m.clone())));
            for decision in e.process(&p)? {
                if decision["type"] == "inference" {
                    println!("{decision}");
                    count += 1;
                }
            }
        } else {
            let p = protocol::parse(line.as_bytes())?;
            if p.message_type != "observation_batch" {
                continue;
            }
            let (s, b, n) = stores.entry(p.run_id.clone()).or_insert((
                EventStore::default(),
                inference::prior(),
                0,
            ));
            s.append(&p)?;
            if p.batch_end {
                let summary = graph::summarize(&s.finish(), *n, "full");
                let (z, d) = m.encode(&summary);
                if summary.observed > 0 {
                    *b = inference::update(
                        b,
                        z,
                        p.entry_id * 2 + usize::from(p.formation_mode == "CHAIN_INSERT"),
                        m,
                    );
                }
                *n += 5;
                println!(
                    "{}",
                    json!({"run_id":p.run_id,"probabilities":b,"abstain":inference::abstain(b,&summary,d,m).0})
                );
                count += 1;
            }
        }
    }
    ensure!(count > 0, "no completed batches");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn engine() -> Engine {
        let model: Model =
            serde_json::from_str(include_str!("../../../../models/m-v1/model.json")).unwrap();
        model.validate().unwrap();
        Engine::new(Arc::new(model))
    }
    fn packet(kind: &str, seq: u64) -> Packet {
        protocol::parse(json!({"protocol_version":"1.0","type":kind,"run_id":"unit-run","seq_no":seq,"sim_step":0,"timestamp_ms":0,"policy":"fixed","policy_seed":42,"remaining_nodes":20}).to_string().as_bytes()).unwrap()
    }
    #[test]
    fn handshake_and_duplicate_batch_are_idempotent() {
        let mut e = engine();
        let ready = e.process(&packet("hello", 0)).unwrap();
        assert_eq!(ready[0]["type"], "ready");
        let mut ack = packet("action_ack", 1);
        ack.action_id = "a-000".into();
        ack.accepted = true;
        e.process(&ack).unwrap();
        assert_eq!(e.deployed, 5);
        let mut observation = packet("observation_batch", 2);
        observation.action_id = "a-000".into();
        observation.entry_id = 0;
        observation.formation_mode = "CLUSTER".into();
        observation.node_count = 5;
        observation.batch_end = true;
        observation.events = vec![protocol::Event {
            node_id: "N00".into(),
            t_ms: 0,
            node_time_ms: 0,
            kind: protocol::Kind::NodeSpawn,
            peer_id: String::new(),
            value: 0.,
            quality: 1.,
        }];
        let first = e.process(&observation).unwrap();
        let belief = e.p;
        assert_eq!(e.process(&observation).unwrap(), first);
        assert_eq!(e.batch, 1);
        assert_eq!(e.deployed, 5);
        assert_eq!(e.p, belief);
    }
    #[test]
    fn observation_without_actuator_ack_is_rejected() {
        let mut e = engine();
        e.process(&packet("hello", 0)).unwrap();
        assert!(e.process(&packet("observation_batch", 1)).is_err());
    }
    #[test]
    fn simulation_clock_jump_is_rejected() {
        let mut e = engine();
        e.process(&packet("hello", 0)).unwrap();
        let mut p = packet("heartbeat", 1);
        p.sim_step = 100_000;
        p.timestamp_ms = 2_000_000;
        assert!(e.process(&p).is_err());
    }
    #[test]
    fn changing_run_or_negative_ack_stops_session() {
        let mut e = engine();
        e.process(&packet("hello", 0)).unwrap();
        let mut ack = packet("action_ack", 1);
        ack.action_id = "a-000".into();
        ack.accepted = false;
        assert!(e.process(&ack).is_err());
        assert_eq!(e.deployed, 0);
        let mut other = packet("heartbeat", 2);
        other.run_id = "different".into();
        assert!(e.process(&other).is_err());
    }
}
