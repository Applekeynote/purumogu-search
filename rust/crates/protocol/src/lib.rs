//! Observation-only wire contract. Unknown fields (including truth) are rejected.
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

pub const VERSION: &str = "1.0";
pub const MAX_LINE: usize = 1_048_576;
pub const MAX_EVENTS: usize = 2000;
pub const LABELS: [&str; 5] = ["STRAIGHT", "T_BRANCH", "DEAD_END", "REJOIN", "NARROWING"];
pub const SCHEMA: &str = "event-graph-v1-16";
pub const MODES: [&str; 2] = ["CLUSTER", "CHAIN_INSERT"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub node_id: String,
    pub t_ms: u64,
    pub node_time_ms: i64,
    pub kind: Kind,
    #[serde(default)]
    pub peer_id: String,
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub quality: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Kind {
    NodeSpawn,
    NodeDeployed,
    Impact,
    StillStart,
    StillEnd,
    PeerHeard,
    PeerTimeout,
    ObservationGap,
    ParentCmd,
    CoupleStart,
    CoupleEnd,
    ThrustOn,
    ThrustOff,
    GateOpened,
    GateClosed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Packet {
    pub protocol_version: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub run_id: String,
    pub seq_no: u64,
    pub sim_step: u64,
    pub timestamp_ms: u64,
    #[serde(default)]
    pub events: Vec<Event>,
    #[serde(default)]
    pub batch_end: bool,
    #[serde(default)]
    pub action_id: String,
    #[serde(default)]
    pub entry_id: usize,
    #[serde(default)]
    pub formation_mode: String,
    #[serde(default)]
    pub node_count: usize,
    #[serde(default)]
    pub remaining_nodes: usize,
    #[serde(default)]
    pub accepted: bool,
    #[serde(default)]
    pub policy: String,
    #[serde(default)]
    pub policy_seed: u64,
}

impl Packet {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.protocol_version == VERSION,
            "protocol version mismatch"
        );
        ensure!(
            !self.run_id.is_empty() && self.run_id.len() <= 96,
            "invalid run id"
        );
        ensure!(
            matches!(
                self.message_type.as_str(),
                "hello" | "action_ack" | "observation_batch" | "heartbeat" | "run_end"
            ),
            "unknown message type"
        );
        ensure!(self.events.len() <= MAX_EVENTS, "too many events");
        ensure!(
            self.entry_id < 3 && self.node_count <= 10 && self.remaining_nodes <= 100,
            "invalid gate/budget"
        );
        ensure!(
            self.timestamp_ms <= self.sim_step.saturating_mul(20).saturating_add(1000),
            "clock inconsistent"
        );
        if !self.formation_mode.is_empty() {
            ensure!(
                MODES.contains(&self.formation_mode.as_str()),
                "unknown formation"
            );
        }
        for e in &self.events {
            ensure!(
                !e.node_id.is_empty() && e.node_id.len() <= 48 && e.peer_id.len() <= 48,
                "invalid node id"
            );
            ensure!(
                e.value.is_finite() && e.quality.is_finite() && (0.0..=1.0).contains(&e.quality),
                "invalid sensor value"
            );
            ensure!(e.t_ms <= self.timestamp_ms + 100, "future event");
            if e.kind == Kind::Impact {
                ensure!(
                    (0.0..=80.0).contains(&e.value),
                    "impact outside sensor range"
                );
            }
            if e.kind == Kind::PeerHeard {
                ensure!(
                    !e.peer_id.is_empty()
                        && e.peer_id != e.node_id
                        && (-120.0..=0.0).contains(&e.value),
                    "invalid radio observation"
                );
            }
        }
        if self.message_type == "hello" {
            ensure!(
                matches!(
                    self.policy.as_str(),
                    "ghost_flow" | "random" | "fixed" | "passive"
                ),
                "unknown policy"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Action {
    pub action_id: String,
    pub entry_id: usize,
    pub node_count: usize,
    pub formation_mode: String,
    pub delay_steps: u64,
    pub policy: String,
}
impl Action {
    pub fn index(&self) -> usize {
        self.entry_id * 2 + usize::from(self.formation_mode == "CHAIN_INSERT")
    }
}

pub fn parse(line: &[u8]) -> Result<Packet> {
    ensure!(line.len() <= MAX_LINE, "line too long");
    let packet: Packet = serde_json::from_slice(line)?;
    packet.validate()?;
    Ok(packet)
}

pub fn valid_budget(action: &Action, remaining: usize) -> Result<()> {
    if action.entry_id >= 3
        || action.node_count == 0
        || action.node_count > 10
        || action.node_count > remaining
        || !MODES.contains(&action.formation_mode.as_str())
    {
        bail!("infeasible action");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truth_and_unknown_are_rejected() {
        let mut v = serde_json::json!({"protocol_version":"1.0","type":"hello","run_id":"a","seq_no":0,"sim_step":0,"timestamp_ms":0,"policy":"fixed"});
        assert!(parse(v.to_string().as_bytes()).is_ok());
        v["world_position"] = serde_json::json!([1, 2, 3]);
        assert!(parse(v.to_string().as_bytes()).is_err());
    }
    #[test]
    fn oversized_and_negative_rejected() {
        assert!(parse(&vec![b' '; MAX_LINE + 1]).is_err());
        assert!(serde_json::from_str::<Packet>(r#"{"node_count":-1}"#).is_err());
    }
    #[test]
    fn budget_checked() {
        let a = Action {
            action_id: "a".into(),
            entry_id: 0,
            node_count: 5,
            formation_mode: "CLUSTER".into(),
            delay_steps: 0,
            policy: "fixed".into(),
        };
        assert!(valid_budget(&a, 4).is_err());
    }
}
