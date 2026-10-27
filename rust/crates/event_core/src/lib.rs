use anyhow::{Result, ensure};
use protocol::{Event, Kind, Packet};
use std::collections::{BTreeSet, HashSet};

/// Lifetime bounded to one batch; identities are used for relations, never as features.
#[derive(Default)]
pub struct EventStore {
    pub events: Vec<Event>,
    pub gaps: u64,
    seen_nodes: HashSet<String>,
}
impl EventStore {
    pub fn append(&mut self, p: &Packet) -> Result<()> {
        ensure!(
            self.events.len() + p.events.len() <= 60_000,
            "batch event limit exceeded"
        );
        for e in &p.events {
            if e.kind == Kind::NodeSpawn {
                ensure!(
                    self.seen_nodes.insert(e.node_id.clone()),
                    "duplicate node definition"
                );
            }
        }
        self.events.extend(p.events.iter().cloned());
        Ok(())
    }
    pub fn finish(&mut self) -> Vec<Event> {
        self.events.sort_by(|a, b| {
            (a.t_ms, &a.node_id, format!("{:?}", a.kind)).cmp(&(
                b.t_ms,
                &b.node_id,
                format!("{:?}", b.kind),
            ))
        });
        std::mem::take(&mut self.events)
    }
}

/// Duplicate packets reuse the exact cached response; old sequence numbers never execute.
#[derive(Default)]
pub struct SequenceGuard {
    last: Option<u64>,
    pub missing: u64,
    pub duplicates: u64,
}
#[derive(Debug, PartialEq)]
pub enum Sequence {
    New,
    Duplicate,
    Stale,
}
impl SequenceGuard {
    pub fn accept(&mut self, seq: u64) -> Sequence {
        if let Some(last) = self.last {
            if seq == last {
                self.duplicates += 1;
                return Sequence::Duplicate;
            }
            if seq < last {
                return Sequence::Stale;
            }
            self.missing += seq - last - 1;
        }
        self.last = Some(seq);
        Sequence::New
    }
}
pub fn nodes(events: &[Event]) -> BTreeSet<&str> {
    events
        .iter()
        .filter(|e| e.node_id != "PARENT" && !e.node_id.starts_with("GATE"))
        .map(|e| e.node_id.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn idempotent_seq_and_gap() {
        let mut g = SequenceGuard::default();
        assert_eq!(g.accept(0), Sequence::New);
        assert_eq!(g.accept(0), Sequence::Duplicate);
        assert_eq!(g.accept(3), Sequence::New);
        assert_eq!(g.missing, 2);
        assert_eq!(g.accept(2), Sequence::Stale);
    }
}
