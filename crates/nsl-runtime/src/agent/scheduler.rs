//! M56 v1 single-threaded logical-time scheduler.
//!
//! Spec §3.1; reference implementation per Q1 brainstorming. v1 fires
//! agents in registration order (caller must register in APG topological
//! order). Outputs written at logical time T are visible at T+1, encoded
//! in `PortMailbox::expected_read_time = stamped_time + 1`.

use std::collections::HashMap;

use crate::agent::mailbox::{PortMailbox, PortMessage};

#[derive(Debug, PartialEq, Eq)]
pub enum StepOutcome {
    Advanced,
    Idle,
}

pub type AgentId = usize;

/// A logical mailbox key. Agents address ports by `(agent_id, port_name)`.
type MailboxKey = (AgentId, String);

/// One mailbox per `(agent_id, port_name)` pair.
pub type MailboxMap = HashMap<MailboxKey, PortMailbox>;

/// Borrowed view passed to each agent's fire closure. The closure should
/// read/write only its own ports via `read_in`/`write_out`; the raw
/// `mailboxes` field is exposed because v2 extension points (e.g., the
/// reactor scheduler's lock-free dispatch) may need direct access. v1
/// closures are author-trusted not to reach into other agents' mailboxes.
pub struct AgentPorts<'a> {
    pub mailboxes: &'a mut MailboxMap,
    pub agent_id: AgentId,
    pub current_time: u64,
}

impl AgentPorts<'_> {
    /// Write `msg` to this agent's output port `port`, stamped at T+1.
    pub fn write_out(&mut self, port: &str, msg: PortMessage) {
        let t = self.current_time + 1;
        let key = (self.agent_id, port.to_string());
        self.mailboxes.entry(key).or_default().write(msg, t);
    }

    /// Read and consume this agent's input port `port`'s message at the
    /// current logical time, if present.
    pub fn read_in(&mut self, port: &str) -> Option<PortMessage> {
        let key = (self.agent_id, port.to_string());
        self.mailboxes.get_mut(&key).and_then(|mb| mb.read())
    }
}

type FireFn = Box<dyn FnMut(&mut AgentPorts)>;

pub struct ReactorScheduler {
    fire_fns: Vec<FireFn>,
    mailboxes: MailboxMap,
    /// Connections: (output: (agent, port_name)) -> (input: (agent, port_name)).
    connections: Vec<(MailboxKey, MailboxKey)>,
    logical_time: u64,
}

impl ReactorScheduler {
    pub fn new() -> Self {
        Self {
            fire_fns: Vec::new(),
            mailboxes: HashMap::new(),
            connections: Vec::new(),
            logical_time: 0,
        }
    }

    /// Register an agent with its fire closure. Returns the agent id.
    /// **The caller must register in topological (DAG) order** — the
    /// scheduler fires in registration order within a single step.
    pub fn register_agent<F: FnMut(&mut AgentPorts) + 'static>(&mut self, fire: F) -> AgentId {
        self.fire_fns.push(Box::new(fire));
        self.fire_fns.len() - 1
    }

    /// Connect agent `from`'s output port to agent `to`'s input port.
    /// After each `step()`, written values propagate from output mailbox
    /// to the downstream input mailbox at the next logical time.
    pub fn connect(&mut self, from: (AgentId, &str), to: (AgentId, &str)) {
        self.connections.push((
            (from.0, from.1.to_string()),
            (to.0, to.1.to_string()),
        ));
    }

    pub fn logical_time(&self) -> u64 {
        self.logical_time
    }

    /// Run one logical-time step. Fires every registered agent in topo
    /// order, then propagates outputs along configured connections, then
    /// advances `logical_time` by 1.
    pub fn step(&mut self) -> StepOutcome {
        // Fire each agent in registration order. Each fire closure may
        // read its inputs (current time) and write its outputs (T+1).
        for id in 0..self.fire_fns.len() {
            // Construct a borrowed AgentPorts view; the closure runs against it.
            let mut ports = AgentPorts {
                mailboxes: &mut self.mailboxes,
                agent_id: id,
                current_time: self.logical_time,
            };
            (self.fire_fns[id])(&mut ports);
        }

        // Propagate outputs along connections.
        // v2-revisit: per-step clone of connections avoids a double-borrow of `self`;
        // acceptable for v1 small pipeline graphs (<20 agents).
        let connections = self.connections.clone();
        for (src, dst) in connections {
            // Move the slot from src to dst, preserving the stamped time.
            let stamped = self.mailboxes.get_mut(&src).and_then(|mb| {
                let t = mb.stamped_time();
                mb.read().map(|msg| (msg, t))
            });
            if let Some((msg, t)) = stamped {
                self.mailboxes.entry(dst).or_default().write(msg, t);
            }
        }

        self.logical_time += 1;
        StepOutcome::Advanced
    }
}

impl Default for ReactorScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::mailbox::{PortMessage, StructPayload};

    #[test]
    fn scheduler_advances_time_after_step() {
        let mut sched = ReactorScheduler::new();
        sched.register_agent(|_ports| {}); // no-op agent
        assert_eq!(sched.logical_time(), 0);
        let outcome = sched.step();
        assert_eq!(outcome, StepOutcome::Advanced);
        assert_eq!(sched.logical_time(), 1);
        sched.step();
        assert_eq!(sched.logical_time(), 2);
    }

    #[test]
    fn scheduler_propagates_along_connection() {
        // A produces a struct payload at T+1; B reads it at T+1 (after one step).
        let mut sched = ReactorScheduler::new();
        let a = sched.register_agent(|ports| {
            // A emits exactly one payload per step.
            ports.write_out(
                "out",
                PortMessage::Struct(Box::new(StructPayload::new(vec![1, 2, 3]))),
            );
        });
        let b_seen = std::sync::Arc::new(std::sync::Mutex::new(false));
        let b_seen_clone = b_seen.clone();
        let _b = sched.register_agent(move |ports| {
            if let Some(PortMessage::Struct(payload)) = ports.read_in("in") {
                if payload.as_bytes() == &[1u8, 2, 3] {
                    *b_seen_clone.lock().unwrap() = true;
                }
            }
        });
        sched.connect((a, "out"), (1, "in"));
        // Step 1: A emits to its `out` mailbox at T=1; propagation moves it
        // to B's `in` mailbox; B fires at T=0 and sees nothing yet (B's read
        // happens within this step, before propagation).
        sched.step();
        assert!(
            !*b_seen.lock().unwrap(),
            "B should not see A's output in the same step it was produced"
        );
        // Step 2: B fires at T=1 and reads from its `in` mailbox, finding A's payload.
        sched.step();
        assert!(
            *b_seen.lock().unwrap(),
            "B should see A's output in the next step (visibility-at-T+1)"
        );
    }

    #[test]
    fn scheduler_deterministic_replay() {
        // Same APG + same fire closures + same input → same output trace,
        // every run.
        fn build_pipeline() -> ReactorScheduler {
            let mut sched = ReactorScheduler::new();
            sched.register_agent(|ports| {
                ports.write_out(
                    "out",
                    PortMessage::Struct(Box::new(StructPayload::new(vec![42]))),
                );
            });
            sched.register_agent(|_ports| {});
            sched.connect((0, "out"), (1, "in"));
            sched
        }

        let mut traces: Vec<Vec<u64>> = Vec::new();
        for _ in 0..50 {
            let mut sched = build_pipeline();
            let mut trace = Vec::new();
            for _ in 0..5 {
                if sched.step() == StepOutcome::Advanced {
                    trace.push(sched.logical_time());
                }
            }
            traces.push(trace);
        }

        // All traces must be identical.
        for w in traces.windows(2) {
            assert_eq!(w[0], w[1], "deterministic replay required");
        }
    }

    #[test]
    fn empty_scheduler_advances_idly() {
        let mut sched = ReactorScheduler::new();
        let outcome = sched.step();
        assert_eq!(outcome, StepOutcome::Advanced);
        assert_eq!(sched.logical_time(), 1);
    }
}
