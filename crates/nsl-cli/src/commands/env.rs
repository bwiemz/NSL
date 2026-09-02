//! `nsl env` — the `NSL_*` environment-variable registry (roadmap A5).
//!
//! `nsl env list` renders `nsl_env::REGISTRY`; `nsl env current` reports
//! which registered variables are set in the calling environment and flags
//! any `NSL_*` that is set but unknown — the check to run before recording
//! a result, since a `behavior`-tier variable changes what a run computed.

use std::process;

use crate::args::EnvCmd;
use nsl_env::{json_str, Current, EnvVar, Tier};

pub(crate) fn run(cmd: EnvCmd) {
    match cmd {
        EnvCmd::List { tier, markdown, json } => {
            if markdown {
                print!("{}", nsl_env::render_markdown());
                return;
            }
            let vars: Vec<&EnvVar> = match tier.as_deref() {
                None => nsl_env::REGISTRY.iter().collect(),
                Some(t) => match Tier::parse(t) {
                    Some(tier) => nsl_env::by_tier(tier).collect(),
                    None => {
                        eprintln!(
                            "error: unknown tier '{t}' — one of: {}",
                            Tier::ALL.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(", ")
                        );
                        process::exit(2);
                    }
                },
            };
            if json {
                print!("{}", nsl_env::render_json(&vars));
            } else {
                print!("{}", nsl_env::render_table(&vars));
            }
        }
        EnvCmd::Current { strict, json } => {
            let set = nsl_env::current();
            let unknown = set
                .iter()
                .filter(|c| matches!(c, Current::Unregistered { .. }))
                .count();
            if json {
                print!("{}", current_json(&set));
            } else if set.is_empty() {
                println!("no NSL_* variables are set");
            } else {
                for c in &set {
                    match c {
                        Current::Registered { var, value } => println!(
                            "{}={}\n    [{}] {}",
                            var.name,
                            value,
                            var.tier.as_str(),
                            var.doc
                        ),
                        Current::Unregistered { name, value } => {
                            println!("{name}={value}\n    [UNREGISTERED] not in the nsl-env registry: a typo, or a variable nothing reads")
                        }
                    }
                }
            }
            if strict && unknown > 0 {
                eprintln!("error: {unknown} set NSL_* variable(s) are not registered (see above)");
                process::exit(1);
            }
        }
    }
}

fn current_json(set: &[Current]) -> String {
    let mut out = String::from("[\n");
    for (i, c) in set.iter().enumerate() {
        let (name, value, registered, tier) = match c {
            Current::Registered { var, value } => (var.name, value.as_str(), true, var.tier.as_str()),
            Current::Unregistered { name, value } => (name.as_str(), value.as_str(), false, ""),
        };
        out.push_str(&format!(
            "  {{\"name\": {}, \"value\": {}, \"registered\": {}, \"tier\": {}}}{}\n",
            json_str(name),
            json_str(value),
            registered,
            json_str(tier),
            if i + 1 < set.len() { "," } else { "" }
        ));
    }
    out.push_str("]\n");
    out
}
