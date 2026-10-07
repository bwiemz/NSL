//! Parameter coverage (NSL V2 plan item 0.4): training updates EVERY
//! trainable tensor of a model, and nothing else, in both AD modes.
//!
//! #806 shipped LoRA adapters that never trained: A and B were in neither AD
//! mode's optimizer parameter list, so the step skipped them and the run
//! exited 0. Nothing in CI could see that class of bug, because every
//! existing check takes its expected set from the same enumeration that
//! builds the parameter list (`enumerate_all_model_tensor_paths`):
//! `--grad-integrity` counts the list, and source AD zero-fills a missing
//! gradient and reports it only in a WARN summary whose denominator is that
//! enumeration. The 1B posture certificate checks every saved field moves,
//! but on the GPU only, and `model_save` cannot see adapters.
//!
//! Here the EXPECTED set comes from the fixture's declarations as the
//! semantic analysis records them (`nsl_cli::loader`): the model types'
//! fields (tensors, sub-models, `[Model; N]` arrays), the `@adapter`
//! configs (one set of synthesized tensors per target-model instance) and
//! the `@freeze` configs, walked by this file. Non-trainable state is the
//! language's: a `Buffer<...>` field (spec/02), and the `_`-prefix /
//! `inv_freq` convention the stdlib's configuration tensors use. The
//! OBSERVATIONS are `model_save` before and after training (struct fields,
//! exact f32) and the fixture's printed adapter tensors (`print` writes an
//! f32 exactly). Both sets must be exactly the expected names.
//!
//! The fixture trains three plain-SGD steps from a fixed seed, under the
//! tape and under `--source-ad`. Each mode must move every element of every
//! trainable tensor and leave every frozen and non-trainable tensor
//! bit-identical; the two modes -- independent AD implementations -- must
//! agree on the trained values.
//!
//! Mutation-checked (2026-10-06); each of these fails the gate:
//! - adapters left out of the parameter enumeration (#806's mechanism);
//! - source AD discarding adapter gradients (zero-filled);
//! - the `[Model; N]` walk skipping its last element;
//! - the parameter list skipping direct sub-model fields;
//! - `@freeze` made ineffective;
//! - source AD's freeze filter reaching a frozen model's adapters.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

use nsl_ast::Symbol;
use nsl_semantic::scope::ScopeId;
use nsl_semantic::types::Type;
use nsl_semantic::wrga::{AdapterKind, FreezeConfig, FreezeTarget};

const FIXTURE: &str = "param_coverage.nsl";
/// The trained binding (`train(model = m, ...)`).
const ROOT_BINDING: &str = "m";
const SEED: &str = "7";

/// Known defects the gate observes and tolerates, as a RATCHET: each listed
/// tensor must still show its defect, so fixing one fails this gate until
/// it is delisted. (path, what is wrong)
const KNOWN_DEFECTS: &[(&str, &str)] = &[(
    "buf",
    "a `Buffer<...>` field is non-trainable per spec/02, but the parameter \
     enumeration selects by leaf name only, so both AD modes train it",
)];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn fixture_path() -> PathBuf {
    repo_root().join("crates/nsl-cli/tests/fixtures").join(FIXTURE)
}

// ---------------------------------------------------------------------------
// The expected set, from the declarations.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    /// Must move, every element.
    Trainable,
    /// `@freeze`-frozen: must stay bit-identical.
    Frozen,
    /// `_`-prefixed or `inv_freq`: configuration, must stay bit-identical.
    Config,
    /// A `Buffer<...>` field: non-trainable state, must stay bit-identical.
    Buffer,
}

#[derive(Clone, Debug)]
enum Field {
    Tensor,
    Buffer,
    Model(String),
    Array(String, usize),
}

/// A model instance in the trained model's tree: its type and its path below
/// the root ("" for the root itself).
#[derive(Clone, Debug)]
struct Instance {
    model: String,
    path: String,
}

struct Expected {
    /// Struct-field tensors (what `model_save` writes), by dotted path below
    /// the root (`blocks.1.w`).
    fields: BTreeMap<String, Role>,
    /// Adapter side-table tensors (always trainable).
    adapters: BTreeSet<String>,
}

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") }
}

/// Run the frontend over the fixture and derive the expected set.
fn expected_from_declarations() -> Expected {
    // The train block's optimizer is an implicit stdlib import, and the
    // loader finds the stdlib through this variable (the cwd fallback would
    // look in crates/nsl-cli).
    // SAFETY: this binary's only test sets it before it spawns anything or
    // starts a thread; nothing else in the process reads the environment
    // concurrently. A second #[test] in this file would break that: give it
    // its own binary, or set the variable once before either runs.
    unsafe { std::env::set_var("NSL_STDLIB_PATH", repo_root().join("stdlib")) };
    let mut source_map = nsl_errors::SourceMap::silent();
    let mut interner = nsl_lexer::Interner::new();
    let graph = nsl_cli::loader::load_all_modules(&fixture_path(), &mut source_map, &mut interner)
        .unwrap_or_else(|e| panic!("{FIXTURE}: frontend failed: {e}"));
    let module = &graph.modules[&graph.entry];
    let name_of = |s: Symbol| interner.resolve(s.0).expect("interned symbol").to_string();

    // A model's declared fields, from its type in the top-level scope (a
    // decorated definition is not among the module's exports).
    let model_fields = |model: &str| -> Vec<(String, Field)> {
        let sym = Symbol(interner.get(model).unwrap_or_else(|| panic!("no symbol {model}")));
        let Some((_, info)) = module.scopes.lookup(ScopeId::ROOT, sym) else {
            panic!("no top-level declaration for model {model}")
        };
        let Type::Model { fields, .. } = &info.ty else { panic!("{model} is not a model: {:?}", info.ty) };
        fields
            .iter()
            .filter_map(|(f, t)| {
                let field = match t {
                    Type::Tensor { .. } | Type::Param { .. } => Field::Tensor,
                    Type::Buffer { .. } => Field::Buffer,
                    Type::Model { name, .. } => Field::Model(name_of(*name)),
                    Type::FixedModelArray { element_model, size } => {
                        Field::Array(name_of(*element_model), usize::try_from(*size).expect("array size"))
                    }
                    // A method's type, or a scalar: not a parameter.
                    Type::Function { .. } | Type::Int | Type::Float | Type::Bool | Type::Str => {
                        return None;
                    }
                    other => panic!("{model}.{}: field type {other:?} -- teach the walk", name_of(*f)),
                };
                Some((name_of(*f), field))
            })
            .collect()
    };
    let root_sym = Symbol(interner.get(ROOT_BINDING).expect("the trained binding is interned"));
    let root = match module.scopes.lookup(ScopeId::ROOT, root_sym) {
        Some((_, info)) => match &info.ty {
            Type::Model { name, .. } => name_of(*name),
            other => panic!("`{ROOT_BINDING}` should be a model, is {other:?}"),
        },
        None => panic!("`{ROOT_BINDING}` is not bound at the top level of {FIXTURE}"),
    };

    // Every model the tree holds, and every model an adapter targets (a
    // target the tree does not hold is reported below, not as a missing
    // declaration).
    let mut models: HashMap<String, Vec<(String, Field)>> = HashMap::new();
    let mut pending = vec![root.clone()];
    for cfg in &module.adapter_configs {
        pending.extend(cfg.targets.iter().filter_map(|t| t.split_once('.')).map(|(model, _)| model.to_string()));
    }
    while let Some(model) = pending.pop() {
        if models.contains_key(&model) {
            continue;
        }
        let fields = model_fields(&model);
        for (_, field) in &fields {
            if let Field::Model(sub) | Field::Array(sub, _) = field {
                pending.push(sub.clone());
            }
        }
        models.insert(model, fields);
    }
    // Walk the tree: every tensor leaf, with the model instances above it.
    let mut leaves: Vec<(String, Field, Vec<Instance>)> = Vec::new();
    let mut instances: Vec<Instance> = Vec::new();
    walk(&models, &root, "", &mut vec![], &mut leaves, &mut instances);

    let mut fields = BTreeMap::new();
    for (path, kind, chain) in leaves {
        let leaf = path.rsplit('.').next().unwrap();
        let role = if leaf.starts_with('_') || leaf == "inv_freq" {
            Role::Config
        } else if matches!(kind, Field::Buffer) {
            Role::Buffer
        } else if module.freeze_configs.iter().any(|f| freezes(f, &path, &chain)) {
            Role::Frozen
        } else {
            Role::Trainable
        };
        fields.insert(path, role);
    }

    // `@adapter(type=K, target=["Model.field", ...])`: every instance of
    // Model gets its own synthesized tensors, named
    // `<prefix>_<Model>_<field>__<kind>` (wrga_adapter_inject.rs).
    let mut adapters = BTreeSet::new();
    for cfg in &module.adapter_configs {
        let (prefixes, suffix): (&[&str], &str) = match cfg.kind {
            AdapterKind::Lora => (&["lora_A", "lora_B"], "lora"),
            AdapterKind::Ia3 => (&["ia3_scale"], "ia3"),
            AdapterKind::GatedLora => (&["lora_A", "lora_B", "gate"], "gatedlora"),
        };
        for target in &cfg.targets {
            let (model, field) = target.split_once('.').unwrap_or_else(|| panic!("adapter target {target:?}"));
            let declared = models.get(model).unwrap_or_else(|| panic!("adapter target {target:?}: no model {model}"));
            assert!(
                declared.iter().any(|(f, k)| f == field && matches!(k, Field::Tensor)),
                "adapter target {target:?}: {model} has no tensor field {field}"
            );
            let hosts: Vec<&Instance> = instances.iter().filter(|i| i.model == model).collect();
            assert!(!hosts.is_empty(), "adapter target {target:?}: the trained model holds no {model}");
            for host in hosts {
                for p in prefixes {
                    adapters.insert(join(&host.path, &format!("{p}_{model}_{field}__{suffix}")));
                }
            }
        }
    }
    Expected { fields, adapters }
}

fn walk(
    models: &HashMap<String, Vec<(String, Field)>>,
    model: &str,
    path: &str,
    chain: &mut Vec<Instance>,
    leaves: &mut Vec<(String, Field, Vec<Instance>)>,
    instances: &mut Vec<Instance>,
) {
    assert!(chain.len() < 16, "model nesting too deep at {path:?}");
    let fields = models.get(model).unwrap_or_else(|| panic!("no declaration for model {model}"));
    let me = Instance { model: model.to_string(), path: path.to_string() };
    instances.push(me.clone());
    chain.push(me);
    for (name, field) in fields {
        let here = join(path, name);
        match field {
            Field::Tensor | Field::Buffer => leaves.push((here, field.clone(), chain.clone())),
            Field::Model(sub) => walk(models, sub, &here, chain, leaves, instances),
            Field::Array(elem, n) => {
                for i in 0..*n {
                    walk(models, elem, &format!("{here}.{i}"), chain, leaves, instances);
                }
            }
        }
    }
    chain.pop();
}

/// The documented `@freeze` semantics (docs/wiki/Glossary.md): on
/// `let m = ...` the patterns see the path below `m`, with or without the
/// `m.`; on `model Blk:` they see the path below each `Blk` instance, or
/// the whole path. `include` freezes what a pattern matches, `exclude` what
/// none matches, a bare `@freeze` everything it reaches. Adapters are never
/// frozen (the caller never asks). The fixture binds `m` once, so the
/// binding's name identifies it.
fn freezes(cfg: &FreezeConfig, path: &str, chain: &[Instance]) -> bool {
    let views: Vec<String> = match cfg.target.as_ref().expect("the checker sets every freeze target") {
        FreezeTarget::Binding { name, .. } if name == ROOT_BINDING => {
            vec![path.to_string(), join(ROOT_BINDING, path)]
        }
        FreezeTarget::Binding { .. } => Vec::new(),
        FreezeTarget::Model(model) => {
            let mut v: Vec<String> = chain
                .iter()
                .filter(|i| &i.model == model)
                .map(|i| if i.path.is_empty() { path.to_string() } else { path[i.path.len() + 1..].to_string() })
                .collect();
            if !v.is_empty() {
                v.push(path.to_string());
            }
            v
        }
    };
    if views.is_empty() {
        return false;
    }
    let hit = |pats: &[String]| pats.iter().any(|p| views.iter().any(|v| glob(p.as_bytes(), v.as_bytes())));
    if !cfg.include.is_empty() {
        hit(&cfg.include)
    } else if !cfg.exclude.is_empty() {
        !hit(&cfg.exclude)
    } else {
        true
    }
}

/// `*` matches any run of characters, `?` any one.
fn glob(p: &[u8], t: &[u8]) -> bool {
    match (p.first(), t.first()) {
        (None, None) => true,
        (Some(b'*'), _) => glob(&p[1..], t) || (!t.is_empty() && glob(p, &t[1..])),
        (Some(b'?'), Some(_)) => glob(&p[1..], &t[1..]),
        (Some(a), Some(b)) if a == b => glob(&p[1..], &t[1..]),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// The observations.
// ---------------------------------------------------------------------------

/// One training run: every observed tensor before and after.
struct Run {
    before: BTreeMap<String, Vec<f32>>,
    after: BTreeMap<String, Vec<f32>>,
}

/// Train the fixture in one AD mode. A run that fails, or is not the AD it
/// claims to be, is an `Err` (reported with the other mode's findings).
fn train(source_ad: bool) -> Result<Run, String> {
    let mode = if source_ad { "source AD" } else { "tape AD" };
    let tmp = tempfile::TempDir::new().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .args(["--seed", SEED])
        .arg(fixture_path())
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let (stdout, stderr) = (String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let tail =
        |text: &str| text.lines().rev().take(20).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n");
    if !out.status.success() {
        return Err(format!("{mode}: {FIXTURE} failed ({}); stderr ends:\n{}", out.status, tail(&stderr)));
    }
    // Each arm is the AD it claims to be.
    if source_ad && !stderr.contains("Using source-to-source AD for backward pass") {
        return Err(format!("{mode} did not engage; stderr ends:\n{}", tail(&stderr)));
    }
    if source_ad && stderr.contains("falling back to tape-based AD") {
        return Err(format!("{mode} fell back to the tape; stderr ends:\n{}", tail(&stderr)));
    }
    if !source_ad && stderr.contains("Using source-to-source AD") {
        return Err(format!("{mode} ran source AD; stderr ends:\n{}", tail(&stderr)));
    }

    let snapshots: Vec<&str> = stdout.split("ADAPTERS_BEGIN").skip(1).collect();
    assert_eq!(snapshots.len(), 2, "{mode}: expected an adapter snapshot before and after training:\n{stdout}");
    let mut before = read_nslm(&tmp.path().join("init.nslm"));
    let mut after = read_nslm(&tmp.path().join("end.nslm"));
    for (map, snap) in [(&mut before, snapshots[0]), (&mut after, snapshots[1])] {
        for (name, values) in adapter_snapshot(snap) {
            assert!(map.insert(name.clone(), values).is_none(), "{mode}: {name} observed twice");
        }
    }
    Ok(Run { before, after })
}

/// The `ADAPTER <path>` / tensor line pairs of one snapshot, `*` replaced by
/// how many times that path has been seen (the array index, in iteration
/// order).
fn adapter_snapshot(snap: &str) -> Vec<(String, Vec<f32>)> {
    let body = snap.split_once("ADAPTERS_END").expect("ADAPTERS_END").0;
    let lines: Vec<&str> = body.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    assert!(lines.len().is_multiple_of(2), "unpaired adapter snapshot lines:\n{body}");
    let mut seen: HashMap<&str, usize> = HashMap::new();
    lines
        .chunks(2)
        .map(|pair| {
            let pattern =
                pair[0].strip_prefix("ADAPTER ").unwrap_or_else(|| panic!("not an ADAPTER line: {:?}", pair[0]));
            let k = seen.entry(pattern).or_default();
            let name = pattern.replace('*', &k.to_string());
            *k += 1;
            (name, parse_printed_tensor(pair[1]))
        })
        .collect()
}

/// `tensor([[a, b], [c, d]])` -> [a, b, c, d]. `print` writes each f32
/// widened to f64 in shortest round-trip form, so this recovers it exactly.
fn parse_printed_tensor(line: &str) -> Vec<f32> {
    let inner = line
        .strip_prefix("tensor(")
        .and_then(|l| l.strip_suffix(')'))
        .unwrap_or_else(|| panic!("not a tensor: {line:?}"));
    inner
        .split(|c: char| c == ',' || c == '[' || c == ']' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(|t| {
            let v: f64 = t.parse().unwrap_or_else(|e| panic!("{t:?} in {line:?}: {e}"));
            let f = v as f32;
            assert!(f as f64 == v || v.is_nan(), "{t} is not an f32 value");
            f
        })
        .collect()
}

/// `model_save`'s file: every saved tensor (f32), by dotted path
/// (`blocks[1].w` -> `blocks.1.w`).
fn read_nslm(path: &Path) -> BTreeMap<String, Vec<f32>> {
    let buf = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(buf.len() >= 16 && &buf[0..4] == b"NSLM", "bad magic in {path:?}");
    let header_end = 16 + usize::try_from(u64::from_le_bytes(buf[8..16].try_into().unwrap())).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&buf[16..header_end]).expect("nslm header json");
    let data = header_end.next_multiple_of(64);
    let mut out = BTreeMap::new();
    for p in header["params"].as_array().expect("params") {
        let name = p["name"].as_str().expect("name").replace('[', ".").replace(']', "");
        assert_eq!(p["dtype"], "f32", "{name} in {path:?}");
        let (offset, nbytes) = (p["offset"].as_u64().unwrap() as usize, p["nbytes"].as_u64().unwrap() as usize);
        let slab = &buf[data + offset..data + offset + nbytes];
        let values = slab.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert!(out.insert(name.clone(), values).is_none(), "{name} saved twice in {path:?}");
    }
    out
}

// ---------------------------------------------------------------------------
// The gate.
// ---------------------------------------------------------------------------

/// What the fixture covers, by construct: the expected set must classify
/// each of these paths as given, so an edit that drops a construct from the
/// fixture (or a walk that stops seeing one) fails here, not silently.
const COVERS: &[(&str, &str, Role)] = &[
    ("tied", "a field used in two places", Role::Trainable),
    ("blocks.0.w", "a [Blk; 2] element's field", Role::Trainable),
    ("blocks.1.b", "the last [Blk; 2] element's field", Role::Trainable),
    ("inner.u", "a nested sub-model's field", Role::Trainable),
    ("head", "a plain field", Role::Trainable),
    ("fz.v", "a field of a `@freeze model`", Role::Frozen),
    ("pinned", "a field frozen by `@freeze(include=...)` on the binding", Role::Frozen),
    ("inner._gain", "a `_`-prefixed configuration tensor", Role::Config),
    ("inv_freq", "an `inv_freq` table", Role::Config),
    ("buf", "a `Buffer<...>` field", Role::Buffer),
];
const COVERS_ADAPTERS: &[(&str, &str)] = &[
    ("blocks.1.lora_A_Blk_w__lora", "LoRA on the last [Blk; 2] element"),
    ("blocks.0.lora_B_Blk_w__lora", "LoRA B (starts at zero) on a [Blk; 2] element"),
    ("lora_B_Net_head__lora", "LoRA on a top-level field"),
    ("fz.lora_A_Frozen_v__lora", "LoRA on a frozen sub-model's weight"),
    ("inner.ia3_scale_Inner_u__ia3", "IA3 on a nested sub-model"),
    ("gate_Net_proj__gatedlora", "GatedLoRA's gate"),
];

#[test]
fn every_parameter_trains_and_nothing_else_does() {
    let expected = expected_from_declarations();
    for (path, what, role) in COVERS {
        assert_eq!(expected.fields.get(*path), Some(role), "{FIXTURE} must cover {what} ({path}) as {role:?}");
    }
    for (path, what) in COVERS_ADAPTERS {
        assert!(expected.adapters.contains(*path), "{FIXTURE} must cover {what} ({path})");
    }
    let expected_names: BTreeSet<String> = expected.fields.keys().chain(&expected.adapters).cloned().collect();

    let mut failures = Vec::new();
    let mut runs = Vec::new();
    for (mode, source_ad) in [("tape AD", false), ("source AD", true)] {
        match train(source_ad) {
            Ok(run) => runs.push((mode, run)),
            Err(e) => failures.push(e),
        }
    }
    let mut report = Vec::new();
    for (mode, run) in &runs {
        for (when, observed) in [("before", &run.before), ("after", &run.after)] {
            let names: BTreeSet<String> = observed.keys().cloned().collect();
            for missing in expected_names.difference(&names) {
                failures.push(format!("{mode}: {missing} is declared but never observed {when} training"));
            }
            for extra in names.difference(&expected_names) {
                failures.push(format!("{mode}: {extra} is observed {when} training but not declared"));
            }
        }
        for name in &expected_names {
            let (Some(v0), Some(v1)) = (run.before.get(name), run.after.get(name)) else {
                continue;
            };
            assert_eq!(v0.len(), v1.len(), "{mode}: {name} changed size");
            let moved = v0.iter().zip(v1).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let travel = v0.iter().zip(v1).map(|(a, b)| f64::from(*b - *a).abs()).fold(0.0, f64::max);
            let role = expected.fields.get(name).copied().unwrap_or(Role::Trainable);
            let role_name = format!("{role:?}");
            report.push(format!(
                "{mode:>9} {name:<30} {role_name:<9} moved {moved:>2}/{:<2} max|Δ| {travel:.2e}",
                v0.len()
            ));
            if !v1.iter().all(|v| v.is_finite()) {
                failures.push(format!("{mode}: {name} is not finite after training"));
            }
            let known = KNOWN_DEFECTS.iter().find(|(p, _)| p == name);
            match (role, known) {
                (_, Some((_, defect))) => {
                    if moved == 0 {
                        failures.push(format!(
                            "{mode}: {name} no longer moves -- the known defect ({defect}) looks fixed; delist it from KNOWN_DEFECTS"
                        ));
                    }
                }
                (Role::Trainable, None) => {
                    if moved != v0.len() {
                        failures.push(format!(
                            "{mode}: trainable {name} was not updated ({moved}/{} elements moved) -- a parameter the step does not reach",
                            v0.len()
                        ));
                    }
                }
                (_, None) => {
                    if moved != 0 {
                        failures.push(format!(
                            "{mode}: {role:?} {name} moved ({moved}/{} elements, max|Δ| {travel:.2e})",
                            v0.len()
                        ));
                    }
                }
            }
        }
    }

    // The two AD implementations start from the same parameters (same seed)
    // and must land on the same ones.
    if let [(_, tape), (_, source)] = &runs[..] {
        for name in &expected_names {
            let (Some(t0), Some(s0), Some(t1), Some(s1)) =
                (tape.before.get(name), source.before.get(name), tape.after.get(name), source.after.get(name))
            else {
                continue;
            };
            if t0 != s0 {
                failures.push(format!("{name}: the two modes start from different values (seed not honoured?)"));
                continue;
            }
            let scale = t0.iter().zip(t1).map(|(a, b)| f64::from(*b - *a).abs()).fold(0.0, f64::max);
            let diff = t1.iter().zip(s1).map(|(a, b)| f64::from(*a - *b).abs()).fold(0.0, f64::max);
            report.push(format!(
                "  modes   {name:<30} max|tape - source| {diff:.2e} = {:.2e} of the update",
                diff / scale.max(f64::MIN_POSITIVE)
            ));
            let magnitude = t1.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
            if diff > MODE_AGREEMENT * scale + MODE_ULPS * f64::from(f32::EPSILON) * magnitude {
                failures.push(format!(
                    "{name}: tape and source AD disagree after training: max|Δ| {diff:.2e} vs an update of {scale:.2e}"
                ));
            }
        }
    }

    let report = report.join("\n");
    eprintln!("{report}");
    assert!(failures.is_empty(), "parameter coverage failed:\n{}\n\nper tensor:\n{report}", failures.join("\n"));
}

/// Bound on max|tape - source| after training: `MODE_AGREEMENT` of the
/// tensor's max|update| plus `MODE_ULPS` f32 ulps of its magnitude. Two AD
/// implementations may sum in different orders; a dropped or doubled
/// gradient contribution (one use of a tied field, say) moves the result by
/// O(update). Measured 2026-10-06: bit-identical in every tensor.
const MODE_AGREEMENT: f64 = 1e-3;
const MODE_ULPS: f64 = 8.0;
