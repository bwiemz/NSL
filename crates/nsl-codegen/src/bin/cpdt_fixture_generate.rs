//! CPDT calibration fixture generator. Produces deterministic synthetic
//! transformer weight files:
//!
//!   calib_tiny   — 2L / d_model=128 / d_ffn=512 / vocab=256,  f32, ~2.2 MB
//!   calib_small  — 8L / d_model=512 / d_ffn=1792 / vocab=8192, f16, ~68 MB
//!   calib_medium — 16L / d_model=1024 / d_ffn=4096 / vocab=32768, f16
//!                   (regenerated at test-time into target/, not committed)
//!
//! Init scheme: Kaiming-normal for projection weights, ones for norm scales,
//! zeros for biases (when present). Box–Muller from deterministic `StdRng`
//! seeded with a fixed constant. Output is little-endian safetensors.
//!
//! Usage (from repo root):
//!   cargo run --features calibrate --bin cpdt_fixture_generate -- <output_dir>

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;

const SEED: u64 = 0xC9D7DA7ACA15B;

#[derive(Clone, Copy)]
struct TransformerShape {
    layers: u32,
    d_model: u32,
    d_ffn: u32,
    vocab: u32,
    tied_embeddings: bool,
    bias_schedule: BiasSchedule,
}

#[derive(Clone, Copy)]
enum BiasSchedule {
    None,
    MixedHalf,
}

fn calib_tiny() -> TransformerShape {
    TransformerShape {
        layers: 2,
        d_model: 128,
        d_ffn: 512,
        vocab: 256,
        tied_embeddings: false,
        bias_schedule: BiasSchedule::None,
    }
}

fn calib_small() -> TransformerShape {
    TransformerShape {
        layers: 8,
        d_model: 512,
        d_ffn: 1792,
        vocab: 8192,
        tied_embeddings: true,
        bias_schedule: BiasSchedule::None,
    }
}

fn calib_medium() -> TransformerShape {
    TransformerShape {
        layers: 16,
        d_model: 1024,
        d_ffn: 4096,
        vocab: 32768,
        tied_embeddings: false,
        bias_schedule: BiasSchedule::MixedHalf,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut include_medium = false;
    let mut output_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--include-medium" => {
                include_medium = true;
                i += 1;
            }
            "--output-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output-dir requires a value");
                    std::process::exit(2);
                }
                output_dir = Some(args[i].clone());
                i += 1;
            }
            other => {
                if other.starts_with("--") {
                    eprintln!("error: unknown flag or missing space after flag name: {other}");
                    std::process::exit(2);
                }
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    // Positional output_dir (back-compat) OR --output-dir (new form).
    let out_dir_string = match (output_dir, positional.first()) {
        (Some(d), _) => d,
        (None, Some(d)) => d.clone(),
        (None, None) => {
            eprintln!("usage: cpdt_fixture_generate [--include-medium] [--output-dir DIR | DIR]");
            std::process::exit(1);
        }
    };
    let out_dir = Path::new(&out_dir_string);
    std::fs::create_dir_all(out_dir).unwrap();

    write_fixture(out_dir, "calib_tiny", calib_tiny(), DType::F32);
    write_fixture(out_dir, "calib_small", calib_small(), DType::F16);

    if include_medium {
        write_fixture(out_dir, "calib_medium", calib_medium(), DType::F16);
    } else {
        eprintln!(
            "calib_medium is regenerated at test-time into target/; pass --include-medium to write here."
        );
    }
}

#[derive(Copy, Clone)]
enum DType {
    F32,
    F16,
}

fn dtype_to_st(d: DType) -> Dtype {
    match d {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
    }
}

struct TensorPayload {
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

fn write_fixture(out_dir: &Path, name: &str, shape: TransformerShape, dtype: DType) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut tensors: HashMap<String, TensorPayload> = HashMap::new();

    let d_model = shape.d_model as usize;
    let d_ffn = shape.d_ffn as usize;
    let vocab = shape.vocab as usize;

    tensors.insert(
        "tok_embeddings.weight".into(),
        TensorPayload {
            shape: vec![vocab, d_model],
            dtype: dtype_to_st(dtype),
            bytes: kaiming_normal_tensor(&mut rng, &[vocab, d_model], dtype, d_model),
        },
    );

    for l in 0..shape.layers {
        let has_bias = match shape.bias_schedule {
            BiasSchedule::None => false,
            BiasSchedule::MixedHalf => l < shape.layers / 2,
        };

        for proj in &["wq", "wk", "wv", "wo"] {
            let w = kaiming_normal_tensor(&mut rng, &[d_model, d_model], dtype, d_model);
            tensors.insert(
                format!("blocks.{l}.attn.{proj}.weight"),
                TensorPayload {
                    shape: vec![d_model, d_model],
                    dtype: dtype_to_st(dtype),
                    bytes: w,
                },
            );
            if has_bias {
                let b = zeros_tensor(&[d_model], dtype);
                tensors.insert(
                    format!("blocks.{l}.attn.{proj}.bias"),
                    TensorPayload {
                        shape: vec![d_model],
                        dtype: dtype_to_st(dtype),
                        bytes: b,
                    },
                );
            }
        }

        for nname in &["attn_norm", "ffn_norm"] {
            tensors.insert(
                format!("blocks.{l}.{nname}.weight"),
                TensorPayload {
                    shape: vec![d_model],
                    dtype: dtype_to_st(dtype),
                    bytes: ones_tensor(&[d_model], dtype),
                },
            );
        }

        for (fname, rows, cols) in &[
            ("w_gate", d_ffn, d_model),
            ("w_up", d_ffn, d_model),
            ("w_down", d_model, d_ffn),
        ] {
            let w = kaiming_normal_tensor(&mut rng, &[*rows, *cols], dtype, *cols);
            tensors.insert(
                format!("blocks.{l}.ffn.{fname}.weight"),
                TensorPayload {
                    shape: vec![*rows, *cols],
                    dtype: dtype_to_st(dtype),
                    bytes: w,
                },
            );
        }
    }

    tensors.insert(
        "norm.weight".into(),
        TensorPayload {
            shape: vec![d_model],
            dtype: dtype_to_st(dtype),
            bytes: ones_tensor(&[d_model], dtype),
        },
    );

    if !shape.tied_embeddings {
        tensors.insert(
            "output.weight".into(),
            TensorPayload {
                shape: vec![vocab, d_model],
                dtype: dtype_to_st(dtype),
                bytes: kaiming_normal_tensor(&mut rng, &[vocab, d_model], dtype, d_model),
            },
        );
    }

    let views: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|(k, payload)| {
            let view = TensorView::new(
                payload.dtype,
                payload.shape.clone(),
                payload.bytes.as_slice(),
            )
            .unwrap();
            (k.clone(), view)
        })
        .collect();
    let bytes = serialize(&views, &None).unwrap();
    let path = out_dir.join(format!("{name}.safetensors"));
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn kaiming_normal_tensor(
    rng: &mut StdRng,
    shape: &[usize],
    dtype: DType,
    fan_in: usize,
) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    let stddev = (2.0_f64 / fan_in as f64).sqrt();
    let mut f32_vals = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f64 = rng.gen_range(1e-10..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        f32_vals.push((z * stddev) as f32);
    }
    encode(dtype, &f32_vals)
}

fn zeros_tensor(shape: &[usize], dtype: DType) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    encode(dtype, &vec![0.0_f32; numel])
}

fn ones_tensor(shape: &[usize], dtype: DType) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    encode(dtype, &vec![1.0_f32; numel])
}

fn encode(dtype: DType, vals: &[f32]) -> Vec<u8> {
    match dtype {
        DType::F32 => vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => vals
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect(),
    }
}
