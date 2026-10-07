//! `nsl` subcommand handlers.
//!
//! This module groups the per-command implementations that were historically
//! inlined in `main.rs`. Each submodule owns one command (or a closely related
//! family); `main.rs` parses arguments and dispatches into these handlers.
//!
//! Extractions are behavior-preserving: code is moved verbatim, with only the
//! visibility (`pub(crate)`) and `use` paths adjusted.

pub(crate) mod autotune;
pub(crate) mod build;
pub(crate) mod cep;
pub(crate) mod check;
pub(crate) mod convert;
pub(crate) mod cpkd_design;
pub(crate) mod env;
pub(crate) mod fmt;
pub(crate) mod init;
pub(crate) mod export;
pub(crate) mod profile_merge;
pub(crate) mod ptx_metadata;
pub(crate) mod run;
pub(crate) mod test;
pub(crate) mod tokenize;

/// Parse and validate `--wggo-prune-layers` and `--wggo-layer-prune-fraction`,
/// shared by `nsl run` and `nsl build` so the two dispatchers cannot drift.
/// Returns the layer names, or the `error: ...` line to print before exiting.
///
/// Only the CLI-shaped checks live here (an empty name, a fraction outside
/// (0, 1)). Whether a name exists, whether the fraction selects anything and
/// whether `--source-ad` / `--wggo` are present are decided by codegen, where
/// the layer graph and the options meet.
pub(crate) fn parse_wggo_layer_prune(
    prune_layers: Option<&str>,
    fraction: Option<f64>,
) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    if let Some(raw) = prune_layers {
        for name in raw.split(',').map(str::trim) {
            if name.is_empty() {
                return Err(format!(
                    "error: --wggo-prune-layers has an empty layer name in '{raw}' \
                     (expected comma-separated names such as blocks.1,blocks.3)"
                ));
            }
            names.push(name.to_string());
        }
    }
    if let Some(f) = fraction
        && !(f > 0.0 && f < 1.0)
    {
        return Err(format!(
            "error: --wggo-layer-prune-fraction must be in (0, 1), got {f}"
        ));
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::parse_wggo_layer_prune;

    #[test]
    fn wggo_layer_prune_flags_parse_and_validate() {
        assert_eq!(
            parse_wggo_layer_prune(Some("blocks.1, blocks.3"), None).unwrap(),
            vec!["blocks.1".to_string(), "blocks.3".to_string()]
        );
        assert!(parse_wggo_layer_prune(None, None).unwrap().is_empty());
        assert!(parse_wggo_layer_prune(Some("blocks.1,,blocks.2"), None)
            .unwrap_err()
            .contains("empty layer name"));
        assert!(parse_wggo_layer_prune(None, Some(0.25)).is_ok());
        for bad in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
            assert!(
                parse_wggo_layer_prune(None, Some(bad))
                    .unwrap_err()
                    .contains("must be in (0, 1)"),
                "{bad}"
            );
        }
    }
}
