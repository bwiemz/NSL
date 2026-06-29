//! Trace-merge helper for `nsl run --profile` (combines memory + kernel traces).
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

pub(crate) fn merge_profile_traces(memory_path: &str, kernel_path: &str, output_path: &str) {
    let mem_json = std::fs::read_to_string(memory_path).unwrap_or_default();
    let kern_json = std::fs::read_to_string(kernel_path).unwrap_or_default();

    let mem_events = extract_trace_events(&mem_json).unwrap_or_default();
    let kern_events = extract_trace_events(&kern_json).unwrap_or_default();

    let merged = format!(
        r#"{{"traceEvents":[{}{}{}],"metadata":{{"merged":true}}}}"#,
        mem_events,
        if !mem_events.is_empty() && !kern_events.is_empty() { "," } else { "" },
        kern_events,
    );

    std::fs::write(output_path, &merged).ok();
    std::fs::remove_file(memory_path).ok();
    std::fs::remove_file(kernel_path).ok();
    eprintln!("[nsl] merged profile written to {}", output_path);
}

fn extract_trace_events(json: &str) -> Option<String> {
    let start = json.find("\"traceEvents\":")? + "\"traceEvents\":".len();
    let bracket_start = json[start..].find('[')? + start;
    let mut depth = 0;
    let mut bracket_end = bracket_start;
    for (i, ch) in json[bracket_start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    bracket_end = bracket_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    Some(json[bracket_start + 1..bracket_end].to_string())
}
