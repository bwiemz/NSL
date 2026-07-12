//! HTML/SVG report renderer for `nsl profile` (dev-tools paper, sections 3.1 and 7).
//!
//! Renders a [`ProfileReport`] as a single self-contained HTML5 document:
//! no external assets, no scripts. The centerpiece is a log-log roofline
//! plot emitted as inline SVG (memory roof, compute roof, ridge point, one
//! point per op colored by bound classification), followed by the per-op
//! cost table with a totals row, per-op roofline-utilization lines,
//! recommendations, and the optional memory timeline. Mirrors the data
//! surface of `profile::render_text`.

use nsl_codegen::cost_model::{format_data_bytes, format_flops, BoundClassification, OpCost};
use nsl_codegen::gpu_specs::GpuSpec;
use nsl_codegen::profiling::types::{
    EntryKind, MemoryTimelineEntry, ProfileReport, Severity,
};

// ---------------------------------------------------------------------------
// Plot geometry and palette
// ---------------------------------------------------------------------------

const SVG_W: f64 = 800.0;
const SVG_H: f64 = 500.0;
const M_LEFT: f64 = 70.0;
const M_RIGHT: f64 = 30.0;
const M_TOP: f64 = 40.0;
const M_BOTTOM: f64 = 60.0;

// Categorical slots 1-3 of the validated reference palette (light surface),
// assigned in fixed order; unknown is muted ink, not a series color.
const COLOR_MEMORY: &str = "#2a78d6";
const COLOR_COMPUTE: &str = "#1baf7a";
const COLOR_BALANCED: &str = "#eda100";
const COLOR_UNKNOWN: &str = "#898781";
const COLOR_SURFACE: &str = "#fcfcfb";
const COLOR_GRID: &str = "#e1e0d9";
const COLOR_AXIS: &str = "#c3c2b7";
const COLOR_MUTED: &str = "#898781";
const COLOR_SECONDARY: &str = "#52514e";
const COLOR_ROOF: &str = "#52514e";

const CSS: &str = "\
body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;\
color:#0b0b0b;max-width:920px;margin:2rem auto;padding:0 1rem;}
h1{font-size:1.4rem;margin-bottom:0.2rem;}
h2{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #e1e0d9;padding-bottom:0.3rem;}
p.meta{color:#52514e;margin-top:0;}
figure{margin:1rem 0;}
p.fine{color:#898781;font-size:0.8rem;}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:0.9rem;}
th,td{padding:5px 10px;border-bottom:1px solid #e1e0d9;text-align:left;}
th.num,td.num{text-align:right;}
tfoot td{font-weight:600;border-top:2px solid #c3c2b7;border-bottom:none;}
span.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;}
ul.util{list-style:none;padding-left:0;font-size:0.9rem;}
ul.util li{padding:2px 0;}
ol.recs li{padding:2px 0;}
li.warn{font-weight:600;}
div.approx{background:#fdf3d7;border:1px solid #eda100;border-radius:4px;\
padding:0.6rem 0.8rem;font-size:0.85rem;margin:0.8rem 0;}
div.timeline{font-size:0.85rem;font-variant-numeric:tabular-nums;}
div.tl-row{display:flex;align-items:center;gap:8px;padding:1px 0;}
span.tl-pp{width:3.5em;color:#52514e;text-align:right;flex:none;}
span.tl-track{flex:1;background:#eceae4;height:12px;border-radius:3px;overflow:hidden;}
span.tl-bar{display:block;height:100%;background:#2a78d6;}
span.tl-mb{width:6.5em;text-align:right;flex:none;}
span.tl-phase{color:#898781;}
";

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

/// Render a profile report as a complete standalone HTML5 document with an
/// inline-SVG roofline plot. No external assets and no scripts are emitted.
///
/// Caller contract: `gpu` must be the spec the report was computed against
/// (`find_gpu(&report.target_gpu)`); the header and roofs describe `gpu`.
pub fn render_html(report: &ProfileReport, gpu: &GpuSpec) -> String {
    let peak = peak_tflops_for_dtype(gpu, &report.dtype);
    let bw_tbs = gpu.peak_bandwidth_gbs / 1000.0;
    let entry = match &report.entry {
        EntryKind::Auto => "auto".to_string(),
        EntryKind::Train => "train".to_string(),
        EntryKind::Function(name) => format!("fn:{name}"),
    };

    let mut html = String::new();
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    html.push_str(
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
    );
    html.push_str(&format!(
        "<title>NSL Predictive Profile &#8211; {}</title>\n",
        escape_html(gpu.name)
    ));
    html.push_str("<style>\n");
    html.push_str(CSS);
    html.push_str("</style>\n</head>\n<body>\n");

    // 1. Header.
    html.push_str("<h1>NSL Predictive Profile</h1>\n");
    html.push_str(&format!(
        "<p class=\"meta\">Target: <strong>{}</strong> &#183; peak {peak:.0} TFLOP/s ({}) \
         &#183; HBM {bw_tbs:.2} TB/s &#183; dtype: {} &#183; entry: {}</p>\n",
        escape_html(gpu.name),
        escape_html(&report.dtype),
        escape_html(&report.dtype),
        escape_html(&entry)
    ));

    // 2. Roofline plot.
    html.push_str("<h2>Roofline</h2>\n");
    html.push_str(&render_roofline_svg(report, gpu));

    // 3. Per-op table with totals row.
    html.push_str("<h2>Per-op cost</h2>\n");
    html.push_str(&render_op_table(report));

    // 4. Per-op utilization lines.
    html.push_str("<h2>Roofline utilization</h2>\n");
    html.push_str(&render_utilization(report, gpu));

    // 5. Recommendations (only when non-empty).
    if !report.recommendations.is_empty() {
        html.push_str("<h2>Recommendations</h2>\n<ol class=\"recs\">\n");
        for rec in &report.recommendations {
            let class = match rec.severity {
                Severity::Warn => " class=\"warn\"",
                Severity::Info => "",
            };
            html.push_str(&format!(
                "<li{class}><strong>[{}]</strong> {}</li>\n",
                escape_html(&rec.code),
                escape_html(&rec.message)
            ));
        }
        html.push_str("</ol>\n");
    }

    // 6. Memory timeline (only when present). On the real-liveness path the
    //    report also carries the paper section 3.2 what-if lines, the real
    //    peak, and the unsized-var honesty counts; the text renderer prints
    //    all three, so the HTML deliverable must too.
    if let Some(tl) = &report.memory_timeline {
        let approximate = report.memory_timeline_approximate == Some(true);
        html.push_str(&render_memory_timeline(tl, approximate, report));
    }

    html.push_str("</body>\n</html>\n");
    html
}

// ---------------------------------------------------------------------------
// Escaping and small numeric helpers
// ---------------------------------------------------------------------------

/// Escape a string for safe interpolation into HTML text and attributes.
fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Replace non-finite values so no NaN or inf ever reaches the output.
fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        fallback
    }
}

/// Peak TFLOP/s for the report dtype. Mirrors the profiling walker: fp8
/// falls back to the fp16 rate on GPUs without fp8 tensor cores.
fn peak_tflops_for_dtype(gpu: &GpuSpec, dtype: &str) -> f64 {
    match dtype {
        "bf16" | "fp16" => gpu.peak_fp16_tflops,
        "fp8" => {
            if gpu.peak_fp8_tflops > 0.0 {
                gpu.peak_fp8_tflops
            } else {
                gpu.peak_fp16_tflops
            }
        }
        _ => gpu.peak_fp32_tflops,
    }
}

/// Achieved performance in TFLOP/s, or 0.0 for degenerate ops (zero FLOPs
/// or zero estimated time) so they can be skipped instead of plotted.
fn achieved_tflops(op: &OpCost) -> f64 {
    if op.flops == 0 || op.estimated_time_us <= 0.0 {
        return 0.0;
    }
    finite_or(op.flops as f64 / (op.estimated_time_us * 1e6), 0.0)
}

fn class_color(c: BoundClassification) -> &'static str {
    match c {
        BoundClassification::MemoryBound => COLOR_MEMORY,
        BoundClassification::ComputeBound => COLOR_COMPUTE,
        BoundClassification::Balanced => COLOR_BALANCED,
        BoundClassification::Unknown => COLOR_UNKNOWN,
    }
}

fn class_label(c: BoundClassification) -> &'static str {
    match c {
        BoundClassification::MemoryBound => "memory",
        BoundClassification::ComputeBound => "compute",
        BoundClassification::Balanced => "balanced",
        BoundClassification::Unknown => "unknown",
    }
}

/// Human label for a power-of-10 axis tick.
fn pow10_label(e: i32) -> String {
    if (0..=6).contains(&e) {
        format!("{}", 10u64.pow(e as u32))
    } else if (-4..0).contains(&e) {
        let mut s = String::from("0.");
        for _ in 0..(-e - 1) {
            s.push('0');
        }
        s.push('1');
        s
    } else {
        format!("1e{e}")
    }
}

// ---------------------------------------------------------------------------
// Roofline SVG
// ---------------------------------------------------------------------------

struct PlotPoint<'a> {
    name: &'a str,
    ai: f64,
    perf: f64,
    time_us: f64,
    class: BoundClassification,
}

fn render_roofline_svg(report: &ProfileReport, gpu: &GpuSpec) -> String {
    let peak = peak_tflops_for_dtype(gpu, &report.dtype).max(1e-9);
    let bw_tbs = (gpu.peak_bandwidth_gbs / 1000.0).max(1e-9);
    // Ridge point: AI where the memory roof (perf = AI * bandwidth) meets
    // the compute roof (perf = peak).
    let ridge_ai = (peak / bw_tbs).max(1e-9);

    // Plottable points: positive AI and positive achieved perf only.
    // Zero-FLOP / zero-time ops are counted and footnoted, never plotted,
    // so no NaN or non-finite coordinate can be produced.
    let mut pts: Vec<PlotPoint> = Vec::new();
    let mut skipped = 0usize;
    for op in &report.ops {
        let perf = achieved_tflops(op);
        if op.arithmetic_intensity > 0.0 && op.arithmetic_intensity.is_finite() && perf > 0.0 {
            pts.push(PlotPoint {
                name: &op.name,
                ai: op.arithmetic_intensity,
                perf,
                time_us: op.estimated_time_us,
                class: op.classification,
            });
        } else {
            skipped += 1;
        }
    }
    // Largest ops first: they get drawn first (small dots overpaint big
    // ones) and the top three get direct labels.
    pts.sort_by(|a, b| {
        b.time_us
            .partial_cmp(&a.time_us)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Log-decade domain covering all points plus the ridge, with at least
    // one decade of roof visible on each side of the ridge.
    let mut ai_lo = ridge_ai;
    let mut ai_hi = ridge_ai;
    let mut perf_lo = peak;
    for p in &pts {
        ai_lo = ai_lo.min(p.ai);
        ai_hi = ai_hi.max(p.ai);
        perf_lo = perf_lo.min(p.perf);
    }
    let mut x_lo_e = ai_lo.log10().floor() as i32;
    let mut x_hi_e = ai_hi.log10().ceil() as i32;
    if 10f64.powi(x_lo_e) >= ridge_ai * 0.99 {
        x_lo_e -= 1;
    }
    if 10f64.powi(x_hi_e) <= ridge_ai * 1.01 {
        x_hi_e += 1;
    }
    x_lo_e = x_lo_e.max(x_hi_e - 8);
    if x_hi_e <= x_lo_e {
        x_hi_e = x_lo_e + 1;
    }

    let mut y_hi_e = peak.log10().ceil() as i32;
    if 10f64.powi(y_hi_e) <= peak * 1.01 {
        y_hi_e += 1;
    }
    // Keep the memory roof's left end in view, but never more than six
    // decades of vertical range.
    perf_lo = perf_lo.min(bw_tbs * 10f64.powi(x_lo_e)).max(1e-12);
    let mut y_lo_e = (perf_lo.log10().floor() as i32).max(y_hi_e - 6);
    if y_lo_e >= y_hi_e {
        y_lo_e = y_hi_e - 1;
    }

    let x_min = 10f64.powi(x_lo_e);
    let x_max = 10f64.powi(x_hi_e);
    let y_min = 10f64.powi(y_lo_e);
    let y_max = 10f64.powi(y_hi_e);
    let plot_w = SVG_W - M_LEFT - M_RIGHT;
    let plot_h = SVG_H - M_TOP - M_BOTTOM;
    let x_span = f64::from(x_hi_e - x_lo_e);
    let y_span = f64::from(y_hi_e - y_lo_e);
    // Out-of-range values are clamped to the plot edge, never dropped as
    // bad coordinates.
    let px = |ai: f64| -> f64 {
        let v = finite_or(ai, x_min).clamp(x_min, x_max);
        M_LEFT + (v.log10() - f64::from(x_lo_e)) / x_span * plot_w
    };
    let py = |perf: f64| -> f64 {
        let v = finite_or(perf, y_min).clamp(y_min, y_max);
        M_TOP + (f64::from(y_hi_e) - v.log10()) / y_span * plot_h
    };

    let mut s = String::new();
    s.push_str(&format!(
        "<figure>\n<svg viewBox=\"0 0 {SVG_W} {SVG_H}\" width=\"{SVG_W}\" height=\"{SVG_H}\" \
         xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" \
         aria-label=\"Roofline plot for {}\">\n",
        escape_html(gpu.name)
    ));
    s.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{SVG_W}\" height=\"{SVG_H}\" fill=\"{COLOR_SURFACE}\"/>\n"
    ));

    // Decade gridlines and tick labels.
    for e in x_lo_e..=x_hi_e {
        let x = px(10f64.powi(e));
        s.push_str(&format!(
            "<line x1=\"{x:.1}\" y1=\"{M_TOP}\" x2=\"{x:.1}\" y2=\"{:.1}\" \
             stroke=\"{COLOR_GRID}\" stroke-width=\"1\"/>\n",
            M_TOP + plot_h
        ));
        s.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{COLOR_MUTED}\" \
             text-anchor=\"middle\">{}</text>\n",
            M_TOP + plot_h + 16.0,
            pow10_label(e)
        ));
    }
    for e in y_lo_e..=y_hi_e {
        let y = py(10f64.powi(e));
        s.push_str(&format!(
            "<line x1=\"{M_LEFT}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" \
             stroke=\"{COLOR_GRID}\" stroke-width=\"1\"/>\n",
            M_LEFT + plot_w
        ));
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{COLOR_MUTED}\" \
             text-anchor=\"end\">{}</text>\n",
            M_LEFT - 8.0,
            y + 4.0,
            pow10_label(e)
        ));
    }

    // Plot frame.
    s.push_str(&format!(
        "<rect x=\"{M_LEFT}\" y=\"{M_TOP}\" width=\"{plot_w}\" height=\"{plot_h}\" \
         fill=\"none\" stroke=\"{COLOR_AXIS}\" stroke-width=\"1\"/>\n"
    ));

    // Roofs: memory roof (slope 1 in log-log) up to the ridge, then the
    // horizontal compute roof. The memory roof is clipped where it enters
    // the plot from below.
    let roof_start_ai = x_min.max(y_min / bw_tbs);
    let (rx0, ry0) = (px(roof_start_ai), py(bw_tbs * roof_start_ai));
    let (rx1, ry1) = (px(ridge_ai), py(peak));
    let (rx2, ry2) = (px(x_max), py(peak));
    s.push_str(&format!(
        "<path d=\"M {rx0:.1} {ry0:.1} L {rx1:.1} {ry1:.1} L {rx2:.1} {ry2:.1}\" \
         fill=\"none\" stroke=\"{COLOR_ROOF}\" stroke-width=\"2\"/>\n"
    ));
    s.push_str(&format!(
        "<line x1=\"{rx1:.1}\" y1=\"{ry1:.1}\" x2=\"{rx1:.1}\" y2=\"{:.1}\" \
         stroke=\"{COLOR_AXIS}\" stroke-width=\"1\" stroke-dasharray=\"4 3\"/>\n",
        M_TOP + plot_h
    ));
    s.push_str(&format!(
        "<circle cx=\"{rx1:.1}\" cy=\"{ry1:.1}\" r=\"3.5\" fill=\"{COLOR_ROOF}\"/>\n"
    ));
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{COLOR_SECONDARY}\">\
         ridge {ridge_ai:.1} FLOP/B</text>\n",
        rx1 + 7.0,
        ry1 + 15.0
    ));
    // Direct labels on the roofs.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" fill=\"{COLOR_SECONDARY}\" \
         text-anchor=\"end\">Memory roof</text>\n",
        (rx0 + rx1) / 2.0,
        (ry0 + ry1) / 2.0 - 8.0
    ));
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" fill=\"{COLOR_SECONDARY}\" \
         text-anchor=\"middle\">Compute roof</text>\n",
        (rx1 + rx2) / 2.0,
        ry1 - 8.0
    ));

    // Op points, largest first, each with a tooltip.
    for p in &pts {
        let x = px(p.ai);
        let y = py(p.perf);
        s.push_str(&format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"5\" fill=\"{}\" \
             stroke=\"{COLOR_SURFACE}\" stroke-width=\"2\">\
             <title>{}: AI={:.2} FLOP/B, {:.2} TFLOP/s, {:.2} us ({})</title>\
             </circle>\n",
            class_color(p.class),
            escape_html(p.name),
            p.ai,
            p.perf,
            p.time_us,
            class_label(p.class)
        ));
    }
    // Direct labels for the three largest ops by predicted time.
    for p in pts.iter().take(3) {
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"11\" fill=\"{COLOR_SECONDARY}\">{}</text>\n",
            px(p.ai) + 8.0,
            py(p.perf) - 7.0,
            escape_html(p.name)
        ));
    }

    // Legend (top-left of the plot area, above the roof lines' usual path).
    let has_balanced = pts
        .iter()
        .any(|p| p.class == BoundClassification::Balanced);
    let has_unknown = pts.iter().any(|p| p.class == BoundClassification::Unknown);
    let mut legend: Vec<(Option<&'static str>, String)> = vec![
        (None, format!("Memory roof: {bw_tbs:.2} TB/s")),
        (
            None,
            format!(
                "Compute roof: {peak:.0} TFLOP/s ({})",
                escape_html(&report.dtype)
            ),
        ),
        (Some(COLOR_MEMORY), "memory-bound op".to_string()),
        (Some(COLOR_COMPUTE), "compute-bound op".to_string()),
    ];
    if has_balanced {
        legend.push((Some(COLOR_BALANCED), "balanced op".to_string()));
    }
    if has_unknown {
        legend.push((Some(COLOR_UNKNOWN), "unclassified op".to_string()));
    }
    let lx = M_LEFT + 12.0;
    let box_h = legend.len() as f64 * 17.0 + 10.0;
    s.push_str(&format!(
        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"230\" height=\"{box_h:.1}\" \
         fill=\"{COLOR_SURFACE}\" fill-opacity=\"0.92\" stroke=\"{COLOR_GRID}\"/>\n",
        lx - 6.0,
        M_TOP + 6.0
    ));
    let mut ly = M_TOP + 22.0;
    for (swatch, label) in &legend {
        match swatch {
            Some(color) => s.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\" fill=\"{color}\" \
                 stroke=\"{COLOR_SURFACE}\" stroke-width=\"1.5\"/>\n",
                lx + 11.0,
                ly - 4.0
            )),
            None => s.push_str(&format!(
                "<line x1=\"{lx:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                 stroke=\"{COLOR_ROOF}\" stroke-width=\"2\"/>\n",
                ly - 4.0,
                lx + 22.0,
                ly - 4.0
            )),
        }
        s.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{ly:.1}\" font-size=\"12\" fill=\"#0b0b0b\">{label}</text>\n",
            lx + 30.0
        ));
        ly += 17.0;
    }

    // Axis titles.
    s.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" fill=\"{COLOR_SECONDARY}\" \
         text-anchor=\"middle\">Arithmetic intensity (FLOP/byte)</text>\n",
        M_LEFT + plot_w / 2.0,
        SVG_H - 18.0
    ));
    s.push_str(&format!(
        "<text x=\"18\" y=\"{:.1}\" font-size=\"12\" fill=\"{COLOR_SECONDARY}\" \
         text-anchor=\"middle\" transform=\"rotate(-90 18 {:.1})\">\
         Attainable performance (TFLOP/s)</text>\n",
        M_TOP + plot_h / 2.0,
        M_TOP + plot_h / 2.0
    ));

    s.push_str("</svg>\n");
    if skipped > 0 {
        s.push_str(&format!(
            "<p class=\"fine\">{skipped} op(s) without a plottable roofline position \
             (zero FLOPs, zero bytes, or zero estimated time) are omitted; \
             see the table below.</p>\n"
        ));
    }
    s.push_str("</figure>\n");
    s
}

// ---------------------------------------------------------------------------
// Per-op table
// ---------------------------------------------------------------------------

fn render_op_table(report: &ProfileReport) -> String {
    let mut s = String::from(
        "<table>\n<thead><tr><th>Op</th><th class=\"num\">FLOPs</th>\
         <th class=\"num\">HBM bytes</th><th class=\"num\">AI</th>\
         <th>Bound</th><th class=\"num\">Predicted time</th></tr></thead>\n<tbody>\n",
    );
    for op in &report.ops {
        let fused = if op.fused { " (fused)" } else { "" };
        let ai = finite_or(op.arithmetic_intensity, 0.0);
        let time = finite_or(op.estimated_time_us, 0.0);
        s.push_str(&format!(
            "<tr><td>{}{fused}</td><td class=\"num\">{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{ai:.2}</td>\
             <td><span class=\"dot\" style=\"background:{}\"></span>{}</td>\
             <td class=\"num\">{time:.2} us</td></tr>\n",
            escape_html(&op.name),
            format_flops(op.flops),
            format_data_bytes(op.bytes_read + op.bytes_written),
            class_color(op.classification),
            class_label(op.classification)
        ));
    }
    let avg_ai = if report.total_hbm_bytes > 0 {
        report.total_flops as f64 / report.total_hbm_bytes as f64
    } else {
        0.0
    };
    let total_us = finite_or(report.total_estimated_us, 0.0);
    s.push_str(&format!(
        "</tbody>\n<tfoot><tr><td>Total</td><td class=\"num\">{}</td>\
         <td class=\"num\">{}</td><td class=\"num\">{avg_ai:.2} avg</td><td></td>\
         <td class=\"num\">{total_us:.2} us</td></tr></tfoot>\n</table>\n",
        format_flops(report.total_flops),
        format_data_bytes(report.total_hbm_bytes)
    ));
    s
}

// ---------------------------------------------------------------------------
// Per-op roofline utilization lines
// ---------------------------------------------------------------------------

fn render_utilization(report: &ProfileReport, gpu: &GpuSpec) -> String {
    let peak = peak_tflops_for_dtype(gpu, &report.dtype).max(1e-9);
    let bw_tbs = (gpu.peak_bandwidth_gbs / 1000.0).max(1e-9);

    let mut s = String::from("<ul class=\"util\">\n");
    for op in &report.ops {
        let name = escape_html(&op.name);
        let ai = finite_or(op.arithmetic_intensity, 0.0);
        if op.flops == 0 {
            s.push_str(&format!(
                "<li><span class=\"dot\" style=\"background:{}\"></span>\
                 {name}: 0 FLOPs, pure data movement &#8212; not on the roofline</li>\n",
                class_color(op.classification)
            ));
            continue;
        }
        let perf = achieved_tflops(op);
        if ai <= 0.0 || perf <= 0.0 {
            // Positive FLOPs but a degenerate AI (zero bytes) or a zero /
            // non-finite time estimate: no roof position exists.
            s.push_str(&format!(
                "<li><span class=\"dot\" style=\"background:{}\"></span>\
                 {name}: utilization unavailable (degenerate cost estimate)</li>\n",
                class_color(op.classification)
            ));
            continue;
        }
        // The governing roof at this AI is the lower of the two roofs.
        let mem_roof = ai * bw_tbs;
        let (roof, roof_name) = if mem_roof < peak {
            (mem_roof, "memory roof")
        } else {
            (peak, "compute roof")
        };
        let util_pct = finite_or(perf / roof * 100.0, 0.0);
        let verdict = match op.classification {
            BoundClassification::MemoryBound => "bottleneck",
            BoundClassification::ComputeBound => "efficient",
            BoundClassification::Balanced => "balanced",
            BoundClassification::Unknown => "unclassified",
        };
        s.push_str(&format!(
            "<li><span class=\"dot\" style=\"background:{}\"></span>\
             {name}: AI={ai:.2}, {util_pct:.0}% of {roof_name} &#8212; {verdict}</li>\n",
            class_color(op.classification)
        ));
    }
    s.push_str("</ul>\n");
    s
}

// ---------------------------------------------------------------------------
// Memory timeline
// ---------------------------------------------------------------------------

fn render_memory_timeline(
    tl: &[MemoryTimelineEntry],
    approximate: bool,
    report: &ProfileReport,
) -> String {
    // Prefer the real-liveness peak (which pins params+grads live to the
    // end) when present; otherwise the bar max is the only peak we have.
    let bar_max = tl.iter().map(|e| e.live_bytes).max().unwrap_or(0);
    let peak = report.memory_peak_bytes.unwrap_or(bar_max);
    let mut s = String::from("<h2>Memory timeline</h2>\n");
    if approximate {
        // Same warning wording as the text renderer.
        s.push_str(
            "<div class=\"approx\">NOTE: APPROXIMATE timeline &#8212; synthesized from a \
             fixed 2-step activation-lifetime heuristic, not real liveness analysis. For \
             training programs the true peak (saved-for-backward activations) is typically \
             much higher than shown.</div>\n",
        );
    }
    s.push_str("<div class=\"timeline\">\n");
    for e in tl {
        let pct = if peak == 0 {
            0.0
        } else {
            finite_or(e.live_bytes as f64 / peak as f64 * 100.0, 0.0)
        };
        let mb = e.live_bytes as f64 / (1024.0 * 1024.0);
        let phase = e
            .phase
            .as_deref()
            .map(escape_html)
            .unwrap_or_default();
        s.push_str(&format!(
            "<div class=\"tl-row\"><span class=\"tl-pp\">{}</span>\
             <span class=\"tl-track\"><span class=\"tl-bar\" style=\"width:{pct:.1}%\"></span></span>\
             <span class=\"tl-mb\">{mb:.1} MB</span>\
             <span class=\"tl-phase\">{phase}</span></div>\n",
            e.program_point
        ));
    }
    s.push_str("</div>\n");
    s.push_str(&format!(
        "<p class=\"fine\">Peak: {:.1} MB</p>\n",
        peak as f64 / (1024.0 * 1024.0)
    ));
    // Paper section 3.2 what-if lines ("With FASE: peak drops to ...").
    if let Some(what_if) = &report.memory_what_if {
        if !what_if.is_empty() {
            s.push_str("<ul class=\"whatif\">\n");
            for w in what_if {
                s.push_str(&format!(
                    "<li>{}: peak drops to {:.1} MB <span class=\"fine\">({})</span></li>\n",
                    escape_html(&w.label),
                    w.peak_bytes as f64 / (1024.0 * 1024.0),
                    escape_html(&w.note),
                ));
            }
            s.push_str("</ul>\n");
        }
    }
    // Lower-bound honesty note: some vars had no concrete compile-time shape,
    // so their bytes are absent from the MB figures. Same contract as the
    // text renderer, which must never be dropped from the HTML deliverable.
    if let (Some(unsized_vars), Some(total_vars)) =
        (report.memory_unsized_vars, report.memory_total_vars)
    {
        if unsized_vars > 0 {
            s.push_str(&format!(
                "<div class=\"approx\">NOTE: {unsized_vars} of {total_vars} vars have no \
                 concrete compile-time shape; their bytes are NOT included &#8212; treat \
                 the MB figures as a lower bound.</div>\n",
            ));
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_codegen::gpu_specs::find_gpu;
    use nsl_codegen::profiling::types::Recommendation;

    fn op(
        name: &str,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
        class: BoundClassification,
        time_us: f64,
    ) -> OpCost {
        let total = bytes_read + bytes_written;
        let ai = if total == 0 {
            0.0
        } else {
            flops as f64 / total as f64
        };
        OpCost {
            name: name.to_string(),
            loc: "test.nsl:1".to_string(),
            input_shapes: vec!["[8,1024]".to_string()],
            output_shape: "[8,1024]".to_string(),
            flops,
            bytes_read,
            bytes_written,
            arithmetic_intensity: ai,
            classification: class,
            fused: false,
            estimated_time_us: time_us,
            origin_node: None,
        }
    }

    fn report(ops: Vec<OpCost>) -> ProfileReport {
        let total_flops = ops.iter().map(|o| o.flops).sum();
        let total_hbm_bytes = ops.iter().map(|o| o.bytes_read + o.bytes_written).sum();
        let total_estimated_us = ops.iter().map(|o| o.estimated_time_us).sum();
        ProfileReport {
            target_gpu: "A100-SXM".to_string(),
            dtype: "fp16".to_string(),
            entry: EntryKind::Auto,
            ops,
            total_flops,
            total_hbm_bytes,
            total_estimated_us,
            fusion: None,
            memory_timeline: None,
            memory_timeline_approximate: None,
            memory_what_if: None,
            memory_peak_bytes: None,
            memory_unsized_vars: None,
            memory_total_vars: None,
            recommendations: vec![],
            wggo_explain: None,
        }
    }

    /// Standard 3-op fixture: memory-bound matmul, compute-bound attention,
    /// and a zero-FLOP embedding lookup (the degenerate case).
    fn sample_report() -> ProfileReport {
        report(vec![
            op(
                "gate_proj",
                23_100_000,
                11_000_000,
                500_000,
                BoundClassification::MemoryBound,
                3.4,
            ),
            op(
                "flash_attn",
                67_100_000,
                400_000,
                100_000,
                BoundClassification::ComputeBound,
                0.07,
            ),
            op(
                "embed",
                0,
                4_000_000,
                4_000_000,
                BoundClassification::MemoryBound,
                3.9,
            ),
        ])
    }

    fn gpu() -> &'static GpuSpec {
        find_gpu("A100-SXM").unwrap()
    }

    fn attr_values(html: &str, attr: &str) -> Vec<f64> {
        let pat = format!("{attr}=\"");
        let mut vals = Vec::new();
        let mut rest = html;
        while let Some(i) = rest.find(&pat) {
            rest = &rest[i + pat.len()..];
            let end = rest.find('"').expect("unterminated attribute");
            vals.push(rest[..end].parse::<f64>().expect("non-numeric attribute"));
            rest = &rest[end..];
        }
        vals
    }

    #[test]
    fn doctype_and_svg_present() {
        let html = render_html(&sample_report(), gpu());
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("</svg>"));
        assert!(html.ends_with("</html>\n"));
    }

    #[test]
    fn roof_legend_labels_present() {
        let html = render_html(&sample_report(), gpu());
        assert!(html.contains("Memory roof"));
        assert!(html.contains("Compute roof"));
        assert!(html.contains("ridge"));
    }

    #[test]
    fn op_names_present() {
        let html = render_html(&sample_report(), gpu());
        for name in ["gate_proj", "flash_attn", "embed"] {
            assert!(html.contains(name), "missing op name {name}");
        }
    }

    #[test]
    fn no_nan_or_inf_with_zero_flop_op() {
        let html = render_html(&sample_report(), gpu());
        assert!(!html.contains("NaN"));
        assert!(!html.contains("inf"));
        assert!(!html.contains("Inf"));

        // Also with a zero-time op and an empty report.
        let mut r = sample_report();
        r.ops
            .push(op("instant", 1000, 0, 0, BoundClassification::Unknown, 0.0));
        let html = render_html(&r, gpu());
        assert!(!html.contains("NaN") && !html.contains("inf") && !html.contains("Inf"));

        let html = render_html(&report(vec![]), gpu());
        assert!(!html.contains("NaN") && !html.contains("inf") && !html.contains("Inf"));
    }

    #[test]
    fn svg_point_coordinates_finite_and_in_bounds() {
        let html = render_html(&sample_report(), gpu());
        let cxs = attr_values(&html, "cx");
        let cys = attr_values(&html, "cy");
        assert!(!cxs.is_empty() && !cys.is_empty());
        for v in &cxs {
            assert!(v.is_finite() && (0.0..=SVG_W).contains(v), "cx out of range: {v}");
        }
        for v in &cys {
            assert!(v.is_finite() && (0.0..=SVG_H).contains(v), "cy out of range: {v}");
        }
    }

    #[test]
    fn escape_helper_escapes_all_specials() {
        assert_eq!(escape_html("&<>\"'"), "&amp;&lt;&gt;&quot;&#39;");
        assert_eq!(escape_html("plain_name"), "plain_name");
    }

    #[test]
    fn op_names_are_escaped_in_output() {
        let mut r = sample_report();
        r.ops[0].name = "q<proj>&\"x\"".to_string();
        let html = render_html(&r, gpu());
        assert!(html.contains("q&lt;proj&gt;&amp;&quot;x&quot;"));
        assert!(!html.contains("q<proj>"));
    }

    #[test]
    fn totals_row_present() {
        let html = render_html(&sample_report(), gpu());
        assert!(html.contains("<tfoot>"));
        assert!(html.contains("Total"));
        // 23.1M + 67.1M + 0 FLOPs and 20.0M total HBM bytes.
        assert!(html.contains("90.2M"));
        assert!(html.contains("20.0M"));
    }

    #[test]
    fn approximate_warning_shown_iff_flag_set() {
        let tl = vec![
            MemoryTimelineEntry {
                program_point: 0,
                live_bytes: 100 * 1024 * 1024,
                phase: Some("forward".to_string()),
            },
            MemoryTimelineEntry {
                program_point: 1,
                live_bytes: 250 * 1024 * 1024,
                phase: None,
            },
        ];

        let mut r = sample_report();
        r.memory_timeline = Some(tl.clone());
        r.memory_timeline_approximate = Some(true);
        let html = render_html(&r, gpu());
        assert!(html.contains("APPROXIMATE"));

        r.memory_timeline_approximate = Some(false);
        let html = render_html(&r, gpu());
        assert!(!html.contains("APPROXIMATE"));

        r.memory_timeline_approximate = None;
        let html = render_html(&r, gpu());
        assert!(!html.contains("APPROXIMATE"));
    }

    #[test]
    fn memory_timeline_and_recommendations_sections_conditional() {
        // Absent by default.
        let html = render_html(&sample_report(), gpu());
        assert!(!html.contains("Memory timeline"));
        assert!(!html.contains("<h2>Recommendations</h2>"));

        // Present when populated.
        let mut r = sample_report();
        r.memory_timeline = Some(vec![MemoryTimelineEntry {
            program_point: 0,
            live_bytes: 42 * 1024 * 1024,
            phase: Some("forward".to_string()),
        }]);
        r.recommendations = vec![Recommendation::memory_bound_batch_hint("gate_proj")];
        let html = render_html(&r, gpu());
        assert!(html.contains("Memory timeline"));
        assert!(html.contains("42.0 MB"));
        assert!(html.contains("forward"));
        assert!(html.contains("<h2>Recommendations</h2>"));
        assert!(html.contains("[R01]"));
        // The R01 message body itself, not the always-present legend label.
        assert!(html.contains("consider increasing batch size"));
    }

    #[test]
    fn utilization_lines_present() {
        let html = render_html(&sample_report(), gpu());
        // gate_proj sits under the memory roof; embed is footnoted as pure
        // data movement.
        assert!(html.contains("% of memory roof"));
        assert!(html.contains("bottleneck"));
        assert!(html.contains("pure data movement"));
    }

    /// The real-liveness path carries what-if lines, a real peak, and the
    /// unsized-var lower-bound note; the HTML deliverable must render all
    /// three (regression for the review finding where they were dropped).
    #[test]
    fn html_carries_real_liveness_what_if_and_lower_bound_note() {
        use nsl_codegen::profiling::real_timeline::WhatIfPeak;
        let mut r = sample_report();
        r.memory_timeline = Some(vec![MemoryTimelineEntry {
            program_point: 0,
            live_bytes: 6 * 1024 * 1024,
            phase: Some("backward begins".to_string()),
        }]);
        r.memory_timeline_approximate = Some(false);
        r.memory_peak_bytes = Some(6 * 1024 * 1024);
        r.memory_what_if = Some(vec![WhatIfPeak {
            label: "With FASE".to_string(),
            peak_bytes: 5 * 1024 * 1024,
            note: "parameter-gradient buffers freed per layer".to_string(),
        }]);
        r.memory_unsized_vars = Some(11);
        r.memory_total_vars = Some(23);
        let html = render_html(&r, gpu());
        assert!(html.contains("With FASE"));
        assert!(html.contains("peak drops to 5.0 MB"));
        assert!(html.contains("11 of 23 vars"));
        assert!(html.contains("lower bound"));
        // Real path never shows the approximate banner.
        assert!(!html.contains("APPROXIMATE"));
        assert!(!html.contains("NaN"));
    }

    /// No what-if lines and no lower-bound note when the report doesn't
    /// carry them (approximate/synthetic path).
    #[test]
    fn html_omits_what_if_when_absent() {
        let mut r = sample_report();
        r.memory_timeline = Some(vec![MemoryTimelineEntry {
            program_point: 0,
            live_bytes: 1024,
            phase: None,
        }]);
        let html = render_html(&r, gpu());
        assert!(!html.contains("peak drops to"));
        assert!(!html.contains("lower bound"));
    }
}
