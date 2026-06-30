//! `nsl init` — new project scaffolding.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::process;

pub(crate) fn run_init(name: &str) {
    let root = std::path::Path::new(name);

    // Refuse to overwrite an existing directory
    if root.exists() {
        eprintln!("error: directory '{}' already exists", name);
        process::exit(1);
    }

    // Create project root and sub-directories
    for dir in &[root.to_path_buf(), root.join("data"), root.join("weights")] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: could not create directory '{}': {e}", dir.display());
            process::exit(1);
        }
    }

    // main.nsl — tensor example
    let main_nsl = "\
# My first NeuralScript program

let x = zeros([2, 3])
let y = ones([2, 3])
let z = x + y

print(z)
print(f\"Sum: {z.sum()}\")
";

    // nsl.toml
    let nsl_toml = format!(
        "\
# NeuralScript Project Configuration
# Reserved for future use by the NSL package manager (v0.2)

[project]
name = \"{name}\"
version = \"0.1.0\"
entry = \"main.nsl\"
"
    );

    // .gitignore
    let gitignore = "\
# Build
*.exe
*.o
.nsl-cache/

# ML artifacts
*.safetensors
*.bin
*.nslm
/weights/
/data/
";

    let files: &[(&str, &str)] = &[
        ("main.nsl", main_nsl),
        ("nsl.toml", &nsl_toml),
        (".gitignore", gitignore),
    ];

    for (filename, contents) in files {
        let path = root.join(filename);
        if let Err(e) = std::fs::write(&path, contents) {
            eprintln!("error: could not write '{}': {e}", path.display());
            process::exit(1);
        }
    }

    println!("Created project '{name}'. Run: cd {name} && nsl run main.nsl");
}
