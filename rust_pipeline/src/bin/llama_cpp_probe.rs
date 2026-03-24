// Standalone GGUF loader probe for the Rust llama_cpp backend.
//
// Prints basic file diagnostics and attempts model loading with multiple
// parameter profiles to help isolate load failures.

#[cfg(feature = "llama_cpp_backend")]
use std::fs::File;
#[cfg(feature = "llama_cpp_backend")]
use std::io::Read;
#[cfg(feature = "llama_cpp_backend")]
use std::path::Path;

#[cfg(feature = "llama_cpp_backend")]
use llama_cpp::{LlamaModel, LlamaParams};

#[cfg(feature = "llama_cpp_backend")]
fn print_header(path: &Path) {
    let mut buf = [0u8; 16];
    match File::open(path).and_then(|mut f| f.read_exact(&mut buf).map(|_| ())) {
        Ok(()) => {
            let hex = buf
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect::<Vec<String>>()
                .join(" ");
            println!("Header[0..16]: {}", hex);
            if &buf[0..4] == b"GGUF" {
                println!("Magic: GGUF (OK)");
            } else {
                println!("Magic: not GGUF (unexpected)");
            }
        }
        Err(e) => {
            eprintln!("Could not read file header: {}", e);
        }
    }
}

#[cfg(feature = "llama_cpp_backend")]
fn try_load(label: &str, params: LlamaParams, model_path: &str) -> bool {
    println!("\nAttempt: {}", label);
    println!(
        "  n_gpu_layers={}, use_mmap={}, use_mlock={}",
        params.n_gpu_layers, params.use_mmap, params.use_mlock
    );

    match LlamaModel::load_from_file(model_path, params) {
        Ok(model) => {
            println!("  Result: SUCCESS");
            let _ = model;
            true
        }
        Err(e) => {
            eprintln!("  Result: FAIL -> {:?}", e);
            false
        }
    }
}

#[cfg(feature = "llama_cpp_backend")]
fn main() {
    let model_path = std::env::args().nth(1).or_else(|| std::env::var("GGUF_MODEL_PATH").ok());
    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Usage: llama_cpp_probe <path-to-model.gguf>");
            eprintln!("   or: set GGUF_MODEL_PATH and run without args");
            std::process::exit(2);
        }
    };

    let path = Path::new(&model_path);
    println!("llama_cpp probe");
    println!("Model path: {}", model_path);

    if !path.exists() {
        eprintln!("ERROR: file not found");
        std::process::exit(1);
    }
    if !path.is_file() {
        eprintln!("ERROR: path is not a regular file");
        std::process::exit(1);
    }

    match std::fs::metadata(path) {
        Ok(meta) => println!("File size: {} bytes", meta.len()),
        Err(e) => eprintln!("Could not read metadata: {}", e),
    }

    print_header(path);

    let mut success = false;

    success |= try_load("default LlamaParams", LlamaParams::default(), &model_path);

    let mut conservative = LlamaParams::default();
    conservative.n_gpu_layers = 0;
    conservative.use_mmap = false;
    conservative.use_mlock = false;
    success |= try_load("conservative CPU params", conservative, &model_path);

    if success {
        println!("\nProbe finished: at least one load attempt succeeded.");
        std::process::exit(0);
    }

    eprintln!("\nProbe finished: all load attempts failed.");
    std::process::exit(1);
}

#[cfg(not(feature = "llama_cpp_backend"))]
fn main() {
    eprintln!("llama_cpp_probe requires feature 'llama_cpp_backend'.");
    eprintln!("Run: cargo run --release --manifest-path rust_pipeline/Cargo.toml --features llama_cpp_backend --bin llama_cpp_probe -- <path-to-model.gguf>");
    std::process::exit(2);
}
