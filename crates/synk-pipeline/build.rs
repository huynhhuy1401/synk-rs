//! Embeds llama.cpp's GGUF converter (`vendor/llama.cpp`) into the binary,
//! so GGUF export never downloads it. Generates a list of
//! `(relative path, bytes)`.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    let root = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor/llama.cpp");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut files = Vec::new();
    walk(&root, &mut files);
    files.sort();
    let mut out = String::from("pub static FILES: &[(&str, &[u8])] = &[\n");
    for path in files {
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        out.push_str(&format!("    ({rel:?}, include_bytes!({:?})),\n", path));
    }
    out.push_str("];\n");
    let dest = Path::new(&env::var("OUT_DIR").unwrap()).join("converter_files.rs");
    fs::write(dest, out).unwrap();
}

fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n != "__pycache__") {
                walk(&path, files);
            }
        } else {
            files.push(path);
        }
    }
}
