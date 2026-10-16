use std::path::Path;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use synk_core::{ErrorCode, Result, SynkError};

/// A command's result, rendered for agents (`data`) and for humans (`human`).
pub struct Output {
    pub data: Value,
    pub human: String,
}

static VERBOSE: AtomicBool = AtomicBool::new(false);

/// `-v`: human output also shows IDs, metric names and advice codes.
pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
}

/// A path as a person would type it: relative to the current directory when
/// it is inside it, absolute otherwise.
pub fn show_path(path: &Path) -> String {
    let cwd = std::env::current_dir().ok();
    match cwd.as_deref().and_then(|cwd| path.strip_prefix(cwd).ok()) {
        Some(rel) if rel.as_os_str().is_empty() => ".".to_owned(),
        Some(rel) => rel.display().to_string(),
        None => path.display().to_string(),
    }
}

/// Version of the `--json` envelope. Within a version, fields and codes are
/// only ever added.
pub const CONTRACT_VERSION: u32 = 1;

/// Exit code for usage errors, as clap uses.
const USAGE_EXIT: u8 = 2;

/// Writes one line to stdout. A reader that stops early (`synk ... | head`)
/// closes the pipe; that isn't synk's error, so it's ignored, not a panic.
fn stdout_line(text: &dyn std::fmt::Display) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout().lock(), "{text}");
}

/// Prints the result. In JSON mode both success and errors go to stdout as
/// a single object, so agents only parse one stream.
pub fn print(json: bool, result: Result<Output>) -> ExitCode {
    match result {
        Ok(out) => {
            if json {
                stdout_line(&json!({ "v": CONTRACT_VERSION, "ok": true, "data": out.data }));
            } else {
                stdout_line(&out.human);
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            if json {
                stdout_line(&json!({ "v": CONTRACT_VERSION, "ok": false, "error": err }));
            } else {
                print_human_error(&err);
            }
            if err.code == ErrorCode::InvalidArgs {
                ExitCode::from(USAGE_EXIT)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

fn print_human_error(err: &SynkError) {
    eprintln!("error[{}]: {}", err.code.as_str(), err.message);
    if let Some(next) = &err.next {
        eprintln!("  next: {next}");
    }
    // Other fixes; the first action repeats `next`.
    let skip = usize::from(err.next.is_some());
    for action in err.actions.iter().skip(skip) {
        match &action.argv {
            Some(argv) => eprintln!("  {}: {}", action.why, argv.join(" ")),
            None => eprintln!("  hint: {}", action.why),
        }
    }
    if verbose() && !err.details.is_null() {
        eprintln!("  details: {}", err.details);
    }
}
