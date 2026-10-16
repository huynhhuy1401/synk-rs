//! Ctrl-C. The first press asks the running operation to stop at its next
//! safe point (the same cancel file `synk cancel` writes; the trainer saves
//! a checkpoint first). The second press stops synk at once: the record is
//! marked cancelled here, and the trainer exits when its lifeline closes.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use synk_core::{ErrorCode, SynkError};

use crate::ops::{OpRecord, State};

static PRESSES: AtomicUsize = AtomicUsize::new(0);
static CURRENT: Mutex<Option<Current>> = Mutex::new(None);

/// The operation running in this process.
struct Current {
    dir: PathBuf,
    lock: PathBuf,
    record: Arc<Mutex<OpRecord>>,
}

/// How many times Ctrl-C was pressed.
pub fn presses() -> usize {
    PRESSES.load(Ordering::SeqCst)
}

/// Records a press and returns the count. The first asks the running
/// operation, if any, to stop at a safe point.
pub fn press() -> usize {
    let n = PRESSES.fetch_add(1, Ordering::SeqCst) + 1;
    if n == 1
        && let Some(current) = CURRENT.lock().expect("interrupt state").as_ref()
    {
        let _ = fs::write(current.dir.join("cancel"), b"");
    }
    n
}

/// Whether an operation is running in this process (so a first press can
/// stop it safely).
pub fn has_operation() -> bool {
    CURRENT.lock().expect("interrupt state").is_some()
}

pub(crate) fn register(dir: PathBuf, lock: PathBuf, record: Arc<Mutex<OpRecord>>) {
    *CURRENT.lock().expect("interrupt state") = Some(Current { dir, lock, record });
}

pub(crate) fn unregister() {
    *CURRENT.lock().expect("interrupt state") = None;
}

/// The error for a stop by Ctrl-C. `how` is `safe_stop` or `immediate`.
pub fn cancelled(id: Option<&str>, how: &str) -> SynkError {
    let what = id.unwrap_or("synk");
    let message = match how {
        "immediate" => format!("{what} was stopped at once (Ctrl-C twice)"),
        _ => format!("{what} stopped at a safe point (Ctrl-C)"),
    };
    let mut details = json!({ "how": how });
    if let Some(id) = id {
        details["id"] = json!(id);
    }
    let err = SynkError::new(ErrorCode::Cancelled, message).with_details(details);
    match id {
        Some(id) if id.starts_with("run:") => {
            err.with_next(format!("synk train {}", crate::ops::short_id(id)))
        }
        _ => err,
    }
}

/// The second press: marks the running operation cancelled now and returns
/// the error to print. The caller exits right after; the OS releases the
/// project lock.
pub fn stop_now() -> SynkError {
    let guard = CURRENT.lock().expect("interrupt state");
    let Some(current) = guard.as_ref() else {
        return cancelled(None, "immediate");
    };
    let mut record = current.record.lock().expect("op record lock");
    let err = cancelled(Some(&record.id), "immediate");
    if !record.done {
        record.state = State::Cancelled;
        record.done = true;
        record.finished_at = Some(crate::ops::now());
        record.progress = None;
        record.error = Some(serde_json::to_value(&err).expect("error serializes"));
        let _ = crate::ops::write_record(&current.dir, &record);
    }
    let _ = fs::remove_file(&current.lock);
    err
}
