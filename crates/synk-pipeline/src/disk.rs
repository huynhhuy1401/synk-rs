//! Free disk space, checked before a run downloads or trains: what each
//! step will write, where, against what each drive has free.
//!
//! The sizes are estimates **[unverified: measure on real runs]**: a
//! download's size times how much it grows when unpacked, and training
//! checkpoints from the model's parameter count.

use std::path::{Path, PathBuf};

use serde::Serialize;

/// Warn (`DISK_SPACE_TIGHT`) when less than this would be left.
pub const TIGHT_MARGIN: u64 = 2_000_000_000;
/// The trainer environment is about this many times its download once
/// installed (PyTorch's wheels are compressed).
pub const TRAINER_UNPACKED: u64 = 3;
/// A dataset is about this many times its documents' text (the corpus of
/// chunks, which overlap, plus the queries and their cache), plus
/// DATASET_FIXED_BYTES. **[unverified]**
pub const DATASET_PER_TEXT_BYTE: u64 = 2;
pub const DATASET_FIXED_BYTES: u64 = 1_000_000;
/// Bytes per parameter a full fine-tuning run keeps: the fp32 model in the
/// checkpoint, AdamW's two moments, the best epoch's copy and the final
/// model.
pub const FULL_BYTES_PER_PARAM: u64 = 20;
/// LoRA keeps the merged best and final models; the adapter is small.
pub const LORA_BYTES_PER_PARAM: u64 = 8;

/// One thing a run will write.
#[derive(Debug, Clone, Serialize)]
pub struct Need {
    pub what: String,
    pub path: PathBuf,
    pub bytes: u64,
}

/// One drive: what it has free and what the run needs on it.
#[derive(Debug, Clone, Serialize)]
pub struct Volume {
    /// The first place on this drive the run writes to.
    pub path: PathBuf,
    pub free_bytes: u64,
    pub need_bytes: u64,
    pub needs: Vec<Need>,
}

impl Volume {
    pub fn short(&self) -> bool {
        self.free_bytes < self.need_bytes
    }

    pub fn tight(&self) -> bool {
        !self.short() && self.free_bytes < self.need_bytes.saturating_add(TIGHT_MARGIN)
    }
}

/// Whether `SYNK_DISK_CHECK=warn` turned a too-small drive into a warning.
pub fn warn_only() -> bool {
    std::env::var("SYNK_DISK_CHECK").as_deref() == Ok("warn")
}

/// Bytes for people, e.g. `2.5 GB`.
pub fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// Groups the needs by drive and reads each drive's free space. A place
/// whose drive can't be read is left out (no false alarms).
pub fn check(needs: Vec<Need>) -> Vec<Volume> {
    let mut volumes: Vec<(String, Volume)> = Vec::new();
    for need in needs.into_iter().filter(|n| n.bytes > 0) {
        let (Some((free, _)), Some(id)) = (space(&need.path), drive(&need.path)) else {
            continue;
        };
        match volumes.iter_mut().find(|(v, _)| *v == id) {
            Some((_, volume)) => {
                volume.need_bytes += need.bytes;
                volume.needs.push(need);
            }
            None => volumes.push((
                id,
                Volume {
                    path: need.path.clone(),
                    free_bytes: free,
                    need_bytes: need.bytes,
                    needs: vec![need],
                },
            )),
        }
    }
    volumes.into_iter().map(|(_, v)| v).collect()
}

/// Which drive holds `path` (or its closest existing parent): the device
/// on Unix, the drive letter or share on Windows. Free space can't tell
/// drives apart: it changes while synk reads it.
fn drive(path: &Path) -> Option<String> {
    let existing = path.ancestors().find(|p| p.exists())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(std::fs::metadata(existing).ok()?.dev().to_string())
    }
    #[cfg(not(unix))]
    {
        let absolute = std::path::absolute(existing).ok()?;
        match absolute.components().next()? {
            std::path::Component::Prefix(p) => Some(p.as_os_str().to_string_lossy().to_uppercase()),
            _ => Some(String::new()),
        }
    }
}

/// Free (to this user) and total bytes of the drive holding `path`, or of
/// its closest existing parent. `SYNK_TEST_FREE_BYTES` pretends a value
/// for tests.
pub fn space(path: &Path) -> Option<(u64, u64)> {
    if let Some(free) = std::env::var("SYNK_TEST_FREE_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return Some((free, u64::MAX));
    }
    let existing = path.ancestors().find(|p| p.exists())?;
    let stats = fs4::statvfs(existing).ok()?;
    Some((stats.available_space(), stats.total_space()))
}

/// The bytes a run's checkpoints and models take, from the base model's
/// parameter count, as the trainer decides between full and LoRA.
pub fn training_bytes(parameters: u64, finetune: &str) -> u64 {
    let lora = match finetune {
        "lora" => true,
        "full" => false,
        // `auto`, as the trainer decides it.
        _ => parameters >= 300_000_000,
    };
    parameters
        * if lora {
            LORA_BYTES_PER_PARAM
        } else {
            FULL_BYTES_PER_PARAM
        }
}

/// `118M` or `0.6B` as a number.
pub fn parse_parameters(text: &str) -> Option<u64> {
    let t = text.trim();
    let (num, scale) = match t.chars().last()? {
        'M' | 'm' => (&t[..t.len() - 1], 1e6),
        'B' | 'b' => (&t[..t.len() - 1], 1e9),
        'K' | 'k' => (&t[..t.len() - 1], 1e3),
        _ => (t, 1.0),
    };
    let n: f64 = num.trim().parse().ok()?;
    Some((n * scale) as u64)
}

/// Where Ollama keeps models on this machine.
pub fn ollama_models_dir() -> Option<PathBuf> {
    std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".ollama").join("models")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_and_parameter_counts() {
        assert_eq!(parse_parameters("118M"), Some(118_000_000));
        assert_eq!(parse_parameters("0.6B"), Some(600_000_000));
        assert_eq!(parse_parameters("big"), None);
        // e5-small, full: ~2.4 GB; bge-m3 (568M) auto picks LoRA: ~4.5 GB.
        assert_eq!(training_bytes(118_000_000, "auto"), 2_360_000_000);
        assert_eq!(training_bytes(568_000_000, "auto"), 4_544_000_000);
        assert_eq!(training_bytes(568_000_000, "full"), 11_360_000_000);
    }

    #[test]
    fn needs_on_one_drive_add_up() {
        let dir = tempfile::tempdir().unwrap();
        let need = |what: &str, bytes| Need {
            what: what.into(),
            path: dir.path().join(what),
            bytes,
        };
        let volumes = check(vec![need("a", 5), need("b", 7), need("none", 0)]);
        assert_eq!(volumes.len(), 1);
        assert_eq!(volumes[0].need_bytes, 12);
        assert_eq!(volumes[0].needs.len(), 2);
        assert!(!volumes[0].short());
    }
}
