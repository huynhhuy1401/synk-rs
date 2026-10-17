//! The Python training sidecar: creating its environment and talking to it.
//!
//! Protocol v1: synk writes `request.json` and runs
//! `python synk_sidecar.py run request.json`. Each stdout line is a JSON
//! event (`progress`, `log`, then one `result` or `error`). stderr goes to a
//! log file. See `docs/adr/0001-python-training-sidecar.md`.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use synk_core::{ErrorCode, Result, SynkError};

pub const PROTOCOL_VERSION: u32 = 2;
const SCRIPT: &str = include_str!("../sidecar/synk_sidecar.py");

// The environment `synk setup` builds. `EXCLUDE_NEWER` makes uv ignore
// anything published later, so every install resolves to the same versions.
const PYTHON_VERSION: &str = "3.12";
const TORCH: &str = "torch==2.14.1";
const SENTENCE_TRANSFORMERS: &str = "sentence-transformers[train]==6.1.0";
/// LoRA, ONNX export (PyTorch's exporter; sentence-transformers' own
/// `[onnx]` extra needs an older transformers) and the GGUF writer for
/// Ollama; resolved as of `EXCLUDE_NEWER`.
const EXTRAS: &[&str] = &[
    "peft",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "gguf",
    "sentencepiece",
    "protobuf",
];
const EXCLUDE_NEWER: &str = "2026-10-14T00:00:00Z";

/// Every package of the environment, pinned with hashes, per backend and
/// platform (`scripts/lock-trainer.sh` writes them from the versions
/// above). Installing from one resolves nothing on the user's machine, so
/// every install is the same, byte for byte.
const LOCKS: &[(&str, &str)] = &[
    (
        "cpu-windows-x86_64",
        include_str!("../locks/cpu-windows-x86_64.txt"),
    ),
    (
        "cpu-linux-x86_64",
        include_str!("../locks/cpu-linux-x86_64.txt"),
    ),
    (
        "cpu-linux-aarch64",
        include_str!("../locks/cpu-linux-aarch64.txt"),
    ),
    (
        "cpu-macos-aarch64",
        include_str!("../locks/cpu-macos-aarch64.txt"),
    ),
    (
        "cuda-windows-x86_64",
        include_str!("../locks/cuda-windows-x86_64.txt"),
    ),
    (
        "cuda-linux-x86_64",
        include_str!("../locks/cuda-linux-x86_64.txt"),
    ),
];

/// The lock file for `backend` on this platform, if synk ships one.
fn lock_file(backend: Backend) -> Option<&'static str> {
    let os = if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        return None;
    };
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        return None;
    };
    let name = format!("{}-{os}-{arch}", backend.as_str());
    LOCKS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, text)| *text)
}

/// Where synk keeps environments shared by all projects: `$SYNK_HOME`, or the
/// platform's local data directory.
pub fn synk_home() -> PathBuf {
    std::env::var_os("SYNK_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::data_local_dir().map(|d| d.join("synk")))
        .unwrap_or_else(|| PathBuf::from(".synk-home"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Cpu,
    Cuda,
}

/// Why synk doesn't install a build for AMD or Apple GPUs.
const UNTESTED_GPU: &str = "synk trains on NVIDIA GPUs (CUDA) or the CPU; AMD (ROCm) and Apple (MPS) GPUs aren't supported yet because they haven't been tested";

impl Backend {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "rocm" | "mps" => Err(SynkError::new(ErrorCode::DeviceUnsupported, UNTESTED_GPU)
                .with_next("synk setup --backend cpu")),
            other => Err(SynkError::new(
                ErrorCode::InvalidArgs,
                format!("unknown backend `{other}`; use auto, cpu or cuda"),
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }

    /// The PyTorch wheel index for this backend on this OS, or `None` for
    /// PyPI. Fails where PyTorch publishes no build for the combination.
    fn torch_index(self) -> Result<Option<&'static str>> {
        let unsupported = |why: &str| {
            Err(SynkError::new(ErrorCode::DeviceUnsupported, why.to_owned())
                .with_next("synk setup --backend cpu"))
        };
        match self {
            // macOS CPU wheels are only on PyPI.
            Self::Cpu if cfg!(target_os = "macos") => Ok(None),
            Self::Cpu => Ok(Some("https://download.pytorch.org/whl/cpu")),
            Self::Cuda if cfg!(target_os = "macos") => {
                unsupported("PyTorch has no CUDA builds for macOS")
            }
            Self::Cuda => Ok(Some("https://download.pytorch.org/whl/cu130")),
        }
    }
}

/// Picks a backend for this machine, and says why.
pub fn detect_backend() -> (Backend, String) {
    if runs("nvidia-smi", &["-L"]) {
        return (Backend::Cuda, "nvidia-smi found an NVIDIA GPU".to_owned());
    }
    (Backend::Cpu, format!("no NVIDIA GPU found; {UNTESTED_GPU}"))
}

fn runs(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// A Python environment the sidecar can run in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerEnv {
    pub python: PathBuf,
    /// `None` for an environment the user brought.
    pub backend: Option<Backend>,
    /// The sidecar's `describe` output: versions, devices, protocol.
    pub describe: Value,
}

impl TrainerEnv {
    /// The environment to train with: `python` from `[trainer]` if set,
    /// otherwise the one `synk setup` recorded.
    pub fn find(python: Option<&Path>) -> Result<Self> {
        if crate::fake_trainer::enabled() {
            return Ok(Self {
                python: PathBuf::from("fake"),
                backend: Some(Backend::Cpu),
                describe: crate::fake_trainer::describe(),
            });
        }
        if let Some(python) = python {
            let describe = describe(python)?;
            return Ok(Self {
                python: python.to_path_buf(),
                backend: None,
                describe,
            });
        }
        let path = synk_home().join("trainer.json");
        let text = fs::read_to_string(&path).map_err(|_| {
            SynkError::new(
                ErrorCode::SidecarNotFound,
                "the training environment is not set up",
            )
            .with_next("synk setup")
        })?;
        let env: Self = serde_json::from_str(&text).map_err(|e| {
            SynkError::new(
                ErrorCode::SidecarNotFound,
                format!("{}: {e}", path.display()),
            )
            .with_next("synk setup --force")
        })?;
        if !env.python.exists() {
            return Err(SynkError::new(
                ErrorCode::SidecarNotFound,
                format!(
                    "the training environment at {} is gone",
                    env.python.display()
                ),
            )
            .with_next("synk setup --force"));
        }
        Ok(env)
    }

    /// Torch, transformers and sentence-transformers versions. Part of every
    /// training run's key, since they change the result.
    pub fn fingerprint(&self) -> Value {
        serde_json::json!({
            "torch": self.describe["torch"],
            "transformers": self.describe["transformers"],
            "sentence_transformers": self.describe["sentence_transformers"],
        })
    }

    /// Runs one task. `work_dir` receives `request.json` and `sidecar.log`.
    pub fn run(
        &self,
        request: &Value,
        work_dir: &Path,
        on_event: &mut dyn FnMut(&Value),
    ) -> Result<Value> {
        if self.python == Path::new("fake") {
            return crate::fake_trainer::run(request, on_event);
        }
        let request_path = work_dir.join("request.json");
        // The trainer exits when its stdin closes, i.e. when synk is gone.
        let mut request = request.clone();
        request["lifeline"] = serde_json::json!(true);
        let text = serde_json::to_string_pretty(&request).expect("request serializes");
        fs::write(&request_path, text).map_err(|e| SynkError::io(request_path.display(), e))?;
        let log_path = work_dir.join("sidecar.log");
        let log = fs::File::create(&log_path).map_err(|e| SynkError::io(log_path.display(), e))?;

        let mut command = sidecar_command(&self.python)?;
        own_process_group(&mut command);
        let mut child = command
            .arg("run")
            .arg(&request_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| {
                SynkError::new(
                    ErrorCode::SidecarNotFound,
                    format!("could not start {}: {e}", self.python.display()),
                )
                .with_next("synk setup --force")
            })?;

        // Held open until the trainer is done; dropped with synk otherwise.
        let lifeline = child.stdin.take();
        let mut outcome = None;
        let stdout = child.stdout.take().expect("piped stdout");
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let Ok(event) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            match event["event"].as_str() {
                Some("result") => outcome = Some(Ok(event["data"].clone())),
                Some("error") => outcome = Some(Err(sidecar_error(&event, &log_path))),
                _ => on_event(&event),
            }
        }
        let status = child.wait().map_err(|e| SynkError::io("sidecar", e))?;
        drop(lifeline);
        match outcome {
            Some(result) => result,
            None => Err(SynkError::new(
                ErrorCode::SidecarFailed,
                format!(
                    "the trainer exited ({status}) without a result. Last log lines:\n{}",
                    log_tail(&log_path)
                ),
            )),
        }
    }
}

/// Starts the trainer in its own process group, so the terminal's Ctrl-C
/// reaches only synk, which then stops the trainer at a safe point instead
/// of it dying mid-step.
fn own_process_group(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
}

fn sidecar_error(event: &Value, log_path: &Path) -> SynkError {
    let message = event["message"]
        .as_str()
        .unwrap_or("unknown error")
        .to_owned();
    let code = match event["code"].as_str() {
        Some("DEVICE_UNSUPPORTED") => ErrorCode::DeviceUnsupported,
        Some("MODEL_NOT_FOUND") => ErrorCode::ModelNotFound,
        Some("DATASET_TOO_SMALL") => ErrorCode::DatasetTooSmall,
        Some("SIDECAR_PROTOCOL") => ErrorCode::SidecarProtocol,
        Some("CANCELLED") => ErrorCode::Cancelled,
        Some("OUT_OF_MEMORY") => ErrorCode::OutOfMemory,
        Some("MODEL_ACCESS_DENIED") => ErrorCode::ModelAccessDenied,
        Some("DOWNLOAD_FAILED") => ErrorCode::DownloadFailed,
        Some("EXPORT_CHECK_FAILED") => ErrorCode::ExportCheckFailed,
        Some("EXPORT_UNSUPPORTED") => ErrorCode::ExportUnsupported,
        Some("INSUFFICIENT_DISK") => ErrorCode::InsufficientDisk,
        _ => ErrorCode::SidecarFailed,
    };
    let err = SynkError::new(code, message);
    match code {
        ErrorCode::SidecarFailed => err
            .with_hint(format!("the trainer's log: {}", log_path.display()))
            .with_details(serde_json::json!({ "log": log_path })),
        ErrorCode::OutOfMemory => err
            .with_action(synk_core::ErrorAction::run(
                "synk config set trainer.batch_size 8",
                synk_core::Applicability::Review,
                "train with smaller batches (changes the result a little)",
            ))
            .with_action(synk_core::ErrorAction::run(
                "synk config set trainer.finetune lora",
                synk_core::Applicability::Review,
                "train a LoRA adapter instead of every weight",
            )),
        ErrorCode::ModelAccessDenied => err.with_hint(
            "accept the model's terms on Hugging Face, then set HF_TOKEN to a token of that account",
        ),
        ErrorCode::ModelNotFound => err
            .with_next("synk model list")
            .with_hint("check model.base: synk config get model.base"),
        ErrorCode::InsufficientDisk => err
            .with_hint("free some space, then run the same command again; finished steps are kept")
            .with_detail("path", event["path"].clone()),
        _ => err,
    }
}

fn log_tail(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(15)..].join("\n")
}

/// `python <script>`, with the script written next to synk's environments.
/// The file name carries a hash of its contents, so a different synk version
/// never runs a stale copy.
fn sidecar_command(python: &Path) -> Result<Command> {
    let dir = synk_home().join("sidecar");
    let hash = &blake3::hash(SCRIPT.as_bytes()).to_hex()[..12];
    let script = dir.join(format!("synk_sidecar-{hash}.py"));
    if !script.is_file() {
        fs::create_dir_all(&dir).map_err(|e| SynkError::io(dir.display(), e))?;
        fs::write(&script, SCRIPT).map_err(|e| SynkError::io(script.display(), e))?;
    }
    let mut cmd = Command::new(python);
    cmd.arg(script)
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUNBUFFERED", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("TOKENIZERS_PARALLELISM", "false");
    // Training on several GPUs at once fails in the trainer (seen on a
    // machine with two T4s), so it sees the first GPU only, unless the user
    // chose GPUs with CUDA_VISIBLE_DEVICES.
    if std::env::var_os("CUDA_VISIBLE_DEVICES").is_none() {
        cmd.env("CUDA_VISIBLE_DEVICES", "0");
    }
    Ok(cmd)
}

fn describe(python: &Path) -> Result<Value> {
    let output = sidecar_command(python)?
        .arg("describe")
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            SynkError::new(
                ErrorCode::SidecarNotFound,
                format!("could not run {}: {e}", python.display()),
            )
            .with_next("synk setup")
        })?;
    let event = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|e| e["event"] == "result" || e["event"] == "error");
    let Some(event) = event.filter(|e| e["event"] == "result") else {
        return Err(SynkError::new(
            ErrorCode::SidecarNotFound,
            format!(
                "{} can't run the trainer; it needs torch and sentence-transformers[train]:\n{}",
                python.display(),
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or_default()
            ),
        )
        .with_next("synk setup"));
    };
    let data = event["data"].clone();
    let supported = data["protocol"].as_array().is_some_and(|v| {
        v.iter()
            .any(|p| p.as_u64() == Some(u64::from(PROTOCOL_VERSION)))
    });
    if !supported {
        return Err(SynkError::new(
            ErrorCode::SidecarProtocol,
            format!(
                "the trainer speaks protocol {}, synk needs {PROTOCOL_VERSION}",
                data["protocol"]
            ),
        )
        .with_next("synk setup --force"));
    }
    Ok(data)
}

pub struct SetupReport {
    pub env: TrainerEnv,
    pub reused: bool,
    pub reason: String,
}

/// Creates (or reuses) the environment for `backend`, records it as the one
/// to train with, and checks it runs. `log` receives each step.
pub fn setup(
    backend: Option<Backend>,
    force: bool,
    uv: Option<&Path>,
    log: &mut dyn FnMut(&str),
) -> Result<SetupReport> {
    let (backend, reason) = match backend {
        Some(b) => (b, "chosen with --backend".to_owned()),
        None => detect_backend(),
    };
    let index = backend.torch_index()?;
    let lock = lock_file(backend);
    let recipe = match lock {
        Some(text) => format!(
            "{PYTHON_VERSION}|lock|{}",
            blake3::hash(text.as_bytes()).to_hex()
        ),
        None => format!(
            "{PYTHON_VERSION}|{TORCH}|{SENTENCE_TRANSFORMERS}|{}|{EXCLUDE_NEWER}|{index:?}",
            EXTRAS.join(",")
        ),
    };
    let env_name = format!(
        "{}-{}",
        backend.as_str(),
        &blake3::hash(recipe.as_bytes()).to_hex()[..8]
    );
    let home = synk_home();
    let env_dir = home.join("envs").join(&env_name);
    let python = if cfg!(windows) {
        env_dir.join("Scripts").join("python.exe")
    } else {
        env_dir.join("bin").join("python")
    };

    // A half-built environment (an interrupted install) is rebuilt.
    let reused = python.exists() && !force && describe(&python).is_ok();
    if !reused {
        let uv = find_uv(uv)?;
        if env_dir.exists() {
            fs::remove_dir_all(&env_dir).map_err(|e| SynkError::io(env_dir.display(), e))?;
        }
        log(&format!(
            "Creating a Python {PYTHON_VERSION} environment in {}",
            env_dir.display()
        ));
        uv_run(&uv, &["python", "install", PYTHON_VERSION])?;
        let env_arg = env_dir.to_string_lossy();
        uv_run(
            &uv,
            &[
                "venv",
                &env_arg,
                "--python",
                PYTHON_VERSION,
                "--python-preference",
                "only-managed",
            ],
        )?;
        let python_arg = python.to_string_lossy();
        log(&format!(
            "Installing PyTorch ({}), sentence-transformers, PEFT and the GGUF writer; this downloads several hundred MB to a few GB",
            backend.as_str()
        ));
        if let Some(text) = lock {
            let lock_path = env_dir.join("synk-lock.txt");
            fs::write(&lock_path, text).map_err(|e| SynkError::io(lock_path.display(), e))?;
            let lock_arg = lock_path.to_string_lossy();
            uv_run(
                &uv,
                &[
                    "pip",
                    "install",
                    "--python",
                    &python_arg,
                    "--require-hashes",
                    "--no-deps",
                    "--index-strategy",
                    "unsafe-best-match",
                    "-r",
                    &lock_arg,
                ],
            )?;
        } else {
            // No lock for this platform: resolve, capped at a date.
            let mut torch = vec!["pip", "install", "--python", &python_arg, TORCH];
            if let Some(index) = index {
                torch.extend(["--index-url", index]);
            }
            uv_run(&uv, &torch)?;
            let mut packages = vec![
                "pip",
                "install",
                "--python",
                &python_arg,
                SENTENCE_TRANSFORMERS,
            ];
            packages.extend(EXTRAS);
            packages.extend(["--exclude-newer", EXCLUDE_NEWER]);
            uv_run(&uv, &packages)?;
        }
    }

    log("Checking the environment");
    let env = TrainerEnv {
        describe: describe(&python)?,
        python,
        backend: Some(backend),
    };
    let record = home.join("trainer.json");
    let text = serde_json::to_string_pretty(&env).expect("env serializes");
    fs::write(&record, text).map_err(|e| SynkError::io(record.display(), e))?;
    Ok(SetupReport {
        env,
        reused,
        reason,
    })
}

fn find_uv(configured: Option<&Path>) -> Result<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut candidates: Vec<PathBuf> = configured.map(Path::to_path_buf).into_iter().collect();
    candidates.push(PathBuf::from("uv"));
    // The official installer's default locations, which may not be on PATH
    // in an already-open shell.
    if let Some(home) = dirs::home_dir() {
        candidates.push(home.join(".local").join("bin").join(exe));
        candidates.push(home.join(".cargo").join("bin").join(exe));
    }
    candidates
        .into_iter()
        .find(|c| runs(&c.to_string_lossy(), &["--version"]))
        .ok_or_else(|| {
            let install = if cfg!(windows) {
                r#"powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex""#
            } else {
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            };
            SynkError::new(
                ErrorCode::SidecarNotFound,
                "uv is not installed; synk uses it to create the Python training environment",
            )
            .with_action(synk_core::ErrorAction::run(
                install,
                synk_core::Applicability::Review,
                "install uv (a Python installer), then run the same synk command again",
            ))
        })
}

fn uv_run(uv: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new(uv)
        .args(args)
        .output()
        .map_err(|e| SynkError::new(ErrorCode::SetupFailed, format!("could not run uv: {e}")))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: Vec<&str> = stderr.lines().rev().take(10).collect();
    Err(SynkError::new(
        ErrorCode::SetupFailed,
        format!(
            "uv {} failed:\n{}",
            args.join(" "),
            tail.into_iter().rev().collect::<Vec<_>>().join("\n")
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backends_only_map_to_builds_pytorch_publishes() {
        assert!(Backend::Cpu.torch_index().is_ok());
        for untested in ["rocm", "mps"] {
            assert_eq!(
                Backend::parse(untested).unwrap_err().code,
                ErrorCode::DeviceUnsupported
            );
        }
        assert_eq!(
            Backend::parse("tpu").unwrap_err().code,
            ErrorCode::InvalidArgs
        );
    }
}
