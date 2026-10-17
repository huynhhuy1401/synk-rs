use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable, machine-readable error codes. Agents branch on these, so never
/// rename one; add new variants instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    InvalidArgs,
    ConfigNotFound,
    ConfigExists,
    ConfigInvalid,
    ConfigKeyUnknown,
    SourceMissing,
    SourceExists,
    SourceOverlap,
    SourceNotFound,
    UnsupportedFormat,
    Io,
    Store,
    SidecarNotFound,
    SidecarProtocol,
    SidecarFailed,
    SetupFailed,
    GeneratorUnavailable,
    RemoteNotAllowed,
    ModelNotFound,
    ModelNotTrainable,
    DeviceUnsupported,
    DatasetNotFound,
    DatasetTooSmall,
    RunIncomplete,
    RunNotFound,
    QueriesInvalid,
    RunsNotComparable,
    IdAmbiguous,
    ConsentRequired,
    JobRunning,
    JobNotFound,
    Cancelled,
    InsufficientDisk,
    OutOfMemory,
    ModelAccessDenied,
    DownloadFailed,
    ExportExists,
    ExportCheckFailed,
    ExportUnsupported,
    RunNameTaken,
    RunNotResumable,
}

/// What kind of failure an error is. Agents that don't know a code branch
/// on this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    /// The command line itself is wrong.
    Usage,
    /// `synk.toml` is missing or invalid.
    Config,
    /// A named thing (source, run, model, ...) doesn't exist.
    NotFound,
    /// The request clashes with existing state.
    Conflict,
    /// Something synk needs isn't installed or allowed yet.
    Prerequisite,
    /// A service synk talks to didn't answer.
    Unavailable,
    /// The input data can't support the request.
    Data,
    /// The machine ran out of something (disk, memory).
    Resource,
    /// The work started and failed.
    Failed,
    /// A bug or an unexpected environment problem.
    Internal,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidArgs => "INVALID_ARGS",
            Self::ConfigNotFound => "CONFIG_NOT_FOUND",
            Self::ConfigExists => "CONFIG_EXISTS",
            Self::ConfigInvalid => "CONFIG_INVALID",
            Self::ConfigKeyUnknown => "CONFIG_KEY_UNKNOWN",
            Self::SourceMissing => "SOURCE_MISSING",
            Self::SourceExists => "SOURCE_EXISTS",
            Self::SourceOverlap => "SOURCE_OVERLAP",
            Self::SourceNotFound => "SOURCE_NOT_FOUND",
            Self::UnsupportedFormat => "UNSUPPORTED_FORMAT",
            Self::Io => "IO",
            Self::Store => "STORE",
            Self::SidecarNotFound => "SIDECAR_NOT_FOUND",
            Self::SidecarProtocol => "SIDECAR_PROTOCOL",
            Self::SidecarFailed => "SIDECAR_FAILED",
            Self::SetupFailed => "SETUP_FAILED",
            Self::GeneratorUnavailable => "GENERATOR_UNAVAILABLE",
            Self::RemoteNotAllowed => "REMOTE_NOT_ALLOWED",
            Self::ModelNotFound => "MODEL_NOT_FOUND",
            Self::ModelNotTrainable => "MODEL_NOT_TRAINABLE",
            Self::DeviceUnsupported => "DEVICE_UNSUPPORTED",
            Self::DatasetNotFound => "DATASET_NOT_FOUND",
            Self::DatasetTooSmall => "DATASET_TOO_SMALL",
            Self::RunIncomplete => "RUN_INCOMPLETE",
            Self::RunNotFound => "RUN_NOT_FOUND",
            Self::QueriesInvalid => "QUERIES_INVALID",
            Self::RunsNotComparable => "RUNS_NOT_COMPARABLE",
            Self::IdAmbiguous => "ID_AMBIGUOUS",
            Self::ConsentRequired => "CONSENT_REQUIRED",
            Self::JobRunning => "JOB_RUNNING",
            Self::JobNotFound => "JOB_NOT_FOUND",
            Self::Cancelled => "CANCELLED",
            Self::InsufficientDisk => "INSUFFICIENT_DISK",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::ModelAccessDenied => "MODEL_ACCESS_DENIED",
            Self::DownloadFailed => "DOWNLOAD_FAILED",
            Self::ExportExists => "EXPORT_EXISTS",
            Self::ExportCheckFailed => "EXPORT_CHECK_FAILED",
            Self::ExportUnsupported => "EXPORT_UNSUPPORTED",
            Self::RunNameTaken => "RUN_NAME_TAKEN",
            Self::RunNotResumable => "RUN_NOT_RESUMABLE",
        }
    }

    /// Every code, for docs and the schema.
    pub const ALL: &[ErrorCode] = &[
        Self::InvalidArgs,
        Self::ConfigNotFound,
        Self::ConfigExists,
        Self::ConfigInvalid,
        Self::ConfigKeyUnknown,
        Self::SourceMissing,
        Self::SourceExists,
        Self::SourceOverlap,
        Self::SourceNotFound,
        Self::UnsupportedFormat,
        Self::Io,
        Self::Store,
        Self::SidecarNotFound,
        Self::SidecarProtocol,
        Self::SidecarFailed,
        Self::SetupFailed,
        Self::GeneratorUnavailable,
        Self::RemoteNotAllowed,
        Self::ModelNotFound,
        Self::ModelNotTrainable,
        Self::DeviceUnsupported,
        Self::DatasetNotFound,
        Self::DatasetTooSmall,
        Self::RunIncomplete,
        Self::RunNotFound,
        Self::QueriesInvalid,
        Self::RunsNotComparable,
        Self::IdAmbiguous,
        Self::ConsentRequired,
        Self::JobRunning,
        Self::JobNotFound,
        Self::Cancelled,
        Self::InsufficientDisk,
        Self::OutOfMemory,
        Self::ModelAccessDenied,
        Self::DownloadFailed,
        Self::ExportExists,
        Self::ExportCheckFailed,
        Self::ExportUnsupported,
        Self::RunNameTaken,
        Self::RunNotResumable,
    ];

    pub fn class(self) -> ErrorClass {
        use ErrorClass as C;
        match self {
            Self::InvalidArgs | Self::IdAmbiguous => C::Usage,
            Self::ConfigNotFound | Self::ConfigInvalid | Self::ConfigKeyUnknown => C::Config,
            Self::SourceNotFound
            | Self::SourceMissing
            | Self::ModelNotFound
            | Self::DatasetNotFound
            | Self::RunNotFound => C::NotFound,
            Self::ConfigExists
            | Self::SourceExists
            | Self::SourceOverlap
            | Self::RunIncomplete
            | Self::RunsNotComparable => C::Conflict,
            Self::SidecarNotFound
            | Self::RemoteNotAllowed
            | Self::ModelNotTrainable
            | Self::DeviceUnsupported => C::Prerequisite,
            Self::GeneratorUnavailable => C::Unavailable,
            Self::UnsupportedFormat | Self::DatasetTooSmall | Self::QueriesInvalid => C::Data,
            Self::SidecarFailed | Self::SetupFailed => C::Failed,
            Self::Io | Self::Store | Self::SidecarProtocol => C::Internal,
            Self::ConsentRequired | Self::ModelAccessDenied | Self::ExportUnsupported => {
                C::Prerequisite
            }
            Self::JobRunning | Self::ExportExists | Self::RunNameTaken | Self::RunNotResumable => {
                C::Conflict
            }
            Self::JobNotFound => C::NotFound,
            Self::Cancelled | Self::ExportCheckFailed => C::Failed,
            Self::InsufficientDisk | Self::OutOfMemory => C::Resource,
            Self::DownloadFailed => C::Unavailable,
        }
    }

    /// Whether running the same command again, unchanged, may succeed.
    pub fn retryable(self) -> bool {
        matches!(
            self,
            Self::GeneratorUnavailable
                | Self::SetupFailed
                | Self::SidecarFailed
                | Self::Io
                | Self::JobRunning
                | Self::Cancelled
                | Self::DownloadFailed
        )
    }
}

/// How safe an action is to take without a person looking (the same
/// vocabulary as advice).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "snake_case")]
pub enum Applicability {
    /// An agent may take it.
    Machine,
    /// A person should look first.
    Review,
    /// Only a person can do it (install something, find a file, ...).
    Manual,
}

/// One way to fix an error: a command to run, or something a person does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorAction {
    /// The command, as arguments, when the fix is a command.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub argv: Option<Vec<String>>,
    pub applicability: Applicability,
    pub why: String,
}

impl ErrorAction {
    /// A command, written as it would be typed.
    pub fn run(command: &str, applicability: Applicability, why: impl Into<String>) -> Self {
        Self {
            argv: Some(command.split_whitespace().map(str::to_owned).collect()),
            applicability,
            why: why.into(),
        }
    }

    /// Something only a person can do.
    pub fn manual(why: impl Into<String>) -> Self {
        Self {
            argv: None,
            applicability: Applicability::Manual,
            why: why.into(),
        }
    }
}

/// An error with a code, a human message, a `next` command that can be run
/// as is, and any other ways to fix it.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{message}")]
pub struct SynkError {
    pub code: ErrorCode,
    pub class: ErrorClass,
    pub retryable: bool,
    pub message: String,
    /// A command that can be run exactly as written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<ErrorAction>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub details: Value,
}

impl SynkError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            class: code.class(),
            retryable: code.retryable(),
            message: message.into(),
            next: None,
            actions: Vec::new(),
            details: Value::Null,
        }
    }

    /// A command that can be run exactly as written. It also becomes the
    /// first action.
    pub fn with_next(mut self, next: impl Into<String>) -> Self {
        let next = next.into();
        // Prose or a placeholder belongs in `with_hint` / `with_action`.
        debug_assert!(
            next.starts_with("synk ") && !next.contains('<'),
            "`next` must be a runnable synk command, got {next:?}"
        );
        self.actions.insert(
            0,
            ErrorAction::run(&next, Applicability::Machine, "the most likely fix"),
        );
        self.next = Some(next);
        self
    }

    /// Another way to fix it: a command with placeholders, or a manual step.
    pub fn with_action(mut self, action: ErrorAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Something only a person can do, in plain words.
    pub fn with_hint(self, why: impl Into<String>) -> Self {
        self.with_action(ErrorAction::manual(why))
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }

    /// Adds one key to `details` (an object), keeping the others.
    pub fn with_detail(mut self, key: &str, value: Value) -> Self {
        if !self.details.is_object() {
            self.details = Value::Object(serde_json::Map::new());
        }
        self.details[key] = value;
        self
    }

    /// A filesystem error. A full disk (or quota) is `INSUFFICIENT_DISK`,
    /// so the fix is clear.
    pub fn io(context: impl fmt::Display, err: std::io::Error) -> Self {
        use std::io::ErrorKind;
        if matches!(
            err.kind(),
            ErrorKind::StorageFull | ErrorKind::QuotaExceeded
        ) {
            return Self::new(
                ErrorCode::InsufficientDisk,
                format!("the disk is full: {context}: {err}"),
            )
            .with_hint("free some space, then run the same command again; finished steps are kept")
            .with_detail("path", serde_json::json!(context.to_string()));
        }
        Self::new(ErrorCode::Io, format!("{context}: {err}"))
    }
}

pub type Result<T, E = SynkError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_listed_once_and_round_trip() {
        let mut seen = std::collections::HashSet::new();
        for code in ErrorCode::ALL {
            assert!(seen.insert(code.as_str()), "{code:?} listed twice");
            let json = serde_json::to_value(code).unwrap();
            assert_eq!(json, code.as_str());
        }
    }

    #[test]
    fn next_is_also_the_first_action() {
        let err = SynkError::new(ErrorCode::RunNotFound, "no run")
            .with_hint("train first")
            .with_next("synk train");
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["class"], "not_found");
        assert_eq!(json["retryable"], false);
        assert_eq!(
            json["actions"][0]["argv"],
            serde_json::json!(["synk", "train"])
        );
        assert_eq!(json["actions"][1]["applicability"], "manual");
        assert!(json.get("details").is_none());
    }
}
