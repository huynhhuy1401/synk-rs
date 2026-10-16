mod commands;
mod config;
mod config_cmd;
mod human;
mod output;
mod pipeline;
mod reference;
mod views;
mod work;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use synk_core::{ErrorCode, SynkError};

use crate::config::Project;

/// The top-level help: commands grouped by what they are for. clap has a
/// single heading for subcommands, so the groups are written here and a test
/// keeps them in sync with the parser.
const HELP_TEMPLATE: &str = "\
{about}

{usage-heading} {usage}

Work:
  status    Where the project stands, and what to do next
  train     Sync, write queries, train and test: one run (--dry-run to preview)
  dataset   Sync and build the training data only
  fetch     Download what training needs: models, the LLM, test sets, the trainer
  wait      Wait for the running train, dataset or fetch to finish
  logs      Progress and log lines of a run, dataset build or fetch
  cancel    Ask the running train, dataset or fetch to stop

Results:
  run       List, show, inspect, compare and report on training runs
  export    Save a tuned model: a folder, ONNX, or GGUF for Ollama

Config:
  init      Create synk.toml
  source    Add, remove, list and sync sources
  config    Show, get, set and unset settings in synk.toml
  queries   Import real queries to test with

Machine:
  setup     Install the Python trainer (once per machine)
  model     List and show base models and rerankers

Reference:
  schema    The JSON contract: commands, codes, events
  skill     The agent skill (SKILL.md) for Claude Code, Codex, Cursor, ...

Options:
{options}

Every command takes --json: one object on stdout, {\"v\": 1, \"ok\": ..., \"data\" | \"error\": ...}.
";

/// Fine-tunes a small search (embedding) model on your own documents,
/// locally, and tests whether it beats the original.
#[derive(Parser)]
#[command(
    name = "synk",
    bin_name = "synk",
    version,
    help_template = HELP_TEMPLATE
)]
struct Cli {
    /// Print one JSON object on stdout: {"v": 1, "ok": true, "data": ...} or {"v": 1, "ok": false, "error": ...}
    #[arg(long, global = true)]
    json: bool,

    /// Path to synk.toml (default: search the current directory and its parents)
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Progress of long steps, on stderr: auto (a live line, quiet with --json), jsonl or none
    #[arg(
        long,
        global = true,
        value_enum,
        default_value = "auto",
        value_name = "MODE"
    )]
    progress: pipeline::ProgressMode,

    /// Also show IDs, metric names and advice codes
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct WorkFlags {
    /// Override a setting for this run only, as `config set` would (repeatable)
    #[arg(long = "set", value_name = "KEY=VALUE")]
    set: Vec<String>,
    /// Build the dataset from at most this many chunks, for a quick run
    #[arg(long, value_name = "N")]
    sample: Option<usize>,
    /// Redo this step even if its result exists: dataset, train or eval
    #[arg(long, value_name = "STEP")]
    force: Option<String>,
    /// Show what would run, why, what it downloads or sends, and how long; changes nothing
    #[arg(long)]
    dry_run: bool,
    /// Consent to the downloads (and any sending of text off this machine) this needs
    #[arg(long)]
    yes: bool,
}

impl WorkFlags {
    fn args(&self, run: Option<&String>, name: Option<&String>) -> work::TrainArgs {
        work::TrainArgs {
            run: run.cloned(),
            set: self.set.clone(),
            name: name.cloned(),
            sample: self.sample,
            force: self.force.clone(),
            dry_run: self.dry_run,
            yes: self.yes,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Where the project stands, and what to do next
    #[command(after_help = "Examples:\n  synk status\n  synk status --json")]
    Status,
    /// Sync, write queries, train and test: one run (--dry-run to preview)
    #[command(
        after_help = "Never downloads or sends text off this machine without --yes: when something\nis missing it stops and prints the exact `synk fetch` command.\n\nExamples:\n  synk train --dry-run                           what would run, and what it needs\n  synk train                                     a run with synk.toml's settings\n  synk train --yes                               download what's missing, then run\n  synk train --set model.base=bge-m3 --name bge  a variant, as its own run\n  synk train run:3                               continue run 3 with its own settings"
    )]
    Train {
        /// Continue this run (`run:3`, `3` or its name) with the settings it started with
        run: Option<String>,
        /// A name for the new run, unique in the project (it becomes part of the ID)
        #[arg(long)]
        name: Option<String>,
        #[command(flatten)]
        flags: WorkFlags,
    },
    /// Sync and build the training data only
    #[command(
        after_help = "Examples:\n  synk dataset --dry-run\n  synk dataset --sample 50      a quick dataset from 50 chunks"
    )]
    Dataset {
        #[command(flatten)]
        flags: WorkFlags,
    },
    /// Download what training needs: models, the LLM, test sets, the trainer
    #[command(
        after_help = "Running it is your consent to download exactly the names given; without names,\neverything `synk train --dry-run` lists.\n\nNames: a catalog model (multilingual-e5-small) or hf:org/name, either with @<commit>;\nollama:TAG; general-check; trainer.\n\nExamples:\n  synk fetch\n  synk fetch multilingual-e5-small@614241f ollama:qwen3:8b general-check\n  synk fetch hf:BAAI/bge-m3"
    )]
    Fetch {
        /// What to download (default: everything training needs now)
        names: Vec<String>,
        /// Settings to work out what is needed, as `train --set` (repeatable)
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// For `trainer`: auto, cpu or cuda (NVIDIA)
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Wait for the running train, dataset or fetch to finish; running out of time is not an error
    #[command(after_help = "Examples:\n  synk wait --timeout 9m --json\n  synk wait run:3")]
    Wait {
        /// A run (`run:3`, `3`, its name), `dataset:N` or `fetch:N` (default: the running one, else the latest)
        id: Option<String>,
        /// Like 30s, 9m or 1h
        #[arg(long, default_value = "9m")]
        timeout: String,
    },
    /// Progress and log lines of a run, dataset build or fetch, numbered
    #[command(after_help = "Examples:\n  synk logs\n  synk logs run:3 --after 120")]
    Logs {
        /// Default: the running one, else the latest
        id: Option<String>,
        /// Only events after this number
        #[arg(long, default_value_t = 0)]
        after: u64,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },
    /// Ask the running train, dataset or fetch to stop at the next safe point
    Cancel {
        /// Default: the running one
        id: Option<String>,
    },
    /// List, show, inspect, compare and report on training runs
    Run {
        #[command(subcommand)]
        action: RunAction,
    },
    /// Save a tuned model: a folder, ONNX, or GGUF for Ollama
    #[command(
        after_help = "Examples:\n  synk export best --out ./notes-model\n  synk export best --out ./notes-onnx --format onnx\n  synk export best --out ./notes-gguf --format gguf --ollama notes-embed"
    )]
    Export {
        /// A run (`run:3`, `3`, its name), `latest` or `best` (default: latest)
        run: Option<String>,
        /// Destination folder; must not exist or be empty (unless --force)
        #[arg(long)]
        out: PathBuf,
        /// st (sentence-transformers folder), onnx or gguf
        #[arg(long, default_value = "st")]
        format: String,
        /// GGUF only: q8_0 (default) or f16
        #[arg(long)]
        quant: Option<String>,
        /// GGUF only: also create this model in your local Ollama
        #[arg(long, value_name = "NAME")]
        ollama: Option<String>,
        /// Replace a non-empty --out folder
        #[arg(long)]
        force: bool,
    },
    /// Create synk.toml
    #[command(after_help = "Examples:\n  synk init\n  synk init ./my-project")]
    Init {
        #[arg(default_value = ".")]
        dir: PathBuf,
    },
    /// Add, remove, list and sync sources
    Source {
        #[command(subcommand)]
        action: SourceAction,
    },
    /// Show, get, set and unset settings in synk.toml
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Import real queries (searches people made) to test with
    Queries {
        #[command(subcommand)]
        action: QueriesAction,
    },
    /// Install the Python trainer now (once per machine; about 1-4 GB). `synk fetch trainer` does the same
    #[command(after_help = "Examples:\n  synk setup\n  synk setup --backend cpu")]
    Setup {
        /// auto, cpu or cuda (NVIDIA)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Recreate the environment even if it exists
        #[arg(long)]
        force: bool,
    },
    /// List and show base models and rerankers
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },
    /// The JSON contract: commands, error and advice codes, events
    #[command(after_help = "Examples:\n  synk schema --json\n  synk schema train --json")]
    Schema {
        /// One command (e.g. `train`, `run show`)
        command: Vec<String>,
    },
    /// The agent skill (SKILL.md) for Claude Code, Codex, Cursor, Gemini CLI, ...
    #[command(
        after_help = "Examples:\n  synk skill                         print SKILL.md\n  synk skill --out .claude/skills/synk"
    )]
    Skill {
        /// Write SKILL.md and references/ into this folder
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum RunAction {
    /// Every run, newest first: state, base model, verdict, and settings that differ from synk.toml
    List,
    /// One run: settings, verdict and advice
    #[command(after_help = "Examples:\n  synk run show\n  synk run show best")]
    Show {
        /// A run (`run:3`, `3`, its name), `latest` or `best` (default: latest)
        run: Option<String>,
    },
    /// The queries a run improved or hurt most
    #[command(after_help = "Examples:\n  synk run inspect\n  synk run inspect best --limit 3")]
    Inspect {
        /// A run (`run:3`, `3`, its name), `latest` or `best` (default: latest)
        run: Option<String>,
        /// How many losses and gains to show
        #[arg(long, default_value_t = 10, value_name = "N")]
        limit: usize,
    },
    /// Test whether run B beats run A on the same queries
    #[command(after_help = "Example:\n  synk run compare 1 2")]
    Compare {
        /// The reference run
        a: String,
        /// The run to test against it
        b: String,
    },
    /// A one-page HTML report of a run: the verdict, and searches before and after
    #[command(after_help = "Example:\n  synk run report best --out report.html")]
    Report {
        run: Option<String>,
        #[arg(long, default_value = "synk-report.html")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum SourceAction {
    /// Add files or folders as sources
    #[command(
        after_help = "Examples:\n  synk source add ./notes\n  synk source add ./notes ./handbook.md\n  synk source add ./wiki --exclude 'drafts/**'"
    )]
    Add {
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Source name (single path only; defaults to the file or folder name)
        #[arg(long)]
        name: Option<String>,
        /// Glob, relative to the source root, to leave out (repeatable)
        #[arg(long, value_name = "GLOB")]
        exclude: Vec<String>,
    },
    /// Remove a source by name or path (its documents are marked deleted)
    Remove { source: String },
    /// List configured sources
    List,
    /// Read the sources now and show what changed (`train` and `dataset` do it too)
    Sync,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Every setting with its value, and which ones synk.toml sets
    Show,
    /// One setting's value
    #[command(after_help = "Example:\n  synk config get generator.model")]
    Get { key: String },
    /// Set a value in synk.toml (comments are kept; invalid values are refused)
    #[command(
        after_help = "Examples:\n  synk config set generator.model qwen3:4b\n  synk config set generator.queries_per_chunk 5\n  synk config set generator.languages '[\"vi\", \"en\"]'"
    )]
    Set { key: String, value: String },
    /// Remove a setting from synk.toml, so its default applies
    Unset { key: String },
}

#[derive(Subcommand)]
enum QueriesAction {
    /// Add queries from a JSONL file: {"query": ..., "doc": <document ID>} and/or "text": <answering passage>
    Import {
        file: PathBuf,
        /// Replace all previously imported queries
        #[arg(long)]
        replace: bool,
    },
    /// List imported queries
    List,
}

#[derive(Subcommand)]
enum ModelAction {
    /// The base models and rerankers synk has tested
    List,
    /// One model: a catalog ID, `hf:org/name[@rev]` or a folder
    #[command(
        after_help = "Examples:\n  synk model show multilingual-e5-small\n  synk model show hf:BAAI/bge-m3"
    )]
    Show { model: String },
}

/// `30s`, `9m`, `1h`, or seconds.
fn parse_duration(text: &str) -> synk_core::Result<Duration> {
    let t = text.trim();
    let (num, mult) = match t.char_indices().last() {
        Some((i, 's')) => (&t[..i], 1),
        Some((i, 'm')) => (&t[..i], 60),
        Some((i, 'h')) => (&t[..i], 3600),
        _ => (t, 1),
    };
    num.parse::<u64>()
        .map(|n| Duration::from_secs(n * mult))
        .map_err(|_| {
            SynkError::new(
                ErrorCode::InvalidArgs,
                format!("`{text}` is not a duration; use e.g. 30s, 9m or 1h"),
            )
        })
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        // With --json, bad arguments are an INVALID_ARGS object like any
        // other error. --help and --version print as usual.
        Err(e) if e.use_stderr() && std::env::args().any(|a| a == "--json") => {
            return output::print(true, Err(usage_error(&e.to_string())));
        }
        Err(e) => e.exit(),
    };
    output::set_verbose(cli.verbose);
    let (json, jsonl) = (cli.json, cli.progress == pipeline::ProgressMode::Jsonl);
    let _ = ctrlc::set_handler(move || output::on_interrupt(json, jsonl));
    // Tests: simulate Ctrl-C presses after these many milliseconds
    // (comma-separated), on every OS.
    if let Ok(delays) = std::env::var("SYNK_TEST_INTERRUPT_AFTER_MS") {
        let delays: Vec<u64> = delays
            .split(',')
            .filter_map(|d| d.trim().parse().ok())
            .collect();
        std::thread::spawn(move || {
            let mut slept = 0;
            for at in delays {
                std::thread::sleep(Duration::from_millis(at.saturating_sub(slept)));
                slept = at;
                output::on_interrupt(json, jsonl);
            }
        });
    }
    let config = cli.config.as_deref();
    let project = || Project::find(config);
    let progress = || pipeline::Progress::new(cli.progress, cli.json);
    let display = || work::Display {
        progress: progress(),
        json: cli.json,
        mode: cli.progress,
    };
    let result = match &cli.command {
        Command::Status => project().and_then(|p| commands::status(&p)),
        Command::Train { run, name, flags } => project().and_then(|p| {
            work::train(
                &p,
                synk_pipeline::ops::OpKind::Run,
                &flags.args(run.as_ref(), name.as_ref()),
                display(),
            )
        }),
        Command::Dataset { flags } => project().and_then(|p| {
            work::train(
                &p,
                synk_pipeline::ops::OpKind::Dataset,
                &flags.args(None, None),
                display(),
            )
        }),
        Command::Fetch {
            names,
            set,
            backend,
        } => project().and_then(|p| work::fetch(&p, names, set, backend, display())),
        Command::Wait { id, timeout } => project().and_then(|p| {
            parse_duration(timeout)
                .and_then(|t| work::wait(&p, id.as_deref(), t, cli.json, cli.progress))
        }),
        Command::Logs { id, after, limit } => {
            project().and_then(|p| work::logs(&p, id.as_deref(), *after, *limit))
        }
        Command::Cancel { id } => project().and_then(|p| work::cancel(&p, id.as_deref())),
        Command::Run { action } => project().and_then(|p| match action {
            RunAction::List => views::runs(&p),
            RunAction::Show { run } => views::show_run(&p, run.as_deref()),
            RunAction::Inspect { run, limit } => views::inspect(&p, run.as_deref(), *limit),
            RunAction::Compare { a, b } => views::compare(&p, a, b),
            RunAction::Report { run, out } => views::html_report(&p, run.as_deref(), out),
        }),
        Command::Export {
            run,
            out,
            format,
            quant,
            ollama,
            force,
        } => project().and_then(|p| {
            pipeline::export(
                &p,
                out,
                run.as_deref(),
                &pipeline::ExportArgs {
                    format: format.clone(),
                    quant: quant.clone(),
                    ollama: ollama.clone(),
                    force: *force,
                },
                progress(),
            )
        }),
        Command::Init { dir } => commands::init(dir),
        Command::Source { action } => match action {
            SourceAction::Add {
                paths,
                name,
                exclude,
            } => project().and_then(|p| commands::add(p, paths, name.as_deref(), exclude)),
            SourceAction::Remove { source } => project().and_then(|p| commands::remove(p, source)),
            SourceAction::List => project().and_then(|p| commands::sources(&p)),
            SourceAction::Sync => project().and_then(|p| commands::sync(&p)),
        },
        Command::Config { action } => project().and_then(|p| match action {
            ConfigAction::Show => config_cmd::show(&p),
            ConfigAction::Get { key } => config_cmd::get(&p, key),
            ConfigAction::Set { key, value } => config_cmd::set(p, key, value),
            ConfigAction::Unset { key } => config_cmd::unset(p, key),
        }),
        Command::Queries { action } => project().and_then(|p| match action {
            QueriesAction::Import { file, replace } => views::import_queries(&p, file, *replace),
            QueriesAction::List => views::list_queries(&p),
        }),
        Command::Setup { backend, force } => {
            pipeline::setup(project(), backend, *force, progress())
        }
        Command::Model { action } => match action {
            ModelAction::List => pipeline::models(project()),
            ModelAction::Show { model } => pipeline::show_model(project(), model),
        },
        Command::Schema { command } => reference::schema(&command.join(" ")),
        Command::Skill { out } => reference::skill(out.as_deref()),
    };
    output::print(cli.json, result)
}

/// A usage error, with clap's "did you mean" as `next` when it names a
/// command, or as a hint otherwise.
fn usage_error(clap_error: &str) -> SynkError {
    let message = clap_error
        .lines()
        .next()
        .unwrap_or_default()
        .trim_start_matches("error: ");
    let err = SynkError::new(ErrorCode::InvalidArgs, message);
    let tip = clap_error
        .lines()
        .find_map(|l| l.trim().strip_prefix("tip: "));
    let quoted = tip.and_then(|t| t.split('\'').nth(1));
    match (tip, quoted) {
        (Some(t), Some(name)) if t.contains("subcommand") => err.with_next(format!("synk {name}")),
        (Some(t), _) => err.with_hint(t).with_next("synk --help"),
        _ => err.with_next("synk --help"),
    }
}

/// The parsed command tree, for `synk schema`.
pub fn cli_command() -> clap::Command {
    use clap::CommandFactory;
    Cli::command()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_lists_every_command() {
        for cmd in Cli::command().get_subcommands() {
            let name = cmd.get_name();
            if cmd.is_hide_set() || name == "help" {
                continue;
            }
            assert!(
                HELP_TEMPLATE.contains(&format!("\n  {name} ")),
                "`{name}` is missing from HELP_TEMPLATE"
            );
        }
    }

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("9m").unwrap(), Duration::from_secs(540));
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_duration("2").unwrap(), Duration::from_secs(2));
        assert!(parse_duration("soon").is_err());
    }
}
