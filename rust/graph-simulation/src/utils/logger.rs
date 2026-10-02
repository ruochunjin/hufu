//! Process-wide logging used by the optional Rust fixed-point backend.
//!
//! Logging must never change Hyper Simulation semantics.  In particular, a
//! read-only working directory or a missing `logs/` directory must not make a
//! library call panic.  The writer therefore creates the parent directory when
//! possible and falls back to standard output when the file cannot be opened.

use env_logger::Target;
use log::LevelFilter;
use serde::{Deserialize, Serialize};
use std::{
    env,
    error::Error,
    fs::{create_dir_all, File},
    io::{self, Write},
    path::Path,
    sync::{Mutex, OnceLock},
};

struct MultiWriter {
    file: Option<Mutex<File>>,
    stdout: io::Stdout,
}

impl Write for MultiWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if let Some(file) = &self.file {
            file.lock().unwrap().write_all(buffer)?;
        }
        self.stdout.write_all(buffer)?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = &self.file {
            file.lock().unwrap().flush()?;
        }
        self.stdout.flush()
    }
}

static LOGGER_INIT: OnceLock<()> = OnceLock::new();

/// Initialize logging once without making file-system availability fatal.
pub fn init_global_logger_once(output_file: &'static str) {
    let path = Path::new(output_file);
    // Trace writers use the same conventional directory.  Repeat this cheap
    // check on every call because a library may be invoked after `chdir`, even
    // though the process-wide logger itself is initialized only once.
    if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
        let _ = create_dir_all(parent);
    }
    LOGGER_INIT.get_or_init(|| {
        let log_file = File::create(path).ok();

        let level = env::var("RUST_LOG")
            .ok()
            .and_then(|value| match value.to_lowercase().as_str() {
                "error" => Some(LevelFilter::Error),
                "warn" | "warning" => Some(LevelFilter::Warn),
                "info" => Some(LevelFilter::Info),
                "debug" => Some(LevelFilter::Debug),
                "trace" => Some(LevelFilter::Trace),
                _ => None,
            })
            .unwrap_or(LevelFilter::Info);

        // Another embedding application may already own the global logger.
        // `try_init` preserves that logger instead of panicking.
        let _ = env_logger::Builder::new()
            .target(Target::Pipe(Box::new(MultiWriter {
                file: log_file.map(Mutex::new),
                stdout: io::stdout(),
            })))
            .filter_level(level)
            .try_init();
    });
}

pub trait TraceLog: Serialize + for<'de> Deserialize<'de> {
    fn store_trace_file(self, filename: &'static str) -> Result<(), Box<dyn Error>>;
    fn get_trace(filename: &'static str) -> Result<Self, Box<dyn Error>>;
}
