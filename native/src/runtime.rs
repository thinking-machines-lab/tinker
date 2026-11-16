//! The process-wide Tokio runtime behind every native operation of the SDK.
//!
//! The runtime starts on first use and lives until the process exits: its worker
//! threads are detached, so nothing waits for them at interpreter shutdown. The
//! workers never touch the Python interpreter, so a task keeps making progress while
//! Python code holds the GIL. The SDK's native work is network I/O, which a few
//! threads multiplex the way one thread runs an `asyncio` loop, so the runtime has
//! two worker threads, the second so that a task is not held up behind a busy one,
//! unless `TINKER_NATIVE_WORKER_THREADS` asks for another count.
//!
//! A Tokio runtime does not survive `fork`: the child inherits the memory of the
//! parent's runtime but none of its threads, so any use of it would block forever.
//! [`runtime`] remembers which process started the runtime and fails in a forked child
//! instead. `multiprocessing` users must start workers with the `spawn` or `forkserver`
//! method, or fork before the parent's first native call.

use std::ffi::OsStr;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::LazyLock;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Default number of worker threads in the Tokio runtime.
/// Override with `TINKER_NATIVE_WORKER_THREADS` environment variable.
pub const DEFAULT_WORKER_THREADS: usize = 2;

const WORKER_THREADS_VAR: &str = "TINKER_NATIVE_WORKER_THREADS";

/// The process that started the runtime, `0` until the first use.
/// Used to detect and fail on forked children.
static OWNER_PID: AtomicU32 = AtomicU32::new(0);

/// Global singleton runtime instance for all native work in the process.
static RUNTIME: LazyLock<Result<Runtime, RuntimeUnavailable>> = LazyLock::new(Runtime::start);

/// The crate's runtime: one per process, started on first use.
pub struct Runtime {
    tokio: tokio::runtime::Runtime,
}

impl Runtime {
    fn start() -> Result<Self, RuntimeUnavailable> {
        let tokio = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads()?.get())
            .thread_name("tinker-native")
            .enable_all()
            .build()
            .expect("the tinker native runtime failed to start");
        Ok(Self { tokio })
    }

    /// The Tokio runtime, for spawning tasks on its worker threads.
    pub fn tokio(&self) -> &tokio::runtime::Runtime {
        &self.tokio
    }
}

/// The runtime of this process, started on first call.
///
/// Fails in a child forked after the parent started the runtime, and when
/// `TINKER_NATIVE_WORKER_THREADS` is set to anything but a positive integer.
pub fn runtime() -> Result<&'static Runtime, RuntimeUnavailable> {
    claim_process()?;
    RUNTIME.as_ref().map_err(Clone::clone)
}

/// Claims the runtime for the current process before it is built, so that a child
/// forked while the parent is mid-build fails rather than blocks.
fn claim_process() -> Result<(), RuntimeUnavailable> {
    let pid = std::process::id();

    match OWNER_PID.compare_exchange(0, pid, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => Ok(()),
        Err(owner) if owner == pid => Ok(()),
        Err(owner) => Err(RuntimeUnavailable {
            message: format!(
                "process {pid} is a fork of process {owner}, which started the tinker \
                 native runtime; a forked child cannot use it. Start worker processes with \
                 the spawn or forkserver method, or fork before the parent's first native \
                 call."
            ),
        }),
    }
}

fn worker_threads() -> Result<NonZeroUsize, RuntimeUnavailable> {
    match std::env::var_os(WORKER_THREADS_VAR) {
        Some(setting) => parse_worker_threads(&setting),
        None => Ok(NonZeroUsize::new(DEFAULT_WORKER_THREADS).expect("a positive default")),
    }
}

fn parse_worker_threads(setting: &OsStr) -> Result<NonZeroUsize, RuntimeUnavailable> {
    setting
        .to_str()
        .and_then(|value| value.trim().parse().ok())
        .ok_or_else(|| RuntimeUnavailable {
            message: format!(
                "{WORKER_THREADS_VAR}={setting:?} is not a positive integer; unset it for \
                 the default of {DEFAULT_WORKER_THREADS}"
            ),
        })
}

/// Single error type for all runtime-related errors.
#[derive(Debug, Clone)]
pub struct RuntimeUnavailable {
    message: String,
}

impl fmt::Display for RuntimeUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeUnavailable {}

impl From<RuntimeUnavailable> for PyErr {
    fn from(error: RuntimeUnavailable) -> Self {
        PyRuntimeError::new_err(error.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn a_forked_child_is_refused() {
        let _ = runtime().unwrap();
        // SAFETY: the child only reads an atomic, formats the refusal and leaves
        // through `_exit`, never returning into the test harness or touching the
        // parent's runtime threads.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork: {}", std::io::Error::last_os_error());
        if pid == 0 {
            let code = match runtime() {
                Err(error) if error.to_string().contains("is a fork of process") => 0,
                Err(_) => 2,
                Ok(_) => 3,
            };
            // SAFETY: `_exit` ends the child without running the parent's exit handlers.
            unsafe { libc::_exit(code) };
        }
        let mut status = 0;
        // SAFETY: reaps the child forked above.
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(libc::WIFEXITED(status), "child did not exit: {status:#x}");
        assert_eq!(
            libc::WEXITSTATUS(status),
            0,
            "2: refused for another reason, 3: the forked child got the runtime"
        );
    }
}
