//! Convert Rust futures to `asyncio.Future` objects via [`spawn_future`].

use std::any::Any;
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};

use futures_util::FutureExt;
use pyo3::panic::PanicException;
use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use tokio::task::AbortHandle;

use crate::event_loop::LoopQueue;
use crate::runtime::runtime;

/// Runs `future` on the shared tokio runtime and returns the `asyncio.Future` that completes
/// with the result of the future on the current running event loop.
/// Cancelling the returned `asyncio.Future` aborts `future`.
pub fn spawn_future<'py, F, T, E>(py: Python<'py>, future: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = Result<T, E>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
    E: Into<PyErr> + Send + 'static,
{
    let runtime = runtime()?;
    let running = running_loop(py)?;
    let event_loop = LoopQueue::of(py, &running)?;
    let py_future = running.call_method0("create_future")?;
    // Weak: a strong reference from the queue would tie the future, and through it
    // the loop, to a queue the loop itself keeps.
    let target = py
        .import("weakref")?
        .call_method1("ref", (&py_future,))?
        .unbind();
    let task = runtime.tokio().spawn(async move {
        // We should never write futures that panic, but for defensiveness, a panic is
        // returned as an error on the future rather than leaving the future hanging.
        let outcome = AssertUnwindSafe(future).catch_unwind().await;

        // schedule completion of the future on the event loop thread without blocking
        // the tokio runtime worker thread
        event_loop.schedule(move |py| complete(py, &target, outcome));
    });
    let abort = Py::new(
        py,
        AbortOnCancel {
            task: task.abort_handle(),
        },
    )?;
    if let Err(error) = py_future.call_method1("add_done_callback", (&abort,)) {
        // If there is an error adding the done callback, treat it as a failure
        // Abort the underlying future and then return the error
        abort.get().task.abort();
        return Err(error);
    }
    Ok(py_future)
}

/// The `asyncio` loop running on the calling thread.
fn running_loop(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("asyncio")?.call_method0("get_running_loop")
}

/// Callback wrapper to pass to asyncio.Future.add_done_callback to
/// abort the underlying future if cancelled from Python.
#[pyclass(frozen)]
struct AbortOnCancel {
    task: AbortHandle,
}

#[pymethods]
impl AbortOnCancel {
    fn __call__(&self, future: &Bound<'_, PyAny>) -> PyResult<()> {
        if future.call_method0("cancelled")?.is_truthy()? {
            self.task.abort();
        }
        Ok(())
    }
}

/// Complete the asyncio.Future object in `target` with the given outcome
fn complete<T, E>(
    py: Python<'_>,
    target: &Py<PyAny>,
    outcome: Result<Result<T, E>, Box<dyn Any + Send>>,
) where
    T: for<'a> IntoPyObject<'a>,
    E: Into<PyErr>,
{
    let target = match target.call0(py) {
        // Gone: nobody is waiting for the outcome.
        Ok(target) if target.is_none(py) => return,
        Ok(target) => target,
        Err(error) => return error.write_unraisable(py, None),
    };
    let result = match outcome {
        // The future ran to its own result: a value or a regular error.
        // catch_unwind is just extra defensiveness to ensure that if anything in the conversion panics
        // we propagate that as an exception on the future rather than
        // leaving the future hanging
        Ok(result) => catch_unwind(AssertUnwindSafe(|| {
            result
                .map_err(Into::into)
                .and_then(|value| value.into_py_any(py))
        }))
        .unwrap_or_else(|panic| Err(panic_exception(panic))),
        // The future panicked.
        Err(panic) => Err(panic_exception(panic)),
    };
    if let Err(error) = set_outcome(py, &target, result) {
        error.write_unraisable(py, Some(target.bind(py)));
    }
}

fn set_outcome(py: Python<'_>, target: &Py<PyAny>, result: PyResult<Py<PyAny>>) -> PyResult<()> {
    // A future the caller cancelled while the task ran takes no outcome.
    if target.call_method0(py, "cancelled")?.is_truthy(py)? {
        return Ok(());
    }
    match result {
        Ok(value) => target.call_method1(py, "set_result", (value,))?,
        Err(error) => target.call_method1(py, "set_exception", (error,))?,
    };
    Ok(())
}

/// The exception a synchronous `#[pyfunction]` raises for the same panic.
fn panic_exception(payload: Box<dyn Any + Send>) -> PyErr {
    let message = if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else {
        "panic from Rust code".to_owned()
    };
    PanicException::new_err(message)
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::time::{Duration, Instant};

    use pyo3::exceptions::PyValueError;
    use tokio::sync::watch;

    use super::*;
    use crate::event_loop::tests::run_on_loop;

    /// Hands out bridged futures that run to their outcome only once it is released,
    /// so a test observes each state of a future without timing assumptions.
    #[pyclass(frozen, name = "Probe")]
    struct Probe {
        released: watch::Sender<bool>,
        /// Futures that have run to their outcome.
        passed: AtomicUsize,
        /// Futures whose task has neither finished nor been aborted.
        in_flight: Mutex<usize>,
        idle: Condvar,
    }

    /// Counts one in-flight future from creation until its task is dropped: on
    /// completion, or on abort.
    struct InFlight {
        probe: Py<Probe>,
    }

    impl InFlight {
        fn new(probe: Py<Probe>) -> Self {
            *probe.get().in_flight.lock().unwrap() += 1;
            Self { probe }
        }
    }

    impl Drop for InFlight {
        fn drop(&mut self) {
            let probe = self.probe.get();
            *probe.in_flight.lock().unwrap() -= 1;
            probe.idle.notify_all();
        }
    }

    #[pymethods]
    impl Probe {
        #[new]
        fn new() -> Self {
            Self {
                released: watch::Sender::new(false),
                passed: AtomicUsize::new(0),
                in_flight: Mutex::new(0),
                idle: Condvar::new(),
            }
        }

        /// Waits, with the GIL released, until no future of the probe has a task
        /// running. A task ends by running to its outcome or by being aborted, so
        /// after a cancellation this returns once the abort has taken effect.
        fn wait_idle(&self, py: Python<'_>) {
            py.detach(|| {
                let in_flight = self.in_flight.lock().unwrap();
                let (in_flight, _) = self
                    .idle
                    .wait_timeout_while(in_flight, Duration::from_secs(10), |n| *n > 0)
                    .unwrap();
                assert_eq!(*in_flight, 0, "tasks still running");
            });
        }

        fn release(&self) {
            self.released.send_replace(true);
        }

        fn passed(&self) -> usize {
            self.passed.load(Ordering::SeqCst)
        }

        /// Spins, holding the GIL, until `count` futures have run to their outcome.
        fn wait_passed(&self, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.passed() < count {
                assert!(
                    Instant::now() < deadline,
                    "{} of {count} futures ran to their outcome while the GIL was held",
                    self.passed()
                );
                std::thread::yield_now();
            }
        }

        fn complete_with<'py>(slf: &Bound<'py, Self>, value: i64) -> PyResult<Bound<'py, PyAny>> {
            let pass = Self::pass(slf);
            spawn_future::<_, _, PyErr>(slf.py(), async move {
                pass.await;
                Ok(value)
            })
        }

        fn fail_with<'py>(slf: &Bound<'py, Self>, message: String) -> PyResult<Bound<'py, PyAny>> {
            let pass = Self::pass(slf);
            spawn_future::<_, (), _>(slf.py(), async move {
                pass.await;
                Err(PyValueError::new_err(message))
            })
        }

        fn panic_with<'py>(slf: &Bound<'py, Self>, message: String) -> PyResult<Bound<'py, PyAny>> {
            let pass = Self::pass(slf);
            spawn_future::<_, (), PyErr>(slf.py(), async move {
                pass.await;
                panic!("{message}");
            })
        }
    }

    impl Probe {
        /// Resolves once the probe is released, then counts the pass.
        fn pass(slf: &Bound<'_, Self>) -> impl Future<Output = ()> + Send + 'static {
            let in_flight = InFlight::new(slf.clone().unbind());
            let mut released = slf.get().released.subscribe();
            async move {
                released
                    .wait_for(|released| *released)
                    .await
                    .expect("the probe outlives its futures");
                in_flight.probe.get().passed.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// Runs `main(probe)` on a fresh loop and returns what it returns.
    fn run<'py>(py: Python<'py>, main_source: &CStr) -> PyResult<Bound<'py, PyAny>> {
        run_with(py, &Probe::new().into_pyobject(py)?, main_source)
    }

    fn run_with<'py>(
        py: Python<'py>,
        probe: &Bound<'py, Probe>,
        main_source: &CStr,
    ) -> PyResult<Bound<'py, PyAny>> {
        run_on_loop(py, main_source, (probe,))
    }

    #[test]
    fn a_value_becomes_the_result() {
        Python::initialize();
        Python::attach(|py| {
            let value: i64 = run(
                py,
                c"import asyncio

async def main(probe):
    future = probe.complete_with(42)
    assert isinstance(future, asyncio.Future)
    probe.release()
    return await asyncio.wait_for(future, 10)
",
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(value, 42);
        });
    }

    #[test]
    fn an_error_becomes_the_exception() {
        Python::initialize();
        Python::attach(|py| {
            let message: String = run(
                py,
                c"import asyncio

async def main(probe):
    future = probe.fail_with('boom')
    probe.release()
    try:
        await asyncio.wait_for(future, 10)
    except ValueError as error:
        return str(error)
    raise AssertionError('no error')
",
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(message, "boom");
        });
    }

    #[test]
    fn a_panic_becomes_an_exception() {
        Python::initialize();
        Python::attach(|py| {
            // Read through `exception()` rather than raised: pyo3 resumes a
            // PanicException it fetches as a Rust panic, which would end the test
            // instead of informing it.
            let exception = run(
                py,
                c"import asyncio

async def main(probe):
    future = probe.panic_with('native panic')
    probe.release()
    await asyncio.wait([future], timeout=10)
    return future.exception()
",
            )
            .unwrap();
            assert!(exception
                .is_instance(&py.get_type::<PanicException>())
                .unwrap());
            assert_eq!(exception.to_string(), "native panic");
        });
    }

    #[test]
    fn tasks_run_to_their_outcome_while_python_holds_the_gil() {
        Python::initialize();
        Python::attach(|py| {
            // `wait_passed` holds the GIL throughout: no Python code, and no native
            // thread waiting for the GIL, can run. The tasks still run to their
            // outcome on the runtime workers; only their completion waits for the loop.
            let values: Vec<i64> = run(
                py,
                c"import asyncio

async def main(probe):
    futures = [probe.complete_with(i) for i in range(8)]
    probe.release()
    probe.wait_passed(8)
    return await asyncio.gather(*futures)
",
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(values, (0..8).collect::<Vec<_>>());
        });
    }

    #[test]
    fn a_completion_queued_after_the_loop_closed_does_not_keep_the_loop_alive() {
        Python::initialize();
        Python::attach(|py| {
            let probe = Probe::new().into_pyobject(py).unwrap();
            // The future is neither awaited nor kept, and its task completes only
            // after the loop has closed: the completion is queued for a loop that
            // will never drain it.
            let loop_ref = run_with(
                py,
                &probe,
                c"import asyncio, weakref

async def main(probe):
    loop = asyncio.get_running_loop()
    probe.complete_with(1)
    return weakref.ref(loop)
",
            )
            .unwrap();
            probe.get().release();
            probe.get().wait_passed(1);
            // The queued completion must not keep the loop alive: it holds the
            // future weakly, and the future is gone with the coroutine's frame.
            let gc = py.import("gc").unwrap();
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                gc.call_method0("collect").unwrap();
                if loop_ref.call0().unwrap().is_none() {
                    break;
                }
                assert!(Instant::now() < deadline, "the closed loop is still alive");
                py.detach(std::thread::yield_now);
            }
        });
    }

    #[test]
    fn cancelling_the_future_aborts_the_task() {
        Python::initialize();
        Python::attach(|py| {
            // The abort has taken effect once the task is gone; releasing the probe
            // afterwards lets nothing through, so the task did not run to its outcome.
            let passed: usize = run(
                py,
                c"import asyncio

async def main(probe):
    future = probe.complete_with(1)
    future.cancel()
    await asyncio.sleep(0)  # the done callback runs on the next iteration
    probe.wait_idle()
    probe.release()
    return probe.passed()
",
            )
            .unwrap()
            .extract()
            .unwrap();
            assert_eq!(passed, 0);
        });
    }
}
