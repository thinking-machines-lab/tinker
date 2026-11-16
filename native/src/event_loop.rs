//! Run code on a Python event loop's thread from any native thread without acquiring a GIL.
//!
//! This is primarily used for completing a Python future from a native thread in order to write
//! native asynchronous functions that can be called from Python and return Python futures.
//!
//! The LoopQueue is a thread-safe queue that is used to queue callbacks and then wake up the
//! loop to run the callbacks once the Python event loop is ready to run the callbacks.
//!
//! The event loop itself needs to acquire the GIL, but the code registering the callbacks
//! never acquires the GIL;.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyAttributeError;
use pyo3::prelude::*;

// Unix builds compile the notifier only for the tests, which run every wake-up.
#[cfg(any(not(unix), test))]
mod notifier_thread;
#[cfg(unix)]
mod socket_pair;

#[cfg(any(not(unix), test))]
pub use notifier_thread::NotifierThreadWakeup;
#[cfg(unix)]
pub use socket_pair::SocketPairWakeup;

/// Platform-dependent type for waking up the event loop to drain the queue.
///
/// On unix, use socket pair wakeup which uses fewer threads and GIL acquisitions
/// compared to the notifier thread wakeup.
#[cfg(unix)]
type PlatformWakeup = SocketPairWakeup;
/// Windows has no Unix sockets, and its default proactor loop no `add_reader` to
/// watch one with.
#[cfg(not(unix))]
type PlatformWakeup = NotifierThreadWakeup;

/// The attribute on the loop object that holds the queue.
/// This ensures that each Python loop object has a single queue.
const QUEUE_ATTR: &str = "_tinker_native_loop_queue";

/// The function that runs on the event loop thread, typically the completion
/// of one Python future with the outcome of native work.
type Callback = Box<dyn FnOnce(Python<'_>) + Send>;

/// Trait representing a way to wake up the event loop to check the queue for
/// new work.
///
/// This is a trait to allow platform-specific implementations for waking up
/// the event loop.
pub trait Wakeup: Send + Sync + 'static {
    /// Create a new instance of the wakeup implementation for the given event loop.
    fn new(event_loop: &Bound<'_, PyAny>) -> PyResult<Self>
    where
        Self: Sized;

    /// Register a function for draining ther queue when the event loop is woken, with the
    /// drain function running on the event loop thread.
    fn register(&self, event_loop: &Bound<'_, PyAny>, drain: &Bound<'_, PyAny>) -> PyResult<()>;

    /// Wake up the event loop to check the queue for new work and drain the queue.
    /// Callable from any thread and the calling thread never acquires the GIL.
    fn wake(&self);

    /// Called by the event loop when it starts to drain the queue to reset any
    /// wakeup state so that the next [`Self::wake`] invocation wakes the loop again.
    fn clear(&self);
}

/// Maintains a queue of callbacks to run on a given event loop thread.
pub struct LoopQueue {
    callbacks: Mutex<Vec<Callback>>,
    wakeup: Box<dyn Wakeup>,
}

impl LoopQueue {
    /// Get the queue for a given Python event loop, creating and attaching a new queue to
    /// the loop if no queue is already attached.
    pub fn of(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<Arc<Self>> {
        match event_loop.getattr(QUEUE_ATTR) {
            Ok(handle) => Ok(handle.extract::<PyRef<'_, QueueHandle>>()?.queue.clone()),
            Err(error) if error.is_instance_of::<PyAttributeError>(py) => {
                Self::attach::<PlatformWakeup>(py, event_loop)
            }
            Err(error) => Err(error),
        }
    }

    /// Create and attach a new queue to the event loop using the specified wakeup implementation.
    pub fn attach<W: Wakeup>(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<Arc<Self>> {
        let this = Arc::new(Self {
            callbacks: Mutex::default(),
            wakeup: Box::new(W::new(event_loop)?),
        });
        let handle = Bound::new(
            py,
            QueueHandle {
                queue: this.clone(),
            },
        )?;
        this.wakeup.register(event_loop, handle.as_any())?;
        event_loop.setattr(QUEUE_ATTR, handle)?;
        Ok(this)
    }

    /// Schedule a new callback to run on the event loop thread.
    /// This method never acquires the GIL, but the callback will not execute until the event loop thread
    /// acquires the GIL.
    ///
    /// The callback should never panic. While a panicking callback will be caught and re-raised on the
    /// event loop, callers should instead handle errors and log or report them appropriately rather than
    /// relying on the event loop exception handler.
    pub fn schedule(&self, callback: impl FnOnce(Python<'_>) + Send + 'static) {
        let was_empty = {
            let mut callbacks = self.lock();
            callbacks.push(Box::new(callback));
            callbacks.len() == 1
        };

        // only wake loop if this is the first callback to be queued to avoid unnecessary wakeups
        if was_empty {
            self.wakeup.wake();
        }
    }

    /// Drain the queue and run all the calbacks on the event loop thread.
    fn drain(&self, py: Python<'_>) {
        self.wakeup.clear();
        // Taken before they run: a callback that queues another leaves it to the next
        // wake-up, and the lock is never held while a callback runs.
        let callbacks = std::mem::take(&mut *self.lock());
        run_all(py, callbacks);
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Callback>> {
        self.callbacks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Run all of the callbacks on the event loop thread, catching any panics and re-raising the first one
/// one the event loop thread.
///
/// This ensures that all callbacks are run and not left hanging if a callback panics.
///
/// In general, callbacks should not panic. No external code should ever be registered as a callback,
/// so if a callback does panic, it is an internal programming error.
fn run_all(py: Python<'_>, callbacks: Vec<Callback>) {
    let panics: Vec<_> = callbacks
        .into_iter()
        // A callback is consumed by its call, so nothing it left broken is used again.
        .filter_map(|callback| catch_unwind(AssertUnwindSafe(|| callback(py))).err())
        .collect();
    if let Some(first) = panics.into_iter().next() {
        resume_unwind(first);
    }
}

/// The "handle" that is attached to the event loop object to ensure each event loop has a single queue.
#[pyclass(frozen, weakref)]
struct QueueHandle {
    queue: Arc<LoopQueue>,
}

#[pymethods]
impl QueueHandle {
    fn __call__(&self, py: Python<'_>) {
        self.queue.drain(py);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::ffi::CStr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    use pyo3::types::{PyCFunction, PyModule, PyTuple};

    use super::*;

    /// Runs `main(*args)` on a fresh loop, as `asyncio.run` would.
    pub(crate) fn run_on_loop<'py>(
        py: Python<'py>,
        main_source: &CStr,
        args: impl IntoPyObject<'py, Target = PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let harness = PyModule::from_code(
            py,
            c"import asyncio

def run(main, args):
    with asyncio.Runner() as runner:
        return runner.run(main(*args))
",
            c"event_loop_test_harness.py",
            c"event_loop_test_harness",
        )?;
        let main = PyModule::from_code(
            py,
            main_source,
            c"event_loop_test_main.py",
            c"event_loop_test_main",
        )?
        .getattr("main")?;
        harness.getattr("run")?.call1((main, args))
    }

    /// Attaches a new queue to a loop.
    type AttachQueue = fn(Python<'_>, &Bound<'_, PyAny>) -> PyResult<Arc<LoopQueue>>;

    /// Every wake-up this platform can run, by name. Each test runs with each of
    /// them, so the notifier thread is tested where it is not the platform's own.
    const WAKEUPS: &[(&str, AttachQueue)] = &[
        #[cfg(unix)]
        ("socket pair", LoopQueue::attach::<SocketPairWakeup>),
        ("notifier thread", LoopQueue::attach::<NotifierThreadWakeup>),
    ];

    fn thread_ident(py: Python<'_>) -> u64 {
        py.import("threading")
            .unwrap()
            .call_method0("get_ident")
            .unwrap()
            .extract()
            .unwrap()
    }

    #[test]
    fn callbacks_run_on_the_loop_thread_in_order() {
        Python::initialize();
        for &(wakeup, attach_queue) in WAKEUPS {
            Python::attach(|py| {
                // `start(loop, seen, done)`: takes the loop's queue and, from a thread
                // that never holds the GIL, queues callbacks that record where they run.
                let start = PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<()> {
                        let py = args.py();
                        let event_loop = attach_queue(py, &args.get_item(0)?)?;
                        assert!(Arc::ptr_eq(
                            &event_loop,
                            &LoopQueue::of(py, &args.get_item(0)?)?
                        ));
                        let seen = args.get_item(1)?;
                        let done = args.get_item(2)?;
                        let targets: Vec<_> = (0..4)
                            .map(|_| (seen.clone().unbind(), done.clone().unbind()))
                            .collect();
                        thread::spawn(move || {
                            for (i, (seen, done)) in targets.into_iter().enumerate() {
                                event_loop.schedule(move |py| {
                                    seen.call_method1(py, "append", ((i, thread_ident(py)),))
                                        .unwrap();
                                    if i == 3 {
                                        done.call_method1(py, "set_result", (py.None(),)).unwrap();
                                    }
                                });
                            }
                        });
                        Ok(())
                    },
                )
                .unwrap();
                let (seen, loop_thread): (Vec<(i64, u64)>, u64) = run_on_loop(
                    py,
                    c"import asyncio, threading

async def main(start):
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    seen = []
    start(loop, seen, done)
    await asyncio.wait_for(done, 10)
    return seen, threading.get_ident()
",
                    (start,),
                )
                .unwrap()
                .extract()
                .unwrap();
                let expected: Vec<_> = (0..4).map(|i| (i, loop_thread)).collect();
                assert_eq!(seen, expected, "{wakeup}");
            });
        }
    }

    #[test]
    fn a_panicking_callback_does_not_stop_the_callbacks_behind_it() {
        Python::initialize();
        for &(wakeup, attach_queue) in WAKEUPS {
            Python::attach(|py| {
                // `start(loop, seen, done)`: queues a callback, one that panics, and the one
                // that completes `done`.
                let start = PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<()> {
                        let py = args.py();
                        let event_loop = attach_queue(py, &args.get_item(0)?)?;
                        let seen = args.get_item(1)?.unbind();
                        let done = args.get_item(2)?.unbind();
                        event_loop.schedule(move |py| {
                            seen.call_method1(py, "append", (0,)).unwrap();
                        });
                        event_loop.schedule(|_| panic!("callback 1 panics"));
                        event_loop.schedule(move |py| {
                            done.call_method1(py, "set_result", (py.None(),)).unwrap();
                        });
                        Ok(())
                    },
                )
                .unwrap();
                let (seen, reported): (Vec<i64>, Vec<String>) = run_on_loop(
                    py,
                    c"import asyncio

async def main(start):
    loop = asyncio.get_running_loop()
    reported = []
    loop.set_exception_handler(
        lambda loop, context: reported.append(str(context['exception']))
    )
    done = loop.create_future()
    seen = []
    start(loop, seen, done)
    await asyncio.wait_for(done, 10)
    return seen, reported
",
                    (start,),
                )
                .unwrap()
                .extract()
                .unwrap();
                assert_eq!(seen, [0], "{wakeup}");
                assert_eq!(reported, ["callback 1 panics"], "{wakeup}");
            });
        }
    }

    #[test]
    fn scheduling_never_waits_for_the_gil() {
        Python::initialize();
        for &(wakeup, attach_queue) in WAKEUPS {
            Python::attach(|py| {
                // `start(loop, done)`: with the GIL held here, a thread that never
                // holds it queues a callback; `start` returns only once that thread is
                // through, which a wake-up that waited for the GIL would never be.
                let start = PyCFunction::new_closure(
                    py,
                    None,
                    None,
                    move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<()> {
                        let event_loop = attach_queue(args.py(), &args.get_item(0)?)?;
                        let done = args.get_item(1)?.unbind();
                        let (scheduled, was_scheduled) = std::sync::mpsc::channel();
                        thread::spawn(move || {
                            event_loop.schedule(move |py| {
                                done.call_method1(py, "set_result", (py.None(),)).unwrap();
                            });
                            scheduled.send(()).unwrap();
                        });
                        was_scheduled
                            .recv_timeout(Duration::from_secs(10))
                            .unwrap_or_else(|_| panic!("scheduling waited for the GIL: {wakeup}"));
                        Ok(())
                    },
                )
                .unwrap();
                run_on_loop(
                    py,
                    c"import asyncio

async def main(start):
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    start(loop, done)
    await asyncio.wait_for(done, 10)
",
                    (start,),
                )
                .unwrap();
            });
        }
    }

    #[test]
    fn a_closed_loop_never_runs_its_callbacks() {
        Python::initialize();
        for &(wakeup, attach_queue) in WAKEUPS {
            Python::attach(|py| {
                let kept: Arc<Mutex<Option<Arc<LoopQueue>>>> = Arc::default();
                let keep = {
                    let kept = kept.clone();
                    PyCFunction::new_closure(
                        py,
                        None,
                        None,
                        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<()> {
                            *kept.lock().unwrap() =
                                Some(attach_queue(args.py(), &args.get_item(0)?)?);
                            Ok(())
                        },
                    )
                    .unwrap()
                };
                run_on_loop(
                    py,
                    c"import asyncio

async def main(keep):
    keep(asyncio.get_running_loop())
",
                    (keep,),
                )
                .unwrap();
                // The loop is closed; the queue outlives it in the hands of a task.
                let event_loop = kept.lock().unwrap().take().unwrap();
                let ran = Arc::new(AtomicBool::new(false));
                let callback_ran = ran.clone();
                py.detach(|| {
                    event_loop.schedule(move |_| callback_ran.store(true, Ordering::SeqCst))
                });
                assert!(!ran.load(Ordering::SeqCst), "{wakeup}");
            });
        }
    }
}
