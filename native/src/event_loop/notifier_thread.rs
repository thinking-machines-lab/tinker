//! Waking a loop from a thread that is allowed to wait for the GIL.

use std::sync::{mpsc, Arc, LazyLock, OnceLock};
use std::thread;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::Wakeup;

/// Wakeup implementation that uses a notifier thread to invoke `call_soon_threadsafe` on the event loop.
/// The notifier thread needs to acquire the GIL to invoke call_soon_threadsafe.
/// However, the calling thread does not need to acquire the GIL and instead just sends
/// a message to a listener on the notifier thread.
///
/// Prefer [`SocketPairWakeup`] on unix systems since it does not require the additional
/// notifier thread and requires one fewer GIL acquisition.
pub struct NotifierThreadWakeup {
    /// the event loop to invoke `call_soon_threadsafe` on and the drain function to
    /// hand it, sent to the notifier thread on each wake up.
    target: Arc<WakeTarget>,
}

impl Wakeup for NotifierThreadWakeup {
    fn new(event_loop: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            target: Arc::new(WakeTarget {
                event_loop: WeakLoop {
                    reference: weak_reference(event_loop)?,
                },
                drain: OnceLock::new(),
            }),
        })
    }

    fn register(&self, _event_loop: &Bound<'_, PyAny>, drain: &Bound<'_, PyAny>) -> PyResult<()> {
        // Only remember the drain function. No other registration is needed since the
        // notifier thread is started on first wake up and begins listening for wakeups there.
        self.target
            .drain
            .set(weak_reference(drain)?)
            .map_err(|_| PyRuntimeError::new_err("the wakeup is already registered"))
    }

    fn wake(&self) {
        // Send the event loop to the notifier thread to be woken
        // this ensures that only the notifier thread acquires the GIL to invoke call_soon_threadsafe
        // and not the thread that calls wake
        drop(NOTIFIER.send(self.target.clone()));
    }

    fn clear(&self) {
        // Each wake-up is one call from the loop, so none is left pending.
    }
}

/// What the notifier thread needs to wake one event loop.
struct WakeTarget {
    event_loop: WeakLoop,
    /// A weak reference to the drain function. Wea since the drain
    /// function keeps a strong reference to the queue, and the queue keeps a strong
    /// reference to this wakeup. The event loop keeps the drain function alive.
    drain: OnceLock<Py<PyAny>>,
}

impl WakeTarget {
    /// Has the event loop run the drain function, if both are still alive.
    fn wake(&self, py: Python<'_>) {
        let Some(drain) = self.drain.get() else {
            // Not registered: there is no drain function to run yet.
            return;
        };
        match drain.call0(py) {
            Ok(drain) if !drain.is_none(py) => self.event_loop.call_soon(py, drain.bind(py)),
            // Collected: nothing left to run the callbacks.
            Ok(_) => {}
            Err(error) => error.write_unraisable(py, None),
        }
    }
}

struct WeakLoop {
    /// A weak reference to the event loop since the event loop keeps a strong reference
    /// to the queue handle, and the queue handle keeps a strong reference to this wakeup.
    reference: Py<PyAny>,
}

impl WeakLoop {
    /// Invoke `call_soon_threadsafe` on the event loop with `callback`.
    fn call_soon(&self, py: Python<'_>, callback: &Bound<'_, PyAny>) {
        let event_loop = match self.reference.call0(py) {
            Ok(event_loop) if !event_loop.is_none(py) => event_loop,
            // Collected: nothing left to run the callbacks.
            Ok(_) => return,
            Err(error) => return error.write_unraisable(py, None),
        };
        if let Err(error) = event_loop.call_method1(py, "call_soon_threadsafe", (callback,)) {
            // A closed loop refuses the call, and never runs its callbacks.
            if !error.is_instance_of::<PyRuntimeError>(py) {
                error.write_unraisable(py, Some(event_loop.bind(py)));
            }
        }
    }
}

/// A `weakref.ref` to `object`.
fn weak_reference(object: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let reference = object
        .py()
        .import("weakref")?
        .call_method1("ref", (object,))?;
    Ok(reference.unbind())
}

/// The notifier thread, started by the first wake-up. It waits for the GIL on behalf
/// of every loop it wakes, so no runtime worker ever does, and attaches once per
/// batch of wake-ups.
static NOTIFIER: LazyLock<mpsc::Sender<Arc<WakeTarget>>> = LazyLock::new(|| {
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("tinker-native-notifier".to_owned())
        .spawn(move || run_notifier(&receiver))
        .expect("the tinker native notifier thread failed to start");
    sender
});

/// Interpreter shutdown is not handled. A wake-up that arrives while the interpreter
/// is finalizing, or after, attaches to it all the same, and may panic this thread or
/// crash the process. Checking first would race with the shutdown on the main thread,
/// so only an exit with no native work left to complete is clean.
fn run_notifier(receiver: &mpsc::Receiver<Arc<WakeTarget>>) {
    while let Ok(first) = receiver.recv() {
        // One attach per batch, and one call per loop: a loop woken twice while this
        // thread waited for the GIL drains once.
        let mut batch = vec![first];
        batch.extend(receiver.try_iter());

        // deduplicate so that we only invoke `call_soon_threadsafe` on each event loop once per batch
        batch.sort_by_key(|target| Arc::as_ptr(target) as usize);
        batch.dedup_by(|a, b| Arc::ptr_eq(a, b));
        Python::attach(|py| {
            for target in &batch {
                target.wake(py);
            }
        });
    }
}
