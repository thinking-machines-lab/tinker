//! Waking a loop through a socket pair it watches.

use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;

use pyo3::prelude::*;

use super::Wakeup;

/// Wakeup implementation that uses a unix socket pair to wake the event loop.
/// Only works on unix-like systems, but has the benefit of not needing any
/// additional notifier thread and only requiring the GIL to be acquired
/// once by the event loop thread.
pub struct SocketPairWakeup {
    /// Wakes the event loop by writing a byte.
    write: UnixStream,

    /// Added as a reader to the event loop to wake it.
    ///
    /// Kept open as long as the write side of the stream is open so that
    /// any callbacks that terminate after the event loop has closed do not
    /// error on a closed socket.
    read: UnixStream,
}

impl Wakeup for SocketPairWakeup {
    fn new(_event_loop: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (read, write) = UnixStream::pair()?;
        read.set_nonblocking(true)?;
        write.set_nonblocking(true)?;
        Ok(Self { write, read })
    }

    fn register(&self, event_loop: &Bound<'_, PyAny>, drain: &Bound<'_, PyAny>) -> PyResult<()> {
        // Register the reader with the event loop via [asyncio.loop.add_reader](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.add_reader) and with the given drain function as the callback to
        // run when the file descriptor is read
        event_loop.call_method1("add_reader", (self.read.as_raw_fd(), drain))?;
        Ok(())
    }

    fn wake(&self) {
        // ignore the result of the write since it means the event loop is already wake
        let _ = (&self.write).write(&[1]);
    }

    fn clear(&self) {
        let mut wakeups = [0; 64];
        // read all bytes to avoid busy looping with an empty queue
        while matches!((&self.read).read(&mut wakeups), Ok(1..)) {}
    }
}
