//! `tinker._native`: the native extension module of the Tinker Python SDK.
//!
//! Built by maturin from the SDK's `pyproject.toml` and shipped inside the
//! `tinker` wheel; `src/tinker/_native.pyi` declares its Python interface.
//!
//! Native work runs on the process-wide [`runtime`] and reaches Python through the
//! [`bridge`], as `asyncio.Future` objects. What needs the interpreter runs on the
//! calling loop's own thread, through [`event_loop`].

use pyo3::prelude::*;

pub mod bridge;
pub mod event_loop;
pub mod runtime;

#[pymodule]
fn _native(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
