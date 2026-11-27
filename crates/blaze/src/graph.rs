//! CUDA capture and replay. Graph keys, warmup and memory budgets belong to the executor.

use crate::gpu::{Stream, check};
use anyhow::{Result, ensure};
use blaze_sys as sys;
use std::ptr::null_mut;

pub(crate) struct Capture(sys::cudaGraph_t);

impl Capture {
    pub fn record(stream: &Stream, enqueue: impl FnOnce() -> Result<()>) -> Result<Self> {
        check(
            unsafe { sys::cudaStreamBeginCapture(stream.0, sys::CAPTURE_MODE_THREAD_LOCAL) },
            "cudaStreamBeginCapture",
        )?;
        let recorded = enqueue();
        let mut capture = Self(null_mut());
        let ended = unsafe { sys::cudaStreamEndCapture(stream.0, &mut capture.0) };
        recorded?;
        check(ended, "cudaStreamEndCapture")?;
        ensure!(!capture.0.is_null(), "CUDA capture produced no graph");
        Ok(capture)
    }

    pub fn instantiate(self) -> Result<CudaGraph> {
        let mut graph = CudaGraph(null_mut());
        check(
            unsafe { sys::cudaGraphInstantiate(&mut graph.0, self.0, 0) },
            "cudaGraphInstantiate",
        )?;
        Ok(graph)
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cudaGraphDestroy(self.0) };
        }
    }
}

pub(crate) struct CudaGraph(sys::cudaGraphExec_t);

impl CudaGraph {
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        check(
            unsafe { sys::cudaGraphLaunch(self.0, stream.0) },
            "cudaGraphLaunch",
        )
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { sys::cudaGraphExecDestroy(self.0) };
        }
    }
}
