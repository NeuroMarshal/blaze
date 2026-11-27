use crate::gpu::{Stream, check};
use anyhow::{Result, bail, ensure};
use blaze_sys as sys;
use std::ptr::null_mut;

pub struct MoeEngine(*mut sys::blz_moe_engine);

unsafe impl Send for MoeEngine {}

impl MoeEngine {
    pub(crate) unsafe fn create(config: &sys::blz_moe_engine_config) -> Result<Self> {
        let mut engine = null_mut();
        let mut error = [0 as std::ffi::c_char; 1024];
        if unsafe { sys::blz_moe_engine_create(config, &mut engine, error.as_mut_ptr(), error.len()) } != 0 {
            let message = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("MoE engine: {message}");
        }
        ensure!(!engine.is_null(), "MoE engine returned a null handle");
        Ok(Self(engine))
    }

    pub(crate) fn layers(&self) -> i32 {
        unsafe { sys::blz_moe_engine_layers(self.0) }
    }

    pub(crate) unsafe fn forward(&self, layer: usize, x: *const u16, tokens: usize, ids: *const i32,
        weights: *const f32, out: *mut u16, stream: &Stream) -> Result<()> {
        check(unsafe {
            sys::blz_moe_engine_forward(self.0, layer as i32, x, tokens as i32, ids, weights, out, stream.0)
        }, "moe")
    }
}

impl Drop for MoeEngine {
    fn drop(&mut self) {
        unsafe { sys::blz_moe_engine_destroy(self.0) };
    }
}
