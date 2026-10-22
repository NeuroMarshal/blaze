//! Owned CUDA buffers, pinned host allocations and streams for the blaze-sys C ABI.

use anyhow::{Result, bail};
use blaze_sys as sys;
use std::ffi::{CStr, c_void};
use std::ptr::null_mut;

pub fn check(err: sys::cudaError_t, what: &str) -> Result<()> {
    if err == sys::CUDA_SUCCESS {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static string.
    let msg = unsafe { CStr::from_ptr(sys::cudaGetErrorString(err)) }.to_string_lossy();
    bail!("{what}: {msg} ({err})")
}

pub fn set_device(device: i32) -> Result<()> {
    check(unsafe { sys::cudaSetDevice(device) }, "cudaSetDevice")
}

/// (free, total) bytes of the current device.
pub fn mem_info() -> Result<(usize, usize)> {
    let (mut free, mut total) = (0usize, 0usize);
    check(unsafe { sys::cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo")?;
    Ok((free, total))
}

pub fn synchronize() -> Result<()> {
    check(unsafe { sys::cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

pub struct Stream(pub sys::cudaStream_t);

impl Stream {
    pub fn new() -> Result<Self> {
        let mut s = null_mut();
        check(unsafe { sys::cudaStreamCreateWithFlags(&mut s, sys::STREAM_NON_BLOCKING) }, "cudaStreamCreate")?;
        Ok(Self(s))
    }

    pub fn sync(&self) -> Result<()> {
        check(unsafe { sys::cudaStreamSynchronize(self.0) }, "cudaStreamSynchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::cudaStreamDestroy(self.0) };
    }
}

/// An event for stream ordering (no timing).
pub struct Event(pub sys::cudaEvent_t);

impl Event {
    pub fn new() -> Result<Self> {
        let mut e = null_mut();
        check(unsafe { sys::cudaEventCreateWithFlags(&mut e, sys::EVENT_DISABLE_TIMING) }, "cudaEventCreate")?;
        Ok(Self(e))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::cudaEventDestroy(self.0) };
    }
}

/// Make `waiter` wait for the work enqueued on `on` so far (through `ev`).
pub fn stream_after(waiter: &Stream, on: &Stream, ev: &Event) -> Result<()> {
    check(unsafe { sys::cudaEventRecord(ev.0, on.0) }, "cudaEventRecord")?;
    check(unsafe { sys::cudaStreamWaitEvent(waiter.0, ev.0, 0) }, "cudaStreamWaitEvent")
}

/// Device memory of `bytes` bytes, freed on drop.
pub struct DevBuf {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: a device allocation is not tied to a host thread.
unsafe impl Send for DevBuf {}
unsafe impl Sync for DevBuf {}

impl DevBuf {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = null_mut();
        check(unsafe { sys::cudaMalloc(&mut ptr, bytes.max(1)) }, &format!("cudaMalloc({bytes} B)"))?;
        Ok(Self { ptr, bytes })
    }

    pub fn zeroed(bytes: usize, s: &Stream) -> Result<Self> {
        let b = Self::new(bytes)?;
        b.zero(s)?;
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let b = Self::new(std::mem::size_of_val(data))?;
        b.upload(data)?;
        Ok(b)
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ptr<T>(&self) -> *mut T {
        self.ptr.cast()
    }

    /// Element `index` of a buffer of `T`s (pointer arithmetic, no bounds check beyond a debug assertion).
    pub fn at<T>(&self, index: usize) -> *mut T {
        debug_assert!((index + 1) * std::mem::size_of::<T>() <= self.bytes.max(std::mem::size_of::<T>()));
        unsafe { self.ptr.cast::<T>().add(index) }
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.upload_at(0, data)
    }

    pub fn upload_at<T: Copy>(&self, byte_offset: usize, data: &[T]) -> Result<()> {
        let n = std::mem::size_of_val(data);
        if byte_offset + n > self.bytes {
            bail!("upload of {n} B at {byte_offset} into a {} B buffer", self.bytes);
        }
        check(
            unsafe { sys::cudaMemcpy(self.ptr.cast::<u8>().add(byte_offset).cast(), data.as_ptr().cast(), n, sys::MEMCPY_H2D) },
            "cudaMemcpy H2D",
        )
    }

    pub fn upload_async<T: Copy>(&self, data: &[T], s: &Stream) -> Result<()> {
        let n = std::mem::size_of_val(data);
        if n > self.bytes {
            bail!("upload of {n} B into a {} B buffer", self.bytes);
        }
        check(unsafe { sys::cudaMemcpyAsync(self.ptr, data.as_ptr().cast(), n, sys::MEMCPY_H2D, s.0) }, "cudaMemcpyAsync H2D")
    }

    pub fn download<T: Copy + Default>(&self, count: usize) -> Result<Vec<T>> {
        let mut out = vec![T::default(); count];
        let n = count * std::mem::size_of::<T>();
        if n > self.bytes {
            bail!("download of {n} B from a {} B buffer", self.bytes);
        }
        check(unsafe { sys::cudaMemcpy(out.as_mut_ptr().cast(), self.ptr, n, sys::MEMCPY_D2H) }, "cudaMemcpy D2H")?;
        Ok(out)
    }

    pub fn zero(&self, s: &Stream) -> Result<()> {
        check(unsafe { sys::cudaMemsetAsync(self.ptr, 0, self.bytes, s.0) }, "cudaMemsetAsync")
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe { sys::cudaFree(self.ptr) };
    }
}

/// Pinned host memory mapped into the device address space (the GPU reads it over PCIe).
pub struct HostMapped {
    host: *mut c_void,
    dev: *mut c_void,
    bytes: usize,
}

unsafe impl Send for HostMapped {}
unsafe impl Sync for HostMapped {}

impl HostMapped {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut host = null_mut();
        check(
            unsafe { sys::cudaHostAlloc(&mut host, bytes.max(1), sys::HOST_ALLOC_MAPPED | sys::HOST_ALLOC_PORTABLE) },
            &format!("cudaHostAlloc({bytes} B)"),
        )?;
        let mut dev = null_mut();
        check(unsafe { sys::cudaHostGetDevicePointer(&mut dev, host, 0) }, "cudaHostGetDevicePointer")?;
        Ok(Self { host, dev, bytes })
    }

    pub fn host_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host.cast(), self.bytes) }
    }

    pub fn dev_ptr<T>(&self) -> *const T {
        self.dev.cast()
    }

    pub fn host_ptr(&self) -> *const std::ffi::c_void {
        self.host
    }
}

impl Drop for HostMapped {
    fn drop(&mut self) {
        unsafe { sys::cudaFreeHost(self.host) };
    }
}
