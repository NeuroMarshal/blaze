//! Raw C ABI of the CUDA runtime functions blaze uses and of the blaze kernels (kernels/include/blaze/*.h).
//! Pointers are device memory unless the header says otherwise; every launch takes the caller's CUDA stream.
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub type cudaStream_t = *mut c_void;
pub type cudaError_t = c_int;
pub type bf16 = u16;

pub const CUDA_SUCCESS: cudaError_t = 0;
pub const MEMCPY_H2D: c_int = 1;
pub const MEMCPY_D2H: c_int = 2;
pub const MEMCPY_D2D: c_int = 3;
pub const HOST_ALLOC_PORTABLE: u32 = 1;
pub const HOST_ALLOC_MAPPED: u32 = 2;
pub const STREAM_NON_BLOCKING: u32 = 1;

// CUDA runtime
unsafe extern "C" {
    pub fn cudaGetErrorString(err: cudaError_t) -> *const c_char;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    pub fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> cudaError_t;
    pub fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> cudaError_t;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int) -> cudaError_t;
    pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn cudaMemcpy2DAsync(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize,
                             height: usize, kind: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn cudaMemsetAsync(dst: *mut c_void, value: c_int, bytes: usize, s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(s: *mut cudaStream_t, flags: u32) -> cudaError_t;
    pub fn cudaStreamDestroy(s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamBeginCapture(s: cudaStream_t, mode: c_int) -> cudaError_t;
    pub fn cudaStreamEndCapture(s: cudaStream_t, graph: *mut cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(exec: *mut cudaGraphExec_t, graph: cudaGraph_t, flags: u64) -> cudaError_t;
    pub fn cudaGraphLaunch(exec: cudaGraphExec_t, s: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(exec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(e: *mut cudaEvent_t, flags: u32) -> cudaError_t;
    pub fn cudaEventDestroy(e: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(e: cudaEvent_t, s: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(s: cudaStream_t, e: cudaEvent_t, flags: u32) -> cudaError_t;
}

pub type cudaEvent_t = *mut c_void;
pub const EVENT_DISABLE_TIMING: u32 = 2;

pub type cudaGraph_t = *mut c_void;
pub type cudaGraphExec_t = *mut c_void;
pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;

// blaze/ops.h
#[repr(C)]
pub struct blz_gemm {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn blz_rmsnorm(x: *const bf16, residual: *mut bf16, w: *const bf16, eps: f32, rows: c_int, dim: c_int,
                       zero_centered: c_int, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gated_rmsnorm(x: *const bf16, ld_x: i64, z: *const bf16, ld_z: i64, w: *const bf16, eps: f32,
                             tokens: c_int, heads: c_int, dim: c_int, act: c_int, out: *mut bf16, s: cudaStream_t)
                             -> cudaError_t;
    pub fn blz_attn_prep(qg: *const bf16, ld_qg: i64, k: *mut bf16, ld_k: i64, positions: *const i64, q_norm: *const bf16, k_norm: *const bf16,
                         eps: f32, zero_centered: c_int, tokens: c_int, heads: c_int, kv_heads: c_int, head_dim: c_int,
                         rotary_dim: c_int, inv_freq: *const f32, rope_scale: f32, q_out: *mut bf16, gate: *mut bf16,
                         s: cudaStream_t) -> cudaError_t;
    pub fn blz_mul_sigmoid(x: *mut bf16, gate: *const bf16, n: i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_silu_mul(gate: *const bf16, ld_gate: i64, up: *const bf16, ld_up: i64, rows: c_int, n: c_int,
                        out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_add_sigmoid_scaled(a: *const bf16, b: *const bf16, gate_logit: *const bf16, ld_gate: i64, rows: c_int,
                                  dim: c_int, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_router_topk(logits: *const bf16, ld: i64, tokens: c_int, experts: c_int, k: c_int, renorm: c_int,
                           ids: *mut i32, w: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_add(a: *const bf16, b: *const bf16, n: i64, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_gate(a: *const bf16, ld_a: i64, b: *const bf16, ld_b: i64, a_log: *const bf16, dt_bias: *const bf16,
                        tokens: c_int, v_heads: c_int, g: *mut f32, beta: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_conv(x: *const bf16, ldx: i64, seq_start: *const i32, slots: *const i32, seqs: c_int,
                        channels: c_int, kernel: c_int, weight: *const bf16, state: *mut bf16, out: *mut bf16,
                        s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_recurrent(qkv: *const bf16, g: *const f32, beta: *const f32, seq_start: *const i32,
                             slots: *const i32, seqs: c_int, tokens: c_int, k_heads: c_int, v_heads: c_int, dk: c_int,
                             dv: c_int, state: *mut f32, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_conv_ex(x: *const bf16, ldx: i64, seq_start: *const i32, slots: *const i32, seqs: c_int,
                           channels: c_int, kernel: c_int, weight: *const bf16, state: *mut bf16, out: *mut bf16,
                           write_state: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_recurrent_ex(qkv: *const bf16, g: *const f32, beta: *const f32, seq_start: *const i32,
                                slots: *const i32, seqs: c_int, tokens: c_int, k_heads: c_int, v_heads: c_int, dk: c_int,
                                dv: c_int, state: *mut f32, out: *mut bf16, write_state: c_int, s: cudaStream_t)
                                -> cudaError_t;
    pub fn blz_argmax_bf16(logits: *const bf16, rows: c_int, vocab: c_int, ld: i64, out: *mut i32,
                           s: cudaStream_t) -> cudaError_t;
    pub fn blz_argmax_part(logits: *const bf16, rows: c_int, vocab: c_int, ld: i64, col0: c_int, pairs: *mut f32,
                           s: cudaStream_t) -> cudaError_t;
    pub fn blz_argmax_merge(pairs: *const f32, world: c_int, rows: c_int, out: *mut i32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_set_req_to_token(req_to_token: *mut i32, index: *const i64, slot: *const i64, n: c_int,
                                s: cudaStream_t) -> cudaError_t;
    pub fn blz_l2_prefetch(p: *const c_void, bytes: usize, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gather_rows(src: *const bf16, ld_src: i64, rows: *const i32, n: c_int, dim: c_int, dst: *mut bf16,
                           s: cudaStream_t) -> cudaError_t;
    pub fn blz_fp8_block_dequant(w: *const u8, scale: *const c_void, scale_is_f32: c_int, n: c_int, k: c_int,
                                 block_n: c_int, block_k: c_int, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gemm_create(out: *mut *mut blz_gemm) -> c_int;
    pub fn blz_gemm_destroy(g: *mut blz_gemm);
    pub fn blz_gemm_bf16(g: *mut blz_gemm, x: *const bf16, ldx: i64, w: *const bf16, out: *mut bf16, ldo: i64,
                         m: c_int, n: c_int, k: c_int, s: cudaStream_t) -> cudaError_t;
}

// KV cache, attention
pub const BLZ_BF16: c_int = 0;
pub const BLZ_I32: c_int = 0;
pub const BLZ_I64: c_int = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct blz_index_ptr {
    pub ptr: *const c_void,
    pub kind: c_int,
}

/// A buffer of fixed-size rows (blaze/tree.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct blz_row_buf {
    pub ptr: *mut c_void,
    pub row_bytes: i64,
}

/// The draft tree's device buffers and shape (blaze/tree.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct blz_draft_tree {
    pub scores: *mut f32,
    pub tokens: *mut i64,
    pub parents: *mut i32,
    pub front_scores: *mut f32,
    pub front_tokens: *mut i64,
    pub front_node: *mut i32,
    pub ancestors: *mut i32,
    pub slot_base: i32,
    pub top_k: i32,
    pub max_depth: i32,
    pub cap: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct blz_nvfp4_pool {
    pub k_data: *const u8,
    pub v_data: *const u8,
    pub k_scale: *const u8,
    pub v_scale: *const u8,
    pub k_g: *const f32,
    pub v_g: *const f32,
    pub heads: c_int,
    pub dim: c_int,
    /// BLZ_KV_NVFP4 / BLZ_KV_FP8 / BLZ_KV_BF16
    pub format: c_int,
}

pub const BLZ_KV_NVFP4: c_int = 0;
pub const BLZ_KV_FP8: c_int = 1;
pub const BLZ_KV_BF16: c_int = 2;

unsafe extern "C" {
    pub fn blz_nvfp4_kv_store(k: *const c_void, v: *const c_void, k_stride: i64, v_stride: i64, dtype: c_int,
                              loc: *const c_void, loc_type: c_int, tokens: i64, heads: c_int, dim: c_int,
                              k_data: *mut u8, v_data: *mut u8, k_scale: *mut u8, v_scale: *mut u8, k_g: *const f32,
                              v_g: *const f32, four_over_six: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_kv_store_dense(k: *const c_void, v: *const c_void, k_stride: i64, v_stride: i64, loc: *const c_void,
                              loc_type: c_int, tokens: i64, heads: c_int, dim: c_int, format: c_int, k_data: *mut u8,
                              v_data: *mut u8, k_g: *const f32, v_g: *const f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_nvfp4_mha_decode(q: *const c_void, q_stride_tok: i64, q_stride_head: i64, q_heads: c_int,
                                pool: blz_nvfp4_pool, req_to_token: *const i32, req_stride: i64,
                                req_indices: blz_index_ptr, kv_lens: blz_index_ptr, batch: c_int, q_len: c_int,
                                splits: c_int, scale: f32, partial_o: *mut f32, partial_ml: *mut f32, dtype: c_int,
                                out: *mut c_void, out_stride_tok: i64, out_stride_head: i64,
                                s: cudaStream_t) -> cudaError_t;
    pub fn blz_nvfp4_mha_parts(q_heads: c_int, kv_heads: c_int, q_len: c_int, splits: c_int) -> c_int;
    pub fn blz_mha_decode_window(q: *const c_void, q_stride_tok: i64, q_stride_head: i64, q_heads: c_int,
                                 pool: blz_nvfp4_pool, req_to_token: *const i32, req_stride: i64,
                                 req_indices: blz_index_ptr, kv_lens: blz_index_ptr, batch: c_int, q_len: c_int,
                                 splits: c_int, scale: f32, window: c_int, causal: c_int, partial_o: *mut f32, partial_ml: *mut f32,
                                 dtype: c_int, out: *mut c_void, out_stride_tok: i64, out_stride_head: i64,
                                 s: cudaStream_t) -> cudaError_t;
    pub fn blz_mha_decode_tail(q: *const c_void, q_stride_tok: i64, q_stride_head: i64, q_heads: c_int,
                               pool: blz_nvfp4_pool, req_to_token: *const i32, req_stride: i64,
                               req_indices: blz_index_ptr, kv_lens: blz_index_ptr, batch: c_int, splits: c_int, scale: f32,
                               window: c_int, tail: *const i32, tail_len: c_int, tail_stride: c_int, partial_o: *mut f32,
                               partial_ml: *mut f32, dtype: c_int, out: *mut c_void, out_stride_tok: i64,
                               out_stride_head: i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_nvfp4_mha_tree(q: *const c_void, q_stride_tok: i64, q_stride_head: i64, q_heads: c_int,
                              pool: blz_nvfp4_pool, req_to_token: *const i32, req_stride: i64,
                              req_indices: blz_index_ptr, kv_lens: blz_index_ptr, batch: c_int, q_len: c_int,
                              splits: c_int, scale: f32, tree_mask: *const u64, partial_o: *mut f32, partial_ml: *mut f32,
                              dtype: c_int, out: *mut c_void, out_stride_tok: i64, out_stride_head: i64,
                              s: cudaStream_t) -> cudaError_t;
    // draft trees (blaze/tree.h)
    pub fn blz_gdn_conv_tree(x: *const bf16, ldx: i64, parent: *const i32, seq_start: *const i32, slots: *const i32,
                             seqs: c_int, channels: c_int, kernel: c_int, weight: *const bf16, state: *const bf16,
                             out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_recurrent_tree(qkv: *const bf16, g: *const f32, beta: *const f32, prog: *const i32,
                                  seq_start: *const i32, slots: *const i32, seqs: c_int, k_heads: c_int, v_heads: c_int,
                                  dk: c_int, dv: c_int, state: *const f32, scratch: *mut f32, levels: c_int,
                                  out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_gdn_tree_scratch_floats(seqs: c_int, v_heads: c_int, dv: c_int, levels: c_int) -> usize;
    pub fn blz_move_rows(bufs: *const blz_row_buf, nbuf: c_int, pairs: *const i32, n: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_candidate_topk(logits: *const f32, ids: *const i32, rows: c_int, c: c_int, k: c_int, probs: *mut f32,
                              tokens: *mut i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_cand_pack(logits: *const f32, ids: *const i32, id_offset: c_int, rows: c_int, c: c_int, rank: c_int,
                         world: c_int, pairs: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_cand_merge_topk(pairs: *const f32, world: c_int, rows: c_int, c: c_int, k: c_int, probs: *mut f32,
                               tokens: *mut i64, record: *mut i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_tree_begin(t: blz_draft_tree, items: c_int, probs: *const f32, tokens: *const i64, root_hidden: *const bf16,
                          dim: c_int, front_hidden: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_tree_step(t: blz_draft_tree, items: c_int, step: c_int, probs: *const f32, tokens: *const i64,
                         hidden: *const bf16, dim: c_int, front_hidden: *mut bf16, s: cudaStream_t) -> cudaError_t;
    // EAGLE-4 draft (blaze/eagle.h)
    pub fn blz_gemv_wide(x: *const c_void, x_stride: i64, w: *const c_void, y: *mut c_void, y_stride: i64, rows: c_int,
                         n: c_int, k: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_rmsnorm_ld(x: *const bf16, ldx: i64, w: *const bf16, eps: f32, rows: c_int, dim: c_int, out: *mut bf16,
                          ldo: i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_rope_neox(x: *mut bf16, ld: i64, tokens: c_int, heads: c_int, head_dim: c_int, positions: *const i64,
                         inv_freq: *const f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_candidate_logits(hidden: *const bf16, ld_hidden: i64, rows: c_int, dim: c_int, table: *const bf16,
                                ids: *const i32, c: c_int, out: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_cand_merge(pairs: *const f32, world: c_int, rows: c_int, k: c_int, ids: *mut i32, vals: *mut f32,
                          s: cudaStream_t) -> cudaError_t;
    pub fn blz_head_rmsnorm(x: *mut bf16, ld: i64, tokens: c_int, heads: c_int, head_dim: c_int, w: *const bf16, eps: f32,
                            s: cudaStream_t) -> cudaError_t;
    pub fn blz_dflash_grouped_conv(x: *const bf16, ldx: i64, delta: *const bf16, delta_row: i64, delta_tap: i64,
                                   base: *const bf16, out: *mut bf16, ldo: i64, rows: c_int, channels: c_int,
                                   block_size: c_int, group_size: c_int, taps: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_dflash_walk(cand: *const i32, unary: *const f32, hproj: *const bf16, anchor: *const i64, pred: *const bf16,
                           succ: *const bf16, items: c_int, steps: c_int, k: c_int, rank: c_int, out: *mut i64,
                           s: cudaStream_t) -> cudaError_t;
    pub fn blz_candidate_pick(logits: *const f32, ids: *const i32, rows: c_int, c: c_int, token: *mut i64,
                              record: *mut i64, s: cudaStream_t) -> cudaError_t;
    pub fn blz_embed_gather(ids: *const i64, n: i64, table: *const u8, rows: i64, row_bytes: i64, out: *mut u8,
                            s: cudaStream_t) -> cudaError_t;
}

// linear layers
pub const BLZ_NVFP4_GEMV_MAX_ROWS: usize = 8;
pub const BLZ_NVFP4_MMA_MAX_ROWS: usize = 64;
pub const BLZ_SKINNY_MAX_ROWS: usize = 16;
pub const BLZ_FP8_MMA_MAX_ROWS: usize = 16;
pub const BLZ_TOPK_MAX: usize = 64;
/// largest k of blz_topk
pub const BLZ_TOPK_LIMIT: usize = 128;
pub const BLZ_GEMV_WIDE_MAX_ROWS: usize = 4;
pub const BLZ_GEMV_WIDE_MAX_K: usize = 16384;
pub const BLZ_GEMV_WIDE_MAX_SMEM: usize = 96 * 1024;

unsafe extern "C" {
    pub fn blz_nvfp4_gemv(x: *const bf16, ldx: i64, w: *const u8, scale: *const u8, alpha: *const f32, y: *mut bf16,
                          ldy: i64, rows: c_int, n: c_int, k: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_nvfp4_gemv_mma(x: *const bf16, ldx: i64, w: *const u8, scale: *const u8, alpha: *const f32, y: *mut bf16,
                              ldy: i64, rows: c_int, n: c_int, k: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_apply_penalties(logits: *mut bf16, ld: i64, rows: c_int, col0: c_int, ncols: c_int, ids: *const i32,
                               counts: *const i32, off: *const i32, presence: *const f32, frequency: *const f32,
                               s: cudaStream_t) -> cudaError_t;
    pub fn blz_topk(logits: *const bf16, ld: i64, rows: c_int, vocab: c_int, col0: c_int, k: c_int, vals: *mut f32,
                    ids: *mut i32, workspace: *mut c_void, s: cudaStream_t) -> cudaError_t;
    pub fn blz_topk_pack(vals: *const f32, ids: *const i32, lse: *const f32, rows: c_int, k: c_int, pairs: *mut f32,
                         s: cudaStream_t) -> cudaError_t;
    pub fn blz_logit_edits(logits: *mut bf16, ld: i64, rows: c_int, col0: c_int, ncols: c_int, ids: *const i32,
                           coef: *const f32, off: *const i32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_token_mask(logits: *mut bf16, ld: i64, rows: c_int, col0: c_int, ncols: c_int, words: c_int, mask: *const u32,
                          mask_row: *const i32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_row_lse(logits: *const bf16, ld: i64, rows: c_int, ncols: c_int, lse: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_fp8_gemv_mma(x: *const bf16, ldx: i64, w: *const u8, block_scale: *const f32, block_n: c_int,
                            alpha: *const f32, y: *mut bf16, ldy: i64, rows: c_int, n: c_int, k: c_int,
                            s: cudaStream_t) -> cudaError_t;
    pub fn blz_fp8_weight_dequant(w: *const u8, block_scale: *const f32, block_n: c_int, alpha: *const f32, n: c_int,
                                  k: c_int, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_nvfp4_weight_dequant(w: *const u8, scale: *const u8, alpha: *const f32, n: c_int, k: c_int,
                                    out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_skinny_gemm(x: *const c_void, x_stride: i64, w: *const c_void, y: *mut c_void, rows: c_int, n: c_int,
                           k: c_int, dtype: c_int, s: cudaStream_t) -> cudaError_t;
}

// host all-reduce
#[repr(C)]
pub struct blz_host_ar {
    _private: [u8; 0],
}

pub const BLZ_AR_BF16: c_int = 0;
pub const BLZ_AR_FP32: c_int = 2;

unsafe extern "C" {
    pub fn blz_host_ar_open(name: *const c_char, rank: c_int, world: c_int, max_bytes: i64, create: c_int,
                            out: *mut *mut blz_host_ar) -> cudaError_t;
    pub fn blz_host_ar_unlink(name: *const c_char);
    pub fn blz_host_ar_close(ar: *mut blz_host_ar);
    pub fn blz_host_ar_rmsnorm(ar: *mut blz_host_ar, x: *const bf16, residual: *mut bf16, w: *const bf16, eps: f32,
                               rows: c_int, dim: c_int, zero_centered: c_int, out: *mut bf16, s: cudaStream_t)
                               -> cudaError_t;
    pub fn blz_host_ar_all_reduce(ar: *mut blz_host_ar, data: *mut c_void, bytes: i64, dtype: c_int,
                                  s: cudaStream_t) -> cudaError_t;
}

// MoE engine
#[repr(C)]
pub struct blz_moe_engine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct blz_moe_engine_config {
    pub model_dir: *const c_char,
    pub topk: c_int,
    pub gpu0: c_int,
    pub gpu1: c_int,
    pub gpu0_gib: f64,
    pub gpu1_gib: f64,
    pub ram_gib: f64,
    pub cpu_threads: c_int,
    pub cpu_cores: *const c_char,
    pub heat_path: *const c_char,
    pub swap_options: *const c_char,
    pub options: *const c_char,
}

unsafe extern "C" {
    pub fn blz_moe_engine_create(cfg: *const blz_moe_engine_config, out: *mut *mut blz_moe_engine, err: *mut c_char,
                                 err_len: usize) -> c_int;
    pub fn blz_moe_engine_destroy(e: *mut blz_moe_engine);
    pub fn blz_moe_engine_layers(e: *const blz_moe_engine) -> c_int;
    pub fn blz_moe_engine_forward(e: *mut blz_moe_engine, layer: c_int, x: *const bf16, tokens: c_int, ids: *const i32,
                                  w: *const f32, out: *mut bf16, s: cudaStream_t) -> cudaError_t;
}

// Qwen4-Exp (qwen4.h, hc_mix.h)
#[repr(C)]
pub struct blz_ple {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn blz_grouped_rmsnorm(x: *const bf16, w: *const bf16, eps: f32, rows: c_int, groups: c_int, dim: c_int,
                               out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_hc_combine(streams: *mut bf16, normed: *const bf16, inject: *const bf16, y: *const bf16, ld_y: i64,
                          rows: c_int, hc: c_int, dim: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_hc_combine_split(streams: *mut bf16, normed: *const bf16, inject: *const bf16, y: *const bf16, ld_y: i64,
                                rows: c_int, hc: c_int, dim: c_int, partial: *mut f32, s: cudaStream_t) -> cudaError_t;
    pub fn blz_hc_widen(x: *const bf16, rows: c_int, hc: c_int, dim: c_int, streams: *mut bf16, s: cudaStream_t)
                        -> cudaError_t;
    pub fn blz_hc_mix_supported(rows: c_int, hc: c_int, hs: c_int, lowrank: c_int) -> c_int;
    pub fn blz_hc_mix_workspace_bytes(rows: c_int, hc: c_int, hs: c_int, lowrank: c_int) -> usize;
    pub fn blz_hc_mix(x: *const bf16, w_down: *const bf16, w_up: *const bf16, t: *mut bf16, out: *mut bf16, rows: c_int,
                      hc: c_int, hs: c_int, lowrank: c_int, workspace: *mut c_void, s: cudaStream_t) -> cudaError_t;
    pub fn blz_ngram_hash(tokens: *const i64, seq_start: *const i32, slots: *const i32, seqs: c_int, max_tokens: c_int,
                          hist: *mut i64, update_hist: c_int, multipliers: *const i64, sizes: *const i64,
                          offsets: *const i64, heads_per_ngram: c_int, eos: i64, ids: *mut i64, s: cudaStream_t)
                          -> cudaError_t;
    pub fn blz_ple_gate(key_n: *const bf16, query_n: *const bf16, value: *const bf16, rows: c_int, hc: c_int, dim: c_int,
                        gated: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_ple_conv(gated_n: *const bf16, gated: *const bf16, seq_start: *const i32, slots: *const i32, seqs: c_int,
                        channels: c_int, kernel: c_int, dilation: c_int, weight: *const bf16, state: *mut bf16,
                        write_state: c_int, streams: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_ple_create(path: *const c_char, max_ids: c_int, threads: c_int, cache_gib: f64, out: *mut *mut blz_ple,
                          err: *mut c_char, err_len: usize) -> c_int;
    pub fn blz_ple_destroy(p: *mut blz_ple);
    pub fn blz_ple_row_dim(p: *const blz_ple) -> c_int;
    pub fn blz_ple_request(p: *mut blz_ple, ids: *const i64, n: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_ple_fetch(p: *mut blz_ple, out: *mut bf16, n: c_int, s: cudaStream_t) -> cudaError_t;
    pub fn blz_ple_warm(p: *mut blz_ple, host_ids: *const i64, n: c_int);
}

// QSA indexer (qsa_index.h)
unsafe extern "C" {
    pub fn blz_qsa_index_q(proj: *const bf16, ld: i64, col: c_int, positions: *const i64, norm_w: *const bf16, eps: f32,
                           tokens: c_int, heads: c_int, dim: c_int, rotary_dim: c_int, inv_freq: *const f32, rope_scale: f32,
                           out: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_qsa_compress(proj: *const bf16, ld: i64, kcol: c_int, positions: *const i64, tok_row: *const i32,
                            tok_kv_len: *const i32, tokens: c_int, dim: c_int, ratio: c_int, norm_w: *const bf16, eps: f32,
                            rotary_dim: c_int, inv_freq: *const f32, rope_scale: f32, req_to_token: *const i32,
                            rt_stride: i64, ring: *mut bf16, comp: *mut bf16, s: cudaStream_t) -> cudaError_t;
    pub fn blz_qsa_scores(q: *const bf16, heads: c_int, dim: c_int, positions: *const i64, tok_row: *const i32, row0: c_int,
                          rows: c_int, same_seq: c_int, ratio: c_int, req_to_token: *const i32, rt_stride: i64,
                          comp: *const bf16, max_blocks: c_int, scores: *mut f32, ld_scores: i64, s: cudaStream_t)
                          -> cudaError_t;
    pub fn blz_qsa_select(scores: *const f32, ld_scores: i64, positions: *const i64, tok_row: *const i32, row0: c_int,
                          rows: c_int, ratio: c_int, block_topk: c_int, req_to_token: *const i32, rt_stride: i64,
                          table: *mut i32, table_stride: i64, counts: *mut i32, s: cudaStream_t) -> cudaError_t;
}
