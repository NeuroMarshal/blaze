// Draft trees (EAGLE-4 style speculative decoding with a tree of drafts instead of a chain).
//
// Target side: a verify step whose rows form a tree per sequence (row 0 the pending token, every later row the child
// of an earlier one) runs the GDN layers through the tree (the convolution window and the recurrent state follow each
// row's ancestors, not the rows before it), and the attention layers with an ancestor mask (blaze/mha.h,
// blz_nvfp4_mha_tree). The commit moves the accepted path's rows (recorded GDN inputs, captured hidden states, KV) in
// front, so that the replay of a chain applies.
//
// Draft side: the tree grows one depth per step from K frontier rows per item; every row's top-K candidates become its
// children, scored by the product of the probabilities on their path, and the best K of the K * K children are the
// next frontier (eagle_harness/core/draft_tree.py).
//
// Raw device pointers, caller's stream.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief blz_gdn_conv_ex over tree rows, the state read only: the window of row t is x[t] and the inputs of its
/// ancestors (parent[t], its parent, ...; parent < 0 past a sequence's root, whose earlier inputs come from the state,
/// the newest last). parent holds absolute row indices; a chain (parent[t] = t - 1, the first row -1) gives exactly
/// blz_gdn_conv_ex's result.
cudaError_t blz_gdn_conv_tree(const uint16_t* x, int64_t ldx, const int32_t* parent, const int32_t* seq_start,
                              const int32_t* slots, int seqs, int channels, int kernel, const uint16_t* weight,
                              const uint16_t* state, uint16_t* out, cudaStream_t stream);

/// \brief blz_gdn_recurrent_ex over tree rows, the state read only. Each sequence's rows are visited in the order of
/// prog (int32 triples [row, load, save] in the sequence's range of seq_start, a depth-first order of its tree): load
/// >= 0 restores the state saved in that level before the row's update (-1: continue from the previous row, its parent),
/// save >= 0 keeps the state after the row in that level (rows with several children). Levels live in `scratch`
/// (float, blz_gdn_tree_scratch_floats(seqs, v_heads, dv, levels)).
cudaError_t blz_gdn_recurrent_tree(const uint16_t* qkv, const float* g, const float* beta, const int32_t* prog,
                                   const int32_t* seq_start, const int32_t* slots, int seqs, int k_heads, int v_heads,
                                   int dk, int dv, const float* state, float* scratch, int levels, uint16_t* out,
                                   cudaStream_t stream);
size_t blz_gdn_tree_scratch_floats(int seqs, int v_heads, int dv, int levels);

/// \brief A buffer of fixed-size rows (row_bytes a multiple of 4).
typedef struct {
  void* ptr;
  int64_t row_bytes;
} blz_row_buf;

/// \brief For every buffer: row pairs[i].dst = row pairs[i].src, in the order of i (so a path compaction in place,
/// src >= dst and increasing, is safe). bufs: device array of nbuf descriptors; pairs: device int32 [n][2] (src, dst).
cudaError_t blz_move_rows(const blz_row_buf* bufs, int nbuf, const int32_t* pairs, int n, cudaStream_t stream);

/// \brief Top-k of every row's softmax over its c candidates: probs [rows][k] f32 (descending, ties: lower index),
/// tokens [rows][k] int64 (ids of the picks). c <= 1024, k <= 16.
cudaError_t blz_candidate_topk(const float* logits, const int32_t* ids, int rows, int c, int k, float* probs,
                               int64_t* tokens, cudaStream_t stream);

/// \brief A vocabulary shard's candidates of every row as pairs for the merge: pairs [world][rows][c][2] floats (logit,
/// id + id_offset; ids < 0 empty), this rank's part written, the others zeroed (an all-reduce sums the ranks' pairs).
cudaError_t blz_cand_pack(const float* logits, const int32_t* ids, int id_offset, int rows, int c, int rank, int world,
                          float* pairs, cudaStream_t stream);

/// \brief Merge the ranks' candidates of every row (eagle_harness merge_candidates): the best c of the world * c pairs
/// by logit (ties: lower id) are the row's candidates; the first k of them with their softmax probabilities over the c
/// (probs [rows][k], may be null) and ids (tokens [rows][k]); `record` (may be null) gets the first id per row.
/// world * c <= 1024, k <= 32.
cudaError_t blz_cand_merge_topk(const float* pairs, int world, int rows, int c, int k, float* probs, int64_t* tokens,
                                int64_t* record, cudaStream_t stream);

/// \brief The draft tree of `items` items, K = top_k frontier rows each, depths 1 .. max_depth. Node n of an item: depth
/// 1 at [0, K), the children of depth d's frontier at K + (d - 1) * K * K (row-major by frontier lane, then candidate).
typedef struct {
  float* scores;        // [items][cap] path probability of every node
  int64_t* tokens;      // [items][cap]
  int32_t* parents;     // [items][cap] parent node (-1: the root)
  float* front_scores;  // [items][K]
  int64_t* front_tokens;  // [items * K] the frontier's tokens (the next step's inputs)
  int32_t* front_node;  // [items][K]
  int32_t* ancestors;   // [items * K][max_depth] draft KV slots of each frontier row's path (depth d at d - 1)
  int32_t slot_base;    // draft KV slot of (item i, depth d, lane l): slot_base + (i * max_depth + d - 1) * K + l
  int32_t top_k;
  int32_t max_depth;
  int32_t cap;          // K + (max_depth - 1) * K * K
} blz_draft_tree;

/// \brief Seed every item's tree from its root candidates (probs/tokens [items][K], descending): depth 1, the frontier,
/// front_hidden rows i * K + l = root_hidden[i] (hidden values, rows `dim` wide), the frontier's depth-1 KV slots.
cudaError_t blz_tree_begin(blz_draft_tree t, int items, const float* probs, const int64_t* tokens,
                           const uint16_t* root_hidden, int dim, uint16_t* front_hidden, cudaStream_t stream);

/// \brief Draft step `step` (1 .. max_depth - 1) ran the frontier rows (depth `step`): add their children (probs/tokens
/// [items * K][K] of every row) as depth step + 1, keep the best K as the next frontier with its hidden inputs
/// (front_hidden row i * K + l = hidden row of its parent lane), path slots and next KV slots.
cudaError_t blz_tree_step(blz_draft_tree t, int items, int step, const float* probs, const int64_t* tokens,
                          const uint16_t* hidden, int dim, uint16_t* front_hidden, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
