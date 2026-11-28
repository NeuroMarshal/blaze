# Extending Blaze

Build each model from the runtime's existing operations. Checkpoint code maps metadata and tensors to a model's layer order, shapes, and math. The runtime owns serving, request scheduling, device placement, memory, and execution.

## Add a model

1. Identify the architecture from the checkpoint configuration. Map tensor names, formats, layout, and stage dependencies in the model loader.
2. Walk the model's forward pass against the existing operations. Reuse the implementation that owns each operation and pass model-specific inputs through its interface.
3. For a missing operation, describe the inputs, outputs, numerical rules, and state it carries across tokens. Extend the closest shared operation when the new behavior has a concrete second caller. Keep formulas specific to the architecture with that architecture.
4. Describe safe partition axes, the reductions they require, and where state has to stay together. Use Blaze's placement and transfer primitives for TP and PP.
5. Connect the model to the existing serving entry point and generation interfaces. Preserve the checkpoint's tokenizer, template, and output protocol.

## Examples

- A routed model can compute its own top-k IDs and coefficients. Pass those to HybridMoE through `expert_runtime`. HybridMoE already owns expert placement, CPU offload, and execution.
- An attention layer describes heads, masks or windows, KV format, and any bias its math requires. FlashAttention is an implementation of the attention operation. A model that uses an existing operation should inherit its compatible implementations.
- A recurrent model uses `layer_walk` to describe repeated visits and their dependencies. Visits can share weights and retain separate KV and convolution history.
- A model can connect DFlash2 by supplying the target embeddings, activation taps, and readout to `dflash::Target`. MTP and n-gram also supply drafts to a target-side verification and commit step. Keep draft acceptance and state rollback in the generation path.
- Use the CUDA graph and GPU transfer primitives when a step repeats or crosses devices. Select the kernels and prepare graph inputs outside the hot loop where their shapes and lifetimes allow it.

## Review a change

- List the existing blocks the model calls. For each new server path, queue, sampler, cache, transfer, or kernel, identify the behavior the existing implementation lacks.
- Keep reusable changes small. Move a shared operation together with its callers and preserve their math, data layout, and resource lifetimes.
- Compare tokens or logits, including seeded sampling when relevant. For stateful changes, check reuse, cancellation, and speculative rollback. Measure prefill and decode on a current model that calls the changed block as well as on the new model.
