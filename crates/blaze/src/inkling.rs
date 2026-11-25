mod cpu;
mod cuda;
mod forward;
pub(crate) mod weights;
pub use forward::{Model, Options, Sequence};
mod generate;
pub use generate::{generate, render_chat, tokenize_prompt};
mod vision;

/// KV formats supported by Inkling's relative attention.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum KvDtype {
    #[default]
    #[value(alias = "auto", alias = "bfloat16")]
    Bf16,
    Nvfp4,
    #[value(name = "nvfp4_4over6", alias = "nvfp4-4over6")]
    Nvfp4FourOverSix,
}
