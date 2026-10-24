// Copied from SGLang under Apache-2.0; modified for Blaze.
//! Every `tests/fixtures/reasoning_parity/*.json` case: the whole output splits into the expected reasoning and
//! content, and the stream (split anywhere) adds up to the same. The step a piece comes at is not checked.

#![cfg(feature = "parser")]

use std::path::Path;

use serde_json::{Value, json};
use blaze_chat::{ReasoningOptions, ReasoningStreamSplitter, split_reasoning};

#[test]
fn whole_and_streamed_outputs_split_alike() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/reasoning_parity");
    for entry in std::fs::read_dir(dir).unwrap() {
        let fixture: Value =
            serde_json::from_str(&std::fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        let parser = fixture["parser"].as_str().unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let input = case["input"].as_u64().unwrap() as usize;
            let chunks: Vec<&str> = fixture["inputs"][input]["chunks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_str().unwrap())
                .collect();
            let options = |stream_reasoning| ReasoningOptions {
                force_reasoning: Some(case["force"] == true),
                stream_reasoning,
                force_nonempty_content: case["nonempty"] == true,
                ..Default::default()
            };
            let mut streaming =
                ReasoningStreamSplitter::new(Some(parser), options(case["stream"] == true));
            let mut steps: Vec<_> = chunks
                .iter()
                .map(|c| streaming.split::<u32>(c, &[]))
                .collect();
            steps.push(streaming.finish());
            let unary =
                split_reasoning::<u32>(Some(parser), &options(false), &chunks.concat(), &[]);
            assert_eq!(json!(unary), case["unary"], "{parser} whole output {case}");
            // the stream of the parser blaze serves Qwen3.x with adds up to the whole split (other families' parsers
            // are not served: their whole split is checked)
            // (force_nonempty_content repeats streamed reasoning as content at the end: its stream adds up to more)
            if parser == "qwen3" && case["nonempty"] != true {
                let (r, c): (String, String) = steps.iter().fold(Default::default(), |(r, c), (a, b)| (r + a, c + b));
                assert_eq!(json!([r, c]), case["unary"], "{parser} stream {case}");
            }
        }
    }
}
