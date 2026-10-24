// Copied from SGLang under Apache-2.0; modified for Blaze.
//! Qwen3 through Qwen3.8: `</think>` ends the reasoning, and so does a `<tool_call>` at the start of a line (a model
//! that calls without closing its thinking); one inside a line is the reasoning quoting the format.

use crate::think::TagConfig;

pub(super) const THINK: TagConfig = TagConfig {
    start: "<think>",
    end: "</think>",
    tool_start: Some("<tool_call>"),
    tool_start_at_line_start: true,
};

#[cfg(test)]
mod tests {
    use super::THINK;
    use crate::think::{ReasoningOptions, TagDetector};

    /// (reasoning, content) of an output streamed char by char and in one piece (the prompt ends inside `<think>`).
    fn split(text: &str) -> (String, String) {
        let opts = ReasoningOptions { force_reasoning: Some(true), stream_reasoning: true, ..Default::default() };
        let run = |chunks: Vec<String>| {
            let mut d = TagDetector::new(THINK, &opts);
            let (mut r, mut c) = (String::new(), String::new());
            for ch in chunks {
                let (a, b) = d.push(&ch);
                r += &a;
                c += &b;
            }
            let (a, b) = d.finish();
            (r + &a, c + &b)
        };
        let whole = run(vec![text.to_owned()]);
        assert_eq!(run(text.chars().map(String::from).collect()), whole, "char by char: {text:?}");
        whole
    }

    #[test]
    fn a_quoted_tool_call_stays_reasoning() {
        let (r, c) = split("I'll use the <tool_call> format here.\n</think>\n\nAnswer");
        assert_eq!(r, "I'll use the <tool_call> format here.\n");
        assert_eq!(c, "\n\nAnswer");
    }

    #[test]
    fn think_after_the_reasoning_is_text() {
        let (r, c) = split("plan</think>\n\nconst S = \"<think>\"; and more <think> text");
        assert_eq!(r, "plan");
        assert_eq!(c, "\n\nconst S = \"<think>\"; and more <think> text");
    }

    #[test]
    fn a_call_at_a_line_start_ends_unclosed_thinking() {
        let (r, c) = split("thinking\n<tool_call>\n<function=f>\n</function>\n</tool_call>");
        assert_eq!(r, "thinking\n");
        assert_eq!(c, "<tool_call>\n<function=f>\n</function>\n</tool_call>");
    }
}
