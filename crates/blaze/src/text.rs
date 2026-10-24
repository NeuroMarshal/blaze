//! Tokenizer (Hugging Face tokenizer.json) and the stop tokens of a model.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::path::Path;

pub struct Text {
    tok: tokenizers::Tokenizer,
    pub eos: Vec<u32>,
    /// the raw bytes of every token (built on first use)
    bytes: std::sync::OnceLock<Vec<Box<[u8]>>>,
}

impl Text {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        // some checkpoints ship a training-time truncation (e.g. max_length 2048) that would cut long prompts
        tok.with_truncation(None).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
        tok.with_padding(None);
        let mut eos = Vec::new();
        // generation_config.json: eos_token_id (int or list); tokenizer_config.json: eos_token (string).
        if let Ok(s) = std::fs::read_to_string(dir.join("generation_config.json")) {
            let v: Value = serde_json::from_str(&s)?;
            match v.get("eos_token_id") {
                Some(Value::Number(n)) => eos.extend(n.as_u64().map(|x| x as u32)),
                Some(Value::Array(a)) => eos.extend(a.iter().filter_map(Value::as_u64).map(|x| x as u32)),
                _ => {}
            }
        }
        if let Ok(s) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            let v: Value = serde_json::from_str(&s)?;
            if let Some(id) = v.get("eos_token").and_then(Value::as_str).and_then(|t| tok.token_to_id(t)) {
                if !eos.contains(&id) {
                    eos.push(id);
                }
            }
        }
        Ok(Self { tok, eos, bytes: Default::default() })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tok.token_to_id(token)
    }

    /// The ids a draft head should always consider, a bit per id: the lowest ids of each script of the vocabulary (the BPE
    /// merges are ordered by frequency within a script): `latin` ASCII tokens, `cyrillic`, `cjk` and `other` ones.
    pub fn frequent_ids(&self, vocab: usize, latin: usize, cyrillic: usize, cjk: usize, other: usize) -> Vec<u64> {
        let mut bits = vec![0u64; vocab.div_ceil(64)];
        let mut left = [latin, cyrillic, cjk, other];
        for id in 0..vocab {
            if left.iter().all(|&l| l == 0) {
                break;
            }
            let Ok(s) = self.tok.decode(&[id as u32], false) else { continue };
            if s.is_empty() || s.contains('\u{fffd}') {
                continue;
            }
            let class = if s.chars().any(|c| ('\u{400}'..='\u{4ff}').contains(&c)) {
                1
            } else if s.chars().any(|c| matches!(c, '\u{4e00}'..='\u{9fff}' | '\u{3040}'..='\u{30ff}' | '\u{ac00}'..='\u{d7af}')) {
                2
            } else if s.is_ascii() {
                0
            } else {
                3
            };
            if left[class] > 0 {
                left[class] -= 1;
                bits[id / 64] |= 1 << (id % 64);
            }
        }
        bits
    }

    /// The raw bytes of token `id`: a byte-level BPE token may hold part of a UTF-8 character, which decoding it
    /// alone turns into U+FFFD.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let table = self.bytes.get_or_init(|| {
            let byte_level = matches!(self.tok.get_decoder(), Some(tokenizers::DecoderWrapper::ByteLevel(_)));
            // GPT-2's byte-to-character map, inverted
            let mut of_char = std::collections::HashMap::new();
            let mut n = 0u32;
            for b in 0..=255u8 {
                let printable = matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
                let c = if printable { b as u32 } else { n += 1; 255 + n };
                of_char.insert(char::from_u32(c).unwrap(), b);
            }
            let added = self.tok.get_added_tokens_decoder();
            (0..self.tok.get_vocab_size(true) as u32)
                .map(|id| {
                    if let Some(a) = added.get(&id) {
                        return a.content.as_bytes().into();
                    }
                    let mapped = byte_level
                        .then(|| self.tok.id_to_token(id))
                        .flatten()
                        .and_then(|t| t.chars().map(|c| of_char.get(&c).copied()).collect::<Option<Vec<u8>>>());
                    mapped.unwrap_or_else(|| self.decode(&[id]).unwrap_or_default().into_bytes()).into_boxed_slice()
                })
                .collect()
        });
        match table.get(id as usize) {
            Some(b) => b.to_vec(),
            None => self.decode(&[id]).unwrap_or_default().into_bytes(),
        }
    }

    /// Token `id` as logprobs show it: its text, the bytes of a partial UTF-8 character written as `\xNN` (as OpenAI
    /// does), so different partial tokens stay different strings.
    pub fn token_str(&self, id: u32) -> (String, Vec<u8>) {
        let b = self.token_bytes(id);
        let mut s = String::new();
        for chunk in b.utf8_chunks() {
            s.push_str(chunk.valid());
            for x in chunk.invalid() {
                s.push_str(&format!("\\x{x:02x}"));
            }
        }
        (s, b)
    }

    /// Decode without the special tokens (chat output: the stop token is not part of the text).
    pub fn decode_skip(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, true).map_err(|e| anyhow!("decode: {e}"))
    }
}
