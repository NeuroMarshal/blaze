//! Tokenizer (Hugging Face tokenizer.json) and the stop tokens of a model.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::path::Path;

pub struct Text {
    tok: tokenizers::Tokenizer,
    pub eos: Vec<u32>,
}

impl Text {
    pub fn load(dir: &Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?;
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
        Ok(Self { tok, eos })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, false).map_err(|e| anyhow!("decode: {e}"))
    }
}
