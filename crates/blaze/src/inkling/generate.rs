use super::Model;
use super::forward::{Prediction, Readout};
use crate::sampler::{Rng, SamplingParams};
use crate::text::Text;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

pub struct Generation {
    pub ids: Vec<u32>,
    pub stop: bool,
    pub prompt_elapsed: std::time::Duration,
    pub decode_elapsed: std::time::Duration,
}

pub enum Input {
    Token(u32),
    Image(Vec<u16>),
}

pub fn tokenize_prompt(model: &Model, text: &Text, prompt: &str, images: &[Vec<u8>]) -> Result<Vec<Input>> {
    let ids = text.encode(prompt)?;
    let placeholder = text
        .token_id("<|unused_200054|>")
        .context("Inkling tokenizer is missing the image token")?;
    let count = ids.iter().filter(|&&id| id == placeholder).count();
    ensure!(
        count == images.len(),
        "expected {} image placeholders, found {count}",
        images.len()
    );
    let mut images = images.iter();
    let mut input = Vec::with_capacity(ids.len());
    for id in ids {
        if id == placeholder {
            input.extend(model.encode_image(images.next().unwrap())?.into_iter().map(Input::Image));
        } else {
            input.push(Input::Token(id));
        }
        ensure!(input.len() < model.context(), "prompt exceeds the model context");
    }
    Ok(input)
}

pub fn render_chat(model: &Path, body: &Value) -> Result<String> {
    let source = std::fs::read_to_string(model.join("chat_template.jinja")).context("reading Inkling chat template")?;
    let markers = regex::Regex::new(r"\{%(-?)\s*(?:end)?generation\s*(-?)%\}")?;
    let source = markers.replace_all(&source, "{%$1 set _unused = none $2%}");
    let mut env = minijinja::Environment::new();
    env.add_function("raise_exception", |message: String| -> Result<(), minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
    });
    env.add_template("inkling", &source)?;
    let mut messages = body["messages"].clone();
    let messages = messages.as_array_mut().context("messages must be an array")?;
    ensure!(!messages.is_empty(), "messages must not be empty");
    for msg in messages.iter_mut() {
        ensure!(msg.is_object(), "each chat message must be an object");
        if msg["role"] == "developer" {
            msg["role"] = Value::from("system");
        }
        if let Some(calls) = msg["tool_calls"].as_array_mut() {
            for call in calls {
                if let Some(args) = call["function"]["arguments"].as_str() {
                    call["function"]["arguments"] = serde_json::from_str(args).context("tool call arguments must be JSON")?;
                }
            }
        }
    }
    let effort = body["chat_template_kwargs"]
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| body.get("reasoning_effort").cloned())
        .unwrap_or(Value::Null);
    Ok(env.get_template("inkling")?.render(minijinja::context! {
        messages => messages, tools => body["tools"], reasoning_effort => effort, add_generation_prompt => true,
    })?)
}

pub fn generate(
    model: &mut Model,
    text: &Text,
    prompt: &[Input],
    max_tokens: usize,
    sampling: &SamplingParams,
    cancelled: impl Fn() -> bool,
    mut publish: impl FnMut(u32) -> bool,
) -> Result<Generation> {
    ensure!(
        max_tokens > 0 && !prompt.is_empty(),
        "prompt and generation budget must be non-empty"
    );
    let capacity = prompt.len().checked_add(max_tokens).context("context budget overflows")?;
    let start = Instant::now();
    ensure!(!cancelled(), "request cancelled");
    let mut seq = model.sequence(capacity)?;
    let mut prediction = None;
    let readout = if sampling.greedy() { Readout::Greedy } else { Readout::Scores };
    let chunk_size = model.prefill_chunk();
    let chunks = prompt.chunks(chunk_size);
    let total_chunks = chunks.len();
    for (i, chunk) in chunks.enumerate() {
        ensure!(!cancelled(), "request cancelled");
        if let Some(out) = model.prefill(&mut seq, chunk, if i + 1 == total_chunks { readout } else { Readout::None })? {
            prediction = Some(out);
        }
    }
    let prompt_elapsed = start.elapsed();
    let mut rng = sampling.seed.map_or_else(Rng::from_time, Rng::new);
    let start = Instant::now();
    let mut ids = Vec::new();
    let mut stop = false;
    let mut drafted = 0;
    let mut accepted = 0;
    let mut draft_time = std::time::Duration::ZERO;
    let mut verify_time = std::time::Duration::ZERO;
    'decode: while ids.len() < max_tokens {
        let id = match prediction.as_ref().context("Inkling generated no prediction")? {
            Prediction::Token(id) => *id,
            Prediction::Scores(logits) => draw(logits, sampling, &mut rng),
            Prediction::Verified(_, _) => anyhow::bail!("unexpected DSpark verdict"),
        };
        ids.push(id);
        if text.eos.contains(&id) {
            stop = true;
            break;
        }
        if !publish(id) {
            break;
        }
        let remaining = max_tokens - ids.len();
        if remaining == 0 { break; }
        let limit = model.draft_limit().min(remaining);
        if limit > 0 {
            let t = Instant::now();
            let draft = model.draft_tokens(&seq, id, limit)?;
            draft_time += t.elapsed();
            let t = Instant::now();
            let (kept, next) = model.verify(&mut seq, id, &draft, readout, |logits| draw(logits, sampling, &mut rng))?;
            verify_time += t.elapsed();
            accepted += kept;
            drafted += draft.len();
            for &token in &draft[..kept] {
                ids.push(token);
                if text.eos.contains(&token) { stop = true; break 'decode; }
                if !publish(token) { break 'decode; }
            }
            prediction = Some(Prediction::Token(next));
        } else {
            prediction = model.forward(&mut seq, id, readout)?;
        }
    }
    if drafted > 0 {
        crate::eprintln!("DSpark proposed in {:.3}s, verified in {:.3}s; DSpark accepted {accepted} / {drafted} draft tokens",
            draft_time.as_secs_f64(), verify_time.as_secs_f64());
    }
    Ok(Generation {
        ids,
        stop,
        prompt_elapsed,
        decode_elapsed: start.elapsed(),
    })
}

fn draw(logits: &[f32], p: &SamplingParams, rng: &mut Rng) -> u32 {
    let (best, &max) = logits
        .iter()
        .enumerate()
        .rev()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    if p.greedy() {
        return best as u32;
    }
    if p.top_k <= 0 && p.top_p == 1.0 && p.min_p == 0.0 {
        let mut weights: Vec<_> = logits.iter().map(|&l| ((l - max) / p.temperature).exp()).collect();
        let mut pick = rng.next_f32() * weights.iter().sum::<f32>();
        for (i, w) in weights.drain(..).enumerate() {
            pick -= w;
            if pick < 0.0 {
                return i as u32;
            }
        }
        return best as u32;
    }
    let mut cands: Vec<_> = logits.iter().enumerate().collect();
    let k = usize::try_from(p.top_k).ok().filter(|&k| k > 0).unwrap_or(cands.len());
    let compare = |(a_id, a): &(usize, &f32), (b_id, b): &(usize, &f32)| {
        b.total_cmp(a).then(a_id.cmp(b_id))
    };
    if k < cands.len() {
        cands.select_nth_unstable_by(k, compare);
        cands.truncate(k);
    }
    cands.sort_unstable_by(compare);
    let weights: Vec<_> = cands.iter().map(|&(_, l)| ((l - max) / p.temperature).exp()).collect();
    let mut keep = weights.len();
    if p.top_p < 1.0 {
        let threshold = weights.iter().sum::<f32>() * p.top_p;
        let mut mass = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            mass += w;
            if mass >= threshold {
                keep = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        keep = weights[..keep].iter().take_while(|&&w| w >= p.min_p).count().max(1);
    }
    let mut pick = rng.next_f32() * weights[..keep].iter().sum::<f32>();
    for (&(id, _), &w) in cands[..keep].iter().zip(&weights) {
        pick -= w;
        if pick < 0.0 {
            return id as u32;
        }
    }
    cands[keep - 1].0 as u32
}
