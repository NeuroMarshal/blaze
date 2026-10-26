//! External benchmark client. Native Rust; never loads a model or touches CUDA.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(about = "Benchmark Blaze HTTP serving: cold/warm prompts, TTFT, TPOT, E2E and concurrency")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    url: String,
    #[arg(long, env = "BLAZE_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long, value_delimiter = ',', default_value = "1024,8192,32768")]
    prompt_tokens: Vec<usize>,
    #[arg(long, default_value_t = 256)]
    output_tokens: usize,
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 1800)]
    timeout: u64,
    #[arg(long)]
    warm_cache: bool,
    /// Local text to use as prompt material (otherwise synthetic prose; results identify the workload)
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Reproduce identical prompts across separate baseline/candidate servers (omit for fresh cold prefixes)
    #[arg(long)]
    prompt_seed: Option<u64>,
    #[arg(long, default_value = "/tmp/blaze-bench.json")]
    output: PathBuf,
    #[arg(long, default_value = "blaze")]
    label: String,
}

struct Client {
    args: Args,
    agent: ureq::Agent,
    model: String,
    base: Vec<u32>,
    suffix: Vec<u32>,
}
impl Client {
    fn request(&self, path: &str, body: Option<&Value>, rid: Option<&str>) -> Result<ureq::Response> {
        let url = format!("{}{path}", self.args.url.trim_end_matches('/'));
        let mut req = if body.is_some() { self.agent.post(&url) } else { self.agent.get(&url) };
        req = req.set("Content-Type", "application/json");
        if let Some(key) = &self.args.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        if let Some(id) = rid {
            req = req.set("X-Request-Id", id);
        }
        let response = match body {
            Some(b) => req.send_string(&b.to_string()),
            None => req.call(),
        };
        match response {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(status, r)) => {
                let mut text = String::new();
                r.into_reader().take(8192).read_to_string(&mut text)?;
                bail!("HTTP {status} {path}: {text}")
            }
            Err(e) => Err(e.into()),
        }
    }
    fn json(&self, path: &str, body: Option<&Value>) -> Result<Value> {
        Ok(serde_json::from_reader(self.request(path, body, None)?.into_reader())?)
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let value = self.json("/tokenize", Some(&json!({"model":self.model,"prompt":text})))?;
        serde_json::from_value(value.get("tokens").or(value.get("token_ids")).context("tokenizer response has no tokens")?.clone())
            .map_err(Into::into)
    }
    fn prime(&self, tokens: &[u32]) -> Result<Value> {
        // A recurrent-state snapshot cannot be rewound to an earlier prefix. One unfed output token leaves
        // the finish snapshot exactly at the input boundary, usable by the warm request's extra token.
        let start = Instant::now();
        let response = self
            .json("/v1/completions", Some(&json!({"model":self.model,"prompt":tokens,"max_tokens":1,"temperature":0,"ignore_eos":true})))?;
        ensure!(response["usage"]["completion_tokens"] == 1, "cache priming did not generate exactly one token");
        Ok(json!({"prompt_tokens":tokens.len(),"elapsed_s":start.elapsed().as_secs_f64(),"usage":response["usage"]}))
    }
    fn run(&self, tokens: &[u32], repeat: usize, client: usize, cache: &str) -> Value {
        let rid = format!("bench-{}", uuid::Uuid::new_v4().simple());
        let prompt_hash = blake3::hash(&serde_json::to_vec(tokens).expect("token IDs serialize")).to_hex().to_string();
        let start = Instant::now();
        let mut row =
            json!({"request_id":rid,"prompt_tokens":tokens.len(),"prompt_hash":prompt_hash,"repeat":repeat,"client":client,"cache":cache});
        let result = (|| -> Result<()> {
            let body = json!({"model":self.model,"prompt":tokens,"max_tokens":self.args.output_tokens,"min_tokens":self.args.output_tokens,
                "temperature":self.args.temperature,"seed":42+client,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}});
            let response = self.request("/v1/completions", Some(&body), Some(&rid))?;
            row["response_request_id"] = json!(response.header("x-request-id"));
            let mut first = None;
            let mut last = None;
            let mut usage = Value::Null;
            let mut output = String::new();
            let mut arrivals = Vec::<f64>::new();
            let mut done = false;
            for line in BufReader::new(response.into_reader()).lines() {
                let line = line?;
                let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
                if data == "[DONE]" {
                    done = true;
                    break;
                }
                let event: Value = serde_json::from_str(data)?;
                if let Some(error) = event.get("error") {
                    bail!("stream error: {error}");
                }
                if let Some(choices) = event["choices"].as_array() {
                    for choice in choices {
                        if let Some(text) = choice["text"].as_str().filter(|s| !s.is_empty()) {
                            let now = start.elapsed().as_secs_f64();
                            first.get_or_insert(now);
                            last = Some(now);
                            arrivals.push(now);
                            output.push_str(text);
                        }
                        if choice["finish_reason"].is_string() {
                            row["finish_reason"] = choice["finish_reason"].clone();
                        }
                    }
                }
                if event["usage"].is_object() {
                    usage = event["usage"].clone();
                }
            }
            let count = usage["completion_tokens"].as_u64().unwrap_or(0);
            ensure!(
                done && first.is_some() && count == self.args.output_tokens as u64,
                "incomplete response: DONE={done}, generated={count}/{}",
                self.args.output_tokens
            );
            let first = first.unwrap();
            let decode = (last.unwrap() - first).max(1e-9);
            let cached = usage.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0);
            row["ok"] = json!(true);
            row["usage"] = usage;
            row["cached_tokens"] = json!(cached);
            row["ttft_s"] = json!(first);
            row["e2e_s"] = json!(start.elapsed().as_secs_f64());
            row["decode_tok_s"] = json!(count.saturating_sub(1) as f64 / decode);
            row["tpot_ms"] = json!(decode * 1000. / count.saturating_sub(1).max(1) as f64);
            row["uncached_prefill_tok_s"] = json!(tokens.len().saturating_sub(cached as usize) as f64 / first.max(1e-9));
            row["chunk_arrivals_s"] = json!(arrivals);
            row["output"] = json!(output);
            if cache == "warm" && cached == 0 {
                row["cache_warning"] = json!("No cached prefix; this result includes cold prefill");
            }
            Ok(())
        })();
        if let Err(error) = result {
            row["ok"] = json!(false);
            row["error"] = json!(format!("{error:#}"));
            row["e2e_s"] = json!(start.elapsed().as_secs_f64());
        }
        row
    }
}

fn percentile(rows: &[Value], field: &str, p: f64) -> Value {
    let mut xs: Vec<f64> = rows.iter().filter(|r| r["ok"] == true).filter_map(|r| r[field].as_f64()).collect();
    xs.sort_by(f64::total_cmp);
    if xs.is_empty() {
        return Value::Null;
    }
    let x = (xs.len() - 1) as f64 * p;
    let lo = x as usize;
    json!(xs[lo] + (xs[(lo + 1).min(xs.len() - 1)] - xs[lo]) * (x - lo as f64))
}

fn save(path: &std::path::Path, report: &Value) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&tmp)?;
    serde_json::to_writer_pretty(&mut file, report)?;
    writeln!(file)?;
    drop(file);
    std::fs::rename(tmp, path)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.concurrency > 0
            && args.concurrency <= 1024
            && args.repeats > 0
            && args.output_tokens > 1
            && args.prompt_tokens.iter().all(|&n| n > 0),
        "invalid benchmark sizes (output tokens >=2 required)"
    );
    ensure!(args.temperature.is_finite() && args.temperature >= 0., "temperature must be finite and nonnegative");
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(args.timeout)).build();
    let mut client = Client { args, agent, model: String::new(), base: vec![], suffix: vec![] };
    client.model = match &client.args.model {
        Some(model) => model.clone(),
        None => client.json("/v1/models", None)?["data"][0]["id"].as_str().context("server has no model")?.to_owned(),
    };
    let material=match &client.args.prompt_file{Some(p)=>std::fs::read_to_string(p)?,None=>"Science advances through careful observation, experiments, mathematical reasoning and debates about evidence. Astronomy, engineering and medicine developed across many societies. Historical discoveries illustrate how methods and institutions influence progress.\n".into()};
    client.base = client.tokenize(&material)?;
    let task = if client.args.prompt_file.is_some() {
        "\n\nAnalyze the supplied material. Explain its main mechanisms, identify concrete weaknesses and propose improvements with examples. Do not repeat the source text.\n\nAnalysis:\n"
    } else {
        "\n\nWrite a new detailed essay comparing scientific institutions and discoveries in different civilizations. Use original examples rather than repeating the supplied background.\n\nEssay:\n"
    };
    client.suffix = client.tokenize(task)?;
    ensure!(!client.base.is_empty(), "prompt material tokenizes to nothing");
    let client = Arc::new(client);
    let started = Instant::now();
    let mut rows = Vec::new();
    let mut primings = Vec::new();
    let mut batches = Vec::new();
    let mut report = json!({"label":client.args.label,"model":client.model,"url":client.args.url,"workload":if client.args.prompt_file.is_some(){"file-material"}else{"synthetic-repeated-background-original-continuation"},
        "material_hash":blake3::hash(material.as_bytes()).to_hex().to_string(),
        "settings":{"prompt_tokens":client.args.prompt_tokens,"output_tokens":client.args.output_tokens,"concurrency":client.args.concurrency,"repeats":client.args.repeats,"temperature":client.args.temperature,"warm_cache":client.args.warm_cache,"prompt_seed":client.args.prompt_seed},"requests":[]});
    for &len in &client.args.prompt_tokens {
        for repeat in 0..client.args.repeats {
            let mut prompts = Vec::new();
            for c in 0..client.args.concurrency {
                let prefix = match client.args.prompt_seed {
                    Some(seed) => format!("Benchmark seed {seed}, length {len}, repeat {repeat}, client {c}.\n"),
                    None => format!("Benchmark {} client {c}.\n", uuid::Uuid::new_v4().simple()),
                };
                let mut prompt = client.tokenize(&prefix)?;
                let prefix_len = prompt.len().min(len);
                while prompt.len() < len {
                    prompt.extend_from_slice(&client.base);
                }
                prompt.truncate(len);
                // Keep the identifying prefix on short prompts too; otherwise the task overwrites it
                // and nominally cold requests for different clients become identical.
                let tail = client.suffix.len().min(len.saturating_sub(prefix_len));
                prompt[len - tail..].copy_from_slice(&client.suffix[client.suffix.len() - tail..]);
                prompts.push(prompt);
            }
            for cache in if client.args.warm_cache { vec!["cold", "warm"] } else { vec!["cold"] } {
                if cache == "warm" {
                    for (index, prompt) in prompts.iter().enumerate() {
                        let mut priming = client.prime(prompt)?;
                        priming["client"] = json!(index);
                        priming["repeat"] = json!(repeat);
                        primings.push(priming);
                    }
                    report["priming_requests"] = json!(primings);
                }
                let lap = Instant::now();
                let results = std::thread::scope(|s| {
                    let workers: Vec<_> = prompts
                        .iter()
                        .enumerate()
                        .map(|(c, p)| {
                            let client = &client;
                            s.spawn(move || {
                                // Identical full-prompt snapshots cannot be used as-is: replay needs at least one new token.
                                let mut tokens = p.clone();
                                if cache == "warm" {
                                    tokens.push(client.suffix[0]);
                                }
                                client.run(&tokens, repeat, c, cache)
                            })
                        })
                        .collect();
                    workers.into_iter().map(|w| w.join().expect("benchmark worker")).collect::<Vec<_>>()
                });
                let elapsed = lap.elapsed().as_secs_f64();
                for row in &results {
                    if row["ok"] == true {
                        println!(
                            "{} prompt={} cache={} cached={} client={} TTFT={:.3}s prefill={:.1} tok/s decode={:.1} tok/s TPOT={:.2}ms E2E={:.3}s",
                            client.args.label,
                            row["prompt_tokens"],
                            cache,
                            row["cached_tokens"],
                            row["client"],
                            row["ttft_s"].as_f64().unwrap(),
                            row["uncached_prefill_tok_s"].as_f64().unwrap(),
                            row["decode_tok_s"].as_f64().unwrap(),
                            row["tpot_ms"].as_f64().unwrap(),
                            row["e2e_s"].as_f64().unwrap()
                        );
                    } else {
                        println!("FAIL {} {}", row["request_id"], row["error"]);
                    }
                }
                let output: u64 = results.iter().filter_map(|r| r.pointer("/usage/completion_tokens").and_then(Value::as_u64)).sum();
                println!("batch output={:.1} tok/s including prefill/queue", output as f64 / elapsed);
                batches.push(json!({"prompt_tokens":len,"repeat":repeat,"cache":cache,"elapsed_s":elapsed,
                    "output_tokens":output,"output_tok_s":output as f64/elapsed,
                    "passed":results.iter().filter(|r|r["ok"]==true).count(),
                    "failed":results.iter().filter(|r|r["ok"]!=true).count()}));
                rows.extend(results);
                report["requests"] = json!(rows);
                report["batches"] = json!(batches);
                save(&client.args.output, &report)?;
            }
        }
    }
    let failed = rows.iter().filter(|r| r["ok"] != true).count();
    report["elapsed_s"] = json!(started.elapsed().as_secs_f64());
    report["summary"] = json!({"passed":rows.len()-failed,"failed":failed,"ttft_p50_s":percentile(&rows,"ttft_s",0.5),"ttft_p95_s":percentile(&rows,"ttft_s",0.95),"tpot_p50_ms":percentile(&rows,"tpot_ms",0.5),"tpot_p95_ms":percentile(&rows,"tpot_ms",0.95)});
    save(&client.args.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report["summary"])?);
    ensure!(failed == 0, "{failed} benchmark requests failed; see {}", client.args.output.display());
    Ok(())
}
