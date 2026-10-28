//! Flat JSON configuration -> the same validated clap arguments as the CLI; `--preset` bundles of options below it.
use anyhow::{Context, Result, bail};
use clap::{ArgAction, CommandFactory, FromArgMatches, Parser};
use serde_json::{Map, Value};
use std::ffi::OsString;

pub fn parse<T: Parser + CommandFactory + FromArgMatches>() -> Result<(T, Value)> {
    let (args, snapshot) = resolve(T::command(), std::env::args_os().collect())?;
    let matches = T::command().get_matches_from(args);
    let parsed = T::from_arg_matches(&matches)?;
    let mut effective = snapshot;
    if let Some(("serve", m)) = matches.subcommand() {
        let command = T::command();
        let serve = command.find_subcommand("serve").context("no serve command")?;
        let obj = effective.as_object_mut().unwrap();
        for id in m.ids() {
            // clap also exposes flattened argument-group IDs, which are not configuration fields.
            if !serve.get_arguments().any(|arg| arg.get_id() == id) {
                continue;
            }
            let key = id.as_str().replace('_', "-");
            if key == "api-key" {
                obj.insert(key, Value::String("[REDACTED]".into()));
                continue;
            }
            if let Some(values) = m.get_raw(id.as_str()) {
                let values: Vec<Value> = values
                    .map(|s| {
                        let s = s.to_string_lossy();
                        serde_json::from_str(&s).unwrap_or_else(|_| Value::String(s.into_owned()))
                    })
                    .collect();
                obj.insert(key, if values.len() == 1 { values[0].clone() } else { Value::Array(values) });
            }
        }
    }
    Ok((parsed, effective))
}

/// Named bundles of serving options (`--preset NAME`): defaults below a --config file and explicit flags. Speed and
/// accuracy trade in the arithmetic of prefill GEMMs (W4A4 rounds the activations to FP4) and attention accumulation.
pub const PRESETS: &[(&str, &[(&str, &str)])] = &[
    // FP4 activations everywhere, FP16 accumulation in attention
    ("fast", &[("gemm", "nvfp4"), ("attn-f16-acc", "on"), ("prefill-a16", "none")]),
    // as fast, but the projections whose outputs persist (KV cache) from 16-bit activations
    ("balanced", &[("gemm", "nvfp4"), ("attn-f16-acc", "auto"), ("prefill-a16", "kv")]),
    // 16-bit activations everywhere (W4A16), FP32 accumulation in attention
    ("quality", &[("gemm", "fp16"), ("attn-f16-acc", "off"), ("prefill-a16", "none")]),
];

fn take_preset(args: &mut Vec<OsString>) -> Result<Option<&'static [(&'static str, &'static str)]>> {
    let mut name = None;
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].to_string_lossy().into_owned();
        if arg == "--preset" {
            if i + 1 >= args.len() {
                bail!("--preset requires a name ({})", PRESETS.iter().map(|p| p.0).collect::<Vec<_>>().join(", "));
            }
            name = Some(args[i + 1].to_string_lossy().into_owned());
            args.drain(i..i + 2);
        } else if let Some(v) = arg.strip_prefix("--preset=") {
            name = Some(v.to_owned());
            args.remove(i);
        } else {
            i += 1;
        }
    }
    let Some(name) = name else { return Ok(None) };
    match PRESETS.iter().find(|p| p.0 == name) {
        Some(p) => Ok(Some(p.1)),
        None => bail!("--preset {name}: {}", PRESETS.iter().map(|p| p.0).collect::<Vec<_>>().join(", ")),
    }
}

fn resolve(command: clap::Command, mut args: Vec<OsString>) -> Result<(Vec<OsString>, Value)> {
    let preset = take_preset(&mut args)?;
    let exe = std::path::Path::new(&args[0]).file_stem().and_then(|x| x.to_str()).unwrap_or("");
    let subcommands: Vec<_> = command.get_subcommands().map(|c| c.get_name()).collect();
    let mut i = 1;
    let mut has_subcommand = false;
    while i < args.len() {
        let arg = args[i].to_string_lossy();
        if arg == "--config" {
            i += 2;
            continue;
        }
        if arg.starts_with("--config=") {
            i += 1;
            continue;
        }
        if subcommands.iter().any(|name| arg == *name) {
            has_subcommand = true;
        }
        break;
    }
    if !has_subcommand
        && (exe == "server"
            || args.iter().any(|s| {
                let s = s.to_string_lossy();
                ["--model", "--model-path", "--config"].iter().any(|name| s == *name || s.starts_with(&format!("{name}=")))
            }))
    {
        args.insert(1, "serve".into());
    }
    let mut path = None;
    let mut i = 1;
    while i < args.len() {
        let arg = args[i].to_string_lossy();
        if arg == "--config" {
            if i + 1 >= args.len() {
                bail!("--config requires a JSON file path");
            }
            path = Some(std::path::PathBuf::from(&args[i + 1]));
            args.drain(i..i + 2);
        } else if let Some(value) = arg.strip_prefix("--config=") {
            path = Some(std::path::PathBuf::from(value));
            args.remove(i);
        } else {
            i += 1;
        }
    }
    if path.is_none() && preset.is_none() {
        return Ok((args, Value::Object(Map::new())));
    }
    if args.get(1).is_none_or(|x| x != "serve") {
        bail!("--config and --preset are supported for serving");
    }
    let mut value = match &path {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("config {}", path.display()))?)?,
        None => Value::Object(Map::new()),
    };
    let obj = value.as_object_mut().context("--config must contain a flat JSON object of CLI options")?;
    // the preset's options under the file's
    for (key, v) in preset.unwrap_or_default() {
        if !obj.keys().any(|k| k.replace('_', "-") == *key) {
            obj.insert((*key).into(), Value::String((*v).into()));
        }
    }
    let obj = &*obj;
    let serve = command.get_subcommands().find(|c| c.get_name() == "serve").context("no serve command")?;
    let mut defaults = Vec::<OsString>::new();
    for (key, value) in obj {
        let key = key.replace('_', "-");
        let arg = serve
            .get_arguments()
            .find(|a| a.get_long() == Some(&key) || a.get_all_aliases().is_some_and(|xs| xs.contains(&key.as_str())))
            .with_context(|| format!("unknown config key '{key}'"))?;
        let names: Vec<_> = std::iter::once(arg.get_long().unwrap()).chain(arg.get_all_aliases().unwrap_or_default()).collect();
        let explicit = args.iter().any(|a| {
            names.iter().any(|name| a.to_string_lossy() == format!("--{name}") || a.to_string_lossy().starts_with(&format!("--{name}=")))
        });
        if explicit || value.is_null() {
            continue;
        }
        let name = format!("--{}", arg.get_long().unwrap());
        if matches!(arg.get_action(), ArgAction::SetTrue) {
            match value.as_bool() {
                Some(true) => defaults.push(name.into()),
                Some(false) => {}
                None => bail!("config '{key}' must be a boolean"),
            }
        } else {
            let values = match value {
                Value::Array(values) if matches!(arg.get_action(), ArgAction::Append) => values.clone(),
                _ => vec![value.clone()],
            };
            for v in values {
                defaults.push(name.clone().into());
                let s = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
                defaults.push(s.into());
            }
        }
    }
    args.splice(2..2, defaults);
    Ok((args, Value::Object(Map::new())))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Cmd,
    }
    #[derive(clap::Subcommand)]
    enum Cmd {
        Serve {
            #[arg(long, alias = "model-path")]
            model: String,
            #[arg(long, default_value_t = 8000)]
            port: u16,
            #[arg(long)]
            no_graphs: bool,
        },
    }
    #[test]
    fn standalone_server_and_blaze_flags_enter_serving() {
        for exe in ["server", "blaze"] {
            let args = vec![exe.into(), "--model".into(), "/tmp/model".into()];
            let (args, _) = resolve(Cli::command(), args).unwrap();
            assert!(matches!(Cli::try_parse_from(args).unwrap().command, Cmd::Serve { .. }));
        }
        let args = vec!["blaze".into(), "--model-path".into(), "/tmp/model".into()];
        let (args, _) = resolve(Cli::command(), args).unwrap();
        assert!(matches!(Cli::try_parse_from(args).unwrap().command, Cmd::Serve { .. }));
        for flag in ["--model=/tmp/model", "--model-path=/tmp/model"] {
            let (args, _) = resolve(Cli::command(), vec!["blaze".into(), flag.into()]).unwrap();
            assert!(matches!(Cli::try_parse_from(args).unwrap().command, Cmd::Serve { .. }));
        }
    }

    #[test]
    fn global_config_before_subcommand_does_not_insert_a_second_serve() {
        let path = std::env::temp_dir().join(format!("blaze-config-{}.json", uuid::Uuid::new_v4().simple()));
        std::fs::write(&path, r#"{"port":8100}"#).unwrap();
        let args =
            vec!["blaze".into(), "--config".into(), path.as_os_str().to_owned(), "serve".into(), "--model".into(), "/tmp/model".into()];
        let (args, _) = resolve(Cli::command(), args).unwrap();
        let Cmd::Serve { model, port, .. } = Cli::try_parse_from(args).unwrap().command;
        assert_eq!(model, "/tmp/model");
        assert_eq!(port, 8100);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn flags_override_config_and_unknown_keys_fail() {
        let path = std::env::temp_dir().join(format!("blaze-config-{}.json", uuid::Uuid::new_v4().simple()));
        std::fs::write(&path, r#"{"model":"from-file","port":8100,"no_graphs":true}"#).unwrap();
        let args = vec!["server".into(), "--config".into(), path.as_os_str().to_owned(), "--model-path".into(), "from-flag".into()];
        let (args, _) = resolve(Cli::command(), args).unwrap();
        let Cmd::Serve { model, port, no_graphs } = Cli::try_parse_from(args).unwrap().command;
        assert_eq!(model, "from-flag");
        assert_eq!(port, 8100);
        assert!(no_graphs);
        std::fs::write(&path, r#"{"unknown-option":true}"#).unwrap();
        assert!(resolve(Cli::command(), vec!["server".into(), "--config".into(), path.as_os_str().to_owned()]).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[derive(Parser)]
    struct PresetCli {
        #[command(subcommand)]
        command: PresetCmd,
    }
    #[derive(clap::Subcommand)]
    enum PresetCmd {
        Serve {
            #[arg(long)]
            model: String,
            #[arg(long, default_value = "auto")]
            gemm: String,
            #[arg(long, default_value = "auto")]
            attn_f16_acc: String,
            #[arg(long, default_value = "none")]
            prefill_a16: String,
        },
    }
    fn preset(args: &[&str]) -> Result<(String, String, String)> {
        let (args, _) = resolve(PresetCli::command(), args.iter().map(OsString::from).collect())?;
        let PresetCmd::Serve { gemm, attn_f16_acc, prefill_a16, .. } = PresetCli::try_parse_from(args)?.command;
        Ok((gemm, attn_f16_acc, prefill_a16))
    }
    #[test]
    fn presets_sit_below_config_and_flags() {
        let s = |a: &str, b: &str, c: &str| (a.to_string(), b.to_string(), c.to_string());
        assert_eq!(preset(&["server", "--model", "m", "--preset", "balanced"]).unwrap(), s("nvfp4", "auto", "kv"));
        assert_eq!(preset(&["server", "--model", "m", "--preset=quality", "--gemm", "bf16"]).unwrap(), s("bf16", "off", "none"));
        let path = std::env::temp_dir().join(format!("blaze-config-{}.json", uuid::Uuid::new_v4().simple()));
        std::fs::write(&path, r#"{"model":"m","attn_f16_acc":"off"}"#).unwrap();
        let p = path.to_str().unwrap();
        assert_eq!(preset(&["server", "--config", p, "--preset", "fast"]).unwrap(), s("nvfp4", "off", "none"));
        std::fs::remove_file(path).unwrap();
        assert!(preset(&["server", "--model", "m", "--preset", "nonesuch"]).is_err());
    }
}
