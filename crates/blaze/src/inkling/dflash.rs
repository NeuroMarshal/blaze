use super::cuda::{Compute, Dense, to_bf16};
use crate::dflash::{DFlash, DFlashConfig, Target};
use crate::gpu::{self, DevBuf, HostMapped, check};
use crate::linear::LinCtx;
use crate::speculation::{DraftItem, DraftPlan};
use anyhow::{Result, ensure};
use blaze_sys as sys;
use std::path::Path;

pub(super) struct Draft {
    engine: DFlash,
    linear: LinCtx,
    embed: HostMapped,
    embed_norm: DevBuf,
    features: DevBuf,
    hidden: usize,
    vocab: usize,
    valid_vocab: usize,
    sms: usize,
}

impl Draft {
    pub fn load(
        path: &Path,
        compute: &Compute,
        device: i32,
        embed: &[u8],
        norm: &[f32],
        vocab: usize,
        valid_vocab: usize,
        visits: usize,
        context: usize,
        chunk: usize,
    ) -> Result<Self> {
        let cfg = DFlashConfig::load(path)?;
        ensure!(
            cfg.target_layers == [5, 23, 41, visits] && visits == 48 && cfg.block == 8,
            "DFlash checkpoint needs the Inkling 48-visit feature layout"
        );
        let hidden = norm.len();
        ensure!(
            embed.len() == vocab * hidden * 2
                && cfg.mask_token >= 0
                && cfg.mask_token < vocab as i64,
            "DFlash target embedding shape or mask token is invalid"
        );
        let linear = compute.linear_context(device);
        let mut table = HostMapped::new(embed.len())?;
        table.host_slice_mut().copy_from_slice(embed);
        let embed_norm =
            DevBuf::from_slice(&norm.iter().copied().map(to_bf16).collect::<Vec<_>>())?;
        let sms = gpu::multiprocessors();
        let stream = &compute.stream;
        let target = Target {
            hidden,
            vocab,
            local_vocab: vocab,
            valid_vocab,
            rank: 0,
            world: 1,
            sms,
            ar: std::ptr::null_mut(),
            stream,
            linear: &linear,
            features: None,
            embedding: table.dev_ptr(),
            embedding_norm: Some(&embed_norm),
            head: &|_, _, _| unreachable!(),
        };
        let stride = context
            .checked_add(cfg.block)
            .ok_or_else(|| anyhow::anyhow!("DFlash context overflows"))?;
        let engine = DFlash::load_target(path, &target, 1, stride)?;
        Ok(Self {
            engine,
            linear,
            embed: table,
            embed_norm,
            features: DevBuf::new(chunk * hidden * 8)?,
            hidden,
            vocab,
            valid_vocab,
            sms,
        })
    }

    pub fn window(&self) -> usize {
        self.engine.cfg.window
    }

    pub fn append(
        &mut self,
        compute: &Compute,
        captures: [*const f32; 4],
        start: usize,
        consumed: usize,
    ) -> Result<()> {
        let Self {
            engine,
            linear,
            features,
            embed,
            embed_norm,
            hidden,
            vocab,
            valid_vocab,
            sms,
        } = self;
        let stream = &compute.stream;
        check(
            unsafe {
                sys::blz_pack_spec_features(
                    captures[0],
                    captures[1],
                    captures[2],
                    captures[3],
                    features.ptr(),
                    consumed as i32,
                    *hidden as i32,
                    stream.0,
                )
            },
            "DFlash target features",
        )?;
        let target = Target {
            hidden: *hidden,
            vocab: *vocab,
            local_vocab: *vocab,
            valid_vocab: *valid_vocab,
            rank: 0,
            world: 1,
            sms: *sms,
            ar: std::ptr::null_mut(),
            stream,
            linear,
            features: Some(features),
            embedding: embed.dev_ptr(),
            embedding_norm: Some(embed_norm),
            head: &|_, _, _| unreachable!(),
        };
        engine.append_target(
            &target,
            &DraftItem {
                row: 0,
                aux_row: 0,
                tokens: vec![0; consumed],
                pos0: start,
            },
        )
    }

    pub fn propose(
        &mut self,
        compute: &Compute,
        head: &Dense,
        position: usize,
        anchor: u32,
        limit: usize,
    ) -> Result<Vec<u32>> {
        ensure!(position > 0, "DFlash needs a nonempty target context");
        let Self {
            engine,
            linear,
            embed,
            embed_norm,
            hidden,
            vocab,
            valid_vocab,
            sms,
            ..
        } = self;
        let head = |input, rows, output| head.run(compute, input, output, rows);
        let target = Target {
            hidden: *hidden,
            vocab: *vocab,
            local_vocab: *vocab,
            valid_vocab: *valid_vocab,
            rank: 0,
            world: 1,
            sms: *sms,
            ar: std::ptr::null_mut(),
            stream: &compute.stream,
            linear,
            features: None,
            embedding: embed.dev_ptr(),
            embedding_norm: Some(embed_norm),
            head: &head,
        };
        let item = DraftItem {
            row: 0,
            aux_row: 0,
            tokens: vec![i64::from(anchor)],
            pos0: position - 1,
        };
        let drafts = engine.run_target(
            &target,
            &DraftPlan {
                items: vec![item],
                k: limit,
                ..Default::default()
            },
        )?;
        Ok(drafts[0]
            .iter()
            .take(limit)
            .map(|&token| token as u32)
            .collect())
    }
}
