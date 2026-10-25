//! Remote pipeline stages use separate TCP connections for commands and hidden rows.
//! Relay threads publish rows through the same pinned-memory links as local stages.

use super::*;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

// wire format

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn tag(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bytes<T: Copy>(&mut self, v: &[T]) {
        self.usize(v.len());
        let n = std::mem::size_of_val(v);
        // SAFETY: plain numbers, read back as the same type
        self.0.extend_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), n) });
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.p + n > self.b.len() {
            bail!("frame ends early");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn usize(&mut self) -> Result<usize> {
        Ok(self.u64()? as usize)
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.take(1)?[0] != 0)
    }
    fn tag(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn vec<T: Copy + Default>(&mut self) -> Result<Vec<T>> {
        let n = self.usize()?;
        let bytes = self.take(n * std::mem::size_of::<T>())?;
        let mut v = vec![T::default(); n];
        // SAFETY: the bytes of n values of T, written by W::bytes
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), v.as_mut_ptr().cast::<u8>(), bytes.len()) };
        Ok(v)
    }
}

fn put_meta(w: &mut W, m: &StepMeta) {
    for v in [&m.tokens, &m.positions, &m.locs, &m.rt_index, &m.pos_h, &m.pos_w] {
        w.bytes(v);
    }
    for v in [&m.seq_start, &m.rows, &m.kv_lens, &m.logit_rows, &m.tok_row, &m.tok_kv_len, &m.pen_ids, &m.pen_off, &m.mask_row, &m.tree_parent,
              &m.tree_prog] {
        w.bytes(v);
    }
    w.bytes(&m.pen_coef);
    w.bytes(&m.mask);
    w.bytes(&m.tree_mask);
    let spans: Vec<u64> = m.spans.iter().flat_map(|s| [s.offset, s.len, s.kv_len, s.row].map(|x| x as u64)).collect();
    w.bytes(&spans);
    for b in [m.sparse, m.sample, m.tree, m.mrope] {
        w.bool(b);
    }
}

fn get_meta(r: &mut R) -> Result<StepMeta> {
    let mut m = StepMeta::default();
    for v in [&mut m.tokens, &mut m.positions, &mut m.locs, &mut m.rt_index, &mut m.pos_h, &mut m.pos_w] {
        *v = r.vec()?;
    }
    for v in [&mut m.seq_start, &mut m.rows, &mut m.kv_lens, &mut m.logit_rows, &mut m.tok_row, &mut m.tok_kv_len, &mut m.pen_ids,
              &mut m.pen_off, &mut m.mask_row, &mut m.tree_parent, &mut m.tree_prog] {
        *v = r.vec()?;
    }
    m.pen_coef = r.vec()?;
    m.mask = r.vec()?;
    m.tree_mask = r.vec()?;
    let spans: Vec<u64> = r.vec()?;
    m.spans = spans.chunks_exact(4).map(|c| Span { offset: c[0] as usize, len: c[1] as usize, kv_len: c[2] as usize, row: c[3] as usize }).collect();
    m.sparse = r.bool()?;
    m.sample = r.bool()?;
    m.tree = r.bool()?;
    m.mrope = r.bool()?;
    // the images are the first stage's (it replaces their rows), never a remote one's
    Ok(m)
}

fn put_cmd(w: &mut W, c: &Cmd) -> Result<()> {
    match c {
        Cmd::Step(m, graph, verify) => {
            w.tag(1);
            put_meta(w, m);
            w.bool(*graph);
            w.bool(*verify);
        }
        Cmd::CopyState(a, b) => {
            w.tag(2);
            w.usize(*a);
            w.usize(*b);
        }
        Cmd::SetKvRow(row, slots) => {
            w.tag(3);
            w.usize(*row);
            w.bytes(slots);
        }
        Cmd::Commit(items) => {
            w.tag(4);
            w.usize(items.len());
            for it in items.iter() {
                for v in [it.row, it.accepted, it.start, it.k, it.offset] {
                    w.usize(v);
                }
                w.bytes(&it.tokens);
                w.bool(it.moves.is_some());
                if let Some((a, b)) = &it.moves {
                    w.bytes(a);
                    w.bytes(b);
                }
            }
        }
        Cmd::PreVerify(p) => {
            w.tag(5);
            let flat: Vec<u64> = p.iter().flat_map(|&(a, b)| [a as u64, b as u64]).collect();
            w.bytes(&flat);
        }
        Cmd::Reset(row) => {
            w.tag(6);
            w.usize(*row);
        }
        Cmd::Logits(row) => {
            w.tag(7);
            w.usize(*row);
        }
        Cmd::Hidden(row) => {
            w.tag(8);
            w.usize(*row);
        }
        Cmd::AllLogits(n) => {
            w.tag(9);
            w.usize(*n);
        }
        Cmd::Batch(cmds) => {
            w.tag(10);
            w.usize(cmds.len());
            for c in cmds {
                put_cmd(w, c)?;
            }
        }
        Cmd::Steps(metas, graph) => {
            w.tag(13);
            w.usize(metas.len());
            for m in metas {
                put_meta(w, m);
            }
            w.bool(*graph);
        }
        Cmd::Quit => w.tag(11),
        // nothing of these runs on a remote stage (drafters, PLE, all-reduce timing are not supported there)
        Cmd::PleWarm(_) | Cmd::TimeAllReduce(_) | Cmd::SetDmaMin(_) => w.tag(12),
        Cmd::Draft(_) => bail!("drafters do not run on a remote pipeline stage"),
    }
    Ok(())
}

fn get_cmd(r: &mut R) -> Result<Cmd> {
    Ok(match r.tag()? {
        1 => {
            let m = get_meta(r)?;
            Cmd::Step(Arc::new(m), r.bool()?, r.bool()?)
        }
        2 => Cmd::CopyState(r.usize()?, r.usize()?),
        3 => Cmd::SetKvRow(r.usize()?, Arc::new(r.vec()?)),
        4 => {
            let n = r.usize()?;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n {
                let (row, accepted, start, k, offset) = (r.usize()?, r.usize()?, r.usize()?, r.usize()?, r.usize()?);
                let tokens = Arc::new(r.vec()?);
                let moves = if r.bool()? { Some((r.vec()?, r.vec()?)) } else { None };
                items.push(CommitItem { row, accepted, start, tokens, k, offset, moves });
            }
            Cmd::Commit(Arc::new(items))
        }
        5 => {
            let flat: Vec<u64> = r.vec()?;
            Cmd::PreVerify(Arc::new(flat.chunks_exact(2).map(|c| (c[0] as usize, c[1] as usize)).collect()))
        }
        6 => Cmd::Reset(r.usize()?),
        7 => Cmd::Logits(r.usize()?),
        8 => Cmd::Hidden(r.usize()?),
        9 => Cmd::AllLogits(r.usize()?),
        10 => {
            let n = r.usize()?;
            Cmd::Batch((0..n).map(|_| get_cmd(r)).collect::<Result<_>>()?)
        }
        11 => Cmd::Quit,
        12 => Cmd::Batch(Vec::new()),
        13 => {
            let n = r.usize()?;
            let metas = (0..n).map(|_| get_meta(r).map(Arc::new)).collect::<Result<Vec<_>>>()?;
            Cmd::Steps(metas, r.bool()?)
        }
        t => bail!("unknown command {t}"),
    })
}

fn put_reply(w: &mut W, r: &Result<Reply>) {
    match r {
        Err(e) => {
            w.tag(0);
            w.bytes(format!("{e:#}").as_bytes());
        }
        Ok(Reply::Tokens(t)) => {
            w.tag(1);
            w.bytes(t);
        }
        Ok(Reply::Cands(c)) => {
            w.tag(2);
            w.bytes(c);
        }
        Ok(Reply::Logits(l)) => {
            w.tag(3);
            w.bytes(l);
        }
        Ok(Reply::AllLogits(rows)) => {
            w.tag(4);
            w.usize(rows.len());
            for r in rows {
                w.bytes(r);
            }
        }
        Ok(Reply::Done) => w.tag(5),
        Ok(Reply::Drafts(_) | Reply::Trees(_)) => {
            w.tag(0);
            w.bytes(b"drafts from a remote stage");
        }
    }
}

fn get_reply(r: &mut R) -> Result<Result<Reply>> {
    Ok(match r.tag()? {
        0 => Err(anyhow!("remote stage: {}", String::from_utf8_lossy(&r.vec::<u8>()?))),
        1 => Ok(Reply::Tokens(r.vec()?)),
        2 => Ok(Reply::Cands(r.vec()?)),
        3 => Ok(Reply::Logits(r.vec()?)),
        4 => {
            let n = r.usize()?;
            Ok(Reply::AllLogits((0..n).map(|_| r.vec()).collect::<Result<_>>()?))
        }
        5 => Ok(Reply::Done),
        t => bail!("unknown reply {t}"),
    })
}

fn send_frame(s: &mut impl Write, b: &[u8]) -> Result<()> {
    s.write_all(&(b.len() as u64).to_le_bytes())?;
    s.write_all(b)?;
    s.flush()?;
    Ok(())
}

fn recv_frame(s: &mut impl Read) -> Result<Vec<u8>> {
    let mut n = [0u8; 8];
    s.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n) as usize;
    if n > 1 << 34 {
        bail!("a frame of {n} B");
    }
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)?;
    Ok(b)
}

/// The header words of a link (flag, bytes of the rows).
/// The header word of the first receiver's last step taken (blaze/pp_link.h).
const PP_ACK: usize = 8;

fn header(link: &HostMapped) -> *mut u64 {
    link.host_ptr().cast::<u64>().cast_mut()
}

// coordinator

/// The rank of a remote stage: a relay of the engine's commands to the stage worker at `addr`, and the forwarder of
/// the rows its link (`prev`, written by the stage before it) receives.
pub(super) fn connect(addr: &str, rank: &crate::model::PpRank, prev: Arc<HostMapped>) -> Result<Worker> {
    let cmd = TcpStream::connect(addr).map_err(|e| anyhow!("pipeline stage {} at {addr}: {e}", rank.stage))?;
    cmd.set_nodelay(true)?;
    let data = TcpStream::connect(addr)?;
    data.set_nodelay(true)?;
    let mut hello = W::default();
    for v in [rank.stage, rank.stages, rank.layers.start, rank.layers.end, prev.bytes()] {
        hello.usize(v);
    }
    let (mut cw, mut cr) = (BufWriter::new(cmd.try_clone()?), BufReader::new(cmd));
    send_frame(&mut cw, &hello.0)?;
    let mut dw = BufWriter::new(data.try_clone()?);
    send_frame(&mut dw, b"data")?;
    let (ctx, crx) = channel::<Cmd>();
    let (rtx, rrx) = channel::<Result<Reply>>();
    let stop = Arc::new(AtomicBool::new(false));
    // rows: wait for each new flag of the link, send the step's rows
    {
        let (stop, prev) = (stop.clone(), prev.clone());
        std::thread::Builder::new().name(format!("blaze-pp-send{}", rank.stage)).spawn(move || {
            let h = header(&prev);
            let mut seen = 0u64;
            // spinning while steps come, then yielding, then short sleeps when idle (a core is not held by an idle
            // server; the first step after a pause waits at most a sleep)
            let mut idle = Instant::now();
            while !stop.load(Ordering::Relaxed) {
                // SAFETY: the link's header, written by the GPU of the stage before
                let flag = unsafe { std::ptr::read_volatile(h) };
                if flag <= seen {
                    match idle.elapsed().as_micros() {
                        0..200 => std::hint::spin_loop(),
                        200..20_000 => std::thread::yield_now(),
                        _ => std::thread::sleep(Duration::from_micros(100)),
                    }
                    continue;
                }
                std::sync::atomic::fence(Ordering::Acquire);
                // every step since the last one sent (at most two: the GPU waits for its slot back), in order
                let slot_bytes = crate::model::pp_slot_bytes(prev.bytes());
                let mut broken = false;
                for v in seen + 1..=flag {
                    let slot = ((v - 1) % 2) as usize;
                    // an abort (u64::MAX) or a size past the slot: the flag goes on without rows
                    let bytes = unsafe { std::ptr::read_volatile(h.add(2 + slot)) } as usize;
                    let bytes = if bytes > slot_bytes { 0 } else { bytes };
                    let at = crate::model::PP_LINK_HEADER + slot * slot_bytes;
                    let rows = unsafe { std::slice::from_raw_parts(prev.host_ptr().cast::<u8>().add(at), bytes) };
                    let mut f = W::default();
                    f.u64(v);
                    f.0.extend_from_slice(rows);
                    // the rows are in the frame: the slot goes back to the GPU
                    std::sync::atomic::fence(Ordering::SeqCst);
                    unsafe { std::ptr::write_volatile(h.add(PP_ACK), v) };
                    if send_frame(&mut dw, &f.0).is_err() {
                        broken = true;
                        break;
                    }
                }
                if broken {
                    break;
                }
                seen = flag;
                idle = Instant::now();
            }
        })?;
    }
    // commands: the load result first, then a reply per command
    let handle = std::thread::Builder::new().name(format!("blaze-pp-relay{}", rank.stage)).spawn(move || {
        // Result<Result<Reply>>: a broken frame or the stage's own load error
        let first: Result<Reply> = recv_frame(&mut cr).and_then(|b| get_reply(&mut R { b: &b, p: 0 })).and_then(|r| r);
        let failed = first.is_err();
        let _ = rtx.send(first);
        if failed {
            return;
        }
        while let Ok(c) = crx.recv() {
            let quit = matches!(c, Cmd::Quit);
            let mut w = W::default();
            let reply = put_cmd(&mut w, &c)
                .and_then(|_| send_frame(&mut cw, &w.0))
                .and_then(|_| recv_frame(&mut cr))
                .and_then(|b| get_reply(&mut R { b: &b, p: 0 }))
                .and_then(|r| r)
                .map_err(|e| e.context("remote pipeline stage"));
            if rtx.send(reply).is_err() || quit {
                break;
            }
        }
        stop.store(true, Ordering::Relaxed);
    })?;
    Ok(Worker { tx: ctx, rx: rrx, handle: Some(handle) })
}

// stage worker

/// `blaze stage-worker`: one pipeline stage for a coordinator, on this machine's GPU 0, until it quits. `opts`: the
/// engine options (the same as the coordinator's), `rows` the most rows of a step and `width` the values of one.
pub fn serve(listen: &str, dir: &Path, mut opts: Options, use_graphs: bool) -> Result<()> {
    let listener = TcpListener::bind(listen)?;
    eprintln!("blaze: stage worker on {listen}");
    let (cmd, peer) = listener.accept()?;
    cmd.set_nodelay(true)?;
    let (mut cw, mut cr) = (BufWriter::new(cmd.try_clone()?), BufReader::new(cmd));
    let hello = recv_frame(&mut cr)?;
    let mut h = R { b: &hello, p: 0 };
    let (stage, stages, l0, l1, peer_link) = (h.usize()?, h.usize()?, h.usize()?, h.usize()?, h.usize()?);
    let (data, _) = listener.accept()?;
    data.set_nodelay(true)?;
    let mut dr = BufReader::new(data);
    if recv_frame(&mut dr)? != b"data" {
        bail!("{peer}: the second connection is not the rows'");
    }
    eprintln!("blaze: stage {stage} of {stages} (layers {l0}..{l1}) for {peer}");
    if stage + 1 != stages {
        bail!("a remote stage must be the last one for now (stage {stage} of {stages})");
    }
    let cfg = crate::config::ModelConfig::load(dir)?;
    let width = cfg.hc.as_ref().map_or(2 * cfg.hidden, |hc| hc.count * cfg.hidden);
    let rows = opts.max_chunk.max(opts.max_seqs * opts.max_verify.max(1)).max(opts.max_seqs);
    // the coordinator's link may be the larger one (other chunk or verify settings): its rows fit here too
    let mut link = HostMapped::new(crate::model::pp_link_bytes(rows, width).max(peer_link))?;
    link.host_slice_mut()[..crate::model::PP_LINK_HEADER].fill(0);
    let link = Arc::new(link);
    opts.pp = Some(crate::model::PpRank { stage, stages, layers: l0..l1, prev: Some(link.clone()), next: None, tp_rank: 0, tp: 1 });
    opts.kv_auto = None;
    // rows from the coordinator into the link, then its flag for the GPU
    {
        let link = link.clone();
        std::thread::Builder::new().name("blaze-pp-recv".into()).spawn(move || {
            let h = header(&link);
            let slot_bytes = crate::model::pp_slot_bytes(link.bytes());
            while let Ok(f) = recv_frame(&mut dr) {
                if f.len() < 8 {
                    eprintln!("blaze: a pipeline frame of {} B", f.len());
                    break;
                }
                let flag = u64::from_le_bytes(f[..8].try_into().unwrap());
                let slot = ((flag.max(1) - 1) % 2) as usize;
                // the slot held step flag - 2: the GPU gives it back once it copied those rows out
                let t0 = Instant::now();
                while unsafe { std::ptr::read_volatile(h.add(PP_ACK)) } + 2 < flag {
                    if t0.elapsed() < Duration::from_micros(200) { std::hint::spin_loop() } else { std::thread::sleep(Duration::from_micros(20)) }
                }
                std::sync::atomic::fence(Ordering::SeqCst);
                let rows = f.len() - 8;
                if rows > slot_bytes {
                    // the flag still goes up: the GPU must not wait for it forever (the step's rows are not there)
                    eprintln!("blaze: pipeline rows of {rows} B for a link slot of {slot_bytes} B");
                    unsafe {
                        std::ptr::write_volatile(h.add(2 + slot), u64::MAX);
                        std::sync::atomic::fence(Ordering::SeqCst);
                        std::ptr::write_volatile(h, flag);
                    }
                    continue;
                }
                // SAFETY: the slot's rows and the header; the GPU reads them only after the flag below
                unsafe {
                    let at = crate::model::PP_LINK_HEADER + slot * slot_bytes;
                    std::ptr::copy_nonoverlapping(f[8..].as_ptr(), link.host_ptr().cast::<u8>().cast_mut().add(at), rows);
                    std::ptr::write_volatile(h.add(2 + slot), rows as u64);
                    std::sync::atomic::fence(Ordering::SeqCst);
                    std::ptr::write_volatile(h, flag);
                }
            }
        })?;
    }
    let embed = Arc::new(Model::load_embedding(dir)?);
    let (ctx, crx) = channel::<Cmd>();
    let (rtx, rrx) = channel::<Result<Reply>>();
    let dir2 = dir.to_path_buf();
    let barrier = Arc::new(RankBarrier::new(1));
    let worker = std::thread::Builder::new()
        .name("blaze-rank0".into())
        .spawn(move || worker_main(0, 1, dir2, opts, embed, None, use_graphs, barrier, crx, rtx, None))?;
    // the load result, then command by command
    let first = rrx.recv().map_err(|_| anyhow!("the stage exited while loading"))?;
    let mut w = W::default();
    put_reply(&mut w, &first);
    send_frame(&mut cw, &w.0)?;
    first?;
    loop {
        let f = match recv_frame(&mut cr) {
            Ok(f) => f,
            Err(_) => {
                let _ = ctx.send(Cmd::Quit);
                break;
            }
        };
        let c = get_cmd(&mut R { b: &f, p: 0 })?;
        let quit = matches!(c, Cmd::Quit);
        ctx.send(c).map_err(|_| anyhow!("the stage exited"))?;
        let reply = rrx.recv().map_err(|_| anyhow!("the stage exited"))?;
        let mut w = W::default();
        put_reply(&mut w, &reply);
        send_frame(&mut cw, &w.0)?;
        if quit {
            break;
        }
    }
    let _ = worker.join();
    Ok(())
}
