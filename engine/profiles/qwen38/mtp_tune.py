"""Fine-tune Qwen3.8's MTP head on the target's own streams -- the served draft chain, teacher-forced (profile).

FastMTP's recipe as vLLM Speculators 0.6 serves it (Red Hat, 2026-09-08: Qwen3-Next-80B's per-position acceptance
0.897 / 0.719 / 0.476 -> 0.912 / 0.776 / 0.616 after a short fine-tune): the head runs `depth` times recursively as the
draft chain does (decode_graphs.draft_chain) -- depth 1 from the target's streams at a position and the token after
it, depth d from the head's own streams one position back and the token there -- and every depth's loss counts with
weight beta^(d-1). The tokens are the text's (teacher forcing: depth d only matters where depths < d were kept, and
there its token is the text's). The labels are the target's own distribution at each position -- its closing mixer
over the tapped streams, then the shared head: KL(target || head) -- and, with `auf`, Spec-AUF's cut (arXiv
2607.01893): a depth's loss counts only where the head's greedy picks at the depths before it were the target's.

The attention of depth d's row i (at position start + i + d - 1) reads depth 1's rows 0..i and its own chain's rows of
depths 2..d: the served context (the observation's rows are real, the chain's provisional rows its own). A window
stays inside the QSA budget's covered reach, where the attention selects nothing (every position up to its own), so the
head's indexer weights are not trained and nothing here scores.

What trains (`TRAINED`): the fuse, the three gated-residual sites, the attention, the router, the shared expert and its
gate -- about 90 M parameters, fp32 masters cast to the compute dtype at each use. The routed experts (the checkpoint's
BF16, 2.5 B parameters), the shared embedding and head, and the target's closing mixer stay as they are.

    python3 -m engine.profiles.qwen38.mtp_tune data   --taps DIR[,DIR..] --out DATA   fleet --tap-mtp-inputs shards ->
                                                                                      contiguous runs a sequence
    python3 -m engine.profiles.qwen38.mtp_tune train  --data DATA --ckpt CKPT --out RUN   (RANK/WORLD_SIZE/MASTER_*:
                                                                                      one process a node, data parallel)
    python3 -m engine.profiles.qwen38.mtp_tune extract --ckpt CKPT --out BASE            the 33 tensors, ~7.8 GB, alone
    python3 -m engine.profiles.qwen38.mtp_tune eval   --data DATA --ckpt CKPT [--tuned RUN/head.safetensors]
    python3 -m engine.profiles.qwen38.mtp_tune export --tuned RUN/head.safetensors --ckpt CKPT --out DIR

`CKPT` is a checkpoint copy with the MTP head in BF16 (srv2 ~/models/qwen38-flash-next-nvfp4: its `mtp.*`, fused BF16
experts, lm_head and the language model's embedding and closing mixer). `export` writes the head's served dense
tensors a rank (specs.mtp_specs over the tuned tensors) beside the rank files: fleet --mtp-tuned DIR serves them.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

L0 = "mtp.layers.0."
SITES = {"attn": L0 + "attn_hyper_connection.", "mlp": L0 + "mlp_hyper_connection."}
CLOSE = "mtp.hyper_connection_mixer."
FUSE = {"embed_norm": "mtp.pre_fc_norm_embedding.weight", "embed_proj": "mtp.fc_embedding.weight",
        "hidden_norm": "mtp.pre_fc_norm_hidden.weight", "hidden_proj": "mtp.fc_hidden.weight"}
TRAINED = (
    *FUSE.values(),
    *(f"{site}{name}.weight" for site in SITES.values()
      for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")),
    *(f"{CLOSE}{name}.weight" for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")),
    *(f"{L0}self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")),
    f"{L0}mlp.gate.weight", f"{L0}mlp.shared_expert_gate.weight",
    *(f"{L0}mlp.shared_expert.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")),
)
EXPERTS = (f"{L0}mlp.experts.gate_up_proj", f"{L0}mlp.experts.down_proj")
HEAD = "lm_head.weight"
LAYOUT = "qwen38-mtp-tuned-v1"                      # the export's rank files (fleet --mtp-tuned)


def target_names(prefix: str) -> "dict[str, str]":
    """The language model's tensors the tuning reads: its embedding and its closing mixer."""
    return {"embed": f"{prefix}embed_tokens.weight",
            **{name: f"{prefix}hyper_connection_mixer.{name}.weight"
               for name in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")}}


@dataclass(frozen=True)
class Shape:
    hidden: int
    hc: int
    heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    theta: float
    eps: float
    experts: int
    topk: int
    normalize: bool
    reach: int                                       # the positions QSA's budget covers: a window's attention stays inside

    @classmethod
    def of(cls, cfg: dict) -> "Shape":
        rope = cfg.get("rope_parameters") or {}
        rotary = int(cfg["head_dim"] * rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)))
        ratio = cfg["indexer_compress_ratio"]
        return cls(cfg["hidden_size"], cfg["hc_count"], cfg["num_attention_heads"], cfg["num_key_value_heads"],
                   cfg["head_dim"], rotary, float(rope.get("rope_theta", cfg.get("rope_theta", 10000.0))),
                   cfg["rms_norm_eps"], cfg["num_experts"], cfg["num_experts_per_tok"],
                   bool(cfg.get("norm_topk_prob", True)), (cfg["indexer_budget"] // ratio + 1) * ratio - 1)


def _key(name: str) -> str:
    return name.replace(".", "__")


class Head(torch.nn.Module):
    """Qwen3.8's MTP head as a trainable module over `tensors(name)` (checkpoint names, full tensors): the `train`
    names as fp32 parameters, cast to `dtype` at every use; everything else it reads frozen in `dtype`."""

    def __init__(self, cfg: dict, tensors, *, prefix: str = "model.", dtype=torch.bfloat16, device="cpu",
                 train=TRAINED):
        super().__init__()
        self.shape, self.dtype, self.prefix = Shape.of(cfg), dtype, prefix
        unknown = set(train) - set(TRAINED)
        if unknown:
            raise ValueError(f"only the head's dense weights train, not {sorted(unknown)}")
        self.trained = tuple(train)
        self.weights = torch.nn.ParameterDict(
            {_key(n): torch.nn.Parameter(tensors(n).to(device=device, dtype=torch.float32)) for n in self.trained})
        frozen = [n for n in TRAINED if n not in self.trained] + [*EXPERTS, HEAD, *target_names(prefix).values()]
        self.frozen = {n: tensors(n).to(device=device, dtype=dtype) for n in frozen}

    def w(self, name: str) -> torch.Tensor:
        key = _key(name)
        return self.weights[key].to(self.dtype) if key in self.weights else self.frozen[name]

    # -- the head's pieces (engine/modules, op for op) -----------------------------------------------------------------
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.frozen[target_names(self.prefix)["embed"]][ids]

    def fuse(self, ids: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
        from engine.modules.mtp import fuse_streams
        S = self.shape
        return fuse_streams(self.embed(ids), given.to(self.dtype), lambda name: self.w(FUSE[name]), hc=S.hc,
                            hidden=S.hidden, eps=S.eps)

    def site(self, site: str, h: torch.Tensor):
        from engine.modules.hyper_connection import gated_residual
        base = SITES[site]
        return gated_residual(h, self.w(base + "hc_norm.weight"), self.w(base + "input_mix_weight_down.weight"),
                              self.w(base + "input_mix_weight_up.weight"), self.w(base + "block_inject_weight.weight"),
                              self.shape.hc, self.shape.eps)

    @staticmethod
    def leave(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        return h + (out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def close(self, h: torch.Tensor) -> torch.Tensor:
        from engine.modules.hyper_connection import gated_residual
        return gated_residual(h, self.w(CLOSE + "hc_norm.weight"), self.w(CLOSE + "input_mix_weight_down.weight"),
                              self.w(CLOSE + "input_mix_weight_up.weight"), None, self.shape.hc, self.shape.eps)

    def target_close(self, streams: torch.Tensor) -> torch.Tensor:
        """The target's closing mixer over its tapped streams: the hidden its head reads (no gradient)."""
        from engine.modules.hyper_connection import gated_residual
        t = target_names(self.prefix)
        f = self.frozen
        return gated_residual(streams.to(self.dtype), f[t["hc_norm"]], f[t["input_mix_weight_down"]],
                              f[t["input_mix_weight_up"]], None, self.shape.hc, self.shape.eps)

    def qkv(self, x: torch.Tensor, positions: torch.Tensor):
        """(q [N, heads, D] normalised and rotated, k and v [N, kv, D] -- k normalised and rotated --, the output gate
        [N, heads*D] before its sigmoid). transformers qwen4_exp's q_proj holds each head's query and gate."""
        from engine.modules.norm import rmsnorm_unit_offset
        from engine.modules.rotary import apply_rope, rope_tables
        S, linear = self.shape, torch.nn.functional.linear
        n, D = x.shape[0], S.head_dim
        wq = self.w(L0 + "self_attn.q_proj.weight").view(S.heads, 2, D, -1)
        q = linear(x, wq[:, 0].reshape(S.heads * D, -1)).view(n, S.heads, D)
        gate = linear(x, wq[:, 1].reshape(S.heads * D, -1))
        k = linear(x, self.w(L0 + "self_attn.k_proj.weight")).view(n, S.kv_heads, D)
        v = linear(x, self.w(L0 + "self_attn.v_proj.weight")).view(n, S.kv_heads, D)
        q = rmsnorm_unit_offset(q, self.w(L0 + "self_attn.q_norm.weight"), S.eps)
        k = rmsnorm_unit_offset(k, self.w(L0 + "self_attn.k_norm.weight"), S.eps)
        if S.rotary:
            cos, sin = rope_tables(positions, S.rotary, S.theta, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v, gate

    def attend(self, q: torch.Tensor, keys: "list[torch.Tensor]", values: "list[torch.Tensor]") -> torch.Tensor:
        """Depth len(keys)'s rows [T, heads, D]: row i attends depth 1's rows 0..i (keys[0]) and its own chain's rows
        of the later depths (keys[1:], row i each) -- scores in the compute dtype, the softmax in fp32, rounded
        (modules/attention.Attention)."""
        S = self.shape
        groups = S.heads // S.kv_heads
        T, scale = q.shape[0], S.head_dim ** -0.5
        qh = q.transpose(0, 1)                                                          # [H, T, D]
        k1 = keys[0].repeat_interleave(groups, dim=1).transpose(0, 1)                   # [H, T, D]
        v1 = values[0].repeat_interleave(groups, dim=1).transpose(0, 1)
        scores = torch.matmul(qh, k1.transpose(1, 2)) * scale                           # [H, T, T]
        causal = torch.arange(T, device=q.device)[None, :] <= torch.arange(T, device=q.device)[:, None]
        scores = scores.masked_fill(~causal[None], torch.finfo(scores.dtype).min)
        if len(keys) > 1:
            own_k = torch.stack([k.repeat_interleave(groups, dim=1) for k in keys[1:]], dim=2)      # [T, H, j, D]
            own = torch.einsum("thd,thjd->htj", q.float(), own_k.float()).to(q.dtype) * scale      # [H, T, j]
            scores = torch.cat([scores, own], dim=-1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs[..., :T].float(), v1.float())                          # [H, T, D]
        if len(keys) > 1:
            own_v = torch.stack([v.repeat_interleave(groups, dim=1) for v in values[1:]], dim=2)    # [T, H, j, D]
            out = out + torch.einsum("htj,thjd->htd", probs[..., T:].float(), own_v.float())
        return out.to(q.dtype).transpose(0, 1)

    def moe(self, x: torch.Tensor) -> torch.Tensor:
        """The routed experts (softmax top-k, renormalised) in ascending id, and the sigmoid-gated shared expert
        (modules/moe.MoE with Qwen3.8's axes)."""
        from engine.modules.moe import gated_mlp, route
        S, linear = self.shape, torch.nn.functional.linear
        ids, weights, _ = route(x, self.w(L0 + "mlp.gate.weight"), score="softmax", topk=S.topk, normalize=S.normalize,
                                scaling=1.0, fp32=False)
        gate_up, down = self.frozen[EXPERTS[0]], self.frozen[EXPERTS[1]]
        out = torch.zeros_like(x)
        n = x.shape[0]
        flat = ids.transpose(0, 1).reshape(-1)                  # slot-major: an expert's rows by slot, then row
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=S.experts).tolist()
        at = 0
        for e, count in enumerate(counts):
            if not count:
                continue
            pick = order[at:at + count]
            at += count
            slots, rows = pick // n, pick % n
            g, u = linear(x[rows], gate_up[e]).chunk(2, dim=-1)
            y = linear(gated_mlp(g, u, "silu"), down[e]) * weights[rows, slots, None]
            out = out.index_add(0, rows, y.to(out.dtype))
        sg, su, sd = (self.w(f"{L0}mlp.shared_expert.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj"))
        shared = linear(gated_mlp(linear(x, sg), linear(x, su), "silu"), sd)
        return out + torch.sigmoid(linear(x, self.w(L0 + "mlp.shared_expert_gate.weight"))) * shared

    # -- the chain ---------------------------------------------------------------------------------------------------
    def chain(self, given: torch.Tensor, tokens: torch.Tensor, start: int, depth: int):
        """One window, `depth` deep: `given` [T, hc*H] the target's streams at positions start..start+T-1, `tokens`
        [>= T + depth - 1] the token after each of those positions -> (hidden [depth, T, H] each depth's rows before
        the head, streams [depth, T, hc*H] each depth's own streams). Depth d's row i sits at start + i + d - 1."""
        S, linear = self.shape, torch.nn.functional.linear
        T = given.shape[0]
        if T + depth - 1 > S.reach:
            raise ValueError(f"a window of {T} positions and {depth} depths passes the {S.reach} QSA covers")
        if tokens.shape[0] < T + depth - 1:
            raise ValueError("a window needs the token after each of its positions and its chain's")
        hidden, streams, keys, values = [], [], [], []
        state = given
        for d in range(1, depth + 1):
            positions = torch.arange(start + d - 1, start + d - 1 + T, device=given.device)
            h = self.fuse(tokens[d - 1:d - 1 + T], state)
            x, inject = self.site("attn", h)
            q, k, v, gate = self.qkv(x, positions)
            keys.append(k)
            values.append(v)
            attended = self.attend(q, keys, values).reshape(T, -1) * torch.sigmoid(gate)
            h = self.leave(h, linear(attended, self.w(L0 + "self_attn.o_proj.weight")), inject)
            x, inject = self.site("mlp", h)
            h = self.leave(h, self.moe(x), inject)
            hidden.append(self.close(h))
            streams.append(h)
            state = h
        return torch.stack(hidden), torch.stack(streams)


def kl_chunk(hidden: torch.Tensor, target: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """sum over rows of KL(softmax(target @ head) || softmax(hidden @ head)), fp32."""
    draft = torch.log_softmax(torch.matmul(hidden, head.T).float(), dim=-1)
    want = torch.log_softmax(torch.matmul(target, head.T).float(), dim=-1)
    return (want.exp() * (want - draft)).sum(-1)


def picks(hidden: torch.Tensor, head: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """The head's greedy ids over rows, in chunks (no gradient)."""
    with torch.no_grad():
        return torch.cat([torch.matmul(hidden[c:c + chunk], head.T).float().argmax(-1)
                          for c in range(0, hidden.shape[0], chunk)])


def window_loss(model: Head, given: torch.Tensor, tokens: torch.Tensor, start: int, depth: int, *, beta: float = 0.6,
                auf: bool = False, chunk: int = 256):
    """(loss, metrics) of one window: `given` [T + depth, hc*H] the target's streams from `start` and `tokens`
    [T + depth] the token after each -- chain starts 0..T-1, depth d's label at row i the target's distribution at
    position start + i + d (its streams there, through its closing mixer and the head) and the text's token there."""
    from torch.utils.checkpoint import checkpoint
    T = given.shape[0] - depth
    if T <= 0:
        raise ValueError(f"a window of {given.shape[0]} positions holds no chain {depth} deep")
    head = model.frozen[HEAD]
    hidden, _ = model.chain(given[:T], tokens, start, depth)
    with torch.no_grad():
        target = model.target_close(given)                                         # [T + depth, H]
    losses, metrics, alive = [], {}, torch.ones(T, dtype=torch.bool, device=given.device)
    kept = torch.ones(T, dtype=torch.bool, device=given.device)
    for d in range(1, depth + 1):
        th, labels = target[d:d + T], tokens[d:d + T]
        mask = (alive if auf else torch.ones_like(alive)).to(torch.float32)
        total = hidden.new_zeros((), dtype=torch.float32)
        for c in range(0, T, chunk):
            rows = checkpoint(kl_chunk, hidden[d - 1, c:c + chunk], th[c:c + chunk], head, use_reentrant=False)
            total = total + (rows * mask[c:c + chunk]).sum()
        losses.append(total / mask.sum().clamp_min(1.0))
        mine, theirs = picks(hidden[d - 1].detach(), head, chunk), picks(th, head, chunk)
        agree = mine == theirs
        alive = alive & agree
        kept = kept & (mine == labels)
        metrics[f"loss_{d}"] = float(losses[-1].detach())
        metrics[f"agree_{d}"] = float(agree.float().mean())
        metrics[f"chain_{d}"] = float(alive.float().mean())                      # every depth to here the target's pick
        metrics[f"text_chain_{d}"] = float(kept.float().mean())                  # ... the text's token (sampled data)
    weights = [beta ** (d - 1) for d in range(1, depth + 1)]
    loss = sum(w * l for w, l in zip(weights, losses)) / sum(weights)
    metrics["tokens_a_step"] = 1.0 + sum(metrics[f"chain_{d}"] for d in range(1, depth + 1))
    return loss, metrics


# -- data: the tap's shards -> runs ------------------------------------------------------------------------------------
def shards(directories) -> "list[Path]":
    """fleet --tap-mtp-inputs shards, in the order they were written."""
    return [shard for directory in directories for shard in sorted(Path(directory).glob("mtp-inputs-*.npz"))]


def load_meta(files) -> dict:
    """{(boot, sequence): {position: (shard index, row, next token, decoded)}} -- a boot's sequence ids are its own, and
    a position seen twice (a prompt prefilled again after a park) keeps its last record."""
    import numpy as np
    table = {}
    for index, shard in enumerate(files):
        boot = "-".join(shard.stem.split("-")[2:4])
        for row, (seq, position, token, decoded) in enumerate(np.load(shard)["meta"].tolist()):
            table.setdefault((boot, seq), {})[position] = (index, row, token, decoded)
    return table


def build_runs(files, out: Path, *, holdout: float = 0.1, min_length: int = 32, seed: int = 0) -> dict:
    """Each sequence's contiguous positions as a run file (streams BF16 as int16 bits, the next tokens, decoded
    flags); a whole sequence goes to evaluation with probability `holdout`. The shards' streams are read one shard at a
    time and each run is written when its last row is in. -> the index, written to runs.json."""
    import numpy as np
    out.mkdir(parents=True, exist_ok=True)
    table = load_meta(files)
    rng = random.Random(seed)
    runs, waiting = [], {}                                  # shard index -> [(run, slot, row)]
    for key in sorted(table):
        positions = table[key]
        split = "eval" if rng.random() < holdout else "train"
        ordered = sorted(positions)
        begin = 0
        for i in range(1, len(ordered) + 1):
            if i == len(ordered) or ordered[i] != ordered[i - 1] + 1:
                span = ordered[begin:i]
                if len(span) >= min_length:
                    picked = [positions[p] for p in span]
                    run = {"file": f"run-{len(runs):06d}.npz", "boot": key[0], "seq": key[1], "start": span[0],
                           "length": len(span), "decoded": sum(d for *_, d in picked), "split": split,
                           "_tokens": [t for _, _, t, _ in picked], "_flags": [d for *_, d in picked],
                           "_rows": [None] * len(span), "_left": len(span)}
                    runs.append(run)
                    for slot, (shard, row, _, _) in enumerate(picked):
                        waiting.setdefault(shard, []).append((run, slot, row))
                begin = i
    for index in sorted(waiting):
        streams = np.load(files[index])["streams"]
        for run, slot, row in waiting.pop(index):
            run["_rows"][slot] = streams[row]
            run["_left"] -= 1
            if not run["_left"]:
                np.savez(out / run["file"], streams=np.stack(run["_rows"]),
                         tokens=np.array(run["_tokens"], dtype=np.int64), decoded=np.array(run["_flags"], dtype=np.uint8))
                for private in ("_tokens", "_flags", "_rows", "_left"):
                    del run[private]
    index = {"runs": runs, "positions": sum(r["length"] for r in runs),
             "train": sum(r["length"] for r in runs if r["split"] == "train"),
             "eval": sum(r["length"] for r in runs if r["split"] == "eval"),
             "decoded": sum(r["decoded"] for r in runs)}
    (out / "runs.json").write_text(json.dumps(index, indent=1) + "\n")
    return index


class Runs:
    """A data directory's runs of one split, sampled as windows of `window` chain starts `depth` deep."""

    def __init__(self, data: Path, split: str, *, window: int, depth: int):
        index = json.loads((Path(data) / "runs.json").read_text())
        self.data, self.depth, self.window = Path(data), depth, window
        self.runs = [r for r in index["runs"] if r["split"] == split and r["length"] > depth + 8]
        if not self.runs:
            raise ValueError(f"{data}: no {split} run longer than {depth + 8} positions")
        self._cache = {}

    def _load(self, run):
        import numpy as np
        if run["file"] not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            data = np.load(self.data / run["file"])
            self._cache[run["file"]] = (torch.from_numpy(data["streams"]).view(torch.bfloat16),
                                        torch.from_numpy(data["tokens"]))
        return self._cache[run["file"]]

    def windows(self, rng: random.Random, count: int):
        """`count` windows drawn by position: (streams [T + depth, hc*H], tokens [T + depth], start)."""
        weights = [r["length"] for r in self.runs]
        for run in rng.choices(self.runs, weights=weights, k=count):
            streams, tokens = self._load(run)
            span = min(self.window + self.depth, run["length"])
            at = rng.randrange(0, run["length"] - span + 1)
            yield streams[at:at + span], tokens[at:at + span], run["start"] + at

    def every_window(self):
        """Each run cut into consecutive windows (evaluation)."""
        for run in self.runs:
            streams, tokens = self._load(run)
            span = self.window + self.depth
            for at in range(0, max(1, run["length"] - self.depth), self.window):
                piece = slice(at, min(at + span, run["length"]))
                if piece.stop - piece.start > self.depth + 1:
                    yield streams[piece], tokens[piece], run["start"] + at


# -- the checkpoint --------------------------------------------------------------------------------------------------
def checkpoint_tensors(ckpt: Path, prefix: str, *, tuned: "Path | None" = None):
    """name -> tensor over a checkpoint copy (engine/base/checkpoint.Checkpoint), the tuned file's names first."""
    from engine.base.checkpoint import Checkpoint
    source = Checkpoint(str(ckpt))
    names = [*TRAINED, *EXPERTS, HEAD, *target_names(prefix).values()]
    loaded = source.load(names)
    if tuned is not None:
        from safetensors.torch import load_file
        loaded.update(load_file(str(tuned)))
    return loaded.__getitem__


def config(ckpt: Path) -> "tuple[dict, str]":
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    return text, "model.language_model." if "text_config" in cfg else "model."


def distributed() -> "tuple[int, int]":
    """(rank, world) of a run torch.distributed started (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT: one process a
    node, NCCL on GPUs), (0, 1) for one process."""
    import os
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return dist.get_rank(), dist.get_world_size()


def all_sum(values: "list[float]", world: int, device) -> "list[float]":
    """Every rank's numbers summed, the same list on each (one all-reduce)."""
    if world <= 1:
        return list(values)
    import torch.distributed as dist
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def average_gradients(params, world: int) -> None:
    """Every rank's gradients averaged in place, one flat all-reduce: the step every rank then takes is the same, so
    the ranks' weights stay the same bits."""
    if world <= 1:
        return
    import torch.distributed as dist
    grads = [p.grad for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    at = 0
    for g in grads:
        g.copy_(flat[at:at + g.numel()].view_as(g))
        at += g.numel()


def evaluate(model: Head, runs: Runs, *, depth: int, limit: int = 0, rank: int = 0, world: int = 1) -> dict:
    """The held-out windows' metrics, position-weighted -- window n on rank n % world, the sums gathered, so every
    rank holds the same answer."""
    sums, weight = {}, 0
    device = next(iter(model.frozen.values())).device
    with torch.no_grad():
        for n, (streams, tokens, start) in enumerate(runs.every_window()):
            if limit and n >= limit:
                break
            if n % world != rank:
                continue
            loss, metrics = window_loss(model, streams.to(device), tokens.to(device), start, depth)
            rows = streams.shape[0] - depth
            weight += rows
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value * rows
    keys = sorted(set(sums) | {f"{m}_{d}" for m in ("loss", "agree", "chain", "text_chain") for d in range(1, depth + 1)}
                  | {"tokens_a_step"})
    total = all_sum([sums.get(k, 0.0) for k in keys] + [float(weight)], world, device)
    weight = total[-1]
    return {key: round(value / max(weight, 1), 5) for key, value in zip(keys, total)} | {"positions": int(weight)}


def train(args) -> None:
    cfg, prefix = config(args.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rank, world = distributed()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed * 1000 + rank)                   # each rank its own windows
    model = Head(cfg, checkpoint_tensors(args.ckpt, prefix), prefix=prefix, device=device)
    train_runs = Runs(args.data, "train", window=args.window, depth=args.depth)
    eval_runs = Runs(args.data, "eval", window=args.window, depth=args.depth)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a") if rank == 0 else None

    def note(record):
        record["t"] = round(time.time(), 1)
        if log is not None:
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)

    note({"event": "start", "args": {k: str(v) for k, v in vars(args).items()}, "world": world,
          "trained_parameters": sum(p.numel() for p in model.weights.values())})
    base = evaluate(model, eval_runs, depth=args.depth, limit=args.eval_windows, rank=rank, world=world)
    note({"event": "eval", "step": 0, **base})
    optimizer = torch.optim.AdamW(model.weights.values(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total = args.steps
    best = base["tokens_a_step"]
    for step in range(1, total + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        seen = {}
        for streams, tokens, start in train_runs.windows(rng, args.accumulate):
            loss, metrics = window_loss(model, streams.to(device), tokens.to(device), start, args.depth,
                                        beta=args.beta, auf=args.auf)
            (loss / args.accumulate).backward()
            for key, value in metrics.items():
                seen[key] = seen.get(key, 0.0) + value / args.accumulate
        average_gradients(list(model.weights.values()), world)
        norm = torch.nn.utils.clip_grad_norm_(model.weights.values(), args.clip)
        optimizer.step()
        if step % args.log_every == 0:
            note({"event": "train", "step": step, "lr": lr, "grad_norm": round(float(norm), 4),
                  **{k: round(v, 5) for k, v in seen.items()}})
        if step % args.eval_every == 0 or step == total:
            result = evaluate(model, eval_runs, depth=args.depth, limit=args.eval_windows, rank=rank, world=world)
            note({"event": "eval", "step": step, **result})
            if result["tokens_a_step"] > best:                      # every rank the same numbers, the same choice
                best = result["tokens_a_step"]
                if rank == 0:
                    save(model, out / "head.safetensors", {"step": step, "eval": result, "base": base, "world": world})
                note({"event": "saved", "step": step, "tokens_a_step": best})
    note({"event": "end", "best_tokens_a_step": best, "base_tokens_a_step": base["tokens_a_step"]})
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def save(model: Head, path: Path, meta: dict) -> None:
    """The trained weights by checkpoint name, BF16 -- what `export` and `eval --tuned` read."""
    from safetensors.torch import save_file
    tensors = {name: model.weights[_key(name)].detach().to(torch.bfloat16).cpu().contiguous() for name in model.trained}
    save_file(tensors, str(path), metadata={"layout": "qwen38-mtp-tuned-head-v1", "meta": json.dumps(meta)})


def export(args) -> None:
    """The tuned head's served dense tensors a rank (specs.mtp_specs over the checkpoint with the tuned names
    replaced), one file a rank under `out` with a manifest -- fleet --mtp-tuned serves them."""
    from engine.base.checkpoint import Checkpoint
    from engine.base.preshard import RankWriter
    from engine.profiles.qwen38 import facts, specs as layout
    from safetensors.torch import load_file
    F = facts.load(args.ckpt_meta or args.ckpt)
    wanted = layout.mtp_specs(F, routed=False)
    sources = sorted({k for s in wanted for k in s.sources})
    tensors = Checkpoint(str(args.ckpt)).load(sources)
    tuned = load_file(str(args.tuned))
    unknown = set(tuned) - set(sources)
    if unknown:
        raise SystemExit(f"{args.tuned}: tensors the served head does not read: {sorted(unknown)[:5]}")
    tensors.update(tuned)
    out = Path(args.out)
    staging = out.with_name(out.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    manifest = {"layout": LAYOUT, "tuned": str(args.tuned), "replaced": sorted(tuned), "ranks": {}}
    for rank in range(facts.TP):
        path = staging / tuned_file(rank)
        writer = RankWriter(path, wanted, {"weight_layout": LAYOUT, "rank": rank, "world": facts.TP})
        for spec in wanted:
            value = spec.build(tensors, rank, facts.TP)
            if tuple(value.shape) != tuple(spec.shape) or value.dtype != spec.dtype:
                raise SystemExit(f"{spec.name}: {tuple(value.shape)} {value.dtype} against the served "
                                 f"{tuple(spec.shape)} {spec.dtype}")
            writer.put(spec.name, value)
        writer.close()
        manifest["ranks"][rank] = {"file": path.name, "tensors": len(wanted), "bytes": path.stat().st_size}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    staging.rename(out)
    print(json.dumps(manifest["ranks"]), flush=True)


def extract(args) -> None:
    """The tensors the tuning reads, out of a checkpoint copy into a small one of its own (one shard, its index, the
    config): what the other nodes train from without the 126 GB copy."""
    import shutil
    from engine.base.checkpoint import Checkpoint
    from engine.base.params import Spec
    from engine.base.preshard import RankWriter
    cfg, prefix = config(args.ckpt)
    names = [*TRAINED, *EXPERTS, HEAD, *target_names(prefix).values()]
    source = Checkpoint(str(args.ckpt))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # one tensor in memory at a time, read through the shard mapping (the largest, the experts' gate_up, is 3.4 GB,
    # copied once by the writer): it runs beside production under a 6 GB cap
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    specs = []
    for name in names:
        header = source.reader(source.weight_map[name]).header[name]
        specs.append(Spec(name, tuple(header["shape"]), dtypes[header["dtype"]]))
    writer = RankWriter(out / "tune-base.safetensors", specs, {"layout": "qwen38-mtp-tune-base-v1"})
    for name in names:
        writer.put(name, source.views([name])[name])            # the mapped bytes: page cache, not this process
    writer.close()
    (out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "tune-base.safetensors" for n in names}}))
    shutil.copy(Path(args.ckpt) / "config.json", out / "config.json")
    print(json.dumps({"tensors": len(names), "bytes": (out / "tune-base.safetensors").stat().st_size}), flush=True)


def tuned_file(rank: int) -> str:
    from engine.profiles.qwen38 import facts
    return f"mtp-tuned-r{rank}of{facts.TP}.safetensors"


def served_names(F) -> "list[str]":
    """The served dense tensors of the head a tuned file replaces (fleet --mtp-tuned)."""
    from engine.profiles.qwen38 import specs as layout
    return [s.name for s in layout.mtp_specs(F, routed=False)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m engine.profiles.qwen38.mtp_tune", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data")
    d.add_argument("--taps", required=True, help="fleet --tap-mtp-inputs directories, comma separated")
    d.add_argument("--out", required=True)
    d.add_argument("--holdout", type=float, default=0.1)
    d.add_argument("--min-length", type=int, default=32)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--data", required=True)
        p.add_argument("--ckpt", required=True, type=Path)
        p.add_argument("--depth", type=int, default=3)
        p.add_argument("--window", type=int, default=1024)
        p.add_argument("--eval-windows", type=int, default=0, help="0: every held-out window")
        p.add_argument("--seed", type=int, default=0)
        if name == "train":
            p.add_argument("--out", required=True)
            p.add_argument("--steps", type=int, default=2000)
            p.add_argument("--accumulate", type=int, default=4)
            p.add_argument("--lr", type=float, default=2e-5)
            p.add_argument("--warmup", type=int, default=100)
            p.add_argument("--beta", type=float, default=0.6)
            p.add_argument("--auf", action="store_true")
            p.add_argument("--clip", type=float, default=1.0)
            p.add_argument("--log-every", type=int, default=10)
            p.add_argument("--eval-every", type=int, default=200)
        else:
            p.add_argument("--tuned", type=Path, default=None)
    x = sub.add_parser("extract")
    x.add_argument("--ckpt", required=True, type=Path)
    x.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--tuned", required=True, type=Path)
    e.add_argument("--ckpt", required=True, type=Path)
    e.add_argument("--ckpt-meta", default=None, type=Path, help="the served checkpoint's config (default: --ckpt)")
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.command == "data":
        index = build_runs(shards([Path(p) for p in a.taps.split(",")]), Path(a.out), holdout=a.holdout,
                           min_length=a.min_length)
        print(json.dumps({k: v for k, v in index.items() if k != "runs"}), flush=True)
    elif a.command == "extract":
        extract(a)
    elif a.command == "train":
        train(a)
    elif a.command == "eval":
        cfg, prefix = config(a.ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Head(cfg, checkpoint_tensors(a.ckpt, prefix, tuned=a.tuned), prefix=prefix, device=device)
        runs = Runs(a.data, "eval", window=a.window, depth=a.depth)
        print(json.dumps(evaluate(model, runs, depth=a.depth, limit=a.eval_windows)), flush=True)
    else:
        export(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
