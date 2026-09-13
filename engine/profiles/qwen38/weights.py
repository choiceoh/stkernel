"""Qwen3.8-Flash-Next's checkpoint as the composition's `tensor(name)` (profile): mapped files, names, encodings.

The checkpoint on srv2 (/home/choiceoh/models/qwen38-flash-next-nvfp4, 206 safetensors shards, HF names under
`model.language_model.`) keeps three encodings, and each reaches the composition in the form its feature reads:

- bf16 matrices, vectors and the GDN parameters (`model-bf16-*`): read once, cast to the requested dtype, kept;
- NVFP4 experts (`layer-L-experts-A-B`: per expert `gate_proj`/`up_proj`/`down_proj` as the modelopt four-tensor
  layout -- U8 packed e2m1 [out, in/2], E4M3 scales [out, in/16], one FP32 global scale, an input scale the W4A4
  kernels use): dequantised by engine/modules/moe.dequant_nvfp4 into (gate_up [2I, H], down [H, I]) on demand and
  kept in a bounded cache -- the reference lane's form; the served lane reads the packed bytes themselves;
- the PLE table (`model-plefp8-*`: 128 shards of [2,500,012, 160] e4m3 rows, the concatenated hashed n-gram
  vocabulary in row order): a row's shard is row // shard_rows, and rows are gathered by index straight off the
  mapped file (engine/modules/lookup_table's engram reader does the same over NVMe), never loaded whole.

Every read goes through the safetensors header (8 bytes, JSON, data): no safetensors library, one numpy memmap per
shard. Nothing here computes.
"""
from __future__ import annotations

import json
import re
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

CKPT = Path("/home/choiceoh/models/qwen38-flash-next-nvfp4")
NUMPY_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8,
                "I64": np.int64, "I32": np.int32}
TORCH_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8,
                "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32}
_EXPERT = re.compile(r"^(?P<layer>.*layers\.\d+\.mlp\.experts)\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<part>\w+)$")
_SHARD = re.compile(r"^(?P<table>.*ngram_embedding)\.shard_(?P<s>\d+)\.weight$")


class Shard:
    """One safetensors file: its header, and any tensor as a numpy view of the mapped data."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self._map = None

    def view(self, name: str) -> "tuple[np.memmap, str]":
        """(the tensor's bytes as a numpy array of its stored element type, the safetensors dtype name)."""
        entry = self.header[name]
        lo, hi = entry["data_offsets"]
        if self._map is None:
            self._map = np.memmap(self.path, dtype=np.uint8, mode="r")
        raw = self._map[self.base + lo:self.base + hi]
        return raw.view(NUMPY_DTYPES[entry["dtype"]]).reshape(entry["shape"]), entry["dtype"]


def as_torch(array: np.ndarray, stored: str) -> torch.Tensor:
    """A stored array as a torch tensor of its real dtype (bf16 and e4m3 travel as 16- and 8-bit integers): a copy,
    off the read-only map."""
    t = torch.from_numpy(np.array(array, copy=True))
    if stored == "BF16":
        return t.view(torch.bfloat16)
    if stored == "F8_E4M3":
        return t.view(torch.float8_e4m3fn)
    return t


class Weights:
    """`tensor(name)` for engine/profiles/qwen38/composition.build over the checkpoint directory.

    `dtype`: what floating tensors are handed out as (the composition's activation dtype). `expert_cache`: how many
    dequantised experts stay resident (each is 2I*H + H*I floats); the served lane will not dequantise at all."""

    def __init__(self, root=CKPT, *, dtype: str = "bfloat16", prefix: str = "model.language_model.", expert_cache: int = 512):
        self.root = Path(root)
        self.dtype = getattr(torch, dtype)
        self.prefix = prefix
        index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.where = {name: file for name, file in index.items() if name.startswith((prefix, "lm_head."))}
        self.shards: dict = {}
        self.tables: dict = {}                       # ngram table name -> (shard rows, [ordered shard tensor names])
        for name in self.where:
            m = _SHARD.match(name)
            if m:
                self.tables.setdefault(m.group("table"), []).append((int(m.group("s")), name))
        self.tables = {table: [n for _, n in sorted(parts)] for table, parts in self.tables.items()}
        self.config = json.loads((self.root / "config.json").read_text())["text_config"]
        self._kept: dict = {}
        self._experts: OrderedDict = OrderedDict()
        self.expert_cache = expert_cache

    def shard(self, file: str) -> Shard:
        if file not in self.shards:
            self.shards[file] = Shard(self.root / file)
        return self.shards[file]

    def raw(self, name: str) -> torch.Tensor:
        """A checkpoint tensor in its stored dtype, by its checkpoint name."""
        if name not in self.where:
            raise KeyError(name)
        array, stored = self.shard(self.where[name]).view(name)
        return as_torch(array, stored)

    def _name(self, name: str) -> str:
        """A composition name (`model.` + the transformers path) -> the checkpoint's (`model.language_model.` + it)."""
        return name if name.startswith("lm_head.") else self.prefix + name[len("model."):]

    def __call__(self, name: str) -> torch.Tensor:
        """The composition's `tensor(name)`: kept floating tensors in `dtype`, integers as stored; a KeyError for a
        name the checkpoint lacks (the builder tries `<name>.weight` before `<name>`)."""
        if name in self._kept:
            return self._kept[name]
        full = self._name(name)
        if full in self.tables:                      # the whole PLE table is never a tensor here
            raise KeyError(name)
        t = self.raw(full)
        if t.is_floating_point():
            t = t.to(self.dtype)
        self._kept[name] = t
        return t

    def table_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """PLE rows by index [..., heads] -> [..., heads, width] in `dtype`, gathered shard by shard off the map."""
        full = self._name(name)
        shards = self.tables[full]
        first, stored = self.shard(self.where[shards[0]]).view(shards[0])
        per = first.shape[0]
        flat = rows.reshape(-1).to(torch.int64)
        out = torch.empty(flat.numel(), first.shape[1], dtype=self.dtype)
        which = flat // per
        for s in torch.unique(which).tolist():
            sel = (which == s).nonzero().reshape(-1)
            array, _ = self.shard(self.where[shards[s]]).view(shards[s])
            picked = np.ascontiguousarray(array[(flat[sel] % per).numpy()])
            out[sel] = as_torch(picked, stored).to(self.dtype)
        return out.reshape(*rows.shape, first.shape[1])

    def expert(self, layer: int, e: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """(gate_up [2I, H], down [H, I]) in `dtype`: the expert's NVFP4 tensors dequantised, cached bounded."""
        key = (layer, e)
        hit = self._experts.get(key)
        if hit is not None:
            self._experts.move_to_end(key)
            return hit
        from engine.modules.moe import dequant_nvfp4
        base = f"{self.prefix}layers.{layer}.mlp.experts.{e}"
        parts = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = self.raw(f"{base}.{proj}.weight")
            scale = self.raw(f"{base}.{proj}.weight_scale")
            scale2 = self.raw(f"{base}.{proj}.weight_scale_2")
            parts[proj] = dequant_nvfp4(w, scale, scale2).to(self.dtype)
        value = (torch.cat([parts["gate_proj"], parts["up_proj"]]), parts["down_proj"])
        self._experts[key] = value
        while len(self._experts) > self.expert_cache:
            self._experts.popitem(last=False)
        return value

    def composition(self):
        """The composition over this checkpoint."""
        from engine.profiles.qwen38 import composition as qc
        cfg = dict(self.config, dtype=str(self.dtype).replace("torch.", ""))
        return qc.build(cfg, self, expert=self.expert, table=self.table_rows)


def random_weights(cfg: dict, seed: int = 0, prefix: str = "model.") -> dict:
    """A synthetic checkpoint: every tensor engine/profiles/qwen38/composition.build reads, by its transformers name,
    drawn at random for a config (a tiny one: tests, the boot's --tiny). The transformers fused expert layout
    (`experts.gate_up_proj` [E, 2I, H], `experts.down_proj` [E, H, I]) and the whole PLE table as one tensor."""
    from engine.modules.ngram_embedding import head_tables
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape, scale=1.0: torch.randn(*shape, generator=g) * scale
    H, hc = cfg["hidden_size"], cfg["hc_count"]
    kd, vd, nk, nv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"], cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    conv_dim = 2 * nk * kd + nv * vd
    heads, kvh, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    ih, ihd = cfg["indexer_n_heads"], cfg["indexer_head_dim"]
    E, I, S = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"]
    w = {f"{prefix}embed_tokens.weight": r(cfg["vocab_size"], H, scale=0.5), "lm_head.weight": r(cfg["vocab_size"], H, scale=0.1)}
    def hyper(base):
        w[f"{base}.hc_norm.weight"] = r(hc * H, scale=0.3)
        w[f"{base}.input_mix_weight_down.weight"] = r(cfg["hc_lowrank"], hc * H, scale=0.1)
        w[f"{base}.input_mix_weight_up.weight"] = r(hc * H, cfg["hc_lowrank"], scale=0.1)
    hyper(f"{prefix}hyper_connection_mixer")
    for L, kind in enumerate(cfg["layer_types"]):
        base = f"{prefix}layers.{L}"
        for site in ("attn_hyper_connection", "mlp_hyper_connection"):
            hyper(f"{base}.{site}")
            w[f"{base}.{site}.block_inject_weight.weight"] = r(hc, hc * H, scale=0.1)
        if kind == "linear_attention":
            la = f"{base}.linear_attn"
            w[f"{la}.in_proj_qkv.weight"] = r(conv_dim, H, scale=0.1)
            w[f"{la}.in_proj_z.weight"] = r(nv * vd, H, scale=0.1)
            w[f"{la}.in_proj_b.weight"] = r(nv, H, scale=0.1)
            w[f"{la}.in_proj_a.weight"] = r(nv, H, scale=0.1)
            w[f"{la}.out_proj.weight"] = r(H, nv * vd, scale=0.1)
            w[f"{la}.norm.weight"] = 1 + r(vd, scale=0.1)
            w[f"{la}.conv1d.weight"] = r(conv_dim, 1, cfg["linear_conv_kernel_dim"], scale=0.3)
            w[f"{la}.dt_bias"] = torch.ones(nv)
            w[f"{la}.A_log"] = torch.empty(nv).uniform_(0.01, 16, generator=g).log()
        else:
            sa = f"{base}.self_attn"
            w[f"{sa}.q_proj.weight"] = r(heads * hd * 2, H, scale=0.1)
            w[f"{sa}.k_proj.weight"] = r(kvh * hd, H, scale=0.1)
            w[f"{sa}.v_proj.weight"] = r(kvh * hd, H, scale=0.1)
            w[f"{sa}.o_proj.weight"] = r(H, heads * hd, scale=0.1)
            w[f"{sa}.q_norm.weight"] = r(hd, scale=0.3)
            w[f"{sa}.k_norm.weight"] = r(hd, scale=0.3)
            w[f"{sa}.indexer.index_qk_proj.weight"] = r((ih + 1) * ihd, H, scale=0.1)
            w[f"{sa}.indexer.q_layernorm.weight"] = r(ihd, scale=0.3)
            w[f"{sa}.indexer.k_layernorm.weight"] = r(ihd, scale=0.3)
        w[f"{base}.mlp.gate.weight"] = r(E, H, scale=0.1)
        w[f"{base}.mlp.experts.gate_up_proj"] = r(E, 2 * I, H, scale=0.1)
        w[f"{base}.mlp.experts.down_proj"] = r(E, H, I, scale=0.1)
        for name, shape in (("gate_proj", (S, H)), ("up_proj", (S, H)), ("down_proj", (H, S))):
            w[f"{base}.mlp.shared_expert.{name}.weight"] = r(*shape, scale=0.1)
        w[f"{base}.mlp.shared_expert_gate.weight"] = r(1, H, scale=0.1)
        if L + 1 in cfg["ple_layer_ids"]:
            ple = f"{base}.ple"
            heads_n = (cfg["ngram_size"] - 1) * cfg["heads_per_ngram"]
            _, _, total = head_tables(cfg["ngram_size"], cfg["heads_per_ngram"], cfg["ngram_vocab_size_base"], 0)
            padded = -(-total // cfg["make_ngram_vocab_size_divisible_by"]) * cfg["make_ngram_vocab_size_divisible_by"]
            w[f"{ple}.ple_embedding.ngram_embedding.weight"] = r(padded, cfg["ple_embed_dim"] // heads_n, scale=0.2)
            w[f"{ple}.key_proj.weight"] = r(hc * H, cfg["ple_embed_dim"], scale=0.1)
            w[f"{ple}.value_proj.weight"] = r(H, cfg["ple_embed_dim"], scale=0.1)
            for norm in ("norm_key", "norm_query", "norm_conv"):
                w[f"{ple}.{norm}.weight"] = r(hc * H, scale=0.3)
            w[f"{ple}.conv1d.weight"] = r(hc * H, 1, cfg["ple_conv_kernel_size"] if "ple_conv_kernel_size" in cfg else 4, scale=0.3)
    return w


__all__ = ["CKPT", "Shard", "Weights", "as_torch", "random_weights"]
