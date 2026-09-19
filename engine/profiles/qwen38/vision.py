"""Qwen3.8-Flash-Next's eyes (profile): the checkpoint's Qwen3-VL vision tower and the processor in front of it, as the
served model runs them -- without vLLM or transformers.

The checkpoint (`Qwen4ExpForConditionalGeneration`) carries a BF16 tower under `model.visual.*` (333 tensors, ~0.45 B
parameters): a Conv3d patch embedding over 2 x 16 x 16 pixel patches, a learned 48 x 48 position table resampled
bilinearly to each image's grid, 27 pre-norm blocks (LayerNorm, full attention with a 2-D rotary over (row, column),
a gelu-tanh MLP), and a merger that folds each 2 x 2 window into one token of the text width (LayerNorm, 4,608 ->
4,608, GELU, -> 2,560). No deepstack (`deepstack_visual_indexes` is empty). The served stack is vLLM's
`qwen3_vl.Qwen3_VisionTransformer` behind transformers' `Qwen2VLImageProcessor` (5.15.1 in the served image,
`vllm/vllm-openai:qwen38-flash-next`); probes/qwen38_vision_reference.py reads both there and writes
tests/fixtures/qwen38_vision_reference.json, which this module is held to.

  * `load` / `specs`: the tower's constants from config.json's vision_config and preprocessor_config.json, checked at
    load (D3); its tensors under their checkpoint names, whole on every rank (0.9 GB; splitting it four ways buys
    nothing), written once as `vision.safetensors` next to the rank files (`write_file`).
  * `Door` (rank 0, no weights): bytes -> the picture as vLLM loads it (engine/modules/pictures) -> resized to the
    processor's grid (`smart_resize`: 32-pixel multiples inside the pixel budget, bicubic antialiased) -> a uint8
    canvas and the placeholder run the text model sees: <|vision_start|> <|image_pad|> x tokens <|vision_end|>.
  * `Vision` (every rank): canvas -> normalised patches in the processor's order -> tower -> [tokens, 2560] rows that
    replace the embedding rows at the placeholder positions. The ranks compute it identically and check that they did.
  * `rope_positions`: the text model's interleaved mRoPE positions for a prompt with pictures (Qwen3-VL's rule).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from engine.base.params import Spec
from engine.modules.pictures import decode as decode_picture

BF = torch.bfloat16
FILE = "vision.safetensors"          # next to the rank files: the tower, whole, under its checkpoint names
PREFIX = "model.visual."
LIMITS = {"image": 4}                # per prompt: GLM-5.3's served limit (PR #431); video is not served here yet
ROPE_BASE = 10000.0                  # the tower's 2-D rotary: get_rope(head_dim, partial_rotary_factor 0.5), neox halves
LN_EPS = 1e-6                        # every LayerNorm in the tower (vLLM passes the text model's rms_norm_eps, 1e-6)
MAX_ASPECT = 200                     # smart_resize refuses a longer side more than 200 times the shorter


@dataclass(frozen=True)
class VisionFacts:
    depth: int
    hidden: int
    heads: int
    inter: int
    out_hidden: int                 # the text model's width: what the merger emits
    patch: int
    temporal: int
    merge: int
    channels: int
    grid_side: int                  # the learned position table is grid_side x grid_side
    image_token: int
    video_token: int
    vision_start: int
    vision_end: int
    mean: tuple
    std: tuple
    min_pixels: int
    max_pixels: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def factor(self) -> int:
        return self.patch * self.merge          # an image's sides are multiples of one merged token

    @property
    def patch_dim(self) -> int:
        return self.channels * self.temporal * self.patch * self.patch

    def tokens(self, grid) -> int:
        t, h, w = grid
        return t * h * w // (self.merge * self.merge)


def load(directory: "str | Path") -> VisionFacts:
    """The checkpoint's (or the rank files') config.json and preprocessor_config.json -> the facts, checked (D3)."""
    c = json.loads((Path(directory) / "config.json").read_text())
    v = c["vision_config"]
    p = json.loads((Path(directory) / "preprocessor_config.json").read_text())
    side = int(math.isqrt(int(v["num_position_embeddings"])))
    V = VisionFacts(
        depth=int(v["depth"]), hidden=int(v["hidden_size"]), heads=int(v["num_heads"]), inter=int(v["intermediate_size"]),
        out_hidden=int(v["out_hidden_size"]), patch=int(v["patch_size"]), temporal=int(v["temporal_patch_size"]),
        merge=int(v["spatial_merge_size"]), channels=int(v["in_channels"]), grid_side=side,
        image_token=int(c["image_token_id"]), video_token=int(c["video_token_id"]),
        vision_start=int(c["vision_start_token_id"]), vision_end=int(c["vision_end_token_id"]),
        mean=tuple(float(x) for x in p["image_mean"]), std=tuple(float(x) for x in p["image_std"]),
        min_pixels=int(p["size"]["shortest_edge"]), max_pixels=int(p["size"]["longest_edge"]),
    )
    # -- what the code assumes, checked against the checkpoint (D3) ----------
    assert c["architectures"] == ["Qwen4ExpForConditionalGeneration"], c.get("architectures")
    assert not c.get("language_model_only", False), "a language-model-only checkpoint has no tower to serve"
    assert v["hidden_act"] == "gelu_pytorch_tanh", "the blocks' MLP is gelu-tanh"
    assert not v.get("deepstack_visual_indexes"), "deepstack features into the text layers are not served"
    assert side * side == int(v["num_position_embeddings"]), "the learned position table is square"
    assert p["processor_class"] == "Qwen3VLProcessor" and p["image_processor_type"] in ("Qwen2VLImageProcessor",
                                                                                       "Qwen2VLImageProcessorFast")
    assert int(p["patch_size"]) == V.patch and int(p["merge_size"]) == V.merge and int(p["temporal_patch_size"]) == V.temporal
    assert p.get("do_rescale", True) and p.get("do_normalize", True) and p.get("do_resize", True)
    assert float(p.get("rescale_factor", 1 / 255)) == 1 / 255
    assert len(V.mean) == len(V.std) == V.channels == 3
    assert V.hidden % V.heads == 0 and V.head_dim % 4 == 0, "the 2-D rope splits each head's rotary half into rows and columns"
    assert V.merge == 2 and V.temporal == 2 and V.depth > 0 and 0 < V.min_pixels <= V.max_pixels
    assert int(c["text_config"]["hidden_size"]) == V.out_hidden, "the merger emits the text model's width"
    return V


# -- the tower's tensors, as the checkpoint names them (whole copies on every rank) ---------------------------------
def specs(V: VisionFacts) -> "list[Spec]":
    def whole(name, shape):
        return Spec(name, tuple(shape), BF, (name,), lambda s, r, W, name=name: s[name].contiguous())
    P, H, M = PREFIX, V.hidden, V.hidden * V.merge * V.merge
    out = [whole(P + "patch_embed.proj.weight", (H, V.channels, V.temporal, V.patch, V.patch)),
           whole(P + "patch_embed.proj.bias", (H,)),
           whole(P + "pos_embed.weight", (V.grid_side * V.grid_side, H))]
    for i in range(V.depth):
        b = f"{P}blocks.{i}."
        out += [whole(b + "norm1.weight", (H,)), whole(b + "norm1.bias", (H,)),
                whole(b + "norm2.weight", (H,)), whole(b + "norm2.bias", (H,)),
                whole(b + "attn.qkv.weight", (3 * H, H)), whole(b + "attn.qkv.bias", (3 * H,)),
                whole(b + "attn.proj.weight", (H, H)), whole(b + "attn.proj.bias", (H,)),
                whole(b + "mlp.linear_fc1.weight", (V.inter, H)), whole(b + "mlp.linear_fc1.bias", (V.inter,)),
                whole(b + "mlp.linear_fc2.weight", (H, V.inter)), whole(b + "mlp.linear_fc2.bias", (H,))]
    out += [whole(P + "merger.norm.weight", (H,)), whole(P + "merger.norm.bias", (H,)),
            whole(P + "merger.linear_fc1.weight", (M, M)), whole(P + "merger.linear_fc1.bias", (M,)),
            whole(P + "merger.linear_fc2.weight", (V.out_hidden, M)), whole(P + "merger.linear_fc2.bias", (V.out_hidden,))]
    return out


def write_file(ckpt: "str | Path", out_dir: "str | Path", log=print) -> int:
    """`vision.safetensors` in the rank files' directory: the tower's tensors, whole, aligned for the arena loader."""
    from engine.base.checkpoint import Checkpoint
    from engine.base.preshard import write_ranks
    V = load(ckpt)
    S = specs(V)
    path = Path(out_dir) / FILE
    sizes = write_ranks([("vision", [s.name for s in S], lambda r: S)], [path], lambda keys: Checkpoint(str(ckpt)).load(keys), 1,
                        metadata={"model": "qwen38", "part": "vision", "layout": "engine.profiles.qwen38.vision"}, log=log)
    return sizes[0]


# -- geometry: the processor's rules, verbatim (transformers qwen2_vl smart_resize) -------------------------------------
def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> "tuple[int, int]":
    """(height, width) for the tower: both sides multiples of `factor`, the area inside [min_pixels, max_pixels], the
    aspect as close as the rounding allows."""
    if min(height, width) <= 0:
        raise ValueError("an image needs a positive height and width")
    if max(height, width) / min(height, width) > MAX_ASPECT:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_ASPECT}, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize(frames: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """uint8 [T, C, h, w] -> uint8 [T, C, H, W]: torchvision's bicubic antialiased resize on the uint8 picture, as the
    processor calls it (TorchvisionBackend.resize: PILImageResampling.BICUBIC, antialias=True)."""
    from torchvision.transforms.v2 import functional as tvF
    return tvF.resize(frames, [H, W], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)


def pixel_values(V: VisionFacts, canvas_u8: np.ndarray, grid, device="cpu") -> torch.Tensor:
    """uint8 [1, C, H, W] -> fp32 [gh*gw, C*temporal*patch*patch], the processor's pixel_values: normalised (the
    processor fuses rescale into normalize: mean and std times 255, one subtract and one divide in fp32), then patches
    in merge-window order (window row, window column, row in window, column in window), each patch's values ordered
    (channel, frame, row, column) with the still picture repeated for the temporal patch."""
    x = torch.from_numpy(np.ascontiguousarray(canvas_u8)).to(device)
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != V.channels:
        raise ValueError("a picture's canvas is uint8 [1, channels, height, width]")
    gt, gh, gw = (int(g) for g in grid)
    if gt != 1 or x.shape[2] != gh * V.patch or x.shape[3] != gw * V.patch or gh % V.merge or gw % V.merge:
        raise ValueError("the canvas and its grid disagree")
    mean = (torch.tensor(V.mean, dtype=torch.float32) * 255.0).view(-1, 1, 1).to(device)
    std = (torch.tensor(V.std, dtype=torch.float32) * 255.0).view(-1, 1, 1).to(device)
    x = x.float().sub(mean).div(std)                                # tvF.normalize on the fp32 picture
    m, P = V.merge, V.patch
    x = x.reshape(1, V.channels, gh // m, m, P, gw // m, m, P).permute(0, 2, 5, 3, 6, 1, 4, 7)
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, V.temporal, -1, -1)
    return x.reshape(gh * gw, V.patch_dim)


# -- the text model's positions for a prompt with pictures (Qwen3-VL's mRoPE rule) -----------------------------------------
def rope_positions(length: int, media: "list[dict]", merge: int) -> "tuple[np.ndarray, int]":
    """([3, length] int64 (t, h, w) positions, delta) for a prompt of `length` tokens whose pictures sit at each record's
    `positions` with `grid` (t, h, w) in patches -- Qwen3-VL's `_get_mrope_input_positions`: text advances all three
    axes together from one past the largest position so far; a picture's tokens take (frame, row, column) of the merged
    grid offset by the text before it. `delta` is (largest position + 1) - length: every later token's position is its
    index plus `delta` (vLLM's mrope_position_delta), the same on all three axes."""
    pieces, st, top = [], 0, -1
    for item in sorted(media, key=lambda m: m["positions"][0]):
        first = int(item["positions"][0])
        t, h, w = (int(g) for g in item["grid"])
        gh, gw = h // merge, w // merge
        count = t * gh * gw
        if first < st or len(item["positions"]) != count or int(item["positions"][-1]) != first + count - 1:
            raise ValueError("a picture's placeholder run and its grid disagree")
        start = top + 1
        if first > st:
            pieces.append(np.broadcast_to(np.arange(first - st, dtype=np.int64), (3, first - st)) + start)
        grid = np.indices((t, gh, gw), dtype=np.int64).reshape(3, -1) + (first - st) + start
        pieces.append(grid)
        top = int(grid.max())
        st = first + count
    if st > length:
        raise ValueError("a picture runs past the prompt")
    if length > st:
        pieces.append(np.broadcast_to(np.arange(length - st, dtype=np.int64), (3, length - st)) + top + 1)
    out = np.concatenate(pieces, axis=1) if pieces else np.zeros((3, 0), dtype=np.int64)
    return np.ascontiguousarray(out), int(out.max()) + 1 - length if length else 0


# -- rank 0's half: bytes -> canvas -> placeholder tokens ----------------------------------------------------------------
class Door:
    """What base/serve.py calls for a multipart chat: `prepare` turns fetched bytes into a canvas record, `expand`
    turns the template's single <|image_pad|> per picture into the run of placeholder tokens the text model sees. No
    weights here; the canvases ride with the request to every rank, where `Vision` encodes them."""
    kinds = ("image",)
    limits = dict(LIMITS)

    def __init__(self, V: VisionFacts):
        self.V = V

    def prepare(self, kind: str, data: bytes) -> dict:
        if kind != "image":
            raise ValueError(f"{kind} is not served by this deployment (pictures only)")
        return self.prepare_image(data)

    def prepare_image(self, data: bytes) -> dict:
        V = self.V
        frames = decode_picture(data)[None]                     # [1, 3, h, w]: engine/modules/pictures, as production loads it
        h, w = frames.shape[-2:]
        H, W = smart_resize(h, w, V.factor, V.min_pixels, V.max_pixels)
        out = resize(frames, H, W)
        grid = (1, H // V.patch, W // V.patch)
        return {"kind": "image", "digest": hashlib.sha256(b"image\0" + data).hexdigest(), "canvas": out.numpy(),
                "grid": grid, "tokens": V.tokens(grid)}

    def expand(self, ids: "list[int]", items: "list[dict]") -> "tuple[list[int], list[dict]]":
        """The template emits <|vision_start|><|image_pad|><|vision_end|> per picture; the text model sees `tokens`
        copies of <|image_pad|> between the markers. Returns the ids and the media records with the positions their
        rows replace (in the order the parts appeared)."""
        V = self.V
        if any(t == V.video_token for t in ids):
            raise ValueError("videos are not served by this deployment")
        images = [it for it in items if it["kind"] == "image"]
        if len(images) != len(items):
            raise ValueError("videos are not served by this deployment")
        out, media, n = [], [], 0
        for i, t in enumerate(ids):
            if t != V.image_token:
                out.append(t)
                continue
            if n >= len(images):
                raise ValueError("the prompt has more image placeholders than images")
            if i == 0 or ids[i - 1] != V.vision_start or i + 1 >= len(ids) or ids[i + 1] != V.vision_end:
                raise ValueError("an image placeholder outside <|vision_start|> ... <|vision_end|>")
            item = images[n]
            n += 1
            positions = list(range(len(out), len(out) + item["tokens"]))
            out.extend([V.image_token] * item["tokens"])
            media.append(self._record(item, positions))
        if n != len(images):
            raise ValueError("the prompt's placeholders and the media parts disagree")
        return out, media

    @staticmethod
    def _record(item: dict, positions: "list[int]") -> dict:
        if len(positions) != item["tokens"]:
            raise ValueError("placeholder length and the canvas grid disagree")
        return {"kind": item["kind"], "digest": item["digest"], "positions": positions, "canvas": item["canvas"],
                "grid": tuple(int(x) for x in item["grid"])}


# -- every rank's half: canvas -> rows ----------------------------------------------------------------------------------
class Vision:
    def __init__(self, V: VisionFacts, views: dict, comm=None):
        from engine.base.params import bind
        self.V, self.comm = V, comm
        self.p = bind(specs(V), views)
        self.device = self.p[PREFIX + "patch_embed.proj.bias"].device
        self._grids = {}

    def pixel_values(self, canvas_u8: np.ndarray, grid) -> torch.Tensor:
        return pixel_values(self.V, canvas_u8, grid, self.device)

    # -- the tower's per-grid tables ------------------------------------------------------------------------------------
    def _merge_order(self, t: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        m = self.V.merge
        return t.reshape(gh // m, m, gw // m, m, *t.shape[2:]).permute(0, 2, 1, 3, *range(4, t.ndim + 2)).reshape(gh * gw, *t.shape[2:])

    def tables(self, gh: int, gw: int) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """(position embeddings [gh*gw, hidden] bf16, cos, sin [gh*gw, head_dim/2] fp32), all in merge-window order.
        The position table is resampled bilinearly with aligned corners (vLLM's pos_embed_interpolate: linspace over
        the 48-cell side, four taps, weights and sum in fp32 here); the rotary gives the first half of each head's
        rotary angles from the row and the second from the column, its cos/sin rounded to bf16 as vLLM's cache is."""
        key = (gh, gw)
        if key in self._grids:
            return self._grids[key]
        V, dev = self.V, self.device
        side = V.grid_side
        hi = torch.linspace(0, side - 1, gh, dtype=torch.float32, device=dev)
        wi = torch.linspace(0, side - 1, gw, dtype=torch.float32, device=dev)
        hf, wf = hi.long(), wi.long()
        hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
        dh, dw = (hi - hf)[:, None], (wi - wf)[None, :]
        table = self.p[PREFIX + "pos_embed.weight"].float()
        pos = (table[(hf[:, None] * side + wf[None, :])] * ((1 - dh) * (1 - dw))[..., None]
               + table[(hf[:, None] * side + wc[None, :])] * ((1 - dh) * dw)[..., None]
               + table[(hc[:, None] * side + wf[None, :])] * (dh * (1 - dw))[..., None]
               + table[(hc[:, None] * side + wc[None, :])] * (dh * dw)[..., None])          # [gh, gw, hidden]
        pos = self._merge_order(pos, gh, gw).to(BF)
        quarter = V.head_dim // 4
        inv = 1.0 / (ROPE_BASE ** (torch.arange(0, 2 * quarter, 2, dtype=torch.float32, device=dev) / (2 * quarter)))
        freqs = torch.outer(torch.arange(max(gh, gw), dtype=torch.float32, device=dev), inv)
        cos, sin = freqs.cos().to(BF).float(), freqs.sin().to(BF).float()
        hpos = self._merge_order(torch.arange(gh, device=dev)[:, None].expand(gh, gw), gh, gw)
        wpos = self._merge_order(torch.arange(gw, device=dev)[None, :].expand(gh, gw), gh, gw)
        out = (pos, torch.cat([cos[hpos], cos[wpos]], -1), torch.cat([sin[hpos], sin[wpos]], -1))
        self._grids = {key: out}                                    # one grid kept: pictures rarely repeat a size
        return out

    # -- the tower --------------------------------------------------------------------------------------------------
    @staticmethod
    def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Neox halves over the whole head, in fp32: out = x * cos + rotate_half(x) * sin."""
        xf = x.float()
        c, s = cos[:, None, :], sin[:, None, :]
        x1, x2 = xf.chunk(2, dim=-1)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)

    @staticmethod
    def _attention(q, k, v, scale: float) -> torch.Tensor:
        """Full attention over one picture [1, heads, n, d]; on CUDA only the fused backends may answer (D3: the math
        backend would materialise an n x n score matrix per head)."""
        if q.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                return Fn.scaled_dot_product_attention(q, k, v, scale=scale)
        return Fn.scaled_dot_product_attention(q, k, v, scale=scale)

    def _block(self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        V, p = self.V, self.p
        b = f"{PREFIX}blocks.{i}."
        L = x.shape[0]
        h = Fn.layer_norm(x, (V.hidden,), p[b + "norm1.weight"], p[b + "norm1.bias"], LN_EPS)
        q, k, v = Fn.linear(h, p[b + "attn.qkv.weight"], p[b + "attn.qkv.bias"]).view(L, 3, V.heads, V.head_dim).unbind(1)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        heads = lambda t: t.transpose(0, 1).unsqueeze(0)          # noqa: E731 -- [L, H, D] -> [1, H, L, D]
        a = self._attention(heads(q), heads(k), heads(v), V.head_dim ** -0.5)
        a = a.squeeze(0).transpose(0, 1).reshape(L, V.hidden)
        x = x + Fn.linear(a, p[b + "attn.proj.weight"], p[b + "attn.proj.bias"])
        h = Fn.layer_norm(x, (V.hidden,), p[b + "norm2.weight"], p[b + "norm2.bias"], LN_EPS)
        h = Fn.gelu(Fn.linear(h, p[b + "mlp.linear_fc1.weight"], p[b + "mlp.linear_fc1.bias"]), approximate="tanh")
        return x + Fn.linear(h, p[b + "mlp.linear_fc2.weight"], p[b + "mlp.linear_fc2.bias"])

    def _merger(self, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm per patch -> each 2 x 2 window's four patches side by side -> fc1, GELU, fc2 into the text width."""
        V, p, P = self.V, self.p, PREFIX
        x = Fn.layer_norm(x, (V.hidden,), p[P + "merger.norm.weight"], p[P + "merger.norm.bias"], LN_EPS)
        x = x.reshape(-1, V.hidden * V.merge * V.merge)
        x = Fn.gelu(Fn.linear(x, p[P + "merger.linear_fc1.weight"], p[P + "merger.linear_fc1.bias"]))
        return Fn.linear(x, p[P + "merger.linear_fc2.weight"], p[P + "merger.linear_fc2.bias"])

    def encode(self, canvas_u8: np.ndarray, grid) -> torch.Tensor:
        """[tokens, out_hidden] bf16 for one picture: the patches attend within the picture."""
        out = self.tower(self.pixel_values(canvas_u8, grid).to(BF), grid)
        self._agree(out)
        return out

    def tower(self, x: torch.Tensor, grid) -> torch.Tensor:
        """bf16 patches [gh*gw, patch_dim] (pixel_values' order) -> [tokens, out_hidden] bf16."""
        V, p = self.V, self.p
        gt, gh, gw = (int(g) for g in grid)
        if gt != 1 or x.shape != (gh * gw, V.patch_dim):
            raise ValueError(f"the tower takes one picture's {gh}x{gw} patches of {V.patch_dim}, got {tuple(x.shape)}")
        x = Fn.linear(x, p[PREFIX + "patch_embed.proj.weight"].view(V.hidden, -1), p[PREFIX + "patch_embed.proj.bias"])
        pos, cos, sin = self.tables(gh, gw)
        x = x + pos
        for i in range(V.depth):
            x = self._block(i, x, cos, sin)
        return self._merger(x)

    def _agree(self, out: torch.Tensor) -> None:
        """Every rank encoded the same canvas: the rows enter the text model's collectives, so a rank that differs
        (a divergent kernel, a corrupted copy) would poison every other rank's answer silently. One max-reduce."""
        comm = self.comm
        if comm is None or int(getattr(comm, "world_size", 1)) <= 1:
            return
        s = out.float().sum()
        mine = torch.stack([s, -s])
        top = comm.all_reduce_max(mine.clone())
        if not torch.equal(top, mine):
            raise RuntimeError(f"the ranks encoded a picture differently (this rank's row sum {s.item():.6g}, the fleet's "
                               f"extremes {top[0].item():.6g} / {-top[1].item():.6g}): the tower is not replicated identically")

    # -- boot ------------------------------------------------------------------------------------------------------------
    def qualify(self) -> dict:
        """The largest picture the processor makes, once, before the door opens (D3): the attention backend, the
        workspace and the tables are proven here, not on the first user. Returns the seconds paid."""
        V = self.V
        side = math.isqrt(V.max_pixels) // V.factor * V.factor           # the largest square the budget admits
        gh = gw = side // V.patch
        t0 = time.perf_counter()
        out = self.encode(self._pattern(side, side), (1, gh, gw))
        if out.shape != (gh * gw // (V.merge * V.merge), V.out_hidden) or not torch.isfinite(out.float()).all():
            raise RuntimeError("the vision tower's largest picture did not encode")
        del out
        self._grids = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {"vision/image": round(time.perf_counter() - t0, 3)}

    @staticmethod
    def _pattern(H: int, W: int) -> np.ndarray:
        """A deterministic non-flat canvas: gradients, so the qualification runs real values, not zeros."""
        y = np.arange(H, dtype=np.int32)[:, None] * 255 // max(H - 1, 1)
        x = np.arange(W, dtype=np.int32)[None, :] * 255 // max(W - 1, 1)
        return np.stack([x + 0 * y, y + 0 * x, (x + y) % 256]).astype(np.uint8)[None]      # [1, 3, H, W]


__all__ = ["FILE", "LIMITS", "PREFIX", "Door", "Vision", "VisionFacts", "load", "pixel_values", "resize", "rope_positions",
           "smart_resize", "specs", "write_file"]
