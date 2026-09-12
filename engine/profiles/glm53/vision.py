"""GLM-5.3-Flash's eyes (profile): the vision tower and what feeds it, ported from what production serves.

Production (vLLM, PR #431: MM_LIMIT image 4 / video 1) runs the checkpoint's BF16 vision tower
(`model.visual.*`, 564 M parameters: a 24-block ViT with 2-D rotary and clamped SwiGLU, a 2x2 conv
downsample and a SwiGLU merger into the text width) behind a training-side preprocessing pipeline
(`vllm/transformers_utils/processors/glm5next.py`: an upward-aligned canvas under a token budget,
pad mode, bicubic antialiased resize, CLIP mean/std, temporal patch 2, Qwen-VL patch order). Both were
read in the ledger (45차 §23 A7) and are reproduced here without vLLM:

  * `load` / `specs`: the tower's constants from config.json's vision_config and processor_config.json,
    checked at load (D3); its 347 tensors declared under their checkpoint names -- replicated on every
    rank (TP-splitting 1.1 GiB across four boxes buys nothing) and written once as `vision.safetensors`
    next to the rank files (preshard.py --vision), so the boot carves them from the arena like any weight (D1).
  * `Door` (rank 0, no weights): bytes -> uint8 canvas (resized and padded exactly as served) -> the
    placeholder tokens the text model sees (one <|image|> per vision token; a video is frame pairs, each
    bracketed and time-stamped). The canvas, not the encoding, travels to the ranks with the request.
  * `Vision` (every rank): canvas -> normalised patches -> tower -> [tokens, hidden] rows that replace the
    embedding rows at the placeholder positions (net.Step.patches). The ranks compute it identically and
    check that they did (D3) before the rows enter the collectives.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from engine.base.params import Spec
from engine.profiles.glm53.lanes import swiglu_clamped
from engine.profiles.glm53.net import rmsnorm

BF = torch.bfloat16
FILE = "vision.safetensors"          # next to the rank files: the tower, whole, under its checkpoint names
PREFIX = "model.visual."
LIMITS = {"image": 4, "video": 1}    # per prompt, as production serves (PR #431's MM_LIMIT)
VIDEO_TOKEN_CAP = 30000              # production caps the checkpoint's 240,000-token video budget (_MAX_VIDEO_TOKENS)
VIDEO_FRAMES_LOADED = 32             # production's loader samples this many frames uniformly before the processor (VideoMediaIO num_frames)
VIDEO_MAX_FRAMES = 2048              # Glm5NextVideoProcessor.max_frame_count_dynamic (a class default: not in processor_config.json)
MAX_IMAGE_PIXELS = 178_956_970       # vLLM refuses a larger decoded image (VLLM_MAX_IMAGE_PIXELS; PIL's decompression-bomb line)
ROPE_BASE, ROPE_MAX = 10000.0, 8192  # get_rope(head_size, max_position=8192, partial_rotary_factor=0.5): a 2-D rope over (h, w)
QK_NORM_EPS = 1e-5                   # the attention's q/k RMSNorm: hard-coded in the served tower, distinct from the block norms
LAYERNORM_EPS = 1e-5                 # merger.post_projection_norm: nn.LayerNorm's default
SLICE_PATCHES = 32768                # the tower runs whole frame groups up to this many patches at once (the workspace bound)


@dataclass(frozen=True)
class VisionFacts:
    depth: int
    hidden: int
    heads: int
    inter: int
    out_hidden: int                 # the text model's width: what the merger emits
    proj_inter: int                 # the merger's SwiGLU width
    patch: int
    temporal: int
    merge: int
    channels: int
    rms_eps: float
    swiglu_limit: float
    image_token: int
    image_start: int
    image_end: int
    video_token: int
    video_start: int
    video_end: int
    mean: tuple
    std: tuple
    image_min_tokens: int
    image_max_tokens: int
    video_min_tokens: int
    video_max_tokens: int
    video_fps: float                # the GLM frame sampler's target rate (processor_config fps / fps_interval)

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def factor(self) -> int:
        return self.patch * self.merge          # the canvas aligns to one merged token (patch_expand_factor 1)

    @property
    def patch_dim(self) -> int:
        return self.channels * self.temporal * self.patch * self.patch

    def pixels(self, kind: str) -> "tuple[int, int]":
        """(min, max) canvas pixels for the kind's token budget: one vision token covers temporal * factor^2 pixels."""
        lo, hi = (self.image_min_tokens, self.image_max_tokens) if kind == "image" else (self.video_min_tokens, self.video_max_tokens)
        per = self.temporal * self.factor * self.factor
        return lo * per, hi * per

    def tokens(self, grid) -> int:
        t, h, w = grid
        return t * h * w // (self.merge * self.merge)


def load(ckpt: "str | Path") -> VisionFacts:
    c = json.loads((Path(ckpt) / "config.json").read_text())
    v = c["vision_config"]
    p = json.loads((Path(ckpt) / "processor_config.json").read_text())
    ip, vp = p["image_processor"], p["video_processor"]
    V = VisionFacts(
        depth=v["depth"], hidden=v["hidden_size"], heads=v["num_heads"], inter=v["intermediate_size"],
        out_hidden=v["out_hidden_size"], proj_inter=v["projection_intermediate_size"],
        patch=v["patch_size"], temporal=v["temporal_patch_size"], merge=v["spatial_merge_size"], channels=v["in_channels"],
        rms_eps=float(v["rms_norm_eps"]), swiglu_limit=float(v["swiglu_limit"]),
        image_token=int(c["image_token_id"]), image_start=int(c["image_start_token_id"]), image_end=int(c["image_end_token_id"]),
        video_token=int(c["video_token_id"]), video_start=int(c["video_start_token_id"]), video_end=int(c["video_end_token_id"]),
        mean=tuple(float(x) for x in ip["image_mean"]), std=tuple(float(x) for x in ip["image_std"]),
        image_min_tokens=int(ip["min_image_tokens"]), image_max_tokens=int(ip["max_image_tokens"]),
        video_min_tokens=int(vp["min_image_tokens"]), video_max_tokens=min(int(vp["max_image_tokens"]), VIDEO_TOKEN_CAP),
        video_fps=float(vp["fps"]),
    )
    # -- what the code assumes, checked against the checkpoint (D3) ----------
    assert c["architectures"] == ["Glm5NextForConditionalGeneration"] and v["model_type"] == "glm5_next_vision", c.get("architectures")
    assert v["attention_bias"] and v["hidden_act"] == "silu", "the tower is biased-qkv, SiLU-gated"
    assert p["processor_class"] == "Glm5NextProcessor" and ip["image_processor_type"] == "Glm5NextImageProcessor" \
        and vp["video_processor_type"] == "Glm5NextVideoProcessor"
    assert ip["patch_expand_factor"] == 1 and vp["patch_expand_factor"] == 1, "the canvas factor is patch * merge"
    assert ip["patch_size"] == vp["patch_size"] == V.patch and ip["merge_size"] == vp["merge_size"] == V.merge
    assert ip["temporal_patch_size"] == vp["temporal_patch_size"] == V.temporal and ip["do_rescale"] and vp["do_rescale"]
    assert tuple(float(x) for x in vp["image_mean"]) == V.mean and tuple(float(x) for x in vp["image_std"]) == V.std
    assert len(V.mean) == len(V.std) == V.channels == 3
    assert V.hidden % V.heads == 0 and V.head_dim % 4 == 0, "the 2-D rope splits each head into four quarters"
    assert V.merge == 2 and V.temporal == 2 and V.depth > 0 and V.image_min_tokens <= V.image_max_tokens
    return V


# -- the tower's tensors, as the checkpoint names them (whole copies on every rank) ---------------------------------
def specs(V: VisionFacts) -> "list[Spec]":
    def whole(name, shape):
        return Spec(name, tuple(shape), BF, (name,), lambda s, r, W, name=name: s[name].contiguous())
    P = PREFIX
    out = [whole(P + "patch_embed.proj.weight", (V.hidden, V.channels, V.temporal, V.patch, V.patch)),
           whole(P + "patch_embed.proj.bias", (V.hidden,))]
    for i in range(V.depth):
        b = f"{P}blocks.{i}."
        out += [whole(b + "norm1.weight", (V.hidden,)), whole(b + "norm2.weight", (V.hidden,)),
                whole(b + "attn.qkv.weight", (3 * V.hidden, V.hidden)), whole(b + "attn.qkv.bias", (3 * V.hidden,)),
                whole(b + "attn.proj.weight", (V.hidden, V.hidden)), whole(b + "attn.proj.bias", (V.hidden,)),
                whole(b + "attn.q_norm.weight", (V.head_dim,)), whole(b + "attn.k_norm.weight", (V.head_dim,)),
                whole(b + "mlp.gate_proj.weight", (V.inter, V.hidden)), whole(b + "mlp.gate_proj.bias", (V.inter,)),
                whole(b + "mlp.up_proj.weight", (V.inter, V.hidden)), whole(b + "mlp.up_proj.bias", (V.inter,)),
                whole(b + "mlp.down_proj.weight", (V.hidden, V.inter)), whole(b + "mlp.down_proj.bias", (V.hidden,))]
    out += [whole(P + "post_layernorm.weight", (V.hidden,)),
            whole(P + "downsample.weight", (V.out_hidden, V.hidden, V.merge, V.merge)), whole(P + "downsample.bias", (V.out_hidden,)),
            whole(P + "merger.proj.weight", (V.out_hidden, V.out_hidden)),
            whole(P + "merger.post_projection_norm.weight", (V.out_hidden,)), whole(P + "merger.post_projection_norm.bias", (V.out_hidden,)),
            whole(P + "merger.gate_proj.weight", (V.proj_inter, V.out_hidden)), whole(P + "merger.up_proj.weight", (V.proj_inter, V.out_hidden)),
            whole(P + "merger.down_proj.weight", (V.out_hidden, V.proj_inter))]
    return out


def write_file(ckpt: "str | Path", out_dir: "str | Path", log=print) -> int:
    """`vision.safetensors` in the rank files' directory: the tower's 347 tensors, whole, aligned for the arena loader."""
    from engine.base.checkpoint import Checkpoint
    from engine.base.preshard import write_ranks
    V = load(ckpt)
    S = specs(V)
    path = Path(out_dir) / FILE
    sizes = write_ranks([("vision", [s.name for s in S], lambda r: S)], [path], lambda keys: Checkpoint(str(ckpt)).load(keys), 1,
                        metadata={"model": "glm53", "part": "vision", "layout": "engine.profiles.glm53.vision"}, log=log)
    return sizes[0]


# -- geometry: the served processor's canvas rules, verbatim ---------------------------------------------------------------
def _ceil_to(value: int, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _fit_within_budget(t: int, h: int, w: int, h_factor: int, w_factor: int, max_pixels: int) -> "tuple[int, int]":
    """The largest proportional content whose upward-aligned canvas fits: a binary search on the content height."""
    if max_pixels < t * h_factor * w_factor:
        raise ValueError(f"max_pixels={max_pixels} is too small for one aligned patch")
    low, high = 1, h
    best_h, best_w = h_factor, w_factor
    while low <= high:
        content_h = (low + high) // 2
        content_w = max(1, math.floor(w * content_h / h))
        aligned_h, aligned_w = _ceil_to(content_h, h_factor), _ceil_to(content_w, w_factor)
        if t * aligned_h * aligned_w <= max_pixels:
            best_h, best_w = aligned_h, aligned_w
            low = content_h + 1
        else:
            high = content_h - 1
    return best_h, best_w


def smart_resize(t: int, h: int, w: int, t_factor: int, h_factor: int, w_factor: int, min_pixels: int, max_pixels: int) -> "tuple[int, int]":
    """(canvas height, width): rounded UP to the factors, refit by search when over budget, scaled up when under."""
    if min(t, h, w, t_factor, h_factor, w_factor) <= 0 or min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError("image dimensions, factors and the pixel budget must be positive and ordered")
    t_bar = max(t_factor, round(t / t_factor) * t_factor)
    h_bar, w_bar = _ceil_to(h, h_factor), _ceil_to(w, w_factor)
    if t_bar * h_bar * w_bar > max_pixels:
        h_bar, w_bar = _fit_within_budget(t_bar, h, w, h_factor, w_factor, max_pixels)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = _ceil_to(max(1, math.ceil(h * beta)), h_factor)
        w_bar = _ceil_to(max(1, math.ceil(w * beta)), w_factor)
        if t_bar * h_bar * w_bar > max_pixels:
            h_bar, w_bar = _fit_within_budget(t_bar, h, w, h_factor, w_factor, max_pixels)
    return h_bar, w_bar


def content_size(h: int, w: int, H: int, W: int, allow_upscale: bool) -> "tuple[int, int]":
    """Aspect-preserving content inside the canvas: shrunk when larger, enlarged only when allowed; the rest is padding."""
    scale = min(H / h, W / w)
    if not allow_upscale:
        scale = min(1.0, scale)
    return max(1, min(H, math.floor(h * scale))), max(1, min(W, math.floor(w * scale)))


def canvas(frames: torch.Tensor, H: int, W: int, allow_upscale: bool) -> torch.Tensor:
    """uint8 [T, C, h, w] -> uint8 [T, C, H, W]: the served pad mode -- bicubic antialiased resize onto the content
    size, zero padding on the right and bottom (torchvision, as the served processor calls it)."""
    from torchvision.transforms.v2 import functional as tvF
    h, w = frames.shape[-2:]
    ch, cw = content_size(h, w, H, W, allow_upscale)
    if (ch, cw) != (h, w):
        frames = tvF.resize(frames, [ch, cw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    return tvF.pad(frames, [0, 0, W - cw, H - ch], fill=0)


def sample_frame_indices(total_frames: int, fps: float, duration: float, *, target_fps: float, max_frame_count: int,
                         temporal: int) -> "list[int]":
    """The GLM frame sampler (training parity, as served): a greedy walk at 1 / (temporal * target_fps) seconds,
    re-spread uniformly when it over- or under-collects, made even for the temporal patch."""
    max_frame_idx = total_frames - 1
    if not duration:
        duration = (round(max_frame_idx / fps) + 1) if fps else 0
    extract_t = min(int(duration * target_fps), int(max_frame_count))
    duration_per_frame = 1 / fps
    timestamps = [i * duration_per_frame for i in range(total_frames)]
    max_second = int(duration)
    if total_frames < extract_t:
        indices = [math.floor(i * total_frames / extract_t) for i in range(extract_t)]
    else:
        indices, current, inv = [], 0.0, 1 / (temporal * target_fps)
        for i in range(total_frames):
            if timestamps[i] >= current:
                current += inv
                indices.append(i)
                if current >= max_second:
                    break
    if len(indices) < extract_t:
        start, end = (0, max(total_frames - 1, 0)) if not indices else (indices[0], indices[-1])
        indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(indices) > extract_t:
        indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
    seen, uniq = set(), []
    for i in indices:
        if i not in seen:
            seen.add(i)
            uniq.append(int(i))
    if len(uniq) & 1:
        uniq.append(uniq[-1])
    return uniq


def to_rgb(image):
    """PIL image -> RGB the way production loads it: transparency composited on white, everything else converted."""
    from PIL import Image
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in getattr(image, "info", {}):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        out = Image.new("RGB", image.size, (255, 255, 255))
        out.paste(image, mask=image.split()[3])
        return out
    return image.convert("RGB")


# -- video frames, decoded in a child process ----------------------------------------------------------------------------
# OpenCV runs in its own interpreter: its stream-buffered capture crashed on release inside the engine process (45차 §23
# A7: SIGSEGV at cap.release() with torch loaded), and a decoder fed untrusted bytes must never take rank 0's door down --
# a child that dies is a 400, not a fleet restart. The child does what production's loader does: at most `n_max` frames,
# uniformly spaced, BGR -> RGB, and reports the source's frame count and rate.
_DECODER = r"""
import json, struct, sys
import numpy as np, cv2
path, n_max = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    sys.exit(3)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS))
if total <= 0:
    sys.exit(4)
n = max(1, min(n_max, total))
wanted = list(range(n)) if n == total else np.linspace(0, total - 1, n, dtype=int).tolist()
targets, frames, picked = set(wanted), [], []
for i in range(wanted[-1] + 1):
    if not cap.grab():
        break
    if i in targets:
        ok, frame = cap.retrieve()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); picked.append(i)
if not frames:
    sys.exit(5)
arr = np.stack(frames)
head = json.dumps({"total": total, "fps": fps, "picked": picked, "shape": list(arr.shape)}).encode()
out = sys.stdout.buffer
out.write(struct.pack("<Q", len(head))); out.write(head); out.write(arr.tobytes()); out.flush()
cap.release()
"""
_DECODER_EXIT = {3: "not a decodable video", 4: "the video has no frames", 5: "no frame of the video could be decoded"}


def decode_video(data: bytes, n_max: int, timeout_s: float = 120.0):
    """(frames uint8 [T, h, w, 3], picked frame indices, total frames, fps) -- production's loader, out of process."""
    with tempfile.NamedTemporaryFile(suffix=".video") as f:
        f.write(data)
        f.flush()
        try:
            r = subprocess.run([sys.executable, "-c", _DECODER, f.name, str(int(n_max))], capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"video decoding exceeded {timeout_s:.0f} s") from exc
    payload = r.stdout
    if len(payload) >= 8:
        n = struct.unpack("<Q", payload[:8])[0]
        if len(payload) >= 8 + n:
            head = json.loads(payload[8:8 + n])
            shape = tuple(int(x) for x in head["shape"])
            body = payload[8 + n:]
            if len(body) == math.prod(shape):                     # complete: the child's exit after writing is its own business
                frames = np.frombuffer(body, dtype=np.uint8).reshape(shape)
                return frames, [int(i) for i in head["picked"]], int(head["total"]), float(head["fps"])
    if r.returncode in _DECODER_EXIT:
        raise ValueError(_DECODER_EXIT[r.returncode])
    tail = r.stderr.decode(errors="replace").strip().splitlines()[-1:] if r.stderr else []
    raise ValueError(f"video decoder failed (exit {r.returncode}){': ' + tail[0] if tail else ''}")


# -- rank 0's half: bytes -> canvas -> placeholder tokens ----------------------------------------------------------------
class Door:
    """What base/serve.py calls for a multipart chat: `prepare` turns fetched bytes into a canvas record, `expand`
    turns the template's single placeholders into the runs of vision tokens the text model sees. No weights here;
    the canvases ride with the request to every rank, where `Vision` encodes them."""
    kinds = ("image", "video")
    limits = dict(LIMITS)

    def __init__(self, V: VisionFacts, tokenizer):
        self.V, self.tok = V, tokenizer          # tokenizers.Tokenizer: the "N.N seconds" stamps between a video's frame pairs

    def prepare(self, kind: str, data: bytes) -> dict:
        if kind == "image":
            return self.prepare_image(data)
        if kind == "video":
            return self.prepare_video(data)
        raise ValueError(f"{kind} is not served")

    def prepare_image(self, data: bytes) -> dict:
        from PIL import Image, ImageOps, UnidentifiedImageError
        V = self.V
        try:
            image = Image.open(io.BytesIO(data))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(f"not a decodable image: {exc}") from exc
        w, h = image.size
        if w * h > MAX_IMAGE_PIXELS:
            raise ValueError(f"image dimensions {w}x{h} exceed the served maximum of {MAX_IMAGE_PIXELS} pixels")
        try:
            image = ImageOps.exif_transpose(image)          # production normalises EXIF orientation at load
        except Exception:                                   # noqa: BLE001 -- as production: a bad EXIF block is ignored
            pass
        image.load()
        image = to_rgb(image)
        frames = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()[None]     # [1, 3, h, w]
        h, w = frames.shape[-2:]
        lo, hi = V.pixels("image")
        H, W = smart_resize(V.temporal, h, w, V.temporal, V.factor, V.factor, lo, hi)
        out = canvas(frames, H, W, allow_upscale=V.temporal * h * w < lo)
        grid = (1, H // V.patch, W // V.patch)
        return {"kind": "image", "digest": hashlib.sha256(b"image\0" + data).hexdigest(), "canvas": out.numpy(), "grid": grid,
                "tokens": V.tokens(grid)}

    def prepare_video(self, data: bytes) -> dict:
        V = self.V
        stack, picked, total, fps = decode_video(data, VIDEO_FRAMES_LOADED)      # [T, h, w, 3] uint8, out of process
        duration = total / fps if fps > 0 else 0.0
        if len(picked) == total:                                    # every frame loaded: the GLM sampler chooses the pairs
            idx = sample_frame_indices(total, fps, duration, target_fps=V.video_fps, max_frame_count=VIDEO_MAX_FRAMES, temporal=V.temporal)
            if not idx:
                raise ValueError("the video is too short to sample a frame pair from")
            stack, picked = stack[idx], idx
        frames_t = torch.from_numpy(np.ascontiguousarray(stack)).permute(0, 3, 1, 2).contiguous()       # [T, 3, h, w]
        T, h, w = frames_t.shape[0], frames_t.shape[-2], frames_t.shape[-1]
        lo, hi = V.pixels("video")
        H, W = smart_resize(T, h, w, V.temporal, V.factor, V.factor, lo, hi)
        out = canvas(frames_t, H, W, allow_upscale=T * h * w < lo)
        if pad := -T % V.temporal:                                  # the processor repeats the last frame; the placeholder its stamp
            out = torch.cat([out, out[-1:].expand(pad, -1, -1, -1)])
            picked = picked + [picked[-1]] * pad
        grid = (out.shape[0] // V.temporal, H // V.patch, W // V.patch)
        groups = [picked[i:i + V.temporal] for i in range(0, len(picked), V.temporal)]
        seconds = [g[0] / (fps or 1.0) for g in groups]
        return {"kind": "video", "digest": hashlib.sha256(b"video\0" + data).hexdigest(), "canvas": out.numpy(), "grid": grid,
                "tokens": V.tokens(grid), "seconds": seconds}

    def expand(self, ids: "list[int]", items: "list[dict]") -> "tuple[list[int], list[dict]]":
        """The template emits one <|image|> per image and <|begin_of_video|><|video|><|end_of_video|> per video; the
        text model sees, per image, `tokens` copies of <|image|>, and per video the served placeholder: each frame pair
        bracketed by the image markers and followed by its "N.N seconds" stamp. Returns the ids and the media records
        with the positions their rows replace (in the order the parts appeared)."""
        V = self.V
        images = [it for it in items if it["kind"] == "image"]
        videos = [it for it in items if it["kind"] == "video"]
        out, media, ii, vi, i = [], [], 0, 0, 0
        while i < len(ids):
            t = ids[i]
            if t == V.video_start and i + 2 < len(ids) and ids[i + 1] == V.video_token and ids[i + 2] == V.video_end:
                if vi >= len(videos):
                    raise ValueError("the prompt has more video placeholders than videos")
                item = videos[vi]; vi += 1
                per = V.tokens((1, item["grid"][1], item["grid"][2]))
                positions = []
                out.append(V.video_start)
                for sec in item["seconds"]:
                    out.append(V.image_start)
                    positions.extend(range(len(out), len(out) + per))
                    out.extend([V.image_token] * per)
                    out.append(V.image_end)
                    out.extend(self.tok.encode(f"{sec:.1f} seconds", add_special_tokens=False).ids)
                out.append(V.video_end)
                media.append(self._record(item, positions))
                i += 3
                continue
            if t == V.image_token:
                if ii >= len(images):
                    raise ValueError("the prompt has more image placeholders than images")
                item = images[ii]; ii += 1
                positions = list(range(len(out), len(out) + item["tokens"]))
                out.extend([V.image_token] * item["tokens"])
                media.append(self._record(item, positions))
                i += 1
                continue
            out.append(t)
            i += 1
        if ii != len(images) or vi != len(videos):
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
        scale = 1.0 / (1 / 255.0)                                             # the served processor fuses rescale into mean/std
        self.mean = (torch.tensor(V.mean, dtype=torch.float32, device=self.device) * scale).view(-1, 1, 1)
        self.std = (torch.tensor(V.std, dtype=torch.float32, device=self.device) * scale).view(-1, 1, 1)
        self._rope = {}

    # -- preprocessing: what the processor does after the canvas -----------------------------------------------------
    def patches(self, canvas_u8: np.ndarray, grid) -> torch.Tensor:
        """uint8 [T, C, H, W] -> bf16 [T*gh*gw, C*temporal*patch*patch] in the served order (frame pair, merge window,
        channel, temporal, patch row, patch column); a still image is its single frame twice."""
        V = self.V
        x = torch.from_numpy(np.ascontiguousarray(canvas_u8)).to(self.device)
        if x.ndim != 4 or x.shape[1] != V.channels:
            raise ValueError("a canvas is uint8 [frames, channels, height, width]")
        if pad := -x.shape[0] % V.temporal:
            x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)])
        gt, gh, gw = grid
        if x.shape[0] != gt * V.temporal or x.shape[2] != gh * V.patch or x.shape[3] != gw * V.patch or gh % V.merge or gw % V.merge:
            raise ValueError("the canvas and its grid disagree")
        x = (x.float() - self.mean) / self.std
        x = x.view(gt, V.temporal, V.channels, gh // V.merge, V.merge, V.patch, gw // V.merge, V.merge, V.patch)
        x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(gt * gh * gw, V.patch_dim).to(BF)

    # -- the tower --------------------------------------------------------------------------------------------------
    def rotary(self, gh: int, gw: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """cos, sin [gh*gw, head_dim/2] in merge-window order: the first half of the angles from the row, the second
        from the column (the served 2-D rope: neox halves, bf16 tables)."""
        key = (gh, gw)
        if key not in self._rope:
            V = self.V
            n = max(gh, gw)
            if n > ROPE_MAX:
                raise ValueError(f"a {gh}x{gw} patch grid exceeds the rope's {ROPE_MAX} positions")
            quarter = V.head_dim // 4
            inv = 1.0 / (ROPE_BASE ** (torch.arange(0, 2 * quarter, 2, dtype=torch.float32, device=self.device) / (2 * quarter)))
            freqs = torch.outer(torch.arange(n, dtype=torch.float32, device=self.device), inv)
            cos, sin = freqs.cos().to(BF).float(), freqs.sin().to(BF).float()
            hpos = torch.arange(gh, device=self.device)[:, None].expand(gh, gw)
            wpos = torch.arange(gw, device=self.device)[None, :].expand(gh, gw)
            m = V.merge
            order = lambda t: t.reshape(gh // m, m, gw // m, m).permute(0, 2, 1, 3).reshape(-1)   # noqa: E731
            hpos, wpos = order(hpos), order(wpos)
            self._rope = {key: (torch.cat([cos[hpos], cos[wpos]], -1), torch.cat([sin[hpos], sin[wpos]], -1))}
        return self._rope[key]

    @staticmethod
    def _rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, S: int) -> torch.Tensor:
        L, H, D = x.shape
        xf = x.view(S, L // S, H, D).float()
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        x1, x2 = xf.chunk(2, dim=-1)
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype).view(L, H, D)

    @staticmethod
    def _attention(q, k, v, scale: float) -> torch.Tensor:
        """Full attention inside each frame group [S, heads, n, d]; on CUDA only the fused backends may answer (D3:
        the math backend would materialise an n x n score matrix per head -- 32 GiB at the largest image)."""
        if q.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
                return Fn.scaled_dot_product_attention(q, k, v, scale=scale)
        return Fn.scaled_dot_product_attention(q, k, v, scale=scale)

    def _block(self, i: int, x: torch.Tensor, S: int, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        V, p = self.V, self.p
        b = f"{PREFIX}blocks.{i}."
        L = x.shape[0]
        h = rmsnorm(x, p[b + "norm1.weight"], V.rms_eps)
        qkv = Fn.linear(h, p[b + "attn.qkv.weight"], p[b + "attn.qkv.bias"]).view(L, 3, V.heads, V.head_dim)
        q, k, v = qkv.unbind(1)
        q = rmsnorm(q, p[b + "attn.q_norm.weight"], QK_NORM_EPS)
        k = rmsnorm(k, p[b + "attn.k_norm.weight"], QK_NORM_EPS)
        q, k = self._rope_apply(q, cos, sin, S), self._rope_apply(k, cos, sin, S)
        n = L // S
        shape = lambda t: t.view(S, n, V.heads, V.head_dim).transpose(1, 2)        # noqa: E731
        a = self._attention(shape(q), shape(k), shape(v), V.head_dim ** -0.5)
        a = a.transpose(1, 2).reshape(L, V.hidden)
        x = x + Fn.linear(a, p[b + "attn.proj.weight"], p[b + "attn.proj.bias"])
        h = rmsnorm(x, p[b + "norm2.weight"], V.rms_eps)
        g = Fn.linear(h, p[b + "mlp.gate_proj.weight"], p[b + "mlp.gate_proj.bias"])
        u = Fn.linear(h, p[b + "mlp.up_proj.weight"], p[b + "mlp.up_proj.bias"])
        return x + Fn.linear(swiglu_clamped(g, u, V.swiglu_limit), p[b + "mlp.down_proj.weight"], p[b + "mlp.down_proj.bias"])

    def _tail(self, x: torch.Tensor) -> torch.Tensor:
        """post norm -> 2x2 conv downsample (a linear over each merge window) -> the merger into the text width."""
        V, p, P = self.V, self.p, PREFIX
        x = rmsnorm(x, p[P + "post_layernorm.weight"], V.rms_eps)
        x = x.view(-1, V.merge, V.merge, V.hidden).permute(0, 3, 1, 2).reshape(-1, V.hidden * V.merge * V.merge)
        x = Fn.linear(x, p[P + "downsample.weight"].view(V.out_hidden, -1), p[P + "downsample.bias"])
        x = Fn.linear(x, p[P + "merger.proj.weight"])
        x = Fn.layer_norm(x, (V.out_hidden,), p[P + "merger.post_projection_norm.weight"], p[P + "merger.post_projection_norm.bias"], LAYERNORM_EPS)
        x = Fn.gelu(x)
        g, u = Fn.linear(x, p[P + "merger.gate_proj.weight"]), Fn.linear(x, p[P + "merger.up_proj.weight"])
        return Fn.linear(swiglu_clamped(g, u, V.swiglu_limit), p[P + "merger.down_proj.weight"])

    def encode(self, canvas_u8: np.ndarray, grid) -> torch.Tensor:
        """[tokens, out_hidden] bf16 for one image or video: every frame group attends within itself; groups run in
        slices of at most SLICE_PATCHES so the largest video costs the workspace of the largest image."""
        V, p = self.V, self.p
        gt, gh, gw = (int(g) for g in grid)
        x = self.patches(canvas_u8, (gt, gh, gw))
        x = Fn.linear(x, p[PREFIX + "patch_embed.proj.weight"].view(V.hidden, -1), p[PREFIX + "patch_embed.proj.bias"])
        cos, sin = self.rotary(gh, gw)
        seg = gh * gw
        per_slice = max(1, SLICE_PATCHES // seg)
        outs = []
        for s0 in range(0, gt, per_slice):
            S = min(per_slice, gt - s0)
            xs = x[s0 * seg:(s0 + S) * seg]
            for i in range(V.depth):
                xs = self._block(i, xs, S, cos, sin)
            outs.append(self._tail(xs))
        out = torch.cat(outs) if len(outs) > 1 else outs[0]
        self._agree(out)
        return out

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
            raise RuntimeError(f"the ranks encoded an image differently (this rank's row sum {s.item():.6g}, the fleet's extremes "
                               f"{top[0].item():.6g} / {-top[1].item():.6g}): the vision tower is not replicated identically")

    # -- boot ------------------------------------------------------------------------------------------------------------
    def qualify(self) -> dict:
        """The largest image and the largest video production accepts, once, before the door opens (D3): the attention
        backend, the workspace and the slice walk are proven here, not on the first user. Returns the seconds paid."""
        V = self.V
        paid = {}
        n = V.image_max_tokens * V.merge * V.merge                                    # patches of the largest image
        gh = int(math.sqrt(n)) // V.merge * V.merge
        gw = n // gh // V.merge * V.merge
        t0 = time.perf_counter()
        out = self.encode(self._pattern((1, V.channels, gh * V.patch, gw * V.patch)), (1, gh, gw))
        if out.shape != (gh * gw // (V.merge * V.merge), V.out_hidden) or not torch.isfinite(out.float()).all():
            raise RuntimeError("the vision tower's largest image did not encode")
        del out
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        paid["vision/image"] = round(time.perf_counter() - t0, 3)
        groups = VIDEO_FRAMES_LOADED // V.temporal                                    # the largest video: 16 pairs under the token cap
        per = V.video_max_tokens * V.merge * V.merge // groups
        gh = int(math.sqrt(per)) // V.merge * V.merge
        gw = per // gh // V.merge * V.merge
        t0 = time.perf_counter()
        out = self.encode(self._pattern((groups * V.temporal, V.channels, gh * V.patch, gw * V.patch)), (groups, gh, gw))
        if out.shape != (groups * gh * gw // (V.merge * V.merge), V.out_hidden) or not torch.isfinite(out.float()).all():
            raise RuntimeError("the vision tower's largest video did not encode")
        del out
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        paid["vision/video"] = round(time.perf_counter() - t0, 3)
        return paid

    @staticmethod
    def _pattern(shape) -> np.ndarray:
        """A deterministic non-flat canvas: gradients, so the qualification runs real values, not zeros."""
        T, C, H, W = shape
        y = np.arange(H, dtype=np.int32)[:, None] * 255 // max(H - 1, 1)
        x = np.arange(W, dtype=np.int32)[None, :] * 255 // max(W - 1, 1)
        base = np.stack([x + 0 * y, y + 0 * x, (x + y) % 256]).astype(np.uint8)          # [3, H, W]
        return np.broadcast_to(base[None], (T, C, H, W)).copy()
