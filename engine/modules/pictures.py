"""A picture as the served door loads it (modules): bytes -> RGB uint8 [3, h, w], the way vLLM's image loader hands it
to a model's processor (vllm/multimodal/media/image.py ImageMediaIO.load_bytes): a pixel-count ceiling, EXIF orientation
applied, transparency composited on white, every other mode converted to RGB. What a processor does after this -- the
canvas, the resize, the normalisation -- is the model's (engine/profiles/<model>/vision.py).
"""
from __future__ import annotations

import io

import numpy as np
import torch

MAX_IMAGE_PIXELS = 178_956_970       # vLLM refuses a larger decoded image (VLLM_MAX_IMAGE_PIXELS; PIL's decompression-bomb line)


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


def decode(data: bytes, max_pixels: int = MAX_IMAGE_PIXELS) -> torch.Tensor:
    """uint8 [3, h, w]: the picture in `data`, loaded as production loads it. A ValueError is the client's (a 400)."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"not a decodable image: {exc}") from exc
    w, h = image.size
    if w * h > max_pixels:
        raise ValueError(f"image dimensions {w}x{h} exceed the served maximum of {max_pixels} pixels")
    try:
        image = ImageOps.exif_transpose(image)          # production normalises EXIF orientation at load
    except Exception:                                   # noqa: BLE001 -- as production: a bad EXIF block is ignored
        pass
    try:
        image.load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"not a decodable image: {exc}") from exc
    image = to_rgb(image)
    return torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1).contiguous()


__all__ = ["MAX_IMAGE_PIXELS", "decode", "to_rgb"]
