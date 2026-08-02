"""Conversion from ComfyUI ``IMAGE`` tensors to data URIs.

llama.cpp's multimodal chat handlers accept OpenAI-style ``image_url`` content
parts, so batched ComfyUI images are encoded as base64 PNG/JPEG.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List


def _to_pil_images(image) -> List[Any]:
    import numpy as np
    from PIL import Image

    array = image
    if hasattr(array, "detach"):  # torch tensor
        array = array.detach().cpu().numpy()
    array = np.asarray(array)

    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Expected an IMAGE tensor of shape [B,H,W,C], got {array.shape}.")

    images = []
    for frame in array:
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            images.append(Image.fromarray(frame, "RGBA").convert("RGB"))
        elif frame.shape[-1] == 1:
            images.append(Image.fromarray(frame[..., 0], "L").convert("RGB"))
        else:
            images.append(Image.fromarray(frame[..., :3], "RGB"))
    return images


def _resize(image, max_size: int):
    if max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    from PIL import Image

    scale = max_size / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, Image.LANCZOS)


def image_to_data_uri(image, *, max_size: int = 1024, quality: int = 90) -> str:
    """Encode a single PIL image as a ``data:`` URI."""
    image = _resize(image, max_size)
    buffer = io.BytesIO()
    if quality >= 100:
        image.save(buffer, format="PNG")
        mime = "image/png"
    else:
        image.save(buffer, format="JPEG", quality=int(quality))
        mime = "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def images_to_content(image, *, max_size: int = 1024, quality: int = 90) -> List[Dict[str, Any]]:
    """Build the ``image_url`` content parts for an IMAGE batch."""
    return [
        {"type": "image_url", "image_url": {"url": image_to_data_uri(
            frame, max_size=max_size, quality=quality)}}
        for frame in _to_pil_images(image)
    ]
