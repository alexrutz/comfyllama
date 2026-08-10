"""A general-purpose empty latent node, sized by aspect ratio and megapixels.

Nothing here touches llama.cpp — it is bundled because it is the companion
node most workflows in this pack end up needing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .common import CATEGORY_LATENT

# The two ratios asked for most often come first; the rest are ordered in
# landscape/portrait pairs.
ASPECT_RATIOS: List[Tuple[str, int, int]] = [
    ("1:1", 1, 1),
    ("2:3", 2, 3),
    ("3:2", 3, 2),
    ("3:4", 3, 4),
    ("4:3", 4, 3),
    ("4:5", 4, 5),
    ("5:4", 5, 4),
    ("9:16", 9, 16),
    ("16:9", 16, 9),
    ("9:21", 9, 21),
    ("21:9", 21, 9),
    ("1:2", 1, 2),
    ("2:1", 2, 1),
]

RATIO_BY_LABEL: Dict[str, Tuple[int, int]] = {
    label: (width, height) for label, width, height in ASPECT_RATIOS
}

RATIO_LABELS = [label for label, _, _ in ASPECT_RATIOS]

# One megapixel means 1024x1024, the convention the SDXL/Flux resolution
# tables are written in.
PIXELS_PER_MEGAPIXEL = 1024 * 1024

# Latents are one eighth of the image resolution in each direction.
LATENT_DOWNSCALE = 8

LATENT_FORMATS = {
    "SD1.5 / SDXL (4 channels)": 4,
    "SD3 / Flux (16 channels)": 16,
}


def resolve_dimensions(aspect_ratio: str, megapixels: float,
                       divisible_by: int = 8) -> Tuple[int, int]:
    """Pixel size closest to ``megapixels`` at the given ratio.

    Both edges are rounded to a multiple of ``divisible_by`` so the result is a
    valid latent size; that rounding means the area lands near, not exactly on,
    the requested megapixels.
    """
    try:
        width_ratio, height_ratio = RATIO_BY_LABEL[aspect_ratio]
    except KeyError:
        raise ValueError(
            f"Unknown aspect ratio '{aspect_ratio}'. Pick one of: "
            f"{', '.join(RATIO_LABELS)}."
        ) from None

    divisible_by = max(1, int(divisible_by))
    pixels = max(float(megapixels), 0.0) * PIXELS_PER_MEGAPIXEL
    scale = math.sqrt(pixels / (width_ratio * height_ratio))
    width = _round_to(width_ratio * scale, divisible_by)
    height = _round_to(height_ratio * scale, divisible_by)
    return width, height


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


class EmptyLatentByAspectRatio:
    """Empty latent sized by aspect ratio and megapixels instead of w/h."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (RATIO_LABELS, {
                    "default": "1:1",
                    "tooltip": "width:height. Both orientations are listed.",
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 64.0, "step": 0.01,
                    "tooltip": "Target area, where 1.0 MP = 1024x1024. SD1.5 "
                               "likes 0.26, SDXL and Flux 1.0.",
                }),
                "divisible_by": ([8, 16, 32, 64], {
                    "default": 8,
                    "tooltip": "Both edges are rounded to a multiple of this. "
                               "8 is the minimum a latent can represent; 64 "
                               "keeps SDXL happy.",
                }),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "latent_format": (list(LATENT_FORMATS), {
                    "default": "SD1.5 / SDXL (4 channels)",
                    "tooltip": "Channel count of the empty latent. SD3, Flux "
                               "and other 16-channel VAEs need the second entry.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CATEGORY_LATENT
    DESCRIPTION = "Empty latent from an aspect ratio and a megapixel budget."

    def generate(self, aspect_ratio, megapixels, divisible_by, batch_size,
                 latent_format="SD1.5 / SDXL (4 channels)"):
        import torch

        width, height = resolve_dimensions(aspect_ratio, megapixels, divisible_by)
        channels = LATENT_FORMATS.get(latent_format, 4)

        device = None
        try:
            import comfy.model_management as model_management

            device = model_management.intermediate_device()
        except Exception:
            device = None

        samples = torch.zeros(
            [batch_size, channels, height // LATENT_DOWNSCALE,
             width // LATENT_DOWNSCALE],
            device=device,
        )
        return ({"samples": samples}, width, height)
