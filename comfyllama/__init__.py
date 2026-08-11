"""comfyllama — llama.cpp nodes for ComfyUI."""

from __future__ import annotations

__version__ = "1.0.0"

from .api import register_routes
from .paths import register_model_folders

register_model_folders()
register_routes()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
