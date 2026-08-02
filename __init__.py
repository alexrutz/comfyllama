"""ComfyUI entry point for comfyllama (llama.cpp nodes).

ComfyUI imports this file when the repository sits in ``custom_nodes/``.
"""

from .comfyllama import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, __version__

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "__version__"]
