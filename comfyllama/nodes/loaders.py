"""Model loader nodes."""

from __future__ import annotations

from typing import Any, Dict

from .. import backend, paths
from .common import CATEGORY, chat_formats


def _load_options(n_ctx: int, n_gpu_layers: int, n_threads: int, n_batch: int,
                  flash_attn: bool, use_mmap: bool, use_mlock: bool,
                  main_gpu: int, chat_format: str) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "n_ctx": int(n_ctx),
        "n_gpu_layers": int(n_gpu_layers),
        "n_batch": int(n_batch),
        "flash_attn": bool(flash_attn),
        "use_mmap": bool(use_mmap),
        "use_mlock": bool(use_mlock),
        "main_gpu": int(main_gpu),
        "verbose": False,
    }
    if n_threads > 0:
        options["n_threads"] = int(n_threads)
    if chat_format and chat_format != "auto":
        options["chat_format"] = chat_format
    return options


_COMMON_TOOLTIPS = {
    "n_gpu_layers": "Layers offloaded to the GPU. -1 offloads everything, 0 is "
                    "CPU only. Lower it if the model does not fit in VRAM.",
    "n_ctx": "Context window in tokens. 0 uses the value stored in the GGUF.",
    "n_threads": "CPU threads. 0 lets llama.cpp decide.",
    "keep_loaded": "How many models this pack keeps in memory. 0 frees the "
                   "model right after the graph finishes.",
}


class LlamaCppLoader:
    """Loads a GGUF language model through llama-cpp-python."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (paths.list_models(paths.LLM_FOLDER), {
                    "tooltip": "GGUF files found in ComfyUI/models/llm.",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1024,
                                         "tooltip": _COMMON_TOOLTIPS["n_gpu_layers"]}),
                "n_ctx": ("INT", {"default": 4096, "min": 0, "max": 1 << 21, "step": 256,
                                  "tooltip": _COMMON_TOOLTIPS["n_ctx"]}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 256,
                                      "tooltip": _COMMON_TOOLTIPS["n_threads"]}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "flash_attn": ("BOOLEAN", {"default": False}),
                "chat_format": (chat_formats(), {
                    "default": "auto",
                    "tooltip": "Chat template. 'auto' uses the template embedded "
                               "in the GGUF, which is almost always correct.",
                }),
                "keep_loaded": ("INT", {"default": 1, "min": 0, "max": 8,
                                        "tooltip": _COMMON_TOOLTIPS["keep_loaded"]}),
            },
            "optional": {
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "main_gpu": ("INT", {"default": 0, "min": 0, "max": 64}),
                "model_path_override": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Absolute path to a .gguf file. Overrides the "
                               "dropdown when set.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a GGUF text model with llama.cpp."

    def load(self, model_name, n_gpu_layers, n_ctx, n_threads, n_batch, flash_attn,
             chat_format, keep_loaded, use_mmap=True, use_mlock=False, main_gpu=0,
             model_path_override=""):
        model_path = paths.resolve_model_path(paths.LLM_FOLDER, model_name, model_path_override)
        options = _load_options(n_ctx, n_gpu_layers, n_threads, n_batch, flash_attn,
                                use_mmap, use_mlock, main_gpu, chat_format)
        model = backend.load_model(
            model_path,
            options=options,
            keep_loaded=keep_loaded,
            display_name=model_name if not model_path_override else model_path,
        )
        return (model,)


class LlamaCppVisionLoader:
    """Loads a GGUF vision-language model plus its multimodal projector."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (paths.list_models(paths.LLM_FOLDER),),
                "mmproj_name": (paths.list_models(paths.MMPROJ_FOLDER), {
                    "tooltip": "The mmproj-*.gguf projector that belongs to the model.",
                }),
                "chat_handler": (backend.available_chat_handlers(), {
                    "default": "llava-1.6",
                    "tooltip": "Must match the model family: llava-1.6 for "
                               "llava/bakllava, moondream2, minicpm-v-2.6, etc.",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 1024,
                                         "tooltip": _COMMON_TOOLTIPS["n_gpu_layers"]}),
                "n_ctx": ("INT", {"default": 4096, "min": 0, "max": 1 << 21, "step": 256,
                                  "tooltip": _COMMON_TOOLTIPS["n_ctx"]}),
                "n_threads": ("INT", {"default": 0, "min": 0, "max": 256}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "keep_loaded": ("INT", {"default": 1, "min": 0, "max": 8,
                                        "tooltip": _COMMON_TOOLTIPS["keep_loaded"]}),
            },
            "optional": {
                "flash_attn": ("BOOLEAN", {"default": False}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "main_gpu": ("INT", {"default": 0, "min": 0, "max": 64}),
                "model_path_override": ("STRING", {"default": "", "multiline": False}),
                "mmproj_path_override": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("LLAMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load a multimodal GGUF model (LLaVA, MiniCPM-V, moondream, ...)."

    def load(self, model_name, mmproj_name, chat_handler, n_gpu_layers, n_ctx, n_threads,
             n_batch, keep_loaded, flash_attn=False, use_mmap=True, use_mlock=False,
             main_gpu=0, model_path_override="", mmproj_path_override=""):
        model_path = paths.resolve_model_path(paths.LLM_FOLDER, model_name, model_path_override)
        mmproj_path = paths.resolve_model_path(
            paths.MMPROJ_FOLDER, mmproj_name, mmproj_path_override)
        options = _load_options(n_ctx, n_gpu_layers, n_threads, n_batch, flash_attn,
                                use_mmap, use_mlock, main_gpu, "auto")
        model = backend.load_model(
            model_path,
            options=options,
            chat_handler_spec=(chat_handler, mmproj_path),
            keep_loaded=keep_loaded,
            vision=True,
            display_name=model_name if not model_path_override else model_path,
        )
        return (model,)


class LlamaCppUnload:
    """Frees a loaded model, optionally every model this pack holds."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLAMA_MODEL",),
                "unload_all": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also free every other model loaded by this pack.",
                }),
            },
            "optional": {
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Chain the generated text through here so the "
                               "unload happens after generation, not before.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "unload"
    CATEGORY = CATEGORY
    DESCRIPTION = "Free VRAM/RAM held by a llama.cpp model."

    def unload(self, model, unload_all, text=""):
        if unload_all:
            backend.MODEL_CACHE.clear()
        else:
            backend.MODEL_CACHE.forget(model)
        backend.free_comfy_memory()
        return (text,)
