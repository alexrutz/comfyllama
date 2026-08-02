"""Input definitions shared by the node classes."""

from __future__ import annotations

from typing import Any, Dict, List

CATEGORY = "llama.cpp"
CATEGORY_ADVANCED = "llama.cpp/advanced"
CATEGORY_UTILS = "llama.cpp/utils"

MAX_SEED = 0xFFFFFFFFFFFFFFFF

# Chat templates that ship with llama-cpp-python.  The list is merged with
# whatever the installed version registers, so newer templates show up
# automatically.
FALLBACK_CHAT_FORMATS: List[str] = [
    "chatml", "llama-2", "llama-3", "mistral-instruct", "mistrallite", "gemma",
    "zephyr", "vicuna", "alpaca", "qwen", "phi-3", "openchat", "chatglm3",
    "openbuddy", "redpajama-incite", "snoozy", "intel", "oasst_llama",
    "baichuan-2", "saiga",
]


def chat_formats() -> List[str]:
    """``auto`` plus every chat template known to the installed binding.

    ``auto`` means "use the template stored in the GGUF metadata", which is the
    right answer for practically every modern model.
    """
    names = set(FALLBACK_CHAT_FORMATS)
    try:
        from llama_cpp.llama_chat_format import LlamaChatCompletionHandlerRegistry

        registry = LlamaChatCompletionHandlerRegistry()
        handlers = getattr(registry, "_chat_handlers", None)
        if isinstance(handlers, dict):
            names.update(str(name) for name in handlers)
    except Exception:
        pass
    return ["auto"] + sorted(names)


def seed_input(tooltip: str = "Sampling seed. -1 draws a new random seed on every run.") -> Any:
    return ("INT", {
        "default": -1,
        "min": -1,
        "max": MAX_SEED,
        "control_after_generate": True,
        "tooltip": tooltip,
    })


def generation_inputs() -> Dict[str, Any]:
    """The handful of sampling controls that belong on every generation node."""
    return {
        "max_tokens": ("INT", {
            "default": 512, "min": 0, "max": 1 << 20,
            "tooltip": "Maximum number of tokens to generate. 0 = until the "
                       "context window is full or a stop sequence is hit.",
        }),
        "temperature": ("FLOAT", {
            "default": 0.7, "min": 0.0, "max": 5.0, "step": 0.01,
            "tooltip": "0 makes sampling greedy/deterministic.",
        }),
        "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
        "seed": seed_input(),
    }


def is_changed_for_seed(seed: int) -> Any:
    """Force a re-run when the seed is random, keep caching otherwise."""
    if seed is not None and int(seed) < 0:
        return float("nan")
    return seed
