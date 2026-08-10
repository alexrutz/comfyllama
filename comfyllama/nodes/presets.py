"""A llama-server chat node holding several system prompts to switch between.

One node carries up to :data:`MAX_SLOTS` system prompts.  ``active`` picks the
one to run, or ``passthrough`` to hand the prompt straight to the output
without contacting the server at all.  Each slot has its own optional extra
prompt input, for system prompts that expect two separate instructions.

The extra inputs and the server connection are declared lazy, so the branches
belonging to inactive slots are never executed — in passthrough mode nothing
upstream of this node runs on the LLM side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..backend import decode_escapes
from .common import CATEGORY_SERVER, generation_inputs, is_changed_for_seed, thinking_input
from .generation import _messages
from .remote import _chat

MAX_SLOTS = 6

PASSTHROUGH = "passthrough"
PASSTHROUGH_ALIASES = {PASSTHROUGH, "none", "off", "bypass", "direct"}

DEFAULT_NAMES = [f"Preset {index}" for index in range(1, MAX_SLOTS + 1)]

DEFAULT_SYSTEM_PROMPTS = {
    1: "You are a prompt engineer for image generation models. Rewrite the "
       "user's idea as a single vivid prompt. Answer with the prompt only.",
}


def slot_names(values: Dict[str, Any]) -> List[str]:
    """The slot names in order, falling back to the defaults."""
    return [str(values.get(f"name_{index}") or DEFAULT_NAMES[index - 1])
            for index in range(1, MAX_SLOTS + 1)]


def resolve_slot(active: str, names: List[str], slot_count: int) -> Optional[int]:
    """Map the ``active`` selection to a slot number, or ``None`` for passthrough.

    Matching is by name so the dropdown can show what the slots were renamed
    to; a plain number or a trailing number ("Preset 3") also works, which is
    what keeps the node usable if the web extension has not loaded.
    """
    label = (active or "").strip()
    if not label or label.lower() in PASSTHROUGH_ALIASES:
        return None

    usable = max(1, min(int(slot_count), MAX_SLOTS))
    for index in range(1, usable + 1):
        if names[index - 1].strip().lower() == label.lower():
            return index

    digits = ""
    for character in reversed(label):
        if character.isdigit():
            digits = character + digits
        elif digits:
            break
    if digits and 1 <= int(digits) <= usable:
        return int(digits)

    available = ", ".join([PASSTHROUGH] + names[:usable])
    raise ValueError(
        f"'{active}' does not match any of this node's active system prompts. "
        f"Available: {available}."
    )


def join_prompt(prompt: str, extra: Optional[str], separator: str) -> str:
    """Append a slot's extra prompt to the incoming one."""
    parts = [part for part in ((prompt or "").strip(), (extra or "").strip()) if part]
    if len(parts) < 2:
        return parts[0] if parts else ""
    return decode_escapes(separator).join(parts)


def _slot_inputs() -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for index in range(1, MAX_SLOTS + 1):
        inputs[f"name_{index}"] = ("STRING", {
            "default": DEFAULT_NAMES[index - 1],
            "tooltip": "Shown in the 'active' dropdown. Keep the names distinct.",
        })
        inputs[f"system_{index}"] = ("STRING", {
            "default": DEFAULT_SYSTEM_PROMPTS.get(index, ""),
            "multiline": True,
        })
    return inputs


def _extra_inputs() -> Dict[str, Any]:
    return {
        f"extra_{index}": ("STRING", {
            "forceInput": True,
            "lazy": True,
            "tooltip": f"Second instruction for '{DEFAULT_NAMES[index - 1]}', "
                       "appended to the prompt. Only evaluated while that "
                       "system prompt is the active one.",
        })
        for index in range(1, MAX_SLOTS + 1)
    }


class LlamaServerPresetChat:
    """Chat against llama-server with a switchable set of system prompts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLAMA_SERVER", {
                    "lazy": True,
                    "tooltip": "Not requested at all in passthrough mode.",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "dynamicPrompts": True}),
                "active": ([PASSTHROUGH] + DEFAULT_NAMES, {
                    "default": PASSTHROUGH,
                    "tooltip": "Which system prompt to run. 'passthrough' "
                               "returns the prompt unchanged without calling "
                               "the model.",
                }),
                "slot_count": ("INT", {
                    "default": 3, "min": 1, "max": MAX_SLOTS,
                    "tooltip": "How many system prompts this node offers. The "
                               "rest are hidden and ignored.",
                }),
                "thinking": thinking_input(),
                **generation_inputs(),
                **_slot_inputs(),
            },
            "optional": {
                "extra_separator": ("STRING", {
                    "default": "\\n\\n",
                    "tooltip": "Put between the prompt and the extra prompt. "
                               "Escapes such as \\n are decoded.",
                }),
                **_extra_inputs(),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "thinking", "active")
    FUNCTION = "generate"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = ("llama-server chat with several system prompts to switch "
                   "between, plus a passthrough that skips the model.")

    @classmethod
    def VALIDATE_INPUTS(cls, active):
        # Taking `active` here tells ComfyUI to skip its own combo check, so
        # renamed presets are accepted as values.
        return True

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def check_lazy_status(self, active="", slot_count=MAX_SLOTS, **kwargs):
        """Only pull in the connection and the active slot's extra prompt."""
        try:
            index = resolve_slot(active, slot_names(kwargs), slot_count)
        except ValueError:
            # Let generate() raise the readable error instead of failing here.
            return []
        if index is None:
            return []

        needed = []
        if kwargs.get("server") is None:
            needed.append("server")
        extra = f"extra_{index}"
        if extra in kwargs and kwargs.get(extra) is None:
            needed.append(extra)
        return needed

    def generate(self, server, prompt, active, slot_count, thinking, max_tokens,
                 temperature, top_p, seed, extra_separator="\\n\\n", sampling=None,
                 grammar=None, **slots):
        names = slot_names(slots)
        index = resolve_slot(active, names, slot_count)

        if index is None:
            # Bypass: hand the prompt straight to the output.
            return (prompt, "", PASSTHROUGH)

        if server is None:
            raise ValueError(
                "No llama-server connection. Connect the 'Connect to "
                "llama-server' node, or set active to 'passthrough'."
            )

        system = str(slots.get(f"system_{index}") or "")
        full_prompt = join_prompt(prompt, slots.get(f"extra_{index}"), extra_separator)
        conversation = _messages(system, full_prompt, None)
        text, thought = _chat(server, conversation, thinking, max_tokens, temperature,
                              top_p, seed, sampling, grammar)
        return (text, thought, names[index - 1])
