"""Text completion and chat nodes."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .. import backend
from ..images import images_to_content
from .common import (CATEGORY, CATEGORY_ADVANCED, generation_inputs,
                     is_changed_for_seed)


def _messages(system: str, prompt: str, history, content=None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    has_system = bool(system and system.strip())
    if has_system:
        messages.append({"role": "system", "content": system})
    for message in history or []:
        # A system prompt on the node wins over one carried in the history.
        if has_system and message.get("role") == "system":
            continue
        messages.append(dict(message))
    if content is not None:
        messages.append({"role": "user", "content": content})
    elif prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


class LlamaCppComplete:
    """Raw text completion — no chat template is applied."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLAMA_MODEL",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "dynamicPrompts": True}),
                **generation_inputs(),
            },
            "optional": {
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "tokens")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Continue a prompt with a llama.cpp model (no chat template)."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, model, prompt, max_tokens, temperature, top_p, seed,
                 sampling=None, grammar=None):
        kwargs = backend.sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                                        top_p=top_p, seed=seed, sampling=sampling)
        text, _ = backend.complete(model, prompt, grammar=backend.build_grammar(grammar),
                                   **kwargs)
        return (text, backend.count_tokens(model, text))


class LlamaCppChat:
    """Chat completion using the model's chat template."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLAMA_MODEL",),
                "system": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "Leave empty to send no system message.",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "dynamicPrompts": True}),
                **generation_inputs(),
            },
            "optional": {
                "messages": ("LLAMA_MESSAGES", {
                    "tooltip": "Prior conversation turns, inserted before the prompt.",
                }),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "LLAMA_MESSAGES", "INT")
    RETURN_NAMES = ("text", "messages", "tokens")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Chat with a llama.cpp model and get the updated history back."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, model, system, prompt, max_tokens, temperature, top_p, seed,
                 messages=None, sampling=None, grammar=None):
        conversation = _messages(system, prompt, messages)
        kwargs = backend.sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                                        top_p=top_p, seed=seed, sampling=sampling)
        text, _ = backend.chat(
            model, conversation,
            grammar=backend.build_grammar(grammar) if _needs_grammar(grammar) else None,
            response_fmt=backend.response_format(grammar),
            **kwargs)
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, history, backend.count_tokens(model, text))


def _needs_grammar(spec) -> bool:
    """GBNF has to go in as a grammar; JSON modes use ``response_format``."""
    return bool(spec) and spec.get("type") == "gbnf"


class LlamaCppVisionChat:
    """Chat about one or more images with a multimodal model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLAMA_MODEL", {
                    "tooltip": "Must come from the vision loader — a plain text "
                               "model cannot see images.",
                }),
                "image": ("IMAGE",),
                "system": ("STRING", {
                    "default": "You are an assistant that describes images accurately.",
                    "multiline": True,
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True, "dynamicPrompts": True,
                }),
                **generation_inputs(),
            },
            "optional": {
                "image_max_size": ("INT", {
                    "default": 1024, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Longest edge the image is scaled to before it is "
                               "sent to the model. 0 disables resizing.",
                }),
                "image_quality": ("INT", {
                    "default": 90, "min": 30, "max": 100,
                    "tooltip": "JPEG quality; 100 sends lossless PNG.",
                }),
                "messages": ("LLAMA_MESSAGES",),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "LLAMA_MESSAGES")
    RETURN_NAMES = ("text", "messages")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Caption or interrogate images with a multimodal llama.cpp model."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, model, image, system, prompt, max_tokens, temperature, top_p, seed,
                 image_max_size=1024, image_quality=90, messages=None, sampling=None,
                 grammar=None):
        if not model.vision:
            raise ValueError(
                "This model was loaded without a multimodal projector. Use the "
                "'Load Vision LLM (llama.cpp)' node for image input."
            )
        content = images_to_content(image, max_size=image_max_size, quality=image_quality)
        content.append({"type": "text", "text": prompt})
        conversation = _messages(system, prompt, messages, content=content)
        kwargs = backend.sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                                        top_p=top_p, seed=seed, sampling=sampling)
        text, _ = backend.chat(
            model, conversation,
            grammar=backend.build_grammar(grammar) if _needs_grammar(grammar) else None,
            response_fmt=backend.response_format(grammar),
            **kwargs)
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, history)


class LlamaCppSampling:
    """Bundles the sampler settings that rarely need touching."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000,
                                  "tooltip": "0 disables top-k filtering."}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0,
                                             "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0,
                                               "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0,
                                                "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2,
                                          "tooltip": "0 off, 1 Mirostat, 2 Mirostat 2.0."}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0,
                                           "step": 0.1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0,
                                           "step": 0.01}),
                "stop_sequences": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "One stop sequence per line. Escapes such as \\n work.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_SAMPLING",)
    RETURN_NAMES = ("sampling",)
    FUNCTION = "build"
    CATEGORY = CATEGORY_ADVANCED
    DESCRIPTION = "Advanced sampler settings for the generation nodes."

    def build(self, top_k, min_p, typical_p, repeat_penalty, presence_penalty,
              frequency_penalty, mirostat_mode, mirostat_tau, mirostat_eta,
              stop_sequences):
        return ({
            "top_k": top_k,
            "min_p": min_p,
            "typical_p": typical_p,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "mirostat_mode": mirostat_mode,
            "mirostat_tau": mirostat_tau,
            "mirostat_eta": mirostat_eta,
            "stop": backend.parse_stop_sequences(stop_sequences),
        },)


class LlamaCppGrammar:
    """Constrains the output to JSON or to a custom GBNF grammar."""

    MODES = ["json_object", "json_schema", "gbnf"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (cls.MODES, {
                    "default": "json_object",
                    "tooltip": "json_object: any valid JSON. json_schema: JSON "
                               "matching the schema below. gbnf: raw grammar.",
                }),
                "definition": ("STRING", {
                    "default": json.dumps({
                        "type": "object",
                        "properties": {"caption": {"type": "string"}},
                        "required": ["caption"],
                    }, indent=2),
                    "multiline": True,
                    "tooltip": "JSON schema or GBNF text, depending on the mode. "
                               "Ignored for json_object.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_GRAMMAR",)
    RETURN_NAMES = ("grammar",)
    FUNCTION = "build"
    CATEGORY = CATEGORY_ADVANCED
    DESCRIPTION = "Force structured output from the generation nodes."

    def build(self, mode, definition):
        if mode == "json_object":
            return ({"type": "json_object"},)
        if mode == "json_schema":
            try:
                schema = json.loads(definition)
            except json.JSONDecodeError as exc:
                raise ValueError(f"The JSON schema is not valid JSON: {exc}") from exc
            return ({"type": "json_schema", "schema": schema},)
        if not definition.strip():
            raise ValueError("GBNF mode needs a grammar in the definition field.")
        return ({"type": "gbnf", "gbnf": definition},)
