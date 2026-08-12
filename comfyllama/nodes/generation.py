"""Text completion and chat nodes."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .. import backend, reasoning
from ..images import images_to_content
from .common import (CATEGORY, CATEGORY_ADVANCED, generation_inputs,
                     image_inputs, is_changed_for_seed, thinking_input)


def user_content(prompt: str, image=None, *, max_size: int = 1024,
                 quality: int = 90):
    """The user turn's content.

    ``None`` when no image was connected, which leaves the turn as plain text;
    otherwise the OpenAI-style parts list, images first, prompt last.
    """
    if image is None:
        return None
    content = images_to_content(image, max_size=max_size, quality=quality)
    content.append({"type": "text", "text": prompt})
    return content


def require_vision_model(model, image) -> None:
    """In-process models can only see images if a projector was loaded."""
    if image is not None and not model.vision:
        raise ValueError(
            "An image is connected, but this model was loaded without a "
            "multimodal projector. Load it with 'Load Vision LLM (llama.cpp)' "
            "instead, or disconnect the image."
        )


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

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "thinking", "tokens")
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
        raw, reasoning_field, _ = backend.complete(
            model, prompt, grammar=backend.build_grammar(grammar), **kwargs)
        text, thinking = reasoning.split_thinking(raw)
        thinking = reasoning.combine(reasoning_field, thinking)
        return (text, thinking, backend.count_tokens(model, text))


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
                "thinking": thinking_input(),
                **generation_inputs(),
            },
            "optional": {
                "messages": ("LLAMA_MESSAGES", {
                    "tooltip": "Prior conversation turns, inserted before the prompt.",
                }),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
                **image_inputs(),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMA_MESSAGES", "INT")
    RETURN_NAMES = ("text", "thinking", "messages", "tokens")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Chat with a llama.cpp model and get the updated history back."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, model, system, prompt, thinking, max_tokens, temperature, top_p,
                 seed, messages=None, sampling=None, grammar=None, image=None,
                 image_max_size=1024, image_quality=90):
        require_vision_model(model, image)
        content = user_content(prompt, image, max_size=image_max_size,
                               quality=image_quality)
        conversation = _messages(system, prompt, messages, content=content)
        text, thought = _run_chat(model, conversation, thinking, max_tokens,
                                  temperature, top_p, seed, sampling, grammar)
        # The chain of thought is not fed back into the next turn.
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, thought, history, backend.count_tokens(model, text))


def _needs_grammar(spec) -> bool:
    """GBNF has to go in as a grammar; JSON modes use ``response_format``."""
    return bool(spec) and spec.get("type") == "gbnf"


def _run_chat(model, conversation, thinking, max_tokens, temperature, top_p, seed,
              sampling, grammar) -> Tuple[str, str]:
    """Run a chat completion and split the answer from the chain of thought."""
    kwargs = backend.sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                                    top_p=top_p, seed=seed, sampling=sampling)
    raw, reasoning_field, _ = backend.chat(
        model, reasoning.apply_control_tag(conversation, thinking),
        grammar=backend.build_grammar(grammar) if _needs_grammar(grammar) else None,
        response_fmt=backend.response_format(grammar),
        **kwargs)
    text, thought = reasoning.split_thinking(raw)
    return text, reasoning.combine(reasoning_field, thought)


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
                "thinking": thinking_input(),
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

    RETURN_TYPES = ("STRING", "STRING", "LLAMA_MESSAGES")
    RETURN_NAMES = ("text", "thinking", "messages")
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Caption or interrogate images with a multimodal llama.cpp model."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, model, image, system, prompt, thinking, max_tokens, temperature,
                 top_p, seed, image_max_size=1024, image_quality=90, messages=None,
                 sampling=None, grammar=None):
        require_vision_model(model, image)
        content = user_content(prompt, image, max_size=image_max_size,
                               quality=image_quality)
        conversation = _messages(system, prompt, messages, content=content)
        text, thought = _run_chat(model, conversation, thinking, max_tokens,
                                  temperature, top_p, seed, sampling, grammar)
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, thought, history)


def _enable(label: str) -> Any:
    """The switch in front of one sampler setting."""
    return ("BOOLEAN", {
        "default": False,
        "label_on": "send",
        "label_off": "leave default",
        "tooltip": f"Send {label} with the request. While off it is left out "
                   "entirely, so the model's own default (or the llama-server "
                   "command line) decides.",
    })


class LlamaCppSampling:
    """Advanced sampler settings, each switched on individually."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_top_k": _enable("top_k"),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000,
                                  "tooltip": "0 disables top-k filtering."}),
                "use_min_p": _enable("min_p"),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_typical_p": _enable("typical_p"),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_repeat_penalty": _enable("repeat_penalty"),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0,
                                             "step": 0.01}),
                "use_presence_penalty": _enable("presence_penalty"),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0,
                                               "step": 0.01}),
                "use_frequency_penalty": _enable("frequency_penalty"),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0,
                                                "step": 0.01}),
                "use_mirostat": _enable("the Mirostat settings"),
                "mirostat_mode": ("INT", {"default": 2, "min": 0, "max": 2,
                                          "tooltip": "0 off, 1 Mirostat, 2 Mirostat 2.0."}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0,
                                           "step": 0.1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0,
                                           "step": 0.01}),
                "use_stop_sequences": _enable("the stop sequences"),
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
    DESCRIPTION = ("Advanced sampler settings for the generation nodes. Each "
                   "setting is only sent while its switch is on.")

    def build(self, use_top_k, top_k, use_min_p, min_p, use_typical_p, typical_p,
              use_repeat_penalty, repeat_penalty, use_presence_penalty, presence_penalty,
              use_frequency_penalty, frequency_penalty, use_mirostat, mirostat_mode,
              mirostat_tau, mirostat_eta, use_stop_sequences, stop_sequences):
        sampling: Dict[str, Any] = {}
        if use_top_k:
            sampling["top_k"] = int(top_k)
        if use_min_p:
            sampling["min_p"] = float(min_p)
        if use_typical_p:
            sampling["typical_p"] = float(typical_p)
        if use_repeat_penalty:
            sampling["repeat_penalty"] = float(repeat_penalty)
        if use_presence_penalty:
            sampling["presence_penalty"] = float(presence_penalty)
        if use_frequency_penalty:
            sampling["frequency_penalty"] = float(frequency_penalty)
        if use_mirostat:
            # tau and eta are meaningless on their own, so they share a switch.
            sampling["mirostat_mode"] = int(mirostat_mode)
            sampling["mirostat_tau"] = float(mirostat_tau)
            sampling["mirostat_eta"] = float(mirostat_eta)
        if use_stop_sequences:
            stops = backend.parse_stop_sequences(stop_sequences)
            if stops:
                sampling["stop"] = stops
        return (sampling,)


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
