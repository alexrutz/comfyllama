"""Nodes that talk to a running ``llama-server`` over HTTP.

These need no llama-cpp-python: the model lives in a separate process (or on
another machine), which keeps ComfyUI's VRAM free and lets several workflows
share one loaded model.
"""

from __future__ import annotations

import json
from typing import Tuple

from ..backend import sampler_kwargs
from ..images import images_to_content
from ..reasoning import combine, split_thinking
from ..server import (LlamaServer, LlamaServerError, apply_grammar, apply_thinking,
                      build_payload, stream_chat, stream_completion)
from .common import (CATEGORY_SERVER, generation_inputs, is_changed_for_seed,
                     thinking_input)
from .generation import _messages


def _chat(connection, conversation, thinking, max_tokens, temperature, top_p, seed,
          sampling, grammar) -> Tuple[str, str]:
    """Run a remote chat completion, split into ``(answer, thinking)``."""
    kwargs = sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                            top_p=top_p, seed=seed, sampling=sampling)
    payload = apply_grammar(build_payload(kwargs, native=False), grammar, native=False)
    payload = apply_thinking(payload, thinking)
    raw, reasoning_field, _ = stream_chat(connection, conversation, payload)
    text, thought = split_thinking(raw)
    return text, combine(reasoning_field, thought)


class LlamaServerConnect:
    """Points the other server nodes at a ``llama-server`` endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {
                    "default": "http://127.0.0.1:8080",
                    "multiline": False,
                    "tooltip": "Where llama-server listens. A trailing /v1 is "
                               "accepted and stripped.",
                }),
                "timeout": ("INT", {
                    "default": 300, "min": 1, "max": 3600,
                    "tooltip": "Seconds to wait for the server. Long generations "
                               "need a generous value.",
                }),
                "check_connection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Query /health when the node runs, so a wrong URL "
                               "fails here instead of halfway through the graph.",
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "auto",
                    "tooltip": "Model name to request. 'auto' uses the first "
                               "model the server reports.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Only needed when llama-server runs with --api-key.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_SERVER", "STRING")
    RETURN_NAMES = ("server", "model")
    FUNCTION = "connect"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = "Connect to a running llama-server instance."

    def connect(self, base_url, timeout, check_connection, model="auto", api_key=""):
        connection = LlamaServer(base_url, api_key=api_key, timeout=timeout, model=model)
        if check_connection:
            status = connection.health().get("status")
            if status and status != "ok":
                raise LlamaServerError(
                    f"llama-server at {connection.base_url} is not ready "
                    f"(status: {status}). It is probably still loading the model."
                )
        return (connection, connection.resolve_model())


class LlamaServerChat:
    """Chat completion against ``/v1/chat/completions``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLAMA_SERVER",),
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
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMA_MESSAGES")
    RETURN_NAMES = ("text", "thinking", "messages")
    FUNCTION = "generate"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = "Chat with a remote llama-server."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, server, system, prompt, thinking, max_tokens, temperature,
                 top_p, seed, messages=None, sampling=None, grammar=None):
        conversation = _messages(system, prompt, messages)
        text, thought = _chat(server, conversation, thinking, max_tokens, temperature,
                              top_p, seed, sampling, grammar)
        # The chain of thought is not fed back into the next turn.
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, thought, history)


class LlamaServerVisionChat:
    """Sends images to a ``llama-server`` started with ``--mmproj``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLAMA_SERVER",),
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
                    "tooltip": "Longest edge the image is scaled to before upload. "
                               "0 disables resizing.",
                }),
                "image_quality": ("INT", {
                    "default": 90, "min": 30, "max": 100,
                    "tooltip": "JPEG quality; 100 uploads lossless PNG.",
                }),
                "messages": ("LLAMA_MESSAGES",),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMA_MESSAGES")
    RETURN_NAMES = ("text", "thinking", "messages")
    FUNCTION = "generate"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = "Caption or interrogate images with a multimodal llama-server."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, server, image, system, prompt, thinking, max_tokens, temperature,
                 top_p, seed, image_max_size=1024, image_quality=90, messages=None,
                 sampling=None, grammar=None):
        content = images_to_content(image, max_size=image_max_size,
                                    quality=image_quality)
        content.append({"type": "text", "text": prompt})
        conversation = _messages(system, prompt, messages, content=content)
        text, thought = _chat(server, conversation, thinking, max_tokens, temperature,
                              top_p, seed, sampling, grammar)
        history = conversation + [{"role": "assistant", "content": text}]
        return (text, thought, history)


class LlamaServerComplete:
    """Raw completion against the native ``/completion`` endpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLAMA_SERVER",),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "dynamicPrompts": True}),
                **generation_inputs(),
            },
            "optional": {
                "cache_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Let the server reuse the KV cache when several "
                               "runs share a prompt prefix.",
                }),
                "sampling": ("LLAMA_SAMPLING",),
                "grammar": ("LLAMA_GRAMMAR",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    FUNCTION = "generate"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = "Continue a prompt on a remote llama-server (no chat template)."

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        return is_changed_for_seed(seed)

    def generate(self, server, prompt, max_tokens, temperature, top_p, seed,
                 cache_prompt=True, sampling=None, grammar=None):
        kwargs = sampler_kwargs(max_tokens=max_tokens, temperature=temperature,
                                top_p=top_p, seed=seed, sampling=sampling)
        payload = apply_grammar(build_payload(kwargs, native=True), grammar, native=True)
        payload["cache_prompt"] = bool(cache_prompt)
        raw, _ = stream_completion(server, prompt, payload)
        return split_thinking(raw)


class LlamaServerTokenCount:
    """Counts tokens with the server's tokenizer via ``/tokenize``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "server": ("LLAMA_SERVER",),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "count"
    CATEGORY = CATEGORY_SERVER

    def count(self, server, text):
        return (len(server.tokenize(text)),)


class LlamaServerInfo:
    """Reports what the server is currently running."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"server": ("LLAMA_SERVER",)}}

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("info", "model", "n_ctx")
    FUNCTION = "info"
    CATEGORY = CATEGORY_SERVER
    DESCRIPTION = "Model name, context size and settings of a llama-server."

    def info(self, server):
        props = server.props()
        settings = props.get("default_generation_settings") or {}
        n_ctx = int(settings.get("n_ctx") or props.get("n_ctx") or 0)
        summary = {
            "base_url": server.base_url,
            "model": server.resolve_model() or props.get("model_path", ""),
            "n_ctx": n_ctx,
            "has_chat_template": bool(props.get("chat_template")),
            "models": server.models(),
        }
        return (json.dumps(summary, indent=2), summary["model"], n_ctx)
