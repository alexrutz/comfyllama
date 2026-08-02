"""Small helper nodes: history editing, templating, token counting, preview."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .. import backend
from .common import CATEGORY, CATEGORY_UTILS

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class LlamaCppMessage:
    """Appends a message to a conversation, or starts a new one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "role": (["user", "assistant", "system"], {"default": "user"}),
                "content": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "messages": ("LLAMA_MESSAGES", {
                    "tooltip": "Existing conversation to append to. Leave "
                               "unconnected to start a new one.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_MESSAGES",)
    RETURN_NAMES = ("messages",)
    FUNCTION = "append"
    CATEGORY = CATEGORY
    DESCRIPTION = "Build a chat history for few-shot prompting or multi-turn chats."

    def append(self, role, content, messages=None):
        history: List[Dict[str, Any]] = [dict(m) for m in (messages or [])]
        history.append({"role": role, "content": content})
        return (history,)


class LlamaCppMessagesToText:
    """Renders a conversation as plain text (for previewing or saving)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LLAMA_MESSAGES",),
                "include_system": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "render"
    CATEGORY = CATEGORY_UTILS

    def render(self, messages, include_system):
        lines = []
        for message in messages or []:
            role = message.get("role", "user")
            if role == "system" and not include_system:
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                # Multimodal content: keep the text parts, note the images.
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        parts.append("[image]")
                content = "\n".join(parts)
            lines.append(f"{role}: {content}")
        return ("\n\n".join(lines),)


class LlamaCppPromptTemplate:
    """Fills ``{placeholders}`` in a template with connected strings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {
                    "default": "Write a caption for: {a}",
                    "multiline": True,
                    "tooltip": "Use {a} {b} {c} {d} to insert the connected strings.",
                }),
            },
            "optional": {
                "a": ("STRING", {"forceInput": True}),
                "b": ("STRING", {"forceInput": True}),
                "c": ("STRING", {"forceInput": True}),
                "d": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "format"
    CATEGORY = CATEGORY_UTILS
    DESCRIPTION = "Compose a prompt from several text inputs."

    def format(self, template, a=None, b=None, c=None, d=None):
        values = {"a": a, "b": b, "c": c, "d": d}

        def substitute(match):
            key = match.group(1)
            if key in values and values[key] is not None:
                return str(values[key])
            return match.group(0)  # unknown placeholders are left untouched

        return (_PLACEHOLDER.sub(substitute, template),)


class LlamaCppTokenCount:
    """Counts tokens with the model's own tokenizer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLAMA_MODEL",),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "count"
    CATEGORY = CATEGORY_UTILS

    def count(self, model, text):
        return (backend.count_tokens(model, text),)


class LlamaCppPreviewText:
    """Shows generated text directly on the node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_UTILS
    DESCRIPTION = "Display text in the graph and pass it through."

    def preview(self, text, unique_id=None):
        return {"ui": {"text": [text]}, "result": (text,)}
