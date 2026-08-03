"""Separating a reasoning model's thinking from its answer.

Reasoning models (Qwen3, DeepSeek-R1, GPT-OSS, Magistral, …) wrap their chain
of thought in ``<think>`` tags.  llama-server can already split it into a
``reasoning_content`` field; in-process models cannot, so the tags are parsed
out of the text here.  Either way the nodes expose thinking and answer on
separate outputs.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Tag names used by the reasoning models in circulation.
TAG_NAMES = ("think", "thinking", "reason", "reasoning", "thought")

_OPEN = re.compile(r"<(" + "|".join(TAG_NAMES) + r")\s*>", re.IGNORECASE)
_ANY_CLOSE = re.compile(r"</(" + "|".join(TAG_NAMES) + r")\s*>", re.IGNORECASE)

THINKING_MODES = ["auto", "on", "off"]

# Control tags understood by Qwen3-style models when the template cannot be
# parameterised (the in-process path).
CONTROL_TAGS = {"on": "/think", "off": "/no_think"}


def split_thinking(text: str) -> Tuple[str, str]:
    """Split raw model output into ``(answer, thinking)``.

    Handles the three shapes seen in practice: a normal ``<think>…</think>``
    block, a block the chat template already opened so only the closing tag is
    generated, and a block that was cut off before it closed.
    """
    if not text:
        return "", ""

    open_match = _OPEN.search(text)
    close_match = _ANY_CLOSE.search(text)

    # Closing tag with no opening one: the template emitted "<think>" as part
    # of the prompt, so the model starts mid-thought.
    if close_match and (open_match is None or close_match.start() < open_match.start()):
        answer, rest = split_thinking(text[close_match.end():])
        thinking = text[:close_match.start()]
        return answer, "\n".join(part for part in (thinking.strip(), rest) if part)

    answer_parts: List[str] = []
    thinking_parts: List[str] = []
    position = 0
    while position < len(text):
        open_match = _OPEN.search(text, position)
        if open_match is None:
            answer_parts.append(text[position:])
            break
        answer_parts.append(text[position:open_match.start()])
        close = re.compile(rf"</{open_match.group(1)}\s*>", re.IGNORECASE).search(
            text, open_match.end())
        if close is None:
            # Generation stopped inside the block; everything left is thinking.
            thinking_parts.append(text[open_match.end():])
            break
        thinking_parts.append(text[open_match.end():close.start()])
        position = close.end()

    return "".join(answer_parts).strip(), "".join(thinking_parts).strip()


def template_kwargs(mode: str) -> Optional[Dict[str, Any]]:
    """``chat_template_kwargs`` for llama-server, or ``None`` for ``auto``."""
    if mode == "on":
        return {"enable_thinking": True}
    if mode == "off":
        return {"enable_thinking": False}
    return None


def apply_control_tag(messages: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Append ``/think`` or ``/no_think`` to the last user message.

    This is the only lever available in-process: llama-cpp-python renders the
    GGUF's chat template without forwarding template arguments, so the switch
    has to travel inside the prompt.  Models that do not know the tag ignore it.
    """
    tag = CONTROL_TAGS.get(mode)
    if not tag:
        return messages

    result = [dict(message) for message in messages]
    for message in reversed(result):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"{content}\n{tag}" if content else tag
            return result
        if isinstance(content, list):
            parts = [dict(part) for part in content]
            for part in reversed(parts):
                if part.get("type") == "text":
                    part["text"] = f"{part.get('text', '')}\n{tag}".strip()
                    message["content"] = parts
                    return result
            parts.append({"type": "text", "text": tag})
            message["content"] = parts
            return result
    return result


def combine(reasoning_field: str, parsed_thinking: str) -> str:
    """Merge server-provided ``reasoning_content`` with tags found in the text."""
    return "\n".join(part for part in (reasoning_field.strip(), parsed_thinking) if part)
