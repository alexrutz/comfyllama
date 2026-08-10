"""Node registry for comfyllama."""

from __future__ import annotations

from .generation import (LlamaCppChat, LlamaCppComplete, LlamaCppGrammar,
                         LlamaCppSampling, LlamaCppVisionChat)
from .latent import EmptyLatentByAspectRatio
from .loaders import LlamaCppLoader, LlamaCppUnload, LlamaCppVisionLoader
from .presets import LlamaServerPresetChat
from .remote import (LlamaServerChat, LlamaServerComplete, LlamaServerConnect,
                     LlamaServerInfo, LlamaServerTokenCount, LlamaServerVisionChat)
from .utils import (LlamaCppMessage, LlamaCppMessagesToText, LlamaCppPreviewText,
                    LlamaCppPromptTemplate, LlamaCppTokenCount)

NODE_CLASS_MAPPINGS = {
    "LlamaCppLoader": LlamaCppLoader,
    "LlamaCppVisionLoader": LlamaCppVisionLoader,
    "LlamaCppUnload": LlamaCppUnload,
    "LlamaCppComplete": LlamaCppComplete,
    "LlamaCppChat": LlamaCppChat,
    "LlamaCppVisionChat": LlamaCppVisionChat,
    "LlamaCppSampling": LlamaCppSampling,
    "LlamaCppGrammar": LlamaCppGrammar,
    "LlamaCppMessage": LlamaCppMessage,
    "LlamaCppMessagesToText": LlamaCppMessagesToText,
    "LlamaCppPromptTemplate": LlamaCppPromptTemplate,
    "LlamaCppTokenCount": LlamaCppTokenCount,
    "LlamaCppPreviewText": LlamaCppPreviewText,
    "LlamaServerConnect": LlamaServerConnect,
    "LlamaServerChat": LlamaServerChat,
    "LlamaServerVisionChat": LlamaServerVisionChat,
    "LlamaServerComplete": LlamaServerComplete,
    "LlamaServerTokenCount": LlamaServerTokenCount,
    "LlamaServerInfo": LlamaServerInfo,
    "LlamaServerPresetChat": LlamaServerPresetChat,
    "EmptyLatentByAspectRatio": EmptyLatentByAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppLoader": "Load LLM (llama.cpp)",
    "LlamaCppVisionLoader": "Load Vision LLM (llama.cpp)",
    "LlamaCppUnload": "Unload LLM (llama.cpp)",
    "LlamaCppComplete": "Text Completion (llama.cpp)",
    "LlamaCppChat": "Chat (llama.cpp)",
    "LlamaCppVisionChat": "Vision Chat (llama.cpp)",
    "LlamaCppSampling": "Sampler Settings (llama.cpp)",
    "LlamaCppGrammar": "Grammar / JSON Output (llama.cpp)",
    "LlamaCppMessage": "Chat Message (llama.cpp)",
    "LlamaCppMessagesToText": "Messages to Text (llama.cpp)",
    "LlamaCppPromptTemplate": "Prompt Template (llama.cpp)",
    "LlamaCppTokenCount": "Token Count (llama.cpp)",
    "LlamaCppPreviewText": "Preview Text (llama.cpp)",
    "LlamaServerConnect": "Connect to llama-server",
    "LlamaServerChat": "Chat (llama-server)",
    "LlamaServerVisionChat": "Vision Chat (llama-server)",
    "LlamaServerComplete": "Text Completion (llama-server)",
    "LlamaServerTokenCount": "Token Count (llama-server)",
    "LlamaServerInfo": "Server Info (llama-server)",
    "LlamaServerPresetChat": "Chat with Prompt Presets (llama-server)",
    "EmptyLatentByAspectRatio": "Empty Latent (Aspect Ratio + Megapixels)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
