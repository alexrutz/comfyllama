"""Thin wrapper around ``llama-cpp-python``.

Everything that touches the llama.cpp bindings lives here so the node modules
stay declarative.  ``llama_cpp`` is imported lazily: ComfyUI must be able to
load the pack (and show a helpful error) even when the binding is missing or
was built for a different backend.
"""

from __future__ import annotations

import gc
import inspect
import json
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

INSTALL_HINT = (
    "llama-cpp-python is not installed in the Python environment running "
    "ComfyUI.\n"
    "Install it with the backend you want, for example:\n"
    "  CPU   : pip install llama-cpp-python\n"
    "  CUDA  : pip install llama-cpp-python "
    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
    "  Metal : CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python\n"
    "Use the same interpreter as ComfyUI (e.g. python_embeded\\python.exe -m pip ...)."
)

_llama_cpp = None
_import_lock = threading.Lock()


def llama_cpp():
    """Import and return the ``llama_cpp`` module, with a readable error."""
    global _llama_cpp
    if _llama_cpp is not None:
        return _llama_cpp
    with _import_lock:
        if _llama_cpp is None:
            try:
                import llama_cpp  # noqa: PLC0415 - deliberately lazy
            except ImportError as exc:
                raise RuntimeError(f"{INSTALL_HINT}\n\nOriginal error: {exc}") from exc
            _llama_cpp = llama_cpp
    return _llama_cpp


def filter_kwargs(func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments the installed llama-cpp-python does not accept.

    The binding's signatures move between releases; passing an unknown sampler
    parameter should degrade gracefully instead of raising ``TypeError``.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


# --------------------------------------------------------------------------
# ComfyUI integration helpers (optional at import time)
# --------------------------------------------------------------------------


def _model_management():
    try:
        import comfy.model_management as model_management
    except ImportError:
        return None
    return model_management


def check_interrupt() -> None:
    """Raise if the user pressed cancel in the ComfyUI UI."""
    mm = _model_management()
    if mm is not None:
        mm.throw_exception_if_processing_interrupted()


def free_comfy_memory() -> None:
    """Unload diffusion models so llama.cpp can claim the VRAM."""
    mm = _model_management()
    if mm is None:
        return
    try:
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass


def progress_bar(total: int):
    if total <= 0:
        return None
    try:
        from comfy.utils import ProgressBar
    except ImportError:
        return None
    try:
        return ProgressBar(total)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Model handle + cache
# --------------------------------------------------------------------------


class LlamaModel:
    """Handle passed between nodes as the ``LLAMA_MODEL`` type."""

    def __init__(self, llm: Any, key: Tuple, name: str, *, chat_format: Optional[str] = None,
                 vision: bool = False, n_ctx: int = 0) -> None:
        self.llm = llm
        self.key = key
        self.name = name
        self.chat_format = chat_format
        self.vision = vision
        self.n_ctx = n_ctx

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = "vision" if self.vision else "text"
        return f"<LlamaModel {self.name!r} ({kind}, n_ctx={self.n_ctx})>"

    def free(self) -> None:
        llm, self.llm = self.llm, None
        if llm is None:
            return
        close = getattr(llm, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        del llm
        gc.collect()

    def require(self) -> Any:
        if self.llm is None:
            raise RuntimeError(
                f"The model '{self.name}' was unloaded. Re-run the loader node "
                "or move the unload node after every node that uses the model."
            )
        return self.llm


class ModelCache:
    """Keeps recently loaded models alive so re-running a graph is instant."""

    def __init__(self, max_entries: int = 1) -> None:
        self._entries: "OrderedDict[Tuple, LlamaModel]" = OrderedDict()
        self._lock = threading.RLock()
        self.max_entries = max_entries

    def get_or_load(self, key: Tuple, factory: Callable[[], LlamaModel],
                    keep_loaded: int = 1) -> LlamaModel:
        with self._lock:
            self.max_entries = max(0, keep_loaded)
            cached = self._entries.get(key)
            if cached is not None and cached.llm is not None:
                self._entries.move_to_end(key)
                return cached
            self._entries.pop(key, None)
            self._trim(self.max_entries - 1 if self.max_entries else 0)

        # Loading happens outside the lock: it is slow and must not block a
        # parallel `unload` from the UI.
        model = factory()

        with self._lock:
            if self.max_entries:
                self._entries[key] = model
                self._entries.move_to_end(key)
                self._trim(self.max_entries)
        return model

    def _trim(self, keep: int) -> None:
        while len(self._entries) > max(0, keep):
            _, victim = self._entries.popitem(last=False)
            victim.free()

    def forget(self, model: LlamaModel) -> None:
        with self._lock:
            cached = self._entries.get(model.key)
            if cached is model:
                self._entries.pop(model.key, None)
        model.free()

    def clear(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            entry.free()
        gc.collect()
        mm = _model_management()
        if mm is not None:
            try:
                mm.soft_empty_cache()
            except Exception:
                pass


MODEL_CACHE = ModelCache()


def load_model(model_path: str, *, options: Dict[str, Any], chat_handler_spec=None,
               keep_loaded: int = 1, vision: bool = False,
               display_name: Optional[str] = None) -> LlamaModel:
    """Load (or reuse) a GGUF model.

    ``chat_handler_spec`` is a ``(handler_name, clip_path)`` tuple for vision
    models; it is part of the cache key but the handler itself is only built
    when the model actually has to be loaded.
    """
    key = (model_path, tuple(sorted(options.items())), chat_handler_spec)

    def factory() -> LlamaModel:
        if options.get("n_gpu_layers", 0) != 0:
            free_comfy_memory()
        module = llama_cpp()
        kwargs = dict(options)
        if chat_handler_spec is not None:
            kwargs["chat_handler"] = build_chat_handler(*chat_handler_spec)
            # A handler takes precedence over the template baked into the GGUF.
            kwargs.pop("chat_format", None)
        kwargs = filter_kwargs(module.Llama.__init__, kwargs)
        llm = module.Llama(model_path=model_path, **kwargs)
        try:
            n_ctx = int(llm.n_ctx())
        except Exception:
            n_ctx = int(options.get("n_ctx", 0))
        return LlamaModel(
            llm,
            key,
            display_name or model_path,
            chat_format=options.get("chat_format"),
            vision=vision,
            n_ctx=n_ctx,
        )

    return MODEL_CACHE.get_or_load(key, factory, keep_loaded=keep_loaded)


# --------------------------------------------------------------------------
# Vision chat handlers
# --------------------------------------------------------------------------

# Maps the friendly name shown in the UI to the handler class in
# ``llama_cpp.llama_chat_format``.  Availability depends on the installed
# version, so the list is filtered at runtime.
CHAT_HANDLERS: Dict[str, str] = {
    "llava-1.5": "Llava15ChatHandler",
    "llava-1.6": "Llava16ChatHandler",
    "moondream2": "MoondreamChatHandler",
    "nanollava": "NanoLlavaChatHandler",
    "llama-3-vision-alpha": "Llama3VisionAlphaChatHandler",
    "minicpm-v-2.6": "MiniCPMv26ChatHandler",
    "qwen2.5-vl": "Qwen25VLChatHandler",
    "obsidian": "ObsidianChatHandler",
}


def available_chat_handlers() -> List[str]:
    """Handler names usable with the installed binding (never empty)."""
    try:
        from llama_cpp import llama_chat_format
    except Exception:
        return list(CHAT_HANDLERS)
    names = [name for name, attr in CHAT_HANDLERS.items()
             if hasattr(llama_chat_format, attr)]
    return names or list(CHAT_HANDLERS)


def build_chat_handler(handler_name: str, clip_model_path: str):
    from llama_cpp import llama_chat_format

    attr = CHAT_HANDLERS.get(handler_name)
    if attr is None:
        raise ValueError(f"Unknown vision chat handler '{handler_name}'.")
    handler_cls = getattr(llama_chat_format, attr, None)
    if handler_cls is None:
        raise RuntimeError(
            f"The installed llama-cpp-python has no '{attr}'. Upgrade it or pick "
            "a different chat handler."
        )
    return handler_cls(clip_model_path=clip_model_path, verbose=False)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

# Sampler parameters the sampling node can switch on.  Nothing here is sent
# unless it was explicitly enabled, so a parameter left off keeps whatever
# default the model, llama-cpp-python or the llama-server command line sets.
SAMPLING_KEYS = (
    "top_k", "min_p", "typical_p", "repeat_penalty", "presence_penalty",
    "frequency_penalty", "mirostat_mode", "mirostat_tau", "mirostat_eta",
)


def decode_escapes(text: str) -> str:
    """Turn a ``\\n`` typed into a widget into a real newline.

    The latin-1/backslashreplace detour keeps non-ASCII characters intact,
    which a plain utf-8 round trip through ``unicode_escape`` would mangle.
    """
    if "\\" not in (text or ""):
        return text or ""
    try:
        return text.encode("latin-1", "backslashreplace").decode("unicode_escape")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def parse_stop_sequences(text: str) -> List[str]:
    """One stop sequence per line; ``\\n`` and friends are unescaped."""
    return [decode_escapes(line) for line in (text or "").splitlines() if line.strip()]




def build_grammar(spec: Optional[Dict[str, Any]]):
    """Turn a ``LLAMA_GRAMMAR`` payload into a ``LlamaGrammar`` instance."""
    if not spec:
        return None
    module = llama_cpp()
    grammar_cls = module.LlamaGrammar
    kind = spec.get("type")
    if kind == "gbnf":
        return grammar_cls.from_string(spec["gbnf"], verbose=False)
    if kind == "json_schema":
        return grammar_cls.from_json_schema(json.dumps(spec["schema"]), verbose=False)
    if kind == "json_object":
        return grammar_cls.from_string(JSON_GBNF, verbose=False)
    raise ValueError(f"Unsupported grammar type '{kind}'.")


def response_format(spec: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """``response_format`` payload for chat completions, if applicable."""
    if not spec:
        return None
    kind = spec.get("type")
    if kind == "json_schema":
        return {"type": "json_object", "schema": spec["schema"]}
    if kind == "json_object":
        return {"type": "json_object"}
    return None


# Minimal JSON grammar used for "any JSON object" constrained output.
JSON_GBNF = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt/] | "u" [0-9a-fA-F]{4}) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]{1,3})? ws
ws     ::= | " " | "\n" [ \t]{0,20}
"""


def sampler_kwargs(*, max_tokens: int, temperature: float, top_p: float, seed: int,
                   sampling: Optional[Dict[str, Any]],
                   extra_stop: Optional[List[str]] = None) -> Dict[str, Any]:
    """Assemble the request parameters for one generation.

    The four controls on the generation node are always sent; everything else
    only appears when the sampling node switched it on.
    """
    kwargs: Dict[str, Any] = {
        "max_tokens": None if max_tokens <= 0 else int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        # A negative seed means "random", which llama.cpp expresses as -1.
        "seed": -1 if seed < 0 else int(seed),
    }

    sampling = sampling or {}
    for key in SAMPLING_KEYS:
        if sampling.get(key) is not None:
            kwargs[key] = sampling[key]

    stop = list(sampling.get("stop") or [])
    for sequence in extra_stop or []:
        if sequence not in stop:
            stop.append(sequence)
    if stop:
        kwargs["stop"] = stop
    return kwargs


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _consume(chunks: Iterator[Dict[str, Any]],
             extract: Callable[[Dict[str, Any]], Tuple[str, str]],
             max_tokens: Optional[int]) -> Tuple[str, str, str]:
    """Stream a completion, returning ``(text, reasoning, finish_reason)``.

    Streaming is used even though nodes return the full string: it is what
    makes the ComfyUI cancel button responsive and drives the progress bar.
    """
    progress = progress_bar(max_tokens or 0)
    pieces: List[str] = []
    reasoning: List[str] = []
    finish_reason = ""
    try:
        for chunk in chunks:
            check_interrupt()
            choices = chunk.get("choices") or [{}]
            piece, reasoning_piece = extract(choices[0])
            if piece:
                pieces.append(piece)
            if reasoning_piece:
                reasoning.append(reasoning_piece)
            if (piece or reasoning_piece) and progress is not None:
                progress.update(1)
            reason = choices[0].get("finish_reason")
            if reason:
                finish_reason = reason
    finally:
        close = getattr(chunks, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    return "".join(pieces), "".join(reasoning), finish_reason


def complete(model: LlamaModel, prompt: str, *, grammar=None,
             **kwargs) -> Tuple[str, str, str]:
    llm = model.require()
    call_kwargs = filter_kwargs(llm.create_completion, kwargs)
    if grammar is not None:
        call_kwargs["grammar"] = grammar
    call_kwargs["stream"] = True
    stream = llm.create_completion(prompt=prompt, **call_kwargs)
    return _consume(stream, lambda choice: (choice.get("text") or "", ""),
                    kwargs.get("max_tokens"))


def chat(model: LlamaModel, messages: List[Dict[str, Any]], *, grammar=None,
         response_fmt: Optional[Dict[str, Any]] = None,
         **kwargs) -> Tuple[str, str, str]:
    llm = model.require()
    call_kwargs = filter_kwargs(llm.create_chat_completion, kwargs)
    if grammar is not None:
        call_kwargs["grammar"] = grammar
    elif response_fmt is not None:
        call_kwargs["response_format"] = response_fmt
    call_kwargs["stream"] = True
    stream = llm.create_chat_completion(messages=messages, **call_kwargs)

    def extract(choice: Dict[str, Any]) -> Tuple[str, str]:
        # Non-streaming fallback for handlers that ignore stream=True.
        source = choice.get("delta") or choice.get("message") or {}
        return (source.get("content") or "",
                source.get("reasoning_content") or "")

    return _consume(stream, extract, kwargs.get("max_tokens"))


def count_tokens(model: LlamaModel, text: str) -> int:
    llm = model.require()
    tokens = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    return len(tokens)
