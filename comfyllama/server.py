"""HTTP client for a running ``llama-server`` instance.

Only the standard library is used so the nodes work without extra
dependencies.  Responses are streamed as server-sent events, which keeps the
ComfyUI cancel button responsive during long generations.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .backend import check_interrupt, progress_bar

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

# How the Authorization header is built.  ``auto`` picks basic when a user name
# is present, bearer when only a token is, and sends nothing otherwise.
AUTH_MODES = ["auto", "bearer", "basic", "none"]

# Values that mean "do not pin a model, let the server choose".
AUTO_MODEL = {"auto", "default", "server default"}

ENV_PREFIX = "env:"


class LlamaServerError(RuntimeError):
    """Raised for transport errors and non-2xx responses.

    ``status`` is the HTTP status code, or ``None`` when the server could not
    be reached at all — the difference matters for deciding whether a failed
    probe is fatal.
    """

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def resolve_secret(value: str, *, field: str = "credential") -> str:
    """Read a credential, following an ``env:NAME`` indirection.

    Workflow JSON travels with shared graphs and embedded images, so a token
    typed into a widget leaks easily.  ``env:LLAMA_TOKEN`` keeps the secret in
    the environment ComfyUI was started with instead.
    """
    value = (value or "").strip()
    if not value.lower().startswith(ENV_PREFIX):
        return value
    name = value[len(ENV_PREFIX):].strip()
    if not name:
        raise ValueError(f"The {field} says '{ENV_PREFIX}' but names no variable.")
    resolved = os.environ.get(name)
    if resolved is None:
        raise ValueError(
            f"The {field} refers to the environment variable '{name}', which is "
            "not set for the process running ComfyUI."
        )
    return resolved


def _split_url(base_url: str):
    url = (base_url or "").strip()
    if not url:
        raise ValueError("The llama-server URL is empty.")
    if "://" not in url:
        url = f"http://{url}"
    url = url.rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname:
        raise ValueError(f"'{base_url}' is not a valid llama-server URL.")
    return parsed


def normalize_base_url(base_url: str) -> str:
    """Clean up a user-typed URL: add the scheme, drop ``/v1`` and any userinfo."""
    parsed = _split_url(base_url)
    netloc = parsed.hostname or ""
    if ":" in netloc:  # IPv6 literal
        netloc = f"[{netloc}]"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    path = parsed.path.rstrip("/")
    # A trailing /v1 is a common copy/paste from OpenAI clients; the endpoint
    # paths below already include it.
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, path, "", "")).rstrip("/")


def credentials_from_url(base_url: str) -> Tuple[str, str]:
    """Pull ``user:password`` out of a ``http://user:pass@host`` style URL."""
    try:
        parsed = _split_url(base_url)
    except ValueError:
        return "", ""
    return (urllib.parse.unquote(parsed.username or ""),
            urllib.parse.unquote(parsed.password or ""))


def build_auth_header(mode: str, *, api_key: str = "", username: str = "",
                      password: str = "") -> Optional[str]:
    """Return the Authorization header value for the chosen mode."""
    api_key = resolve_secret(api_key, field="API key")
    username = resolve_secret(username, field="user name")
    password = resolve_secret(password, field="password")

    if mode == "none":
        return None
    if mode == "auto":
        mode = "basic" if username else ("bearer" if api_key else "none")
        if mode == "none":
            return None

    if mode == "bearer":
        if not api_key:
            raise ValueError(
                "Authentication is set to 'bearer' but the api_key field is "
                "empty. Fill it in, or switch auth to 'none'."
            )
        return f"Bearer {api_key}"
    if mode == "basic":
        if not username:
            raise ValueError(
                "Authentication is set to 'basic' but the username field is "
                "empty. Fill it in, or switch auth to 'none'."
            )
        token = base64.b64encode(f"{username}:{password}".encode("utf-8"))
        return f"Basic {token.decode('ascii')}"
    raise ValueError(f"Unknown authentication mode '{mode}'.")


class LlamaServer:
    """Connection details for one ``llama-server`` endpoint."""

    def __init__(self, base_url: str, *, api_key: str = "", username: str = "",
                 password: str = "", auth: str = "auto", timeout: float = 300.0,
                 model: str = "") -> None:
        self.base_url = normalize_base_url(base_url)
        self.timeout = float(timeout)
        self.model = (model or "").strip()
        self._model_list: Optional[List[str]] = None

        if not (username or password):
            # Credentials typed straight into the URL count as basic auth.
            username, password = credentials_from_url(base_url)
        self.auth = auth
        self._authorization = build_auth_header(
            auth, api_key=api_key, username=username, password=password)
        self._opener = self._build_opener()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        # Never include the credential itself.
        auth = "authenticated" if self._authorization else "no auth"
        return f"<LlamaServer {self.base_url} model={self.model or 'default'!r} {auth}>"

    def _build_opener(self):
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        if host in LOOPBACK_HOSTS:
            # An HTTP_PROXY meant for the internet must not swallow requests to
            # a server running on this machine.  An empty ProxyHandler registers
            # no *_open methods, so the opener ends up without proxy support.
            return urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return urllib.request.build_opener()

    # -- transport ---------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._authorization:
            headers["Authorization"] = self._authorization
        return headers

    def _open(self, path: str, payload: Optional[Dict[str, Any]], method: str):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(url, data=data, headers=self._headers(),
                                         method=method)
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            detail = _error_detail(exc)
            raise LlamaServerError(
                f"{url} returned {exc.code}: {detail}"
                f"{self._auth_hint(exc.code)}"
                f"{self._model_hint(path, exc.code, detail, payload)}",
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise LlamaServerError(
                f"Could not reach llama-server at {self.base_url} ({exc.reason}). "
                "Check that it is running and that the URL is correct, e.g. "
                "`llama-server -m model.gguf --host 127.0.0.1 --port 8080`."
            ) from exc
        except OSError as exc:
            raise LlamaServerError(f"Could not reach llama-server at "
                                   f"{self.base_url} ({exc}).") from exc

    def _model_hint(self, path: str, status: int, detail: str,
                    payload: Optional[Dict[str, Any]]) -> str:
        """Name the models a router actually offers when one was rejected."""
        if status not in (400, 404, 422, 503):
            return ""
        if path == "/v1/models" or "model" not in detail.lower():
            return ""
        requested = (payload or {}).get("model")
        available = self.available_models()
        if not available:
            return ""
        listing = ", ".join(available)
        if requested:
            return (f" The server does not appear to offer '{requested}'. "
                    f"Available: {listing}.")
        return f" Models this server offers: {listing}."

    def _auth_hint(self, status: int) -> str:
        """Extra guidance for the two status codes that mean 'credentials'."""
        if status not in (401, 403):
            return ""
        if not self._authorization:
            return (" The connect node sent no credentials — set auth to 'bearer' "
                    "and fill in api_key, or to 'basic' with username/password.")
        kind = self._authorization.split(" ", 1)[0].lower()
        return f" The {kind} credentials from the connect node were rejected."

    def request(self, path: str, payload: Optional[Dict[str, Any]] = None,
                method: str = "POST") -> Any:
        with self._open(path, payload, method) as response:
            body = response.read().decode("utf-8", "replace")
        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LlamaServerError(
                f"llama-server sent a non-JSON reply from {path}: {body[:200]}") from exc

    def stream(self, path: str, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Yield decoded SSE payloads, checking for interruption per event."""
        response = self._open(path, dict(payload, stream=True), "POST")
        try:
            for line in response:
                check_interrupt()
                event = _decode_sse_line(line)
                if event is None:
                    continue
                if event is _DONE:
                    return
                yield event
        finally:
            response.close()

    # -- endpoints ---------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self.request("/health", method="GET") or {}

    def props(self) -> Dict[str, Any]:
        return self.request("/props", method="GET") or {}

    def models(self) -> List[str]:
        """Model ids from ``/v1/models``. Raises if the endpoint is unusable."""
        payload = self.request("/v1/models", method="GET") or {}
        names = [str(entry.get("id")) for entry in payload.get("data", [])
                 if entry.get("id")]
        self._model_list = names
        return names

    def available_models(self) -> List[str]:
        """Cached model list that never raises — for hints and dropdowns.

        A router front end can list a dozen models, and the list is stable for
        the lifetime of a connection, so it is fetched at most once.
        """
        if self._model_list is None:
            try:
                self.models()
            except LlamaServerError:
                self._model_list = []
        return list(self._model_list or [])

    def resolve_model(self, override: str = "") -> str:
        """The model name to send with a request.

        The node's own override wins over the one set on the connection.  When
        both say ``auto`` the server is asked what it serves and the first
        model it reports is used, which is what a router needs — it dispatches
        on this field.  Only a server that reports nothing leaves the field out.
        """
        for candidate in (override, self.model):
            candidate = (candidate or "").strip()
            if candidate and candidate.lower() not in AUTO_MODEL:
                return candidate

        available = self.available_models()
        return available[0] if available else ""

    def probe(self) -> str:
        """Check the endpoint answers, and report what it said.

        Raises only when the server cannot be reached or rejects the
        credentials.  A router in front of llama-server often has no
        ``/health`` at all, or reports "not ready" until the first request
        loads a model, and neither is a reason to fail the graph.
        """
        try:
            status = str(self.health().get("status") or "ok")
        except LlamaServerError as exc:
            if exc.status is None or exc.status in (401, 403):
                raise  # unreachable, or the credentials are wrong
            # Reachable but /health is not usable here. Confirm with the
            # OpenAI-compatible route every router implements.
            try:
                self.models()
            except LlamaServerError as models_exc:
                if models_exc.status is None or models_exc.status in (401, 403):
                    raise
                raise LlamaServerError(
                    f"{self.base_url} answered, but neither /health nor "
                    f"/v1/models worked ({exc.status} and {models_exc.status}). "
                    "Is this really a llama-server endpoint?",
                    status=models_exc.status,
                ) from models_exc
            return "unknown"
        return status

    def tokenize(self, text: str) -> List[int]:
        payload = self.request("/tokenize", {"content": text}) or {}
        return list(payload.get("tokens", []))


_DONE = object()


def _decode_sse_line(line: bytes) -> Any:
    text = line.decode("utf-8", "replace").strip()
    if not text or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[len("data:"):].strip()
    if text == "[DONE]":
        return _DONE
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:
        return exc.reason or ""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body[:300] or (exc.reason or "")
    error = payload.get("error", payload)
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


# --------------------------------------------------------------------------
# Payload construction
# --------------------------------------------------------------------------

# llama-server accepts the llama.cpp sampler names alongside the OpenAI ones,
# with a few spelling differences from llama-cpp-python.
_SAMPLER_RENAMES = {"mirostat_mode": "mirostat"}
_NATIVE_RENAMES = {"mirostat_mode": "mirostat", "max_tokens": "n_predict"}


def build_payload(kwargs: Dict[str, Any], *, native: bool) -> Dict[str, Any]:
    """Translate ``backend.sampler_kwargs`` output into a server payload."""
    renames = _NATIVE_RENAMES if native else _SAMPLER_RENAMES
    payload: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue  # "generate until the context is full"
        if key == "stop" and not value:
            continue
        payload[renames.get(key, key)] = value
    if native and "n_predict" not in payload:
        payload["n_predict"] = -1
    return payload


def apply_model(payload: Dict[str, Any], server: "LlamaServer",
                override: str = "") -> Dict[str, Any]:
    """Pin the request to a model, or leave the choice to the server.

    Router front ends dispatch on this field — including on the native
    endpoints, which plain llama-server simply ignores it on.
    """
    model = server.resolve_model(override)
    if model:
        payload["model"] = model
    return payload


def apply_thinking(payload: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Ask the server's chat template to enable or disable reasoning.

    ``chat_template_kwargs`` is what llama-server forwards into the Jinja chat
    template, which is how Qwen3-style models expose the switch.  Templates
    without the variable simply ignore it.
    """
    from .reasoning import template_kwargs

    kwargs = template_kwargs(mode)
    if kwargs:
        merged = dict(payload.get("chat_template_kwargs") or {})
        merged.update(kwargs)
        payload["chat_template_kwargs"] = merged
    return payload


def apply_grammar(payload: Dict[str, Any], spec: Optional[Dict[str, Any]], *,
                  native: bool) -> Dict[str, Any]:
    """Attach grammar/JSON constraints in the form the endpoint expects."""
    if not spec:
        return payload
    kind = spec.get("type")
    if kind == "gbnf":
        payload["grammar"] = spec["gbnf"]
    elif kind == "json_schema":
        if native:
            payload["json_schema"] = spec["schema"]
        else:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": spec["schema"],
                                "strict": True},
            }
    elif kind == "json_object":
        if native:
            payload["json_schema"] = {"type": "object"}
        else:
            payload["response_format"] = {"type": "json_object"}
    return payload


def stream_chat(server: LlamaServer, messages: List[Dict[str, Any]],
                payload: Dict[str, Any]) -> Tuple[str, str, str]:
    """Run ``/v1/chat/completions``.

    Returns ``(text, reasoning, finish_reason)``.  ``reasoning`` is filled when
    the server splits the chain of thought off itself, which it does when it
    runs with ``--reasoning-format deepseek``; otherwise the thinking stays
    inside the text as ``<think>`` tags and is parsed by the caller.
    """
    body = dict(payload)
    body["messages"] = messages

    pieces: List[str] = []
    reasoning: List[str] = []
    finish_reason = ""
    progress = progress_bar(payload.get("max_tokens") or 0)
    for event in server.stream("/v1/chat/completions", body):
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        # Non-streaming servers answer with "message" instead of "delta".
        source = choice.get("delta") or choice.get("message") or {}
        piece = source.get("content")
        reasoning_piece = source.get("reasoning_content")
        if piece:
            pieces.append(piece)
        if reasoning_piece:
            reasoning.append(reasoning_piece)
        if (piece or reasoning_piece) and progress is not None:
            progress.update(1)
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
    return "".join(pieces), "".join(reasoning), finish_reason


def stream_completion(server: LlamaServer, prompt: str,
                      payload: Dict[str, Any]) -> Tuple[str, str]:
    """Run the native ``/completion`` endpoint."""
    body = dict(payload)
    body["prompt"] = prompt

    pieces: List[str] = []
    finish_reason = ""
    n_predict = payload.get("n_predict")
    progress = progress_bar(n_predict if isinstance(n_predict, int) and n_predict > 0 else 0)
    for event in server.stream("/completion", body):
        piece = event.get("content")
        if piece:
            pieces.append(piece)
            if progress is not None:
                progress.update(1)
        if event.get("stop"):
            if event.get("stopped_eos"):
                finish_reason = "stop"
            elif event.get("stopped_word"):
                finish_reason = "stop_word"
            elif event.get("stopped_limit"):
                finish_reason = "length"
    return "".join(pieces), finish_reason
