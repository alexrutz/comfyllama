"""Tests for the llama-server nodes.

A stub HTTP server that speaks the llama.cpp endpoints is started on a random
port, so the nodes are exercised over a real socket, including SSE streaming.
"""

from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

# Importing test_nodes also installs the ComfyUI stubs.
from test_nodes import HAVE_IMAGING, sampling_args  # noqa: F401

from comfyllama.nodes import remote
from comfyllama.nodes.generation import LlamaCppGrammar, LlamaCppSampling
from comfyllama.server import (LlamaServer, LlamaServerError, apply_grammar,
                               apply_thinking, build_payload, normalize_base_url)


class StubHandler(BaseHTTPRequestHandler):
    """Implements just enough of llama-server for the nodes under test."""

    server_version = "llama.cpp-stub"

    def log_message(self, *args):  # silence the test output
        pass

    # -- helpers -----------------------------------------------------------

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _record(self, path, payload):
        state = self.server.state
        state["requests"].append({
            "path": path,
            "payload": payload,
            "authorization": self.headers.get("Authorization"),
        })

    # -- routes ------------------------------------------------------------

    def do_GET(self):
        state = self.server.state
        if self.path == "/health":
            self._record(self.path, None)
            self._json(state["health"], state["health_status"])
        elif self.path == "/v1/models":
            self._record(self.path, None)
            self._json({"data": [{"id": name} for name in state["models"]]})
        elif self.path == "/props":
            self._record(self.path, None)
            self._json(state["props"])
        else:
            self._json({"error": {"message": "not found"}}, 404)

    def do_POST(self):
        state = self.server.state
        payload = self._body()
        self._record(self.path, payload)

        if self.path == "/tokenize":
            return self._json({"tokens": list(range(len(payload["content"].split())))})
        if self.path == "/v1/chat/completions":
            if state["error"]:
                return self._json({"error": {"message": state["error"]}}, 400)
            events = [{"choices": [{"delta": {"reasoning_content": piece}}]}
                      for piece in state["reasoning_pieces"]]
            events += [{"choices": [{"delta": {"content": piece}}]}
                       for piece in state["pieces"]]
            events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            return self._sse(events)
        if self.path == "/completion":
            if state["error"]:
                return self._json({"error": {"message": state["error"]}}, 400)
            events = [{"content": piece, "stop": False} for piece in state["pieces"]]
            events.append({"content": "", "stop": True, "stopped_eos": True})
            return self._sse(events)
        self._json({"error": {"message": "not found"}}, 404)


class QuietHTTPServer(HTTPServer):
    """Swallows the broken-pipe noise from interrupted streams."""

    def handle_error(self, request, client_address):
        pass


class StubServer:
    def __init__(self):
        self.httpd = QuietHTTPServer(("127.0.0.1", 0), StubHandler)
        self.httpd.state = {
            "requests": [],
            "pieces": ["Hello", " world"],
            "reasoning_pieces": [],
            "models": ["stub-model"],
            "health": {"status": "ok"},
            "health_status": 200,
            "props": {"default_generation_settings": {"n_ctx": 8192},
                      "chat_template": "chatml"},
            "error": None,
        }
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def state(self):
        return self.httpd.state

    @property
    def url(self):
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class ServerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stub = StubServer()

    @classmethod
    def tearDownClass(cls):
        cls.stub.stop()

    def setUp(self):
        self.stub.state["requests"].clear()
        self.stub.state["pieces"] = ["Hello", " world"]
        self.stub.state["reasoning_pieces"] = []
        self.stub.state["error"] = None
        self.stub.state["health"] = {"status": "ok"}
        self.stub.state["health_status"] = 200

    def connect(self, **kwargs):
        options = {"base_url": self.stub.url, "timeout": 10, "check_connection": False}
        options.update(kwargs)
        return remote.LlamaServerConnect().connect(**options)[0]

    def requests_to(self, path):
        return [r for r in self.stub.state["requests"] if r["path"] == path]


class TestUrlHandling(unittest.TestCase):
    def test_scheme_is_added_and_trailing_v1_stripped(self):
        self.assertEqual(normalize_base_url("127.0.0.1:8080"), "http://127.0.0.1:8080")
        self.assertEqual(normalize_base_url("http://host:8080/v1/"), "http://host:8080")
        self.assertEqual(normalize_base_url(" http://host:8080/ "), "http://host:8080")

    def test_empty_or_invalid_url_is_rejected(self):
        for value in ("", "   ", "http://"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_base_url(value)

    def test_loopback_bypasses_the_proxy(self):
        def has_proxy_handler(connection):
            return any(type(h).__name__ == "ProxyHandler"
                       for h in connection._opener.handlers)

        previous = os.environ.get("http_proxy")
        os.environ["http_proxy"] = "http://proxy.invalid:3128"
        try:
            self.assertFalse(has_proxy_handler(LlamaServer("http://127.0.0.1:8080")))
            self.assertTrue(has_proxy_handler(LlamaServer("http://gpu-box.lan:8080")))
        finally:
            if previous is None:
                del os.environ["http_proxy"]
            else:
                os.environ["http_proxy"] = previous


class TestPayloads(unittest.TestCase):
    def test_openai_payload_keeps_max_tokens_and_sends_nothing_extra(self):
        from comfyllama.backend import sampler_kwargs

        payload = build_payload(
            sampler_kwargs(max_tokens=32, temperature=0.5, top_p=0.9, seed=3,
                           sampling=None), native=False)
        self.assertEqual(payload["max_tokens"], 32)
        # Nothing beyond the node's own controls is sent unless switched on.
        self.assertEqual(set(payload), {"max_tokens", "temperature", "top_p", "seed"})

    def test_enabled_mirostat_is_renamed_for_the_server(self):
        from comfyllama.backend import sampler_kwargs

        sampling = LlamaCppSampling().build(**sampling_args(
            use_mirostat=True, mirostat_mode=2))[0]
        payload = build_payload(
            sampler_kwargs(max_tokens=32, temperature=0.5, top_p=0.9, seed=3,
                           sampling=sampling), native=False)
        self.assertEqual(payload["mirostat"], 2)
        self.assertNotIn("mirostat_mode", payload)

    def test_native_payload_uses_n_predict(self):
        from comfyllama.backend import sampler_kwargs

        payload = build_payload(
            sampler_kwargs(max_tokens=0, temperature=0.5, top_p=0.9, seed=3,
                           sampling=None), native=True)
        self.assertEqual(payload["n_predict"], -1)  # 0 tokens means "no limit"
        self.assertNotIn("max_tokens", payload)

    def test_grammar_modes_map_to_the_right_fields(self):
        gbnf = LlamaCppGrammar().build("gbnf", 'root ::= "a"')[0]
        schema = LlamaCppGrammar().build("json_schema", '{"type": "object"}')[0]
        obj = LlamaCppGrammar().build("json_object", "")[0]

        self.assertEqual(apply_grammar({}, gbnf, native=False)["grammar"], 'root ::= "a"')
        self.assertEqual(apply_grammar({}, gbnf, native=True)["grammar"], 'root ::= "a"')

        chat_schema = apply_grammar({}, schema, native=False)["response_format"]
        self.assertEqual(chat_schema["type"], "json_schema")
        self.assertEqual(chat_schema["json_schema"]["schema"], {"type": "object"})
        self.assertEqual(apply_grammar({}, schema, native=True)["json_schema"],
                         {"type": "object"})

        self.assertEqual(apply_grammar({}, obj, native=False)["response_format"],
                         {"type": "json_object"})
        self.assertEqual(apply_grammar({}, None, native=False), {})


class TestConnectNode(ServerTestCase):
    def test_health_check_and_model_resolution(self):
        connection, model = remote.LlamaServerConnect().connect(
            base_url=self.stub.url, timeout=10, check_connection=True)
        self.assertEqual(model, "stub-model")
        self.assertTrue(self.requests_to("/health"))

    def test_explicit_model_is_kept(self):
        _, model = remote.LlamaServerConnect().connect(
            base_url=self.stub.url, timeout=10, check_connection=False,
            model="my-model")
        self.assertEqual(model, "my-model")
        self.assertFalse(self.requests_to("/v1/models"))

    def test_loading_server_reports_a_clear_error(self):
        self.stub.state["health"] = {"status": "loading model"}
        with self.assertRaises(LlamaServerError) as ctx:
            remote.LlamaServerConnect().connect(
                base_url=self.stub.url, timeout=10, check_connection=True)
        self.assertIn("loading model", str(ctx.exception))

    def test_unreachable_server_names_the_url(self):
        with self.assertRaises(LlamaServerError) as ctx:
            remote.LlamaServerConnect().connect(
                base_url="http://127.0.0.1:1", timeout=2, check_connection=True)
        self.assertIn("127.0.0.1:1", str(ctx.exception))
        self.assertIn("llama-server", str(ctx.exception))

    def test_api_key_is_sent_as_a_bearer_token(self):
        remote.LlamaServerConnect().connect(
            base_url=self.stub.url, timeout=10, check_connection=True,
            api_key="secret")
        self.assertEqual(self.requests_to("/health")[0]["authorization"],
                         "Bearer secret")


class TestChatNode(ServerTestCase):
    def test_streamed_chunks_are_joined_and_history_returned(self):
        text, thinking, messages = remote.LlamaServerChat().generate(
            self.connect(), "be brief", "hi", thinking="auto", max_tokens=16,
            temperature=0.2, top_p=0.9, seed=5)
        self.assertEqual(text, "Hello world")
        self.assertEqual(thinking, "")
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant"])

        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["model"], "stub-model")
        self.assertEqual(sent["seed"], 5)
        self.assertEqual(sent["max_tokens"], 16)
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])

    def test_sampling_and_grammar_reach_the_server(self):
        sampling = LlamaCppSampling().build(**sampling_args(
            use_top_k=True, top_k=20, use_mirostat=True, mirostat_mode=2,
            use_stop_sequences=True, stop_sequences="END"))[0]
        grammar = LlamaCppGrammar().build("json_object", "")[0]
        remote.LlamaServerChat().generate(
            self.connect(), "", "hi", thinking="auto", max_tokens=8, temperature=0.0,
            top_p=1.0, seed=0, sampling=sampling, grammar=grammar)

        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["top_k"], 20)
        self.assertEqual(sent["mirostat"], 2)
        self.assertEqual(sent["stop"], ["END"])
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        # Settings left switched off never reach the server.
        self.assertNotIn("min_p", sent)
        self.assertNotIn("repeat_penalty", sent)

    def test_history_is_forwarded(self):
        history = [{"role": "user", "content": "first"},
                   {"role": "assistant", "content": "reply"}]
        *_, messages = remote.LlamaServerChat().generate(
            self.connect(), "", "second", thinking="auto", max_tokens=8,
            temperature=0.0, top_p=1.0, seed=0, messages=history)
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual([m["content"] for m in sent["messages"]],
                         ["first", "reply", "second"])
        self.assertEqual(len(messages), 4)

    def test_server_error_message_is_surfaced(self):
        self.stub.state["error"] = "context shift is disabled"
        with self.assertRaises(LlamaServerError) as ctx:
            remote.LlamaServerChat().generate(
                self.connect(), "", "hi", thinking="auto", max_tokens=8,
                temperature=0.0, top_p=1.0, seed=0)
        self.assertIn("context shift is disabled", str(ctx.exception))
        self.assertIn("400", str(ctx.exception))

    def test_generation_is_interruptible(self):
        import comfy.model_management as mm

        self.stub.state["pieces"] = ["a"] * 50
        mm.interrupted = True
        try:
            with self.assertRaises(KeyboardInterrupt):
                remote.LlamaServerChat().generate(
                    self.connect(), "", "hi", thinking="auto", max_tokens=64,
                    temperature=0.0, top_p=1.0, seed=0)
        finally:
            mm.interrupted = False


class TestThinking(ServerTestCase):
    def test_switch_is_sent_as_chat_template_kwargs(self):
        for mode, expected in (("on", True), ("off", False)):
            with self.subTest(mode=mode):
                self.stub.state["requests"].clear()
                remote.LlamaServerChat().generate(
                    self.connect(), "", "hi", thinking=mode, max_tokens=8,
                    temperature=0.0, top_p=1.0, seed=0)
                sent = self.requests_to("/v1/chat/completions")[0]["payload"]
                self.assertEqual(sent["chat_template_kwargs"],
                                 {"enable_thinking": expected})

    def test_auto_sends_nothing_so_the_template_decides(self):
        remote.LlamaServerChat().generate(
            self.connect(), "", "hi", thinking="auto", max_tokens=8,
            temperature=0.0, top_p=1.0, seed=0)
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertNotIn("chat_template_kwargs", sent)

    def test_reasoning_content_lands_on_the_thinking_output(self):
        self.stub.state["reasoning_pieces"] = ["step one ", "step two"]
        self.stub.state["pieces"] = ["The answer."]
        text, thinking, messages = remote.LlamaServerChat().generate(
            self.connect(), "", "hi", thinking="on", max_tokens=32,
            temperature=0.0, top_p=1.0, seed=0)
        self.assertEqual(text, "The answer.")
        self.assertEqual(thinking, "step one step two")
        self.assertEqual(messages[-1], {"role": "assistant", "content": "The answer."})

    def test_think_tags_in_the_stream_are_split_out(self):
        # A server without --reasoning-format leaves the tags in the content.
        self.stub.state["pieces"] = ["<think>", "hmm", "</think>", "Answer."]
        text, thinking, _ = remote.LlamaServerChat().generate(
            self.connect(), "", "hi", thinking="auto", max_tokens=32,
            temperature=0.0, top_p=1.0, seed=0)
        self.assertEqual((text, thinking), ("Answer.", "hmm"))

    def test_completion_node_also_splits_thinking(self):
        self.stub.state["pieces"] = ["<think>plan</think>", "done"]
        text, thinking = remote.LlamaServerComplete().generate(
            self.connect(), "prompt", max_tokens=32, temperature=0.0, top_p=1.0,
            seed=0)
        self.assertEqual((text, thinking), ("done", "plan"))

    def test_existing_template_kwargs_are_preserved(self):
        payload = apply_thinking({"chat_template_kwargs": {"foo": 1}}, "off")
        self.assertEqual(payload["chat_template_kwargs"],
                         {"foo": 1, "enable_thinking": False})


class TestCompletionNode(ServerTestCase):
    def test_native_endpoint_is_used(self):
        text, _ = remote.LlamaServerComplete().generate(
            self.connect(), "Once upon", max_tokens=24, temperature=0.8, top_p=0.9,
            seed=1)
        self.assertEqual(text, "Hello world")
        sent = self.requests_to("/completion")[0]["payload"]
        self.assertEqual(sent["prompt"], "Once upon")
        self.assertEqual(sent["n_predict"], 24)
        self.assertTrue(sent["cache_prompt"])
        self.assertNotIn("max_tokens", sent)

    def test_gbnf_grammar_is_sent_verbatim(self):
        grammar = LlamaCppGrammar().build("gbnf", 'root ::= "yes" | "no"')[0]
        remote.LlamaServerComplete().generate(
            self.connect(), "answer:", max_tokens=4, temperature=0.0, top_p=1.0,
            seed=0, cache_prompt=False, grammar=grammar)
        sent = self.requests_to("/completion")[0]["payload"]
        self.assertEqual(sent["grammar"], 'root ::= "yes" | "no"')
        self.assertFalse(sent["cache_prompt"])


class TestInfoAndTokenizeNodes(ServerTestCase):
    def test_token_count_uses_the_server_tokenizer(self):
        count, = remote.LlamaServerTokenCount().count(self.connect(),
                                                      "one two three")
        self.assertEqual(count, 3)

    def test_info_reports_model_and_context(self):
        info, model, n_ctx = remote.LlamaServerInfo().info(self.connect())
        self.assertEqual(model, "stub-model")
        self.assertEqual(n_ctx, 8192)
        self.assertEqual(json.loads(info)["models"], ["stub-model"])


@unittest.skipUnless(HAVE_IMAGING, "numpy and Pillow are required")
class TestVisionNode(ServerTestCase):
    def test_images_are_uploaded_as_data_uris(self):
        import numpy as np

        text, _, _ = remote.LlamaServerVisionChat().generate(
            self.connect(), np.zeros((1, 8, 8, 3), dtype=np.float32), "sys",
            "what is this?", thinking="auto", max_tokens=16, temperature=0.1,
            top_p=0.9, seed=2)
        self.assertEqual(text, "Hello world")

        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        content = sent["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[-1], {"type": "text", "text": "what is this?"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
