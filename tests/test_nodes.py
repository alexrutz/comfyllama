"""Tests that run without ComfyUI, torch or llama-cpp-python installed.

ComfyUI's ``folder_paths`` and ``comfy.*`` modules are stubbed so the pack can
be imported and its pure logic exercised in CI.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def install_comfy_stubs(models_dir: str) -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = models_dir
    folder_paths.folder_names_and_paths = {}

    def get_filename_list(folder):
        entry = folder_paths.folder_names_and_paths.get(folder)
        if not entry:
            return []
        names = []
        for directory in entry[0]:
            if os.path.isdir(directory):
                names.extend(name for name in os.listdir(directory)
                             if name.lower().endswith(".gguf"))
        return names

    def get_full_path(folder, name):
        entry = folder_paths.folder_names_and_paths.get(folder)
        for directory in (entry[0] if entry else []):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate):
                return candidate
        return None

    folder_paths.get_filename_list = get_filename_list
    folder_paths.get_full_path = get_full_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    model_management = types.ModuleType("comfy.model_management")
    model_management.interrupted = False

    def throw_exception_if_processing_interrupted():
        if model_management.interrupted:
            raise KeyboardInterrupt("interrupted")

    model_management.throw_exception_if_processing_interrupted = (
        throw_exception_if_processing_interrupted)
    model_management.unload_all_models = lambda: None
    model_management.soft_empty_cache = lambda: None

    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, value):
            self.current += value

    utils.ProgressBar = ProgressBar

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = utils


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_models")
os.makedirs(os.path.join(MODELS_DIR, "llm"), exist_ok=True)
install_comfy_stubs(MODELS_DIR)

sys.path.insert(0, os.path.dirname(REPO_ROOT))
sys.path.insert(0, REPO_ROOT)

import comfyllama  # noqa: E402
from comfyllama import backend, paths, reasoning  # noqa: E402
from comfyllama.nodes import generation, utils  # noqa: E402


class TestRegistration(unittest.TestCase):
    def test_mappings_are_complete(self):
        self.assertTrue(comfyllama.NODE_CLASS_MAPPINGS)
        self.assertEqual(set(comfyllama.NODE_CLASS_MAPPINGS),
                         set(comfyllama.NODE_DISPLAY_NAME_MAPPINGS))

    def test_nodes_declare_the_comfyui_interface(self):
        for name, node in comfyllama.NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=name):
                inputs = node.INPUT_TYPES()
                self.assertIn("required", inputs)
                self.assertTrue(hasattr(node, "RETURN_TYPES"))
                self.assertTrue(hasattr(node, "CATEGORY"))
                self.assertTrue(callable(getattr(node, node.FUNCTION)))
                if hasattr(node, "RETURN_NAMES"):
                    self.assertEqual(len(node.RETURN_NAMES), len(node.RETURN_TYPES))

    def test_model_folders_are_registered(self):
        import folder_paths

        for key in (paths.LLM_FOLDER, paths.MMPROJ_FOLDER):
            self.assertIn(key, folder_paths.folder_names_and_paths)
            self.assertIn(".gguf", folder_paths.folder_names_and_paths[key][1])

    def test_registration_is_idempotent(self):
        import folder_paths

        before = list(folder_paths.folder_names_and_paths[paths.LLM_FOLDER][0])
        paths.register_model_folders()
        self.assertEqual(before,
                         folder_paths.folder_names_and_paths[paths.LLM_FOLDER][0])


class TestPaths(unittest.TestCase):
    def test_empty_folder_yields_placeholder(self):
        names = paths.list_models(paths.LLM_FOLDER)
        self.assertTrue(names)

    def test_placeholder_selection_raises_a_helpful_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            paths.resolve_model_path(paths.LLM_FOLDER, "<no .gguf files found>")
        self.assertIn("models/llm", str(ctx.exception))

    def test_override_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            paths.resolve_model_path(paths.LLM_FOLDER, "x.gguf", "/nope/model.gguf")

    def test_override_wins_over_dropdown(self):
        target = os.path.join(MODELS_DIR, "llm", "dummy.gguf")
        with open(target, "wb") as handle:
            handle.write(b"")
        try:
            self.assertEqual(
                paths.resolve_model_path(paths.LLM_FOLDER, "other.gguf", f' "{target}" '),
                target)
            self.assertEqual(paths.resolve_model_path(paths.LLM_FOLDER, "dummy.gguf"),
                             target)
        finally:
            os.remove(target)


SAMPLING_DEFAULTS = {
    "use_top_k": False, "top_k": 40,
    "use_min_p": False, "min_p": 0.05,
    "use_typical_p": False, "typical_p": 1.0,
    "use_repeat_penalty": False, "repeat_penalty": 1.1,
    "use_presence_penalty": False, "presence_penalty": 0.0,
    "use_frequency_penalty": False, "frequency_penalty": 0.0,
    "use_mirostat": False, "mirostat_mode": 2, "mirostat_tau": 5.0,
    "mirostat_eta": 0.1,
    "use_stop_sequences": False, "stop_sequences": "",
}


def sampling_args(**overrides):
    """Widget values for the sampling node, everything switched off by default."""
    return {**SAMPLING_DEFAULTS, **overrides}


class TestSampling(unittest.TestCase):
    def test_stop_sequences_are_split_and_unescaped(self):
        self.assertEqual(backend.parse_stop_sequences("</s>\n\\n\\n\n\n  \n"),
                         ["</s>", "\n\n"])
        self.assertEqual(backend.parse_stop_sequences(""), [])

    def test_without_a_sampling_node_only_the_node_controls_are_sent(self):
        kwargs = backend.sampler_kwargs(max_tokens=64, temperature=0.5, top_p=0.9,
                                        seed=7, sampling=None)
        self.assertEqual(set(kwargs), {"max_tokens", "temperature", "top_p", "seed"})
        self.assertEqual(kwargs["max_tokens"], 64)
        self.assertEqual(kwargs["seed"], 7)

    def test_disabled_settings_are_left_out_entirely(self):
        sampling = generation.LlamaCppSampling().build(**sampling_args(
            use_repeat_penalty=True, repeat_penalty=1.3))[0]
        self.assertEqual(sampling, {"repeat_penalty": 1.3})

        kwargs = backend.sampler_kwargs(max_tokens=8, temperature=1.0, top_p=1.0,
                                        seed=1, sampling=sampling)
        self.assertEqual(kwargs["repeat_penalty"], 1.3)
        for key in ("top_k", "min_p", "typical_p", "mirostat_mode", "stop"):
            self.assertNotIn(key, kwargs)

    def test_all_switches_off_is_the_same_as_no_node(self):
        sampling = generation.LlamaCppSampling().build(**sampling_args())[0]
        self.assertEqual(sampling, {})
        self.assertEqual(
            backend.sampler_kwargs(max_tokens=8, temperature=1.0, top_p=1.0, seed=1,
                                   sampling=sampling),
            backend.sampler_kwargs(max_tokens=8, temperature=1.0, top_p=1.0, seed=1,
                                   sampling=None))

    def test_mirostat_switch_covers_tau_and_eta(self):
        sampling = generation.LlamaCppSampling().build(**sampling_args(
            use_mirostat=True, mirostat_mode=1, mirostat_tau=4.0, mirostat_eta=0.3))[0]
        self.assertEqual(sampling, {"mirostat_mode": 1, "mirostat_tau": 4.0,
                                    "mirostat_eta": 0.3})

    def test_stop_sequences_need_their_switch(self):
        self.assertNotIn("stop", generation.LlamaCppSampling().build(**sampling_args(
            stop_sequences="END"))[0])
        self.assertEqual(generation.LlamaCppSampling().build(**sampling_args(
            use_stop_sequences=True, stop_sequences="END"))[0]["stop"], ["END"])
        # An enabled but empty field must not send an empty stop list.
        self.assertNotIn("stop", generation.LlamaCppSampling().build(**sampling_args(
            use_stop_sequences=True, stop_sequences="  "))[0])

    def test_zero_max_tokens_means_unlimited_and_negative_seed_random(self):
        kwargs = backend.sampler_kwargs(max_tokens=0, temperature=0.0, top_p=1.0,
                                        seed=-1, sampling=None)
        self.assertIsNone(kwargs["max_tokens"])
        self.assertEqual(kwargs["seed"], -1)

    def test_enabled_settings_and_extra_stops_merge(self):
        sampling = generation.LlamaCppSampling().build(**sampling_args(
            use_top_k=True, top_k=10, use_mirostat=True, mirostat_mode=2,
            use_stop_sequences=True, stop_sequences="END\nEND"))[0]
        kwargs = backend.sampler_kwargs(max_tokens=8, temperature=1.0, top_p=1.0,
                                        seed=1, sampling=sampling,
                                        extra_stop=["END", "###"])
        self.assertEqual(kwargs["top_k"], 10)
        self.assertEqual(kwargs["mirostat_mode"], 2)
        self.assertEqual(kwargs["stop"], ["END", "END", "###"])
        self.assertNotIn("min_p", kwargs)

    def test_filter_kwargs_drops_unknown_parameters(self):
        def target(a, b=1):
            return a, b

        self.assertEqual(backend.filter_kwargs(target, {"a": 1, "zzz": 2}), {"a": 1})

        def flexible(a, **kwargs):
            return a

        self.assertEqual(backend.filter_kwargs(flexible, {"a": 1, "zzz": 2}),
                         {"a": 1, "zzz": 2})


class TestGrammar(unittest.TestCase):
    def test_json_object_mode_ignores_the_definition(self):
        spec = generation.LlamaCppGrammar().build("json_object", "not json")[0]
        self.assertEqual(spec, {"type": "json_object"})
        self.assertEqual(backend.response_format(spec), {"type": "json_object"})

    def test_json_schema_is_parsed(self):
        spec = generation.LlamaCppGrammar().build("json_schema", '{"type": "object"}')[0]
        self.assertEqual(spec["schema"], {"type": "object"})
        self.assertEqual(backend.response_format(spec),
                         {"type": "json_object", "schema": {"type": "object"}})

    def test_invalid_schema_reports_the_field(self):
        with self.assertRaises(ValueError):
            generation.LlamaCppGrammar().build("json_schema", "{oops}")

    def test_gbnf_requires_content_and_bypasses_response_format(self):
        with self.assertRaises(ValueError):
            generation.LlamaCppGrammar().build("gbnf", "   ")
        spec = generation.LlamaCppGrammar().build("gbnf", "root ::= \"a\"")[0]
        self.assertTrue(generation._needs_grammar(spec))
        self.assertIsNone(backend.response_format(spec))

    def test_no_grammar_is_a_no_op(self):
        self.assertIsNone(backend.build_grammar(None))
        self.assertIsNone(backend.response_format(None))


class TestMessages(unittest.TestCase):
    def test_system_prompt_replaces_the_one_in_the_history(self):
        history = [{"role": "system", "content": "old"},
                   {"role": "user", "content": "hi"}]
        messages = generation._messages("new", "what now?", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "user"])
        self.assertEqual(messages[0]["content"], "new")
        self.assertEqual(messages[-1]["content"], "what now?")

    def test_history_system_prompt_survives_an_empty_system_field(self):
        history = [{"role": "system", "content": "old"}]
        messages = generation._messages("  ", "hi", history)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_history_is_copied_not_mutated(self):
        history = [{"role": "user", "content": "hi"}]
        messages = generation._messages("", "again", history)
        messages[0]["content"] = "changed"
        self.assertEqual(history[0]["content"], "hi")

    def test_message_node_appends(self):
        node = utils.LlamaCppMessage()
        first = node.append("user", "hello")[0]
        second = node.append("assistant", "hi there", first)[0]
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 2)
        self.assertEqual(second[1]["role"], "assistant")

    def test_messages_to_text_handles_multimodal_content(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "what is this?"},
            ]},
        ]
        rendered = utils.LlamaCppMessagesToText().render(messages, include_system=False)[0]
        self.assertNotIn("sys", rendered)
        self.assertIn("[image]", rendered)
        self.assertIn("what is this?", rendered)


class TestThinkingSplit(unittest.TestCase):
    def test_plain_answer_has_no_thinking(self):
        self.assertEqual(reasoning.split_thinking("just an answer"),
                         ("just an answer", ""))
        self.assertEqual(reasoning.split_thinking(""), ("", ""))

    def test_think_block_is_removed_from_the_answer(self):
        answer, thinking = reasoning.split_thinking(
            "<think>weighing options</think>\n\nThe answer is 42.")
        self.assertEqual(answer, "The answer is 42.")
        self.assertEqual(thinking, "weighing options")

    def test_closing_tag_without_opening_is_treated_as_thinking(self):
        # The chat template already emitted "<think>", so the model only
        # generates the closing tag.
        answer, thinking = reasoning.split_thinking("still reasoning</think>Done.")
        self.assertEqual(answer, "Done.")
        self.assertEqual(thinking, "still reasoning")

    def test_unterminated_block_keeps_everything_as_thinking(self):
        answer, thinking = reasoning.split_thinking("<think>cut off mid thought")
        self.assertEqual(answer, "")
        self.assertEqual(thinking, "cut off mid thought")

    def test_multiple_blocks_are_merged_and_tags_are_case_insensitive(self):
        answer, thinking = reasoning.split_thinking(
            "<Thinking>one</Thinking>A<think>two</think>B")
        self.assertEqual(answer, "AB")
        self.assertEqual(thinking, "onetwo")

    def test_text_before_the_block_is_kept(self):
        answer, thinking = reasoning.split_thinking("Sure. <think>hmm</think> Here:")
        self.assertEqual(answer, "Sure.  Here:".strip())
        self.assertEqual(thinking, "hmm")

    def test_reasoning_field_and_parsed_tags_are_combined(self):
        self.assertEqual(reasoning.combine("from server", "from tags"),
                         "from server\nfrom tags")
        self.assertEqual(reasoning.combine("", "from tags"), "from tags")
        self.assertEqual(reasoning.combine("  ", ""), "")


class TestThinkingControlTag(unittest.TestCase):
    def test_auto_leaves_the_conversation_untouched(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(reasoning.apply_control_tag(messages, "auto"), messages)

    def test_tag_is_appended_to_the_last_user_message(self):
        messages = [{"role": "system", "content": "sys"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"}]
        tagged = reasoning.apply_control_tag(messages, "off")
        self.assertEqual(tagged[-1]["content"], "second\n/no_think")
        self.assertEqual(tagged[1]["content"], "first")
        self.assertEqual(messages[-1]["content"], "second")  # input not mutated

        self.assertEqual(
            reasoning.apply_control_tag(messages, "on")[-1]["content"],
            "second\n/think")

    def test_tag_goes_into_the_text_part_of_multimodal_content(self):
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text", "text": "what is this?"},
        ]}]
        tagged = reasoning.apply_control_tag(messages, "off")
        content = tagged[0]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[-1]["text"], "what is this?\n/no_think")

    def test_template_kwargs_map_to_the_server_field(self):
        self.assertEqual(reasoning.template_kwargs("on"), {"enable_thinking": True})
        self.assertEqual(reasoning.template_kwargs("off"), {"enable_thinking": False})
        self.assertIsNone(reasoning.template_kwargs("auto"))


class TestUtilNodes(unittest.TestCase):
    def test_template_substitutes_connected_inputs_only(self):
        result = utils.LlamaCppPromptTemplate().format(
            "{a} and {b} and {missing}", a="x")[0]
        self.assertEqual(result, "x and {b} and {missing}")

    def test_preview_returns_ui_and_passthrough(self):
        result = utils.LlamaCppPreviewText().preview("hello")
        self.assertEqual(result["ui"], {"text": ["hello"]})
        self.assertEqual(result["result"], ("hello",))


try:
    import numpy  # noqa: F401
    from PIL import Image  # noqa: F401

    HAVE_IMAGING = True
except ImportError:
    HAVE_IMAGING = False


@unittest.skipUnless(HAVE_IMAGING, "numpy and Pillow are required")
class TestImageEncoding(unittest.TestCase):
    def test_batch_is_encoded_as_data_uris(self):
        import numpy as np

        from comfyllama.images import images_to_content

        batch = np.zeros((2, 8, 8, 3), dtype=np.float32)
        batch[1] = 1.0
        content = images_to_content(batch, max_size=4, quality=100)
        self.assertEqual(len(content), 2)
        for part in content:
            self.assertEqual(part["type"], "image_url")
            self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_jpeg_is_used_below_full_quality(self):
        import numpy as np

        from comfyllama.images import images_to_content

        content = images_to_content(np.zeros((1, 4, 4, 4), dtype=np.float32), quality=80)
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_rejects_a_wrongly_shaped_tensor(self):
        import numpy as np

        from comfyllama.images import images_to_content

        with self.assertRaises(ValueError):
            images_to_content(np.zeros((4, 4), dtype=np.float32))


class FakeLlama:
    """Minimal stand-in for ``llama_cpp.Llama``."""

    def __init__(self, pieces, finish_reason="stop"):
        self.pieces = pieces
        self.finish_reason = finish_reason
        self.calls = []
        self.closed = False

    def create_completion(self, prompt, max_tokens=None, temperature=0.0, top_p=1.0,
                          stop=None, seed=None, stream=False, grammar=None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "seed": seed,
                           "stop": stop, "grammar": grammar})
        return self._stream("text")

    def create_chat_completion(self, messages, max_tokens=None, temperature=0.0,
                               top_p=1.0, stop=None, seed=None, stream=False,
                               grammar=None, response_format=None):
        self.calls.append({"messages": messages, "response_format": response_format,
                           "grammar": grammar, "seed": seed})
        return self._stream("delta")

    def _stream(self, kind):
        for index, piece in enumerate(self.pieces):
            payload = {"text": piece} if kind == "text" else {"delta": {"content": piece}}
            if index == len(self.pieces) - 1:
                payload["finish_reason"] = self.finish_reason
            yield {"choices": [payload]}

    def tokenize(self, text, add_bos=False, special=False):
        return text.split()

    def close(self):
        self.closed = True


class ReasoningLlama(FakeLlama):
    """A model whose handler splits the chain of thought off itself."""

    def __init__(self, pieces, reasoning_pieces):
        super().__init__(list(pieces))
        self.reasoning_pieces = list(reasoning_pieces)

    def create_chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})

        def stream():
            for piece in self.reasoning_pieces:
                yield {"choices": [{"delta": {"reasoning_content": piece}}]}
            for piece in self.pieces:
                yield {"choices": [{"delta": {"content": piece}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return stream()


def fake_model(pieces=("Hello", " world")):
    llm = FakeLlama(list(pieces))
    return backend.LlamaModel(llm, ("key",), "fake"), llm


class TestGeneration(unittest.TestCase):
    def test_completion_streams_and_counts_tokens(self):
        model, llm = fake_model()
        text, thinking, tokens = generation.LlamaCppComplete().generate(
            model, "prompt", max_tokens=16, temperature=0.2, top_p=0.9, seed=3)
        self.assertEqual(text, "Hello world")
        self.assertEqual(thinking, "")
        self.assertEqual(tokens, 2)
        self.assertEqual(llm.calls[0]["seed"], 3)
        self.assertEqual(llm.calls[0]["max_tokens"], 16)

    def test_chat_returns_the_updated_history(self):
        model, llm = fake_model(["Hi"])
        text, _, messages, _ = generation.LlamaCppChat().generate(
            model, "be nice", "hello", thinking="auto", max_tokens=8, temperature=0.1,
            top_p=1.0, seed=0)
        self.assertEqual(text, "Hi")
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], "Hi")

    def test_chat_passes_json_response_format(self):
        model, llm = fake_model(["{}"])
        grammar = generation.LlamaCppGrammar().build("json_object", "")[0]
        generation.LlamaCppChat().generate(
            model, "", "hello", thinking="auto", max_tokens=8, temperature=0.0,
            top_p=1.0, seed=0, grammar=grammar)
        self.assertEqual(llm.calls[0]["response_format"], {"type": "json_object"})
        self.assertIsNone(llm.calls[0]["grammar"])

    def test_generation_is_interruptible(self):
        import comfy.model_management as mm

        model, _ = fake_model(["a", "b"])
        mm.interrupted = True
        try:
            with self.assertRaises(KeyboardInterrupt):
                generation.LlamaCppComplete().generate(
                    model, "prompt", max_tokens=4, temperature=0.0, top_p=1.0, seed=0)
        finally:
            mm.interrupted = False

    def test_vision_node_rejects_a_text_only_model(self):
        model, _ = fake_model()
        with self.assertRaises(ValueError) as ctx:
            generation.LlamaCppVisionChat().generate(
                model, object(), "sys", "describe", thinking="auto", max_tokens=8,
                temperature=0.0, top_p=1.0, seed=0)
        self.assertIn("multimodal projector", str(ctx.exception))

    def test_unloaded_model_reports_a_useful_error(self):
        model, _ = fake_model()
        model.free()
        with self.assertRaises(RuntimeError) as ctx:
            generation.LlamaCppComplete().generate(
                model, "prompt", max_tokens=4, temperature=0.0, top_p=1.0, seed=0)
        self.assertIn("unloaded", str(ctx.exception))

    def test_chat_separates_thinking_from_the_answer_and_history(self):
        model, _ = fake_model(["<think>", "let me see", "</think>", "42"])
        text, thinking, messages, _ = generation.LlamaCppChat().generate(
            model, "", "what is 6*7?", thinking="auto", max_tokens=32,
            temperature=0.0, top_p=1.0, seed=0)
        self.assertEqual(text, "42")
        self.assertEqual(thinking, "let me see")
        # The chain of thought must not leak into the next turn.
        self.assertEqual(messages[-1], {"role": "assistant", "content": "42"})

    def test_completion_separates_thinking_and_counts_only_the_answer(self):
        model, _ = fake_model(["<think>a b c</think>", "final answer"])
        text, thinking, tokens = generation.LlamaCppComplete().generate(
            model, "prompt", max_tokens=32, temperature=0.0, top_p=1.0, seed=0)
        self.assertEqual((text, thinking), ("final answer", "a b c"))
        self.assertEqual(tokens, 2)

    def test_thinking_switch_adds_the_control_tag_to_the_prompt(self):
        model, llm = fake_model(["ok"])
        generation.LlamaCppChat().generate(
            model, "sys", "hello", thinking="off", max_tokens=8, temperature=0.0,
            top_p=1.0, seed=0)
        sent = llm.calls[0]["messages"]
        self.assertEqual(sent[-1]["content"], "hello\n/no_think")
        self.assertEqual(sent[0]["content"], "sys")

    def test_reasoning_content_field_is_used_when_present(self):
        model, _ = fake_model()
        model.llm = ReasoningLlama(["answer"], ["deliberating"])
        text, thinking, _, _ = generation.LlamaCppChat().generate(
            model, "", "hi", thinking="auto", max_tokens=8, temperature=0.0,
            top_p=1.0, seed=0)
        self.assertEqual((text, thinking), ("answer", "deliberating"))

    def test_chat_sends_a_connected_image_as_content_parts(self):
        if not HAVE_IMAGING:
            self.skipTest("numpy and Pillow are required")
        import numpy as np

        model, llm = fake_model(["ok"])
        model.vision = True
        generation.LlamaCppChat().generate(
            model, "sys", "what is this?", thinking="auto", max_tokens=8,
            temperature=0.0, top_p=1.0, seed=0,
            image=np.zeros((1, 8, 8, 3), dtype=np.float32))
        content = llm.calls[0]["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertEqual(content[-1], {"type": "text", "text": "what is this?"})

    def test_chat_without_an_image_stays_plain_text(self):
        model, llm = fake_model(["ok"])
        generation.LlamaCppChat().generate(
            model, "sys", "hello", thinking="auto", max_tokens=8, temperature=0.0,
            top_p=1.0, seed=0)
        self.assertEqual(llm.calls[0]["messages"][-1]["content"], "hello")

    def test_an_image_on_a_text_only_model_is_refused(self):
        model, _ = fake_model(["ok"])  # loaded without a projector
        with self.assertRaises(ValueError) as ctx:
            generation.LlamaCppChat().generate(
                model, "", "hi", thinking="auto", max_tokens=8, temperature=0.0,
                top_p=1.0, seed=0, image=object())
        self.assertIn("multimodal projector", str(ctx.exception))

    def test_random_seed_busts_the_comfyui_cache(self):
        self.assertNotEqual(generation.LlamaCppChat.IS_CHANGED(seed=-1),
                            generation.LlamaCppChat.IS_CHANGED(seed=-1))
        self.assertEqual(generation.LlamaCppChat.IS_CHANGED(seed=5), 5)


class TestModelCache(unittest.TestCase):
    def test_second_request_reuses_the_loaded_model(self):
        cache = backend.ModelCache()
        loads = []

        def factory():
            loads.append(1)
            return backend.LlamaModel(FakeLlama([]), ("a",), "a")

        first = cache.get_or_load(("a",), factory)
        second = cache.get_or_load(("a",), factory)
        self.assertIs(first, second)
        self.assertEqual(len(loads), 1)

    def test_a_different_key_evicts_and_frees_the_previous_model(self):
        cache = backend.ModelCache()
        first = cache.get_or_load(
            ("a",), lambda: backend.LlamaModel(FakeLlama([]), ("a",), "a"))
        llm = first.llm
        cache.get_or_load(("b",), lambda: backend.LlamaModel(FakeLlama([]), ("b",), "b"))
        self.assertTrue(llm.closed)
        self.assertIsNone(first.llm)

    def test_keep_loaded_zero_never_caches(self):
        cache = backend.ModelCache()
        model = cache.get_or_load(
            ("a",), lambda: backend.LlamaModel(FakeLlama([]), ("a",), "a"),
            keep_loaded=0)
        self.assertEqual(len(cache._entries), 0)
        self.assertIsNotNone(model.llm)  # still usable for this run

    def test_unload_node_passes_text_through(self):
        cache_model = backend.LlamaModel(FakeLlama([]), ("a",), "a")
        backend.MODEL_CACHE.get_or_load(("a",), lambda: cache_model)
        from comfyllama.nodes.loaders import LlamaCppUnload

        result = LlamaCppUnload().unload(cache_model, unload_all=False, text="keep me")
        self.assertEqual(result, ("keep me",))
        self.assertIsNone(cache_model.llm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
