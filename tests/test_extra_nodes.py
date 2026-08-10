"""Tests for the nodes that are not part of the llama.cpp chain.

The preset chat node is exercised against the same stub llama-server as the
other remote nodes.
"""

from __future__ import annotations

import contextlib
import sys
import unittest

# Importing test_nodes installs the ComfyUI stubs before comfyllama is loaded.
from test_nodes import HAVE_IMAGING  # noqa: F401
from test_server_nodes import ServerTestCase

from comfyllama.backend import decode_escapes
from comfyllama.nodes.latent import (LATENT_FORMATS, RATIO_LABELS, EmptyLatentByAspectRatio,
                                     resolve_dimensions)
from comfyllama.nodes.presets import (MAX_SLOTS, LlamaServerPresetChat, join_prompt,
                                      resolve_slot, slot_names)

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class TestAspectRatios(unittest.TestCase):
    def test_the_two_requested_ratios_lead_the_list(self):
        self.assertEqual(RATIO_LABELS[:2], ["1:1", "2:3"])

    def test_square_megapixel_is_the_familiar_size(self):
        self.assertEqual(resolve_dimensions("1:1", 1.0), (1024, 1024))
        self.assertEqual(resolve_dimensions("1:1", 4.0), (2048, 2048))

    def test_ratio_is_honoured_and_area_lands_near_the_target(self):
        for label in RATIO_LABELS:
            with self.subTest(ratio=label):
                width, height = resolve_dimensions(label, 1.0)
                wanted_w, wanted_h = (int(part) for part in label.split(":"))
                self.assertAlmostEqual(width / height, wanted_w / wanted_h, delta=0.02)
                megapixels = (width * height) / (1024 * 1024)
                self.assertAlmostEqual(megapixels, 1.0, delta=0.02)

    def test_portrait_and_landscape_are_mirror_images(self):
        self.assertEqual(resolve_dimensions("2:3", 1.0),
                         tuple(reversed(resolve_dimensions("3:2", 1.0))))

    def test_both_edges_are_multiples_of_the_divisor(self):
        for divisor in (8, 16, 32, 64):
            with self.subTest(divisible_by=divisor):
                width, height = resolve_dimensions("9:16", 1.0, divisor)
                self.assertEqual(width % divisor, 0)
                self.assertEqual(height % divisor, 0)

    def test_tiny_requests_still_produce_a_usable_latent(self):
        # 0.01 MP is ~102px square, rounded up to the nearest multiple of 64.
        self.assertEqual(resolve_dimensions("1:1", 0.01, 64), (128, 128))
        self.assertEqual(resolve_dimensions("1:1", 0.0), (8, 8))

    def test_unknown_ratio_lists_the_valid_ones(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_dimensions("7:5", 1.0)
        self.assertIn("1:1", str(ctx.exception))


class FakeTensor:
    def __init__(self, shape, device):
        self.shape = tuple(shape)
        self.device = device


class FakeTorch:
    """Stands in for torch so the node's shape logic is testable anywhere."""

    def __init__(self):
        self.calls = []

    def zeros(self, shape, device=None):
        self.calls.append((tuple(shape), device))
        return FakeTensor(shape, device)


@contextlib.contextmanager
def fake_torch():
    previous = sys.modules.get("torch")
    module = FakeTorch()
    sys.modules["torch"] = module
    try:
        yield module
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


class TestEmptyLatentNode(unittest.TestCase):
    def test_latent_shape_follows_the_ratio_and_batch(self):
        with fake_torch():
            latent, width, height = EmptyLatentByAspectRatio().generate(
                "2:3", 1.0, 8, 2)
        self.assertEqual((width, height), resolve_dimensions("2:3", 1.0))
        self.assertEqual(latent["samples"].shape, (2, 4, height // 8, width // 8))

    def test_sixteen_channel_format(self):
        label = "SD3 / Flux (16 channels)"
        with fake_torch():
            latent, _, _ = EmptyLatentByAspectRatio().generate("1:1", 1.0, 8, 1, label)
        self.assertEqual(latent["samples"].shape[1], LATENT_FORMATS[label])

    def test_every_ratio_produces_a_whole_number_latent(self):
        with fake_torch() as torch_stub:
            for label in RATIO_LABELS:
                EmptyLatentByAspectRatio().generate(label, 1.0, 8, 1)
        for shape, _ in torch_stub.calls:
            self.assertTrue(all(isinstance(dimension, int) and dimension > 0
                                for dimension in shape), shape)

    @unittest.skipUnless(HAVE_TORCH, "torch is required")
    def test_the_real_tensor_is_zeroed(self):
        latent, _, _ = EmptyLatentByAspectRatio().generate("1:1", 1.0, 8, 1)
        self.assertEqual(float(latent["samples"].abs().sum()), 0.0)


class TestPresetSelection(unittest.TestCase):
    def setUp(self):
        self.names = [f"Preset {index}" for index in range(1, MAX_SLOTS + 1)]

    def test_passthrough_and_its_aliases(self):
        for value in ("passthrough", "PASSTHROUGH", " none ", "off", ""):
            with self.subTest(value=value):
                self.assertIsNone(resolve_slot(value, self.names, 3))

    def test_selection_by_name_is_case_insensitive(self):
        names = ["Enhance", "Translate", "Summarise"]
        names += self.names[3:]
        self.assertEqual(resolve_slot("translate", names, 3), 2)

    def test_selection_falls_back_to_a_trailing_number(self):
        # Keeps the node usable when the web extension has not renamed the
        # dropdown entries.
        self.assertEqual(resolve_slot("Preset 3", self.names, 3), 3)
        self.assertEqual(resolve_slot("2", self.names, 3), 2)

    def test_slots_beyond_the_count_are_not_selectable(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_slot("Preset 5", self.names, 3)
        available = str(ctx.exception).split("Available:")[1]
        self.assertIn("Preset 3", available)
        self.assertNotIn("Preset 5", available)

    def test_unknown_name_lists_what_is_available(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_slot("nope", self.names, 2)
        message = str(ctx.exception)
        self.assertIn("passthrough", message)
        self.assertIn("Preset 1", message)

    def test_slot_names_fall_back_to_the_defaults(self):
        names = slot_names({"name_1": "Custom", "name_2": ""})
        self.assertEqual(names[0], "Custom")
        self.assertEqual(names[1], "Preset 2")
        self.assertEqual(len(names), MAX_SLOTS)


class TestExtraPromptJoining(unittest.TestCase):
    def test_extra_is_appended_with_the_decoded_separator(self):
        self.assertEqual(join_prompt("a cat", "in watercolour", "\\n\\n"),
                         "a cat\n\nin watercolour")
        self.assertEqual(join_prompt("a cat", "sharp", " | "), "a cat | sharp")

    def test_a_missing_side_means_no_separator(self):
        self.assertEqual(join_prompt("a cat", "", "\\n\\n"), "a cat")
        self.assertEqual(join_prompt("a cat", None, "\\n\\n"), "a cat")
        self.assertEqual(join_prompt("", "extra", "\\n\\n"), "extra")
        self.assertEqual(join_prompt("  ", None, "\\n\\n"), "")

    def test_escape_decoding_keeps_non_ascii_intact(self):
        self.assertEqual(decode_escapes("caf\\u00e9\\n"), "café\n")
        self.assertEqual(decode_escapes("café"), "café")
        self.assertEqual(decode_escapes(""), "")


class TestPresetChatNode(ServerTestCase):
    NODE = LlamaServerPresetChat

    def widgets(self, **overrides):
        values = {
            "prompt": "a lighthouse",
            "active": "passthrough",
            "slot_count": 3,
            "thinking": "auto",
            "max_tokens": 64,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 0,
        }
        for index in range(1, MAX_SLOTS + 1):
            values[f"name_{index}"] = f"Preset {index}"
            values[f"system_{index}"] = f"system {index}"
        values.update(overrides)
        return values

    def test_passthrough_returns_the_prompt_without_calling_the_server(self):
        text, thinking, active = self.NODE().generate(
            server=None, **self.widgets(prompt="unchanged"))
        self.assertEqual((text, thinking, active), ("unchanged", "", "passthrough"))
        self.assertEqual(self.stub.state["requests"], [])

    def test_active_preset_supplies_its_own_system_prompt(self):
        text, _, active = self.NODE().generate(
            server=self.connect(), **self.widgets(active="Preset 2"))
        self.assertEqual(text, "Hello world")
        self.assertEqual(active, "Preset 2")
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["messages"][0],
                         {"role": "system", "content": "system 2"})
        self.assertEqual(sent["messages"][1]["content"], "a lighthouse")

    def test_renamed_preset_is_selectable_by_its_new_name(self):
        _, _, active = self.NODE().generate(
            server=self.connect(),
            **self.widgets(active="Enhance", name_2="Enhance"))
        self.assertEqual(active, "Enhance")
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["messages"][0]["content"], "system 2")

    def test_the_active_slots_extra_prompt_is_appended(self):
        self.NODE().generate(
            server=self.connect(),
            **self.widgets(active="Preset 2", extra_1="ignored", extra_2="in ink"))
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["messages"][1]["content"], "a lighthouse\n\nin ink")

    def test_an_inactive_slots_extra_prompt_is_ignored(self):
        self.NODE().generate(
            server=self.connect(),
            **self.widgets(active="Preset 1", extra_2="must not appear"))
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertNotIn("must not appear", sent["messages"][1]["content"])

    def test_a_custom_separator_is_used(self):
        self.NODE().generate(
            server=self.connect(),
            **self.widgets(active="Preset 1", extra_1="b", extra_separator=" -- "))
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["messages"][1]["content"], "a lighthouse -- b")

    def test_thinking_is_split_out_like_the_other_chat_nodes(self):
        self.stub.state["pieces"] = ["<think>hmm</think>", "Answer."]
        text, thinking, _ = self.NODE().generate(
            server=self.connect(), **self.widgets(active="Preset 1"))
        self.assertEqual((text, thinking), ("Answer.", "hmm"))

    def test_a_missing_connection_is_reported_clearly(self):
        with self.assertRaises(ValueError) as ctx:
            self.NODE().generate(server=None, **self.widgets(active="Preset 1"))
        self.assertIn("passthrough", str(ctx.exception))


class TestPresetLazyEvaluation(unittest.TestCase):
    """The lazy inputs are what keep inactive branches from running."""

    def setUp(self):
        self.node = LlamaServerPresetChat()
        self.slots = {f"name_{index}": f"Preset {index}"
                      for index in range(1, MAX_SLOTS + 1)}

    def test_passthrough_requests_nothing_at_all(self):
        self.assertEqual(
            self.node.check_lazy_status(active="passthrough", slot_count=3,
                                        server=None, extra_1=None, **self.slots),
            [])

    def test_only_the_active_slots_extra_input_is_requested(self):
        needed = self.node.check_lazy_status(
            active="Preset 2", slot_count=3, server=None,
            extra_1=None, extra_2=None, extra_3=None, **self.slots)
        self.assertEqual(needed, ["server", "extra_2"])

    def test_nothing_is_requested_twice(self):
        needed = self.node.check_lazy_status(
            active="Preset 2", slot_count=3, server=object(), extra_2="already",
            **self.slots)
        self.assertEqual(needed, [])

    def test_an_unresolvable_selection_defers_to_generate(self):
        # generate() raises the readable error; check_lazy_status must not.
        self.assertEqual(
            self.node.check_lazy_status(active="nope", slot_count=3, **self.slots),
            [])

    def test_the_lazy_inputs_are_declared(self):
        inputs = LlamaServerPresetChat.INPUT_TYPES()
        self.assertTrue(inputs["required"]["server"][1]["lazy"])
        for index in range(1, MAX_SLOTS + 1):
            self.assertTrue(inputs["optional"][f"extra_{index}"][1]["lazy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
