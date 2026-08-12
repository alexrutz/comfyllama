"""Tests for the nodes that are not part of the llama.cpp chain.

The preset chat node is exercised against the same stub llama-server as the
other remote nodes.
"""

from __future__ import annotations

import contextlib
import re
import sys
import unittest

# Importing test_nodes installs the ComfyUI stubs before comfyllama is loaded.
from test_nodes import HAVE_IMAGING  # noqa: F401
from test_server_nodes import ServerTestCase

from comfyllama.backend import decode_escapes
from comfyllama.nodes.latent import (LATENT_FORMATS, RATIO_LABELS,
                                     EmptyLatentByAspectRatio, resolve_dimensions)
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
        self.assertEqual(latent["samples"].shape[1], LATENT_FORMATS[label].channels)

    def test_an_unknown_format_falls_back_instead_of_failing(self):
        with fake_torch():
            latent, _, _ = EmptyLatentByAspectRatio().generate(
                "1:1", 1.0, 8, 1, "something else")
        self.assertEqual(latent["samples"].shape[1], 4)

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


class TestKrea2Format(unittest.TestCase):
    """Krea 2 decodes through a 16-channel f8 autoencoder, patchified 2x2."""

    LABEL = "Krea 2 (16 channels)"

    def test_the_format_is_offered(self):
        self.assertIn(self.LABEL, LATENT_FORMATS)
        spec = LATENT_FORMATS[self.LABEL]
        self.assertEqual((spec.channels, spec.downscale), (16, 8))

    def test_latent_shape_matches_a_16_channel_f8_autoencoder(self):
        with fake_torch():
            latent, width, height = EmptyLatentByAspectRatio().generate(
                "2:3", 1.0, 8, 1, self.LABEL)
        self.assertEqual(latent["samples"].shape, (1, 16, height // 8, width // 8))

    def test_edges_stay_on_a_16_pixel_grid(self):
        for ratio in RATIO_LABELS:
            with self.subTest(ratio=ratio):
                width, height = resolve_dimensions(
                    ratio, 1.5, 8, LATENT_FORMATS[self.LABEL].minimum_multiple)
                self.assertEqual((width % 16, height % 16), (0, 0))

    def test_the_grid_floor_is_applied_even_at_divisible_by_8(self):
        spec = LATENT_FORMATS[self.LABEL]
        # 2:3 at 1 MP is 840x1256 on an 8 grid, which is not a multiple of 16.
        self.assertEqual(resolve_dimensions("2:3", 1.0, 8), (840, 1256))
        self.assertEqual(resolve_dimensions("2:3", 1.0, 8, spec.minimum_multiple),
                         (832, 1248))

    def test_a_coarser_choice_still_wins(self):
        spec = LATENT_FORMATS[self.LABEL]
        width, height = resolve_dimensions("2:3", 1.0, 64, spec.minimum_multiple)
        self.assertEqual((width % 64, height % 64), (0, 0))

    def test_reported_size_matches_the_latent_that_was_built(self):
        with fake_torch() as torch_stub:
            _, width, height = EmptyLatentByAspectRatio().generate(
                "16:9", 2.0, 8, 1, self.LABEL)
        shape, _ = torch_stub.calls[0]
        self.assertEqual((shape[3] * 8, shape[2] * 8), (width, height))
        self.assertEqual((width % 16, height % 16), (0, 0))


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
            values[f"model_{index}"] = ""
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

    def test_each_preset_can_name_its_own_model(self):
        # The point of a router: a small model for one job, a big one for another.
        self.NODE().generate(
            server=self.connect(),
            **self.widgets(active="Preset 2", model_1="small", model_2="big-model"))
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["model"], "big-model")

    def test_a_preset_without_a_model_falls_back_to_the_connection(self):
        self.NODE().generate(
            server=self.connect(model="stub-model"),
            **self.widgets(active="Preset 2"))
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["model"], "stub-model")

    def test_no_per_slot_widget_is_required(self):
        """Regression: 'Required input is missing (model_5)' over the API.

        The web extension hides the slots above slot_count, and a hidden
        widget does not survive an "export (API)". Declaring any of them
        required makes such a workflow fail validation before it runs.
        """
        required = self.NODE.INPUT_TYPES()["required"]
        hideable = re.compile(r"^(name|system|model|extra)_\d+$")
        offenders = [key for key in required if hideable.match(key)]
        self.assertEqual(offenders, [])

    def test_an_api_payload_missing_the_hidden_slots_still_runs(self):
        # What ComfyUI sends when slots 4-6 are hidden: they are simply absent.
        payload = {
            "prompt": "a lighthouse", "active": "Preset 2", "slot_count": 3,
            "thinking": "auto", "max_tokens": 64, "temperature": 0.2,
            "top_p": 0.9, "seed": 0,
        }
        for index in range(1, 4):
            payload[f"name_{index}"] = f"Preset {index}"
            payload[f"system_{index}"] = f"system {index}"

        text, _, active = self.NODE().generate(server=self.connect(), **payload)
        self.assertEqual(text, "Hello world")
        self.assertEqual(active, "Preset 2")
        sent = self.requests_to("/v1/chat/completions")[0]["payload"]
        self.assertEqual(sent["messages"][0]["content"], "system 2")
        # model_2 was absent too, so it falls all the way back to the model the
        # server reports.
        self.assertEqual(sent["model"], "stub-model")

    def test_the_bare_minimum_payload_runs(self):
        # Only the required inputs, every optional one absent.
        text, _, active = self.NODE().generate(
            server=None, prompt="straight through", active="passthrough",
            slot_count=3, thinking="auto", max_tokens=64, temperature=0.2,
            top_p=0.9, seed=0)
        self.assertEqual((text, active), ("straight through", "passthrough"))

    @unittest.skipUnless(HAVE_IMAGING, "numpy and Pillow are required")
    def test_a_connected_image_reaches_the_active_preset(self):
        import numpy as np

        self.NODE().generate(
            server=self.connect(),
            image=np.zeros((1, 8, 8, 3), dtype=np.float32),
            **self.widgets(active="Preset 2", extra_2="in ink"))
        content = self.requests_to("/v1/chat/completions")[0]["payload"]["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "image_url")
        # The extra prompt is still appended, inside the text part.
        self.assertEqual(content[-1]["text"], "a lighthouse\n\nin ink")

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
        self.assertTrue(inputs["optional"]["image"][1]["lazy"])
        for index in range(1, MAX_SLOTS + 1):
            self.assertTrue(inputs["optional"][f"extra_{index}"][1]["lazy"])

    def test_passthrough_does_not_pull_in_the_image(self):
        self.assertEqual(
            self.node.check_lazy_status(active="passthrough", slot_count=3,
                                        server=None, image=None, **self.slots),
            [])

    def test_an_active_preset_pulls_in_the_image(self):
        needed = self.node.check_lazy_status(
            active="Preset 1", slot_count=3, server=None, extra_1=None, image=None,
            **self.slots)
        self.assertEqual(needed, ["server", "extra_1", "image"])

    def test_an_unconnected_image_is_never_requested(self):
        # An input that is not wired up is absent from the kwargs entirely.
        needed = self.node.check_lazy_status(
            active="Preset 1", slot_count=3, server=object(), **self.slots)
        self.assertEqual(needed, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
