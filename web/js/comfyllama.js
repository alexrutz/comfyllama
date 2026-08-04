import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Nodes that return {"ui": {"text": [...]}} get a read-only textarea widget so
// the generated text is visible in the graph without opening the console.
const TEXT_OUTPUT_NODES = new Set(["LlamaCppPreviewText"]);

function setDisplayText(node, value) {
	const text = Array.isArray(value) ? value.join("") : String(value ?? "");
	let widget = node.widgets?.find((w) => w.name === "llamacpp_output");

	if (!widget) {
		widget = ComfyWidgets["STRING"](
			node,
			"llamacpp_output",
			["STRING", { multiline: true }],
			app,
		).widget;
		widget.inputEl.readOnly = true;
		widget.inputEl.style.opacity = 0.75;
		widget.serializeValue = () => undefined; // keep it out of the workflow json
	}

	widget.value = text;
	// Grow the node once so short captions are readable straight away.
	const size = node.computeSize();
	node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
	app.graph.setDirtyCanvas(true, false);
}

app.registerExtension({
	name: "comfyllama.textPreview",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!TEXT_OUTPUT_NODES.has(nodeData.name)) {
			return;
		}

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			if (message?.text !== undefined) {
				setDisplayText(this, message.text);
			}
		};
	},
});

// Each sampler setting has a switch in front of it; the value widgets it
// controls are greyed out while the switch is off, because they are then left
// out of the request entirely.
const SAMPLING_SWITCHES = {
	use_top_k: ["top_k"],
	use_min_p: ["min_p"],
	use_typical_p: ["typical_p"],
	use_repeat_penalty: ["repeat_penalty"],
	use_presence_penalty: ["presence_penalty"],
	use_frequency_penalty: ["frequency_penalty"],
	use_mirostat: ["mirostat_mode", "mirostat_tau", "mirostat_eta"],
	use_stop_sequences: ["stop_sequences"],
};

function applySamplingSwitches(node) {
	for (const [switchName, targets] of Object.entries(SAMPLING_SWITCHES)) {
		const toggle = node.widgets?.find((w) => w.name === switchName);
		if (!toggle) {
			continue;
		}
		for (const name of targets) {
			const widget = node.widgets.find((w) => w.name === name);
			if (!widget) {
				continue;
			}
			widget.disabled = !toggle.value;
			if (widget.inputEl) {
				widget.inputEl.style.opacity = toggle.value ? "" : "0.4";
			}
		}
	}
	node.setDirtyCanvas?.(true, false);
}

app.registerExtension({
	name: "comfyllama.samplingSwitches",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "LlamaCppSampling") {
			return;
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			for (const switchName of Object.keys(SAMPLING_SWITCHES)) {
				const toggle = this.widgets?.find((w) => w.name === switchName);
				if (!toggle) {
					continue;
				}
				const original = toggle.callback;
				toggle.callback = (...args) => {
					const result = original?.apply(toggle, args);
					applySamplingSwitches(this);
					return result;
				};
			}
			applySamplingSwitches(this);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			onConfigure?.apply(this, arguments);
			applySamplingSwitches(this);
		};
	},
});
