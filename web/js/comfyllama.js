import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
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

// --- Polling a llama-server for its model list -----------------------------
// The list lives on the remote server, so it cannot come from INPUT_TYPES.
// The button asks the backend route, which makes the same /v1/models request
// the nodes make when they run.
const CONNECT_NODE = "LlamaServerConnect";
const MODEL_PICKER_NODES = new Set([
	CONNECT_NODE,
	"LlamaServerChat",
	"LlamaServerVisionChat",
	"LlamaServerComplete",
	"LlamaServerPresetChat",
]);

function readWidget(node, name, fallback = "") {
	const widget = node?.widgets?.find((w) => w.name === name);
	return widget === undefined ? fallback : widget.value;
}

// The URL and credentials live on the connect node, which for a generation
// node sits on the other end of its `server` input.
function findConnectNode(node) {
	if (node.type === CONNECT_NODE) {
		return node;
	}
	const slot = node.inputs?.findIndex((input) => input.name === "server");
	if (slot === undefined || slot < 0) {
		return null;
	}
	let origin = node.getInputNode?.(slot);
	// Step through reroute nodes, which pass the link straight through.
	for (let hops = 0; origin && origin.type !== CONNECT_NODE && hops < 8; hops++) {
		origin = origin.getInputNode?.(0);
	}
	return origin?.type === CONNECT_NODE ? origin : null;
}

async function pollModels(node) {
	const connect = findConnectNode(node);
	if (!connect) {
		return { models: [], error: "Connect this node to a 'Connect to llama-server' node first." };
	}
	try {
		const response = await api.fetchApi("/comfyllama/models", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				base_url: readWidget(connect, "base_url"),
				auth: readWidget(connect, "auth", "auto"),
				api_key: readWidget(connect, "api_key"),
				username: readWidget(connect, "username"),
				password: readWidget(connect, "password"),
				timeout: 15,
			}),
		});
		return await response.json();
	} catch (error) {
		return { models: [], error: String(error) };
	}
}

// Which widget a picked model should be written into.
function modelTargetWidget(node) {
	if (node.type !== "LlamaServerPresetChat") {
		return node.widgets?.find((w) => w.name === "model");
	}
	// On the preset node each preset has its own model; target the active one.
	const activeWidget = node.widgets?.find((w) => w.name === "active");
	const slotCount = Number(readWidget(node, "slot_count", 1)) || 1;
	const names = presetNames(node, Math.min(slotCount, MAX_PRESET_SLOTS));
	const index = names.indexOf(activeWidget?.value) + 1;
	return node.widgets?.find((w) => w.name === `model_${index || 1}`);
}

async function showModelMenu(node, event) {
	const target = modelTargetWidget(node);
	if (!target) {
		return;
	}
	const result = await pollModels(node);
	const entries = result.models?.length
		? ["auto", ...result.models]
		: [`⚠ ${result.error || "no models reported"}`];

	new LiteGraph.ContextMenu(entries, {
		event,
		title: result.models?.length ? `Models on ${result.base_url}` : "Could not list models",
		callback: (value) => {
			if (typeof value !== "string" || value.startsWith("⚠")) {
				return;
			}
			target.value = value;
			target.callback?.(value);
			node.setDirtyCanvas?.(true, true);
		},
	});
}

function addModelPicker(nodeType) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function () {
		onNodeCreated?.apply(this, arguments);
		this.addWidget(
			"button",
			"⟳ fetch models",
			null,
			(_value, _widget, node, _pos, event) => showModelMenu(node ?? this, event),
			// Never serialised: it would shift every saved widget value.
			{ serialize: false },
		);
	};
}

// --- Chat with Prompt Presets ---------------------------------------------
// The node carries a fixed number of system prompt slots. Only `slot_count` of
// them are in use, so the rest are hidden, and the `active` dropdown is filled
// from whatever the slots were renamed to.
const MAX_PRESET_SLOTS = 6;
const PASSTHROUGH = "passthrough";
const HIDDEN_TYPE = "comfyllama-hidden";

function widgetByName(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

function showWidget(widget, visible) {
	if (!widget) {
		return;
	}
	// Newer frontends honour `hidden` on their own; the type swap covers the
	// older ones that draw every widget unconditionally.
	widget.hidden = !visible;
	if (!visible) {
		if (widget.type !== HIDDEN_TYPE) {
			widget.originalType = widget.type;
			widget.originalComputeSize = widget.computeSize;
			widget.type = HIDDEN_TYPE;
			widget.computeSize = () => [0, -4];
		}
		if (widget.inputEl) {
			widget.inputEl.style.display = "none";
		}
	} else if (widget.type === HIDDEN_TYPE) {
		widget.type = widget.originalType;
		widget.computeSize = widget.originalComputeSize;
		if (widget.inputEl) {
			widget.inputEl.style.display = "";
		}
	}
}

function presetNames(node, slotCount) {
	const names = [];
	for (let index = 1; index <= slotCount; index++) {
		const widget = widgetByName(node, `name_${index}`);
		const name = String(widget?.value ?? "").trim();
		names.push(name || `Preset ${index}`);
	}
	return names;
}

function applyPresetState(node) {
	const slotCountWidget = widgetByName(node, "slot_count");
	const activeWidget = widgetByName(node, "active");
	if (!slotCountWidget || !activeWidget) {
		return;
	}
	const slotCount = Math.max(1, Math.min(Number(slotCountWidget.value) || 1, MAX_PRESET_SLOTS));
	const names = presetNames(node, slotCount);

	// Keep the dropdown in sync with the slot names.
	const options = [PASSTHROUGH, ...names];
	activeWidget.options = { ...(activeWidget.options ?? {}), values: options };
	if (!options.includes(activeWidget.value)) {
		activeWidget.value = PASSTHROUGH;
	}

	const activeIndex = names.indexOf(activeWidget.value) + 1; // 0 = passthrough

	for (let index = 1; index <= MAX_PRESET_SLOTS; index++) {
		const inUse = index <= slotCount;
		showWidget(widgetByName(node, `name_${index}`), inUse);
		showWidget(widgetByName(node, `system_${index}`), inUse);
		showWidget(widgetByName(node, `model_${index}`), inUse);

		// The matching extra input is only read while its slot is the active
		// one; say so in the label rather than dropping the connection.
		const input = node.inputs?.find((i) => i.name === `extra_${index}`);
		if (input) {
			input.label = index === activeIndex
				? `extra_${index}`
				: `extra_${index} (inactive)`;
		}
	}

	node.setSize([node.size[0], node.computeSize()[1]]);
	node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
	name: "comfyllama.modelPicker",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (MODEL_PICKER_NODES.has(nodeData.name)) {
			addModelPicker(nodeType);
		}
	},
});

app.registerExtension({
	name: "comfyllama.presetChat",
	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== "LlamaServerPresetChat") {
			return;
		}

		const watched = ["slot_count", "active"];
		for (let index = 1; index <= MAX_PRESET_SLOTS; index++) {
			watched.push(`name_${index}`);
		}

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);
			for (const name of watched) {
				const widget = widgetByName(this, name);
				if (!widget) {
					continue;
				}
				const original = widget.callback;
				widget.callback = (...args) => {
					const result = original?.apply(widget, args);
					applyPresetState(this);
					return result;
				};
			}
			applyPresetState(this);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function () {
			onConfigure?.apply(this, arguments);
			// The saved `active` value may name a renamed preset, so restore the
			// options before validating it.
			applyPresetState(this);
		};

		const onConnectionsChange = nodeType.prototype.onConnectionsChange;
		nodeType.prototype.onConnectionsChange = function () {
			onConnectionsChange?.apply(this, arguments);
			applyPresetState(this);
		};
	},
});

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
