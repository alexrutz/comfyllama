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
