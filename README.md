# comfyllama

llama.cpp nodes for ComfyUI — run GGUF language and vision models directly in
your graph. Write or rewrite prompts with a local LLM, caption images with a
multimodal model, and force structured JSON output, all without an external API.

## Nodes

| Node | What it does |
| --- | --- |
| **Load LLM (llama.cpp)** | Loads a GGUF text model. GPU offload, context size, threads, chat template. |
| **Load Vision LLM (llama.cpp)** | Loads a GGUF model plus its `mmproj` projector (LLaVA, MiniCPM-V, moondream, …). |
| **Chat (llama.cpp)** | Chat completion using the model's chat template. Returns the text and the updated history. |
| **Text Completion (llama.cpp)** | Raw completion, no chat template applied. |
| **Vision Chat (llama.cpp)** | Sends an `IMAGE` (or a whole batch) plus a prompt to a multimodal model. |
| **Sampler Settings (llama.cpp)** | top_k, min_p, typical_p, repetition/presence/frequency penalties, Mirostat, stop sequences. |
| **Grammar / JSON Output (llama.cpp)** | Constrains output to JSON, a JSON schema, or a custom GBNF grammar. |
| **Chat Message (llama.cpp)** | Builds a conversation for multi-turn chats or few-shot prompting. |
| **Messages to Text (llama.cpp)** | Renders a conversation as plain text. |
| **Prompt Template (llama.cpp)** | Fills `{a} {b} {c} {d}` placeholders from connected strings. |
| **Token Count (llama.cpp)** | Counts tokens with the model's own tokenizer. |
| **Preview Text (llama.cpp)** | Shows the generated text on the node and passes it through. |
| **Unload LLM (llama.cpp)** | Frees the model so the VRAM goes back to your diffusion models. |

## Install

Clone into `ComfyUI/custom_nodes/` and install `llama-cpp-python` **into the same
Python environment ComfyUI runs in**:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexrutz/comfyllama.git
```

Then pick the build that matches your hardware:

```bash
# CPU
pip install llama-cpp-python

# NVIDIA / CUDA 12.4 (prebuilt wheels)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Apple Silicon / Metal
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

On the Windows portable build, prefix the command with the bundled interpreter:

```bat
python_embeded\python.exe -m pip install llama-cpp-python
```

Restart ComfyUI afterwards. If the binding is missing, the nodes still load and
tell you exactly what to install when you run them.

## Models

Put `.gguf` files into `ComfyUI/models/llm/`. The folder is created on first
start, and `models/LLM`, `models/gguf` and `models/llama` are picked up too if
you already use them. Multimodal projectors can live in `models/llm/` or
`models/mmproj/`. Both folder keys work in `extra_model_paths.yaml`:

```yaml
comfyui:
  base_path: /data/ComfyUI
  llm: models/llm
  mmproj: models/mmproj
```

Every loader also has a `model_path_override` field if you want to point at a
file somewhere else entirely.

Good starting points: any `*-instruct-*Q4_K_M.gguf` model in the 3–8B range for
text, and LLaVA 1.6 or MiniCPM-V 2.6 (model **plus** its `mmproj-*.gguf`) for
image captioning.

## Usage notes

- **VRAM.** Set `n_gpu_layers` to `-1` to offload everything, `0` for CPU only.
  Before a GPU load the pack unloads ComfyUI's diffusion models, so a text step
  in front of your image generation does not fight over VRAM. `keep_loaded`
  controls how many models stay resident; put an **Unload** node after the last
  node that uses the model to release it immediately.
- **Caching.** Re-running a graph reuses the loaded model as long as the loader
  settings do not change. A seed of `-1` re-runs generation every time; a fixed
  seed lets ComfyUI cache the result.
- **Cancelling.** Generation streams token by token, so the ComfyUI cancel
  button stops it right away and the progress bar tracks `max_tokens`.
- **Structured output.** Connect the **Grammar** node to force valid JSON, JSON
  matching a schema, or any GBNF grammar — handy for feeding parsed values into
  the rest of a workflow.
- **Chat templates.** Leave `chat_format` on `auto` unless the GGUF has no
  embedded template; then pick the matching one (`chatml`, `llama-3`, …).

Example graphs are in [`example_workflows/`](example_workflows) — drag a JSON
file onto the ComfyUI canvas.

## Tests

The suite stubs ComfyUI, so it runs anywhere:

```bash
python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
