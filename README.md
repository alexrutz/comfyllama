# comfyllama

llama.cpp nodes for ComfyUI — run GGUF language and vision models directly in
your graph. Write or rewrite prompts with a local LLM, caption images with a
multimodal model, and force structured JSON output, all without an external API.

Models can run **in the ComfyUI process** (via `llama-cpp-python`) or on a
**running `llama-server`** reached over HTTP. Both sets of nodes share the same
sampler, grammar and chat-history types, so you can swap between them without
rebuilding the graph.

## Nodes

### In-process (needs `llama-cpp-python`)

| Node | What it does |
| --- | --- |
| **Load LLM (llama.cpp)** | Loads a GGUF text model. GPU offload, context size, threads, chat template. |
| **Load Vision LLM (llama.cpp)** | Loads a GGUF model plus its `mmproj` projector (LLaVA, MiniCPM-V, moondream, …). |
| **Chat (llama.cpp)** | System prompt, user prompt and a thinking switch. Returns `text`, `thinking` and the updated history. |
| **Text Completion (llama.cpp)** | Raw completion, no chat template and no system prompt. |
| **Vision Chat (llama.cpp)** | Sends an `IMAGE` (or a whole batch) plus a system and user prompt to a multimodal model. |
| **Sampler Settings (llama.cpp)** | top_k, min_p, typical_p, repetition/presence/frequency penalties, Mirostat, stop sequences — each switched on individually. |
| **Grammar / JSON Output (llama.cpp)** | Constrains output to JSON, a JSON schema, or a custom GBNF grammar. |
| **Chat Message (llama.cpp)** | Builds a conversation for multi-turn chats or few-shot prompting. |
| **Messages to Text (llama.cpp)** | Renders a conversation as plain text. |
| **Prompt Template (llama.cpp)** | Fills `{a} {b} {c} {d}` placeholders from connected strings. |
| **Token Count (llama.cpp)** | Counts tokens with the model's own tokenizer. |
| **Preview Text (llama.cpp)** | Shows the generated text on the node and passes it through. |
| **Unload LLM (llama.cpp)** | Frees the model so the VRAM goes back to your diffusion models. |

### Remote (needs only a running `llama-server`)

| Node | What it does |
| --- | --- |
| **Connect to llama-server** | URL, timeout, model name and authentication (bearer token or user/password). Checks `/health` so a wrong URL fails immediately. |
| **Chat (llama-server)** | System prompt, user prompt and a thinking switch, via `/v1/chat/completions`. Returns `text`, `thinking` and the updated history. |
| **Vision Chat (llama-server)** | Uploads an `IMAGE` batch to a server started with `--mmproj`. |
| **Text Completion (llama-server)** | Raw completion via the native `/completion` endpoint, with prompt-cache reuse. |
| **Token Count (llama-server)** | Counts tokens via `/tokenize`. |
| **Server Info (llama-server)** | Model name, context size and available models as JSON. |

## Install

Clone into `ComfyUI/custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexrutz/comfyllama.git
```

The **llama-server nodes work right away** — they only use the standard library.
The in-process nodes additionally need `llama-cpp-python` **in the same Python
environment ComfyUI runs in**; pick the build that matches your hardware:

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

## Sampling

`max_tokens`, `temperature`, `top_p` and `seed` sit on the generation nodes and
are always sent. Everything else lives on the **Sampler Settings** node, where
each setting has its own switch:

- **Switch off** (the default): the parameter is left out of the request
  entirely, so whatever the model, llama-cpp-python or your `llama-server`
  command line sets stays in effect. Disabled rows are greyed out on the node.
- **Switch on**: the value next to it is sent.

So a Sampler Settings node with only `use_repeat_penalty` on changes the
repetition penalty and nothing else — it will not quietly pin `top_k` or
`min_p` to this node's defaults. With every switch off it behaves exactly like
not connecting the node at all. `mirostat_tau` and `mirostat_eta` are
meaningless without the mode, so those three share one switch.

## Prompts and reasoning models

All four chat nodes (in-process and remote) take a **`system`** prompt above the
`prompt` field; leave it empty to send no system message. The two Text
Completion nodes deliberately have none — they send the prompt verbatim with no
chat template, so there are no roles to put a system message in. Use a Chat node
if you want one. Every widget can also be driven from another node: drag a
connection onto it, or right-click the node → *Convert widget to input* on older
frontends.

Chat nodes have a **`thinking`** switch and a separate **`thinking`** output:

| | |
| --- | --- |
| `auto` | Sends nothing — the model's own default applies. |
| `on` / `off` | Requests thinking explicitly. Remote: sent as `chat_template_kwargs {"enable_thinking": …}`, which is what Qwen3-style templates read. In-process: appended to the prompt as `/think` or `/no_think`, because llama-cpp-python renders the GGUF's template without forwarding arguments. Models whose template ignores the switch keep their default. |

The chain of thought is always separated, whatever the switch is set to:

- `text` — the answer with the reasoning removed.
- `thinking` — the reasoning, from the server's `reasoning_content` field when
  llama-server runs with `--reasoning-format deepseek`, otherwise parsed out of
  `<think>` tags. Blocks the template opened itself and blocks cut off by
  `max_tokens` are both handled.
- `messages` — the history carries the answer only, so the reasoning is not fed
  back into the next turn.

To turn thinking off server-side regardless of the request, start llama-server
with `--reasoning-budget 0`.

## Using a running llama-server

Start the server yourself and point the **Connect to llama-server** node at it:

```bash
llama-server -m model.gguf --host 127.0.0.1 --port 8080 -ngl 99
# multimodal:
llama-server -m llava.gguf --mmproj mmproj.gguf --host 127.0.0.1 --port 8080
```

Worth doing when you want the LLM to stay loaded between runs, need it on
another machine, want several ComfyUI instances (or other tools) to share one
model, or simply want the model out of ComfyUI's process. The nodes speak both
the OpenAI-compatible `/v1/chat/completions` API and llama.cpp's native
`/completion`, so llama.cpp's samplers, GBNF grammars and JSON schemas all work.

Notes:

- The URL may include a trailing `/v1` — it is stripped. `model` can stay on
  `auto`, which asks the server what it has loaded.
- Requests to `localhost`/`127.0.0.1` deliberately bypass any `HTTP_PROXY` set
  in the environment.
- `timeout` is per request; raise it for long generations on slow hardware.
- Cancelling in ComfyUI aborts the stream immediately.

### Authentication

The connect node speaks two schemes, chosen with the `auth` widget:

| `auth` | Header sent |
| --- | --- |
| `auto` (default) | Basic when `username` is filled in, Bearer when only `api_key` is, nothing when neither. |
| `bearer` | `Authorization: Bearer <api_key>` — for llama-server started with `--api-key`. |
| `basic` | `Authorization: Basic <base64 user:password>` — for an nginx/Caddy/Traefik reverse proxy in front of llama-server. |
| `none` | Never sends credentials, even with the fields filled in. |

Forcing `bearer` or `basic` without its field filled in fails with a clear
error instead of silently sending an unauthenticated request. Credentials may
also be written straight into the URL (`http://user:pass@host:8080`); they are
stripped from the URL and used as basic auth. A 401/403 reply says whether
credentials were rejected or never sent.

**Keeping secrets out of the workflow file:** widget values are saved into
workflow JSON in plain text, and that JSON travels with shared graphs and gets
embedded in generated PNGs. All three credential fields therefore accept an
`env:NAME` indirection that reads the value from the environment ComfyUI runs
in:

```
api_key:  env:LLAMA_API_KEY
username: llama
password: env:LLAMA_PASSWORD
```

## Models

For the in-process nodes, put `.gguf` files into `ComfyUI/models/llm/`. The
folder is created on first start, and `models/LLM`, `models/gguf` and
`models/llama` are picked up too if you already use them. Multimodal projectors can live in `models/llm/` or
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
