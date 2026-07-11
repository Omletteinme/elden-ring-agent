"""HF Spaces entry point (Gradio SDK).

Docker SDK on Hugging Face Spaces requires a paid plan on this account;
CPU Basic hardware does too (on this account, as of this deploy). ZeroGPU
is the free tier that's actually available, but it requires at least one
function decorated with @spaces.GPU to exist at startup -- without one,
the platform's ZeroGPU wrapper fails to initialize and reports "No
@spaces.GPU function detected", which also produced a port 7860
double-bind against our own uvicorn.run() below.

_unused_gpu_placeholder exists ONLY to satisfy that startup check. It is
never called anywhere in the real request path -- this app is entirely
CPU-bound (local sentence-transformers embeddings; the actual LLM calls
go to Groq's remote API, not a local GPU) and doesn't need or use a GPU
at all. This is a standard, documented pattern for CPU-only apps that
want to run on ZeroGPU hardware for the free allocation.

Gradio SDK is free (once ZeroGPU is satisfied). Gradio apps are FastAPI
apps under the hood (gr.mount_gradio_app returns the combined app), so
this mounts our real /chat and /health API (src/api.py, unchanged)
alongside a minimal informational Gradio page -- the API contract the
React frontend expects doesn't change at all, this file only exists to
satisfy HF's "must be a Gradio app" requirement for the free tier.
"""
import sys
from pathlib import Path

import gradio as gr
import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from api import app as fastapi_app  # noqa: E402


@spaces.GPU
def _unused_gpu_placeholder():
    pass

with gr.Blocks(title="Elden Ring Agent") as demo:
    gr.Markdown(
        """
        # Elden Ring Agent -- backend

        This Space hosts the API for the Elden Ring Agent project.
        It isn't meant to be used from this page -- the chat UI is a
        separate React frontend that calls this Space's `/chat` endpoint.

        - `GET /health`
        - `POST /chat` -- `{"question": "..."}`

        Source: https://github.com/Omletteinme/elden-ring-agent
        """
    )

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
