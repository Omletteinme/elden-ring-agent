"""HF Spaces entry point (Gradio SDK).

Docker SDK on Hugging Face Spaces requires a paid plan on this account;
Gradio SDK is free. Gradio apps are FastAPI apps under the hood
(gr.mount_gradio_app returns the combined app), so this mounts our real
/chat and /health API (src/api.py, unchanged) alongside a minimal
informational Gradio page -- the API contract the React frontend expects
doesn't change at all, this file only exists to satisfy HF's "must be a
Gradio app" requirement for the free tier.
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from api import app as fastapi_app  # noqa: E402

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
