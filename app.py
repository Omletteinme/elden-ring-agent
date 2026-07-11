"""HF Spaces entry point (Gradio SDK).

Docker SDK and CPU Basic hardware both require a paid plan on this
account; ZeroGPU is the only free tier available for Gradio SDK Spaces
here. ZeroGPU's function-registration hook is implemented by monkey-
patching gr.Blocks.launch() itself (spaces/zero/gradio.py's `one_launch`
wraps gr.Blocks.launch to run a registration task before the real
launch) -- it only fires if .launch() is actually called.

Three earlier attempts, in order:
  1. gr.mount_gradio_app() + manual uvicorn.run() (Gradio's own documented
     pattern for embedding a UI in a larger FastAPI app) never calls
     .launch() at all, so the hook never fires -- failed with "No
     @spaces.GPU function detected" (confirmed not a stale build by
     checking the Space's Files tab and forcing a manual restart).
  2. Same, but with the @spaces.GPU function properly wired to a Gradio
     event handler instead of standalone -- same failure, confirming the
     issue was .launch() never being called, not the decorator wiring.
  3. Called demo.launch() for real and attached our routes onto
     demo.app (gradio.routes.App, a FastAPI subclass) -- the hook fired,
     but Gradio's own catch-all route (registered at Blocks construction,
     serving its UI/assets at "/") intercepted /health and /chat before
     they reached our handlers, and reordering demo.app.routes didn't
     fix it either (FastAPI's include_router() wraps included routes in
     an internal object, not flat matchable Route entries the way a
     naive list-reorder assumes).

This version sidesteps the routing fight entirely: Gradio launches on a
throwaway internal-only port purely to trigger the ZeroGPU registration
hook (confirmed via spaces/zero/gradio.py that the hook is an out-of-
band call, not tied to which port Gradio's own server binds), and our
real, completely unmodified API (src/api.py's `app`, identical to what
already worked in local Docker testing) serves the actual exposed port.
The two never share a routing table, so there's nothing left to conflict.

/chat 500'd on the first live deploy of this version even though /health
worked: the Docker deploy path (abandoned for the ZeroGPU/Gradio path
above) had an explicit `RUN python index.py` build step that rebuilt the
vector store + BM25 index from the committed chunks.jsonl; this Gradio
SDK path has no equivalent build phase, so data/chroma/ and bm25.pkl
(both gitignored -- regeneratable, not committed) never got created at
all, and retrieval.py's first real query crashed trying to open them.
Building them at import time here, guarded so a restart doesn't rebuild
unnecessarily.
"""
import sys
from pathlib import Path

import gradio as gr
import spaces
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import index as index_builder  # noqa: E402
from api import app as fastapi_app  # noqa: E402

if not index_builder.CHROMA_DIR.exists() or not index_builder.BM25_PATH.exists():
    print("Vector/keyword index not found -- building from committed chunks.jsonl...")
    index_builder.main()
else:
    print("Vector/keyword index already present, skipping build.")

EXTERNAL_PORT = 7860
GRADIO_INTERNAL_PORT = 7861


@spaces.GPU
def _unused_gpu_placeholder():
    return "unused"


with gr.Blocks(title="Elden Ring Agent (internal)") as demo:
    gr.Markdown("Internal Gradio instance -- exists only to satisfy ZeroGPU's registration hook, not user-facing.")
    with gr.Row(visible=False):
        _gpu_trigger = gr.Button()
        _gpu_output = gr.Textbox()
        _gpu_trigger.click(fn=_unused_gpu_placeholder, outputs=_gpu_output)


def _run_gradio_for_zerogpu_registration():
    demo.launch(
        server_name="127.0.0.1",
        server_port=GRADIO_INTERNAL_PORT,
        prevent_thread_lock=True,
        quiet=True,
    )


if __name__ == "__main__":
    _run_gradio_for_zerogpu_registration()
    uvicorn.run(fastapi_app, host="0.0.0.0", port=EXTERNAL_PORT)
