"""FastAPI backend exposing the agent over HTTP for the frontend.

Not streaming (yet) -- the agent loop below runs to completion (including
all tool-call rounds) before returning, since restructuring the
tool-calling loop to yield partial tokens mid-round is real additional
work and streaming isn't needed to demonstrate the core RAG/agent
behavior. Noted as a follow-up in the README rather than silently skipped.
"""
import re

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import ask

# openai/gpt-oss-20b emits full-width-bracket citation markers (e.g.
# "【search_wiki 1】") from its training format; they're meaningless outside
# that context (we render our own Sources list), so strip them for display.
_CITATION_MARKER_RE = re.compile(r"【[^】]*】")


def _clean_answer(text: str) -> str:
    return _CITATION_MARKER_RE.sub("", text).strip()


# a router, not a full FastAPI app, so it can be mounted either on its own
# standalone app (local dev, below) or included into Gradio's own app on
# HF Spaces (see app.py at the repo root -- ZeroGPU's registration hook
# only fires on gr.Blocks.launch(), which our own uvicorn-served app never
# calls, so the HF deploy runs Gradio's app directly and borrows its routes
# rather than the other way around).
router = APIRouter()


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    search_trace: list[dict]
    rounds: int


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = ask(req.question)
    return ChatResponse(answer=_clean_answer(result["answer"] or ""), search_trace=result["search_trace"], rounds=result["rounds"])


@router.get("/health")
def health():
    return {"status": "ok"}


# standalone app for local dev: `uvicorn api:app --port 8000 --reload`
app = FastAPI(title="Elden Ring Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local portfolio project, not handling sensitive data
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
