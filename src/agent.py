"""Agentic tool-use loop over the Elden Ring wiki corpus.

Unlike a static RAG pipeline (always retrieve top-k, then generate), the
LLM here decides whether it needs to search at all, what to search for,
and can call search_wiki multiple times -- e.g. for "how has X changed
across patches" questions that need facts pulled from more than one
lookup. This is the "agentic RAG" pattern: retrieval is a tool the model
chooses to use, not a fixed step before generation.

Every claim in the final answer must trace back to a search_wiki result;
the system prompt enforces this so faithfulness (see eval/) is measurable.
"""
import json
import os
import time

from dotenv import load_dotenv
from groq import BadRequestError, Groq

from retrieval import search

load_dotenv()

MODEL = "openai/gpt-oss-20b"
MAX_TOOL_ROUNDS = 5

SYSTEM_PROMPT = """You are an assistant that answers questions about Elden Ring using ONLY \
information retrieved via the search_wiki tool -- never your own training-data memory of the \
game, since it may be incomplete or outdated relative to the indexed wiki.

The indexed corpus ONLY covers: Weapons, Bosses, Talismans, Sorceries, and Incantations. It \
does NOT include Armor, Ashes of War, NPCs, Locations, or Classes. If a question is about a \
topic outside this list, say directly that it isn't in the indexed corpus -- do not search \
for it at all.

Rules:
- Always call search_wiki before answering a factual question, even if you think you know \
the answer -- but only for topics within the corpus's scope above.
- If a question requires comparing things or spans multiple topics (e.g. "how has X changed", \
"compare X and Y"), call search_wiki multiple times -- once per topic/sub-question -- rather \
than guessing from a single search.
- If your first 1-2 searches don't return relevant results, STOP searching -- do not keep \
retrying rephrased versions of the same query. Tell the user this information doesn't appear \
to be in the indexed corpus instead.
- If search results don't contain the answer, say so explicitly instead of filling the gap \
from memory.
- Never claim two differently-named things are the same, related, or alternate names for each \
other unless a search result explicitly states that connection. A similar-sounding name is NOT \
evidence of a connection -- if you can't find the exact name asked about, say it isn't in the \
corpus rather than substituting the closest match you did find.
- Cite your sources: after each claim, note which page it came from (the tool results include \
titles and URLs). End your answer with a "Sources:" list of the page titles and URLs you used.
- Be concise and factual -- this is a game-mechanics reference tool, not a conversation."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Search the Elden Ring wiki corpus (weapons, bosses, talismans, sorceries, "
                "incantations) for information relevant to a query. Returns the top matching "
                "passages with their source page title and URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query, e.g. an item/boss name plus what you want to know about it.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


TOOL_CALL_RETRY_ATTEMPTS = 3


def _create_completion(client: Groq, **kwargs):
    """Wraps chat.completions.create with retries for tool_use_failed.

    Found via eval: openai/gpt-oss-20b occasionally leaks an internal
    format token into the generated tool name (e.g.
    "search_wiki<|channel|>commentary"), which Groq's tool-call validator
    rejects outright as a 400. This is a sampling-level glitch, not a
    deterministic one -- retrying the same request with fresh sampling
    succeeds the great majority of the time, so we retry a few times
    before giving up rather than failing the whole question on one bad
    generation.
    """
    last_error = None
    for attempt in range(TOOL_CALL_RETRY_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise
            last_error = e
            time.sleep(0.5 * (attempt + 1))
    raise last_error


def _run_search_wiki(query: str) -> str:
    results = search(query, k=5)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"[{r['title']} — {r['section']}] {r['text']}\nSource: {r['url']}")
    return "\n\n".join(lines)


def ask(question: str, verbose: bool = False) -> dict:
    """Runs the agent loop for one question. Returns the final answer plus
    a trace of which searches were made (useful for eval and for
    displaying "how the agent got this answer" in the frontend)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")
    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    search_trace: list[dict] = []

    for round_num in range(MAX_TOOL_ROUNDS):
        response = _create_completion(
            client, model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return {"answer": msg.content, "search_trace": search_trace, "rounds": round_num + 1}

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            query = args.get("query", "")
            if verbose:
                print(f"  [tool call] search_wiki({query!r})")
            result_text = _run_search_wiki(query)
            search_trace.append({"query": query, "result_preview": result_text[:200]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    # ran out of rounds -- force a final answer from what's been gathered so far.
    # tool_choice="none" is required here, not just omitting `tools`: Groq
    # rejects the response if the model attempts a tool call while `tools`
    # isn't in the request, which it readily does given the tool-call-heavy
    # history at this point.
    messages.append({"role": "user", "content": "Stop searching. Based only on what you've found so far, either answer now or say the information isn't in the indexed corpus."})
    response = _create_completion(client, model=MODEL, messages=messages, tools=TOOLS, tool_choice="none")
    return {"answer": response.choices[0].message.content, "search_trace": search_trace, "rounds": MAX_TOOL_ROUNDS}


if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or "What is the FP cost of Ancient Death Rancor?"
    print(f"Q: {question}\n")
    result = ask(question, verbose=True)
    print(f"\nA: {result['answer']}")
    print(f"\n({result['rounds']} round(s), {len(result['search_trace'])} search call(s))")
