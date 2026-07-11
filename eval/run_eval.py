"""Eval harness: retrieval accuracy + answer correctness/faithfulness.

Two things are measured per question, separately, because a wrong final
answer can come from either failure mode and they need different fixes:
  - Retrieval: did search return the right source page(s) at all?
  - Answer: given what was (or wasn't) retrieved, is the final answer
    correct -- and for out-of-scope/adversarial questions, does the agent
    correctly decline instead of guessing or hallucinating?

expected_facts entries are substrings the answer must contain. An entry
can be a plain string (required) or a list of strings (an OR-group --
alternate acceptable phrasings of the same fact, e.g. "4,393" vs "4393").

Run: python eval/run_eval.py [--k 5] [--out results.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import search  # noqa: E402
from agent import ask  # noqa: E402

QA_PATH = Path(__file__).resolve().parent / "qa_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

REFUSAL_MARKERS = [
    "not in the indexed", "isn't in the indexed", "is not in the indexed",
    "isn't included in", "is not included in", "doesn't include", "does not include",
    "not covered in", "isn't covered", "is not covered",
    "doesn't appear to be in", "does not appear to be in",
    "don't have", "do not have", "doesn't contain", "does not contain",
    "no information", "not indexed", "outside the scope", "not available in the indexed",
    "couldn't find", "could not find", "unable to find",
]


def _normalize(text: str) -> str:
    # the model outputs curly quotes (e.g. U+2019 in "couldn't"); our
    # marker strings use straight ASCII apostrophes, which silently fails
    # every apostrophe-containing substring match unless both sides are
    # normalized the same way -- found via q24 scoring as a false FAIL
    # despite the agent's answer being correct ("I couldn't find...").
    return text.lower().replace("’", "'").replace("‘", "'")


def check_facts(answer: str, expected_facts: list) -> tuple[bool, list[str]]:
    answer_norm = _normalize(answer)
    missing = []
    for entry in expected_facts:
        options = entry if isinstance(entry, list) else [entry]
        if not any(_normalize(opt) in answer_norm for opt in options):
            missing.append(" OR ".join(options))
    return len(missing) == 0, missing


def check_refusal(answer: str) -> bool:
    answer_norm = _normalize(answer)
    return any(_normalize(marker) in answer_norm for marker in REFUSAL_MARKERS)


def eval_retrieval(question: str, expected_titles: list[str], k: int) -> dict:
    results = search(question, k=k)
    retrieved_titles = {r["title"] for r in results}
    hits = [t for t in expected_titles if t in retrieved_titles]
    return {
        "expected": expected_titles,
        "retrieved": sorted(retrieved_titles),
        "hit_rate": len(hits) / len(expected_titles) if expected_titles else None,
        "full_hit": len(hits) == len(expected_titles),
    }


def run(k: int = 5) -> dict:
    qa_set = json.loads(QA_PATH.read_text(encoding="utf-8"))
    results = []

    for qa in qa_set:
        print(f"[{qa['id']}] {qa['question']}")
        entry = {"id": qa["id"], "type": qa["type"], "question": qa["question"]}

        if not qa.get("expected_refusal"):
            entry["retrieval"] = eval_retrieval(qa["question"], qa["expected_titles"], k)

        try:
            agent_result = ask(qa["question"])
            answer = agent_result["answer"] or ""
            search_rounds = agent_result["rounds"]
        except Exception as e:
            answer = ""
            search_rounds = None
            entry["error"] = str(e)

        entry["answer"] = answer
        entry["search_rounds"] = search_rounds

        if qa.get("expected_refusal"):
            entry["correct"] = check_refusal(answer)
            entry["expected"] = "refusal"
        else:
            correct, missing = check_facts(answer, qa["expected_facts"])
            entry["correct"] = correct
            entry["missing_facts"] = missing

        status = "PASS" if entry["correct"] else "FAIL"
        print(f"  -> {status}" + (f" (missing: {entry.get('missing_facts')})" if not entry["correct"] and "missing_facts" in entry else ""))
        results.append(entry)
        time.sleep(0.3)  # be gentle on the Groq rate limit

    non_refusal = [r for r in results if r["type"] != "out_of_scope" and r["type"] != "adversarial"]
    refusal = [r for r in results if r["type"] in ("out_of_scope", "adversarial")]

    summary = {
        "total_questions": len(results),
        "answer_accuracy": sum(r["correct"] for r in results) / len(results),
        "retrieval_full_hit_rate": sum(r["retrieval"]["full_hit"] for r in non_refusal) / len(non_refusal) if non_refusal else None,
        "refusal_accuracy": sum(r["correct"] for r in refusal) / len(refusal) if refusal else None,
        "by_type": {},
    }
    for t in set(r["type"] for r in results):
        subset = [r for r in results if r["type"] == t]
        summary["by_type"][t] = {
            "count": len(subset),
            "answer_accuracy": sum(r["correct"] for r in subset) / len(subset),
        }

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    output = run(k=args.k)
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    s = output["summary"]
    print("\n" + "=" * 50)
    print(f"Total questions: {s['total_questions']}")
    print(f"Answer accuracy (overall): {s['answer_accuracy']:.0%}")
    print(f"Retrieval full-hit rate (non-refusal Qs): {s['retrieval_full_hit_rate']:.0%}")
    print(f"Refusal accuracy (out-of-scope/adversarial): {s['refusal_accuracy']:.0%}")
    print("\nBy type:")
    for t, stats in s["by_type"].items():
        print(f"  {t}: {stats['answer_accuracy']:.0%} ({stats['count']} questions)")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
