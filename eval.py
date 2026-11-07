"""Evaluate the copilot.

  python eval.py          -> retrieval metrics + threshold sweep (free, no LLM calls)
  python eval.py --e2e    -> also runs the full pipeline + faithfulness check (uses Groq)
"""
import json, sys
import rag

rag.ensure_ingested()
data = json.load(open("eval_set.json", encoding="utf-8"))
inscope = [d for d in data if d["should_answer"]]
outscope = [d for d in data if not d["should_answer"]]

for d in data:
    d["chunks"] = rag.retrieve(d["question"])
    d["best"] = d["chunks"][0]["distance"]

# ------------------------------------------------------------
# 1) Retrieval quality (free)
#    hit@k      : is the right document anywhere in the top k?
#    precision@k: what share of the top k chunks come from the right document?
#    MRR        : how high is the first right chunk? (rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33)
# ------------------------------------------------------------
hits = 0
precision_sum = 0
mrr_sum = 0
for d in inscope:
    sources = [c["source"] for c in d["chunks"]]
    expected = d["expected_source"]
    if expected in sources:
        hits += 1
        mrr_sum += 1 / (sources.index(expected) + 1)
    precision_sum += sources.count(expected) / len(sources)

n = len(inscope)
print(f"Retrieval hit@{rag.TOP_K}:   {hits}/{n} = {hits/n:.0%}")
print(f"Retrieval precision@{rag.TOP_K}: {precision_sum/n:.2f}")
print(f"Retrieval MRR:        {mrr_sum/n:.2f}\n")

# ------------------------------------------------------------
# 2) Threshold sweep (free): where should the 'evidence is good enough' line be?
# ------------------------------------------------------------
print("threshold | in-scope answered | out-of-scope wrongly answered | accuracy")
for t in [x / 100 for x in range(30, 95, 5)]:
    ok_in = sum(d["best"] <= t for d in inscope)
    bad_out = sum(d["best"] <= t for d in outscope)
    acc = (ok_in + len(outscope) - bad_out) / len(data)
    print(f"  {t:.2f}    |   {ok_in:2d}/{len(inscope)}           |      {bad_out}/{len(outscope)}                      |  {acc:.0%}")
print(f"\nCurrent threshold = {rag.THRESHOLD}. Set DIST_THRESHOLD in .env to change it.\n")


# ------------------------------------------------------------
# 3) End-to-end (optional, uses the LLM)
#    - route accuracy + keyword check
#    - faithfulness: is every claim in the answer supported by the retrieved text?
# ------------------------------------------------------------
def is_faithful(answer, chunks):
    """LLM judge. Returns True if the answer only says things found in the context."""
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = (
        "You are a strict fact checker.\n"
        "Is EVERY claim in the answer supported by the context?\n"
        "Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}"
    )
    return rag.llm(prompt).upper().startswith("SUPPORTED")


if "--e2e" in sys.argv:
    correct = 0
    answered_count = 0
    faithful_count = 0

    for d in data:
        r = rag.ask(d["question"], log=False)
        answered = r["status"] == "answered"
        ok = answered == d["should_answer"]
        if ok and answered:  # crude correctness proxy: expected keywords present
            ok = all(k.lower() in r["answer"].lower() for k in d["keywords"])
        correct += ok

        note = ""
        if answered:
            answered_count += 1
            if is_faithful(r["answer"], d["chunks"]):
                faithful_count += 1
            else:
                note = "  [NOT FAITHFUL]"
        print(("PASS" if ok else "FAIL"), "|", d["question"], "->", r["status"], note)

    print(f"\nEnd-to-end correct: {correct}/{len(data)} = {correct/len(data):.0%}")
    if answered_count:
        print(f"Faithfulness:       {faithful_count}/{answered_count} = {faithful_count/answered_count:.0%}")