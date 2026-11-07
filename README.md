# IT Helpdesk Copilot (Self-Checking RAG)

A small RAG assistant that answers IT questions **only from a company knowledge base**.
If the evidence is weak, it rewrites the query once, and if that fails it **escalates instead of guessing**.

**Stack:** Python, ChromaDB (local vector DB), MiniLM embeddings (Chroma default), Groq LLM, Streamlit.

## How it works
```
Question -> retrieve top 3 chunks -> Check 1: best cosine distance <= threshold?
              |-- no  -> rewrite query with LLM -> retry once -> still no -> ESCALATE
              |-- yes -> LLM answers from context only
                          Check 2: LLM replies NOT_IN_KB? -> ESCALATE
                          else -> ANSWER + sources
```
Every decision is shown in the UI ("How I decided") and logged to `logs/queries.jsonl`.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env        # add your free Groq key (console.groq.com)
streamlit run app.py        # KB is indexed automatically on first run
python eval.py              # retrieval + threshold sweep (no API calls)
python eval.py --e2e        # full pipeline test (uses Groq)
```

## Evaluation (fill in YOUR numbers after running)
- `eval_set.json`: 15 in-scope questions + 6 out-of-scope (including hard ones such as "guest Wi-Fi password" and "Teams outage status").
- Retrieval hit@3: __%
- Threshold chosen: __ (from the sweep table; trades off wrongly-answered out-of-scope vs. missed in-scope)
- End-to-end pass rate: __/21

## Design choices (be ready to explain)
- **Escalate instead of web search:** avoids sending internal questions to a third party and avoids unverified answers.
- **Two checks:** a cheap distance threshold first, then the LLM's own "NOT_IN_KB" guard.
- **Section-based chunking:** each `##` section is one self-contained answer.
- **No framework:** plain Python functions, so the control flow is easy to read.

## Known limitations
Small KB, keyword-based answer checks are crude, no access control per document, no multi-turn memory.

## Ideas to extend
Add a feedback button, add LangGraph, add a PDF upload, add per-document permissions, compare embedding models using eval.py.
