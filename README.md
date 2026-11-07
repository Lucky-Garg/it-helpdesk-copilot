# 🛠️ IT Helpdesk Copilot

A self-checking RAG assistant that answers IT questions **only from a company knowledge base**. When the evidence is weak, it rewrites the question once and retries. If that fails, it **escalates to the service desk instead of guessing**.

Built with Python, ChromaDB, Groq (LLM) and Streamlit. Includes an evaluation script that measures retrieval quality, answer/refuse accuracy and faithfulness.

![Demo](docs/screenshot.png)

## Why this project

Most RAG demos answer every question, even when the knowledge base has nothing relevant. That is how hallucinations reach users. This project focuses on the refusal path: it decides when it knows enough to answer, shows how it decided, and measures how well it does that.

## How it works

```
Question
   │
   ▼
Retrieve top 3 chunks from ChromaDB
   │
   ▼
Check 1: is the best match close enough? (cosine distance ≤ threshold)
   │ no ──► rewrite the question with the LLM ──► retrieve again (once)
   │ yes                                              │ still weak
   ▼                                                  ▼
LLM answers using ONLY the retrieved context     ESCALATE to service desk
   │
   ▼
Check 2: did the LLM reply NOT_IN_KB?
   │ yes ──► (rewrite and retry, or escalate)
   │ no
   ▼
ANSWER + source documents
```

Every decision is shown in the UI under **"How I decided"** and logged to `logs/queries.jsonl` as an audit trail.

## Features

- **Grounded answers** from a local knowledge base, with source file shown
- **Two-layer evidence check:** a distance threshold, then the LLM's own `NOT_IN_KB` guard
- **Query rewrite and one retry** for vague questions
- **Safe escalation** with a draft ticket instead of a guess
- **Transparent decision trace** in the UI and in the log file
- **Evaluation harness** with 21 labeled questions

## Tech stack

| Part | Tool |
|---|---|
| Vector database | ChromaDB (local, persistent, cosine distance) |
| Embeddings | Chroma default (`all-MiniLM-L6-v2`) |
| LLM | Groq (`llama-3.1-8b-instant` by default) |
| UI | Streamlit |
| Config | python-dotenv |

## Project structure

```
it-helpdesk-copilot/
├── app.py            # Streamlit chat UI
├── rag.py            # Retrieval, checks, rewrite, answer, escalation, logging
├── eval.py           # Evaluation script
├── eval_set.json     # 21 labeled test questions (15 answerable, 6 not)
├── kb/               # Sample company IT documents (markdown)
├── requirements.txt
└── .env.example
```

## Getting started

**Requirements:** Python 3.10+ and a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone and enter the project
git clone https://github.com/Lucky-Garg/it-helpdesk-copilot.git
cd it-helpdesk-copilot

# 2. Create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env        # Windows: copy .env.example .env
# then edit .env and set GROQ_API_KEY

# 5. Run the app
streamlit run app.py
```

The first run downloads a small embedding model (about 80 MB) and indexes the documents in `kb/`. If you edit the documents later, delete the `chroma_db/` folder and restart so they are re-indexed.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none (required) | Your Groq key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model used for answers and rewriting |
| `DIST_THRESHOLD` | `0.6` | Maximum cosine distance to treat evidence as good |

## Try these questions

| Question | Expected behavior |
|---|---|
| How do I connect to the VPN from home? | Answered from `vpn_guide.md` |
| Can IT support ask me for my MFA code? | Answered: never |
| What are the password rules? | Answered with length, expiry and reuse rules |
| My work chat app is acting strange after the update | Query rewrite, then Teams answer or escalation |
| What is the guest Wi-Fi password? | Escalated (not in the KB) |
| Who won the cricket match yesterday? | Escalated |

## Evaluation

```bash
python eval.py          # retrieval metrics + threshold sweep (free, no LLM calls)
python eval.py --e2e    # also runs the full pipeline and faithfulness check (uses Groq)
```

The test set (`eval_set.json`) has 21 questions: 15 the knowledge base can answer and 6 it cannot, including tricky IT-sounding ones such as the guest Wi-Fi password and Teams outage status.

### Metrics

| Metric | What it measures |
|---|---|
| **Hit@3** | Is the correct document anywhere in the top 3 results? |
| **Precision@3** | What share of the top 3 chunks come from the correct document? |
| **MRR** | How high does the first correct chunk rank? (rank 1 = 1.0, rank 2 = 0.5) |
| **Threshold sweep** | Answerable questions accepted vs. out-of-scope questions wrongly accepted, for each distance threshold |
| **Answer/refuse accuracy** (`--e2e`) | Does the system answer when it should and escalate when it should? Answered questions also need the expected keywords |
| **Faithfulness** (`--e2e`) | LLM-judged: is every claim in the answer supported by the retrieved text? |

### Results

| Metric | Result |
|---|---|
| Hit@3 | **100%** (15/15) |
| Precision@3 | TBD |
| MRR | TBD |
| Answer/refuse accuracy, distance check only (threshold 0.60) | **95%** (20/21) |
| Answer/refuse accuracy, full pipeline | TBD |
| Faithfulness (LLM-judged) | TBD |

### Threshold sweep

| Threshold | Answerable accepted | Out-of-scope wrongly accepted | Accuracy |
|---|---|---|---|
| 0.40 | 11/15 | 0/6 | 81% |
| 0.50 | 13/15 | 1/6 | 86% |
| **0.60** | **15/15** | **1/6** | **95%** |
| 0.70 | 15/15 | 3/6 | 86% |
| 0.90 | 15/15 | 6/6 | 71% |

A threshold of 0.60 accepts every answerable question while letting one out-of-scope question through the distance check. The second check (the LLM's `NOT_IN_KB` guard) exists to catch cases like that.

## Design decisions

- **Escalate instead of searching the web.** Internal questions stay inside the company, and the system never presents unverified public information as company policy.
- **Two checks instead of one.** A distance score measures similarity, not whether a document really answers the question, so the LLM gets a veto.
- **Section-based chunking.** Each `##` section of a document is one self-contained chunk.
- **No orchestration framework.** Plain Python functions keep the control flow easy to read and debug.
- **Audit log.** Every question, its decision path and its outcome are saved for review.

## Limitations

- Small sample knowledge base and a small test set (21 questions), so percentages move a lot with a single question.
- The threshold was tuned on the same questions it is scored on, so results look slightly better than they would on unseen questions.
- Faithfulness is judged by a small LLM, which can be lenient.
- Keyword matching is a crude proxy for answer correctness.
- No per-user document permissions and no multi-turn memory.

## Future improvements

- Optional web-search fallback with an LLM grader for results, behind a flag
- Held-out evaluation questions and a larger, real knowledge base
- Cross-check metrics with an established framework (e.g. DeepEval or RAGAS)
- Document upload, per-document access control and user feedback buttons