"""Self-checking RAG for IT helpdesk questions.

Flow: retrieve -> check evidence -> (rewrite query & retry once) -> answer or escalate.
"""
import json, os, re, time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

KB_DIR = Path("kb")
DB_DIR = "chroma_db"
LOG_FILE = Path("logs/queries.jsonl")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
THRESHOLD = float(os.getenv("DIST_THRESHOLD", "0.6"))  # tune with eval.py
TOP_K = 3

_llm_client = None


def llm(prompt: str) -> str:
    global _llm_client
    if _llm_client is None:
        from groq import Groq
        _llm_client = Groq()  # reads GROQ_API_KEY
    resp = _llm_client.chat.completions.create(
        model=MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

_db = None

def get_collection():
    global _db
    if _db is None:
        import chromadb
        _db = chromadb.PersistentClient(path=DB_DIR)
    return _db.get_or_create_collection("it_kb", metadata={"hnsw:space": "cosine"})

def ingest() -> int:
    """Split each markdown file into '## section' chunks and store them."""
    ids, docs, metas = [], [], []
    for f in sorted(KB_DIR.glob("*.md")):
        for i, section in enumerate(re.split(r"\n(?=## )", f.read_text(encoding="utf-8"))):
            if section.strip():
                ids.append(f"{f.stem}-{i}")
                docs.append(section.strip())
                metas.append({"source": f.name})
    get_collection().upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def ensure_ingested():
    if get_collection().count() == 0:
        ingest()


def retrieve(query: str) -> list[dict]:
    res = get_collection().query(query_texts=[query], n_results=TOP_K)
    return [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def evidence_is_good(chunks: list[dict]) -> bool:
    """Layer 1 check: is the best chunk close enough to the question?"""
    return bool(chunks) and chunks[0]["distance"] <= THRESHOLD


def rewrite(question: str) -> str:
    return llm(
        "Rewrite this IT helpdesk question as a short, specific search query "
        "using standard IT terms. Return only the query.\n\nQuestion: " + question
    )


def generate(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return llm(
        "You are an IT helpdesk assistant. Answer ONLY using the context below. "
        "If the context does not contain the answer, reply with exactly: NOT_IN_KB\n"
        "Be concise (max 5 sentences) and mention the source file name.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def ask(question: str, log: bool = True) -> dict:
    trace, query = [], question
    result = None
    for attempt in (1, 2):  # original query, then one rewritten retry
        chunks = retrieve(query)
        best = chunks[0]["distance"] if chunks else float("nan")
        trace.append(f"Attempt {attempt}: query='{query}' | best distance={best:.3f} (limit {THRESHOLD})")
        if evidence_is_good(chunks):
            answer = generate(question, chunks)  # Layer 2 check: LLM may say NOT_IN_KB
            if "NOT_IN_KB" not in answer:
                trace.append("Evidence good -> answered from KB")
                result = {"status": "answered", "answer": answer,
                          "sources": sorted({c["source"] for c in chunks})}
                break
            trace.append("LLM judged the context insufficient")
        else:
            trace.append("Evidence too weak")
        if attempt == 1:
            query = rewrite(question)
            trace.append("Rewrote the query and retrying")

    if result is None:
        trace.append("No reliable answer -> escalated")
        result = {
            "status": "escalated",
            "answer": ("I couldn't find a reliable answer in the IT knowledge base, so I won't guess. "
                       "Please contact the service desk (ext 5555).\n\n"
                       f"Draft ticket: \"{question}\""),
            "sources": [],
        }
    result["trace"] = trace
    if log:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "question": question,
                                "status": result["status"], "trace": trace}) + "\n")
    return result
