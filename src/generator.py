"""
Answer generation for the three systems compared in the paper:

  1. no_rag      -- LLM answers from parametric knowledge only
  2. naive_rag   -- LLM answers using vector-retrieved text chunks only
  3. graph_rag   -- LLM answers using text chunks + graph-traversed triples

All three share the same underlying LLM call so any difference in answer
quality is attributable to the retrieval strategy, not model capability.
"""

from src.llm import generate
from config import LLM_TEMPERATURE_GENERATION

NO_RAG_PROMPT = """Answer the question as accurately as you can using your own knowledge.
If you are not sure, say so explicitly rather than guessing.

Question: {question}

Answer:"""

RAG_PROMPT = """You are a domain question-answering assistant. Answer the question using the
context below (text passages and, if present, knowledge graph relations).
Some of the context may be irrelevant to this specific question -- that is
normal, ignore whatever does not help and use whatever does. Do not use
outside knowledge beyond what is in the context. Cite which source(s) you
used in parentheses.

Only say "The provided context does not contain enough information to answer
this" if none of the context, taken together, actually addresses the
question -- do not say this just because some parts of the context are
irrelevant or because the knowledge graph relations alone seem sparse; check
the text passages too before concluding the context is insufficient.

{context_block}

Question: {question}

Answer:"""


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["Text passages:"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {c['source']}) {c['text']}")
    return "\n".join(lines)


def _format_triples(triples: list[dict]) -> str:
    if not triples:
        return ""
    lines = ["Knowledge graph relations:"]
    for t in triples:
        lines.append(f"- {t['subject']} --{t['relation']}--> {t['object']} (source: {t.get('source')})")
    return "\n".join(lines)


def answer_no_rag(question: str) -> dict:
    prompt = NO_RAG_PROMPT.format(question=question)
    answer = generate(prompt, temperature=LLM_TEMPERATURE_GENERATION)
    return {"answer": answer, "context": {"chunks": [], "triples": []}}


def answer_with_context(question: str, context: dict) -> dict:
    chunk_block = _format_chunks(context.get("chunks", []))
    triple_block = _format_triples(context.get("triples", []))
    context_block = "\n\n".join(block for block in (chunk_block, triple_block) if block)
    prompt = RAG_PROMPT.format(context_block=context_block, question=question)
    answer = generate(prompt, temperature=LLM_TEMPERATURE_GENERATION)
    return {"answer": answer, "context": context}


def answer_naive_rag(question: str, retriever) -> dict:
    context = retriever.retrieve(question)
    return answer_with_context(question, context)


def answer_bm25_hybrid(question: str, retriever) -> dict:
    context = retriever.retrieve(question)
    return answer_with_context(question, context)


def answer_graph_rag(question: str, retriever) -> dict:
    context = retriever.retrieve(question)
    return answer_with_context(question, context)
