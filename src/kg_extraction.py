"""
Use the local LLM to extract (subject, relation, object) triples from each
text chunk. This is the "Knowledge Graph Construction" step of the pipeline.
"""

from tqdm import tqdm

from src.llm import generate_json
from config import LLM_TEMPERATURE_EXTRACTION

EXTRACTION_PROMPT = """You are an information-extraction system for building a knowledge graph.

Read the passage below and extract factual (subject, relation, object) triples.
Rules:
- Only extract relations that are explicitly stated or directly implied by the text.
- Use concise, canonical entity names (e.g. "Graph-RAG" not "the Graph-RAG framework mentioned earlier").
- Use short, snake_case relation labels (e.g. "improves", "is_a", "part_of", "causes", "measured_by").
- Extract at most 12 triples. Skip the passage if it has no clear factual relations.
- Return ONLY a JSON array, no prose, no markdown fences. Example format:
[{{"subject": "Graph-RAG", "relation": "reduces", "object": "hallucination rate"}}]

Passage:
\"\"\"{chunk_text}\"\"\"

JSON array:"""


def extract_triples_from_chunk(chunk_text: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(chunk_text=chunk_text[:3000])
    result = generate_json(prompt, temperature=LLM_TEMPERATURE_EXTRACTION)
    if not isinstance(result, list):
        return []
    cleaned = []
    for item in result:
        if not isinstance(item, dict):
            continue
        s, r, o = item.get("subject"), item.get("relation"), item.get("object")
        if s and r and o:
            cleaned.append({
                "subject": str(s).strip(),
                "relation": str(r).strip().lower().replace(" ", "_"),
                "object": str(o).strip(),
            })
    return cleaned


def extract_triples(chunks: list[dict]) -> list[dict]:
    """
    chunks: [{"chunk_id", "source", "text"}, ...]
    returns: [{"subject", "relation", "object", "chunk_id", "source"}, ...]
    """
    all_triples = []
    for chunk in tqdm(chunks, desc="Extracting entities/relations"):
        triples = extract_triples_from_chunk(chunk["text"])
        for t in triples:
            t["chunk_id"] = chunk["chunk_id"]
            t["source"] = chunk["source"]
            all_triples.append(t)
    print(f"[kg_extraction] Extracted {len(all_triples)} triples from {len(chunks)} chunks.")
    return all_triples
