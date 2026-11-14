"""
Central configuration for the Graph-RAG pipeline.
Everything here runs locally: Ollama for the LLM, sentence-transformers for
embeddings, and NetworkX as an in-process graph store (no server to install).
"""

import os

# --- LLM (Ollama) ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
LLM_TEMPERATURE_EXTRACTION = 0.0   # deterministic for entity/relation extraction
LLM_TEMPERATURE_GENERATION = 0.2   # low temperature for answer generation
LLM_TEMPERATURE_JUDGE = 0.0        # deterministic for evaluation scoring

# --- Embeddings ---
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40

# --- Retrieval ---
TOP_K_CHUNKS = 4          # vector search top-k
TOP_K_ENTITIES = 5        # how many query-matched entities to seed graph walk from
GRAPH_HOPS = 1            # how many hops to traverse from seed entities (1 = direct relations only;
                           # 2 pulled in too many loosely-related triples in initial testing and diluted
                           # the context rather than adding useful multi-hop signal)
MAX_GRAPH_TRIPLES = 12    # cap on triples injected into the prompt context (was 25 -- fewer, higher-
                           # precision triples outperformed a large noisy set in initial testing)
MIN_ENTITY_MATCH_RATIO = 0.6  # fraction of a node's tokens that must appear in the query/chunk to seed from it
MAX_ENTITY_NODE_WORDS = 6     # skip graph nodes longer than this (usually mis-extracted sentence fragments)

# --- Baselines ---
BM25_TOP_K = 4             # lexical top-k for the BM25 hybrid baseline
BM25_VECTOR_WEIGHT = 0.5   # fusion weight for vector score vs BM25 score (0.5 = equal weight)

# --- Corpus exclusions ---
# technical_http_rfc2616.txt is the full, un-trimmed RFC (~9,800 lines) kept in data/docs/
# for reference/provenance only. Ingesting it alongside technical_http_rfc2616_trim.txt would
# duplicate ~95% of its content into the corpus and blow up build time on a laptop (one LLM
# extraction call per chunk). technical_http_rfc2616_trim.txt is the curated version actually
# used for the experiment -- it was re-curated to include the Method Definitions, Status Code
# Definitions, and Caching/Host header sections so that the HTTP questions in questions.json
# are actually answerable from retrieved context, not just the model's parametric knowledge.
EXCLUDE_DOCS = {"technical_http_rfc2616.txt"}

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "index")
GRAPH_PATH = os.path.join(INDEX_DIR, "knowledge_graph.pkl")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
VECTOR_PATH = os.path.join(INDEX_DIR, "vectors.npy")
RESULTS_DIR = os.environ.get("RESULTS_DIR") or os.path.join(BASE_DIR, "results")
if not os.path.isabs(RESULTS_DIR): RESULTS_DIR = os.path.join(BASE_DIR, RESULTS_DIR)
QUESTIONS_PATH = os.path.join(BASE_DIR, "questions.json")

for d in (DATA_DIR, DOCS_DIR, INDEX_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
