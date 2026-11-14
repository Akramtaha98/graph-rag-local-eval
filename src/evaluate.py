"""
LLM-as-judge evaluation, giving you the Faithfulness / Answer Relevance /
Context Precision numbers the paper blueprint asks for -- without requiring
an OpenAI key (Ragas' default judge). The same local LLM scores each answer
against its own retrieved context and the question.

These are heuristic, not a substitute for human validation -- report them in
the paper as "LLM-judge scores (evaluator: <model>)" and note this as a
methodological choice in your Limitations section.

Faithfulness uses claim decomposition, not a single holistic 1-5 rating (see
_score_faithfulness_claims below). Three earlier attempts at a single-call
holistic "rate faithfulness 1-5" prompt all collapsed toward a constant value
(first 0.0 as a JSON-parse-failure artifact, then literally the example number
in the prompt template, then a soft default of 4.0 on ~96% of rows regardless
of content) when validated against a blinded human rater -- Pearson r was
consistently near zero or negative. This is a known weak spot for small local
models: holistic scoring anchors heavily on whatever plausible number appears
nearby in the prompt. Decomposing the answer into atomic claims and asking a
binary FOUND/NOT_FOUND per claim is a much easier task for a small model, and
the final 1-5 number is computed deterministically in Python from the found
fraction -- the model never outputs the final score itself, so it can't anchor
on an example value for it.
"""

import os
import re

from src.llm import generate, generate_json
from config import LLM_TEMPERATURE_JUDGE, RESULTS_DIR

_FAILURE_LOG_PATH = os.path.join(RESULTS_DIR, "judge_failures.log")


def _log_failure(question: str, stage: str, raw_responses: list[str]) -> None:
    """Append raw (unparseable) judge responses to a debug log, so a failure can
    actually be diagnosed later instead of guessed at from the outside."""
    with open(_FAILURE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\nSTAGE: {stage}\nQUESTION: {question}\n")
        for i, raw in enumerate(raw_responses, 1):
            f.write(f"--- attempt {i} raw response ---\n{raw}\n")


def _context_to_text(context: dict) -> str:
    parts = []
    for c in context.get("chunks", []):
        parts.append(f"[{c['source']}] {c['text']}")
    for t in context.get("triples", []):
        parts.append(f"{t['subject']} --{t['relation']}--> {t['object']}")
    return "\n".join(parts) if parts else "(no context retrieved)"


# ---------------------------------------------------------------------------
# Relevance + context_precision: a single holistic call. These two dimensions
# validated fine against the human sample (relevance r=0.59, 98% within 1
# point) so they're left as a normal 1-5 rating -- only faithfulness needed
# the claim-decomposition rework.
# ---------------------------------------------------------------------------

RELEVANCE_CONTEXT_PROMPT = """Score the ANSWER below on two dimensions, each 1 (worst) to 5 (best):

- relevance: does ANSWER directly and completely address QUESTION?
- context_precision: of the information in CONTEXT, how much was actually relevant
  and necessary to answer QUESTION? (context_precision = 1 if CONTEXT is empty)

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Compute your own scores for THIS specific answer -- do not reuse any number from
these instructions. Return ONLY this JSON shape with placeholders replaced:

{{"relevance": <your score 1-5>, "context_precision": <your score 1-5>, "justification": "<brief reason, no quotes>"}}
"""

_RC_PATTERN = re.compile(r'"?{key}"?\s*[:=]\s*"?([0-9](?:\.[0-9])?)"?', re.IGNORECASE)


def _regex_fallback_rc(raw_text: str) -> dict | None:
    scores = {}
    for key in ("relevance", "context_precision"):
        match = re.search(_RC_PATTERN.pattern.format(key=key), raw_text, re.IGNORECASE)
        if not match:
            return None
        scores[key] = float(match.group(1))
    just = re.search(r'"justification"\s*:\s*"([^"]{0,300})', raw_text)
    scores["justification"] = just.group(1) if just else "(recovered via regex fallback)"
    return scores


def _judge_relevance_context(question: str, context_text: str, answer: str) -> dict:
    prompt = RELEVANCE_CONTEXT_PROMPT.format(question=question, context=context_text, answer=answer)
    result = generate_json(prompt, temperature=LLM_TEMPERATURE_JUDGE)

    raw_attempts = []
    if not isinstance(result, dict) or not result:
        for _ in range(2):
            raw = generate(prompt, temperature=LLM_TEMPERATURE_JUDGE)
            raw_attempts.append(raw)
            fallback = _regex_fallback_rc(raw)
            if fallback:
                result = fallback
                break
        else:
            result = {}

    if not result:
        _log_failure(question, "relevance_context", raw_attempts)
        return {"relevance": None, "context_precision": None, "justification": ""}

    return {
        "relevance": float(result["relevance"]),
        "context_precision": float(result["context_precision"]),
        "justification": result.get("justification", ""),
    }


# ---------------------------------------------------------------------------
# Faithfulness (non-empty context): claim decomposition.
# Stage A -- list the atomic factual claims in ANSWER (independent of context).
# Stage B -- for each claim, binary FOUND / NOT_FOUND against CONTEXT.
# The 1-5 score is then computed in Python from the found fraction.
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_PROMPT = """List each distinct factual claim made in the ANSWER below as a
JSON array of short strings, one per claim. A claim is any specific fact, number, name, date,
or assertion. Do not include hedging phrases, meta-commentary, or citations themselves as
claims -- only the factual content. If the answer makes no factual claims at all (e.g. it only
says information is unavailable), return an empty array [].

ANSWER:
{answer}

Return ONLY a JSON array, nothing else. Example shape (do not reuse this example's content):
["claim text 1", "claim text 2"]
"""

CLAIM_VERIFICATION_PROMPT = """Below is a CONTEXT passage and a JSON array of CLAIMS extracted
from an answer. For each claim, in order, determine whether CONTEXT actually states or directly
supports it. A claim being generally true does not count -- only whether CONTEXT itself supports it.

CONTEXT:
{context}

CLAIMS:
{claims_json}

For each claim in order, output "FOUND" if CONTEXT supports it, or "NOT_FOUND" if it does not.
Return ONLY a JSON array of "FOUND"/"NOT_FOUND" strings, same length and order as CLAIMS, nothing
else. Example shape (do not reuse this example's content): ["FOUND", "NOT_FOUND"]
"""

# No-context branch (no_rag): there's nothing to check groundedness against, so this
# stays a simple holistic call, same as before -- this branch never showed the
# collapse-to-constant problem in validation (it's a much simpler, single-fact task:
# "is this true by general knowledge", not "is this the specific text in front of you").
NO_CONTEXT_FAITHFULNESS_PROMPT = """The ANSWER below was generated with NO retrieved context --
using only the model's own general world knowledge. Judge whether its claims are factually
accurate by general world knowledge (1 = mostly wrong, 5 = accurate).

QUESTION:
{question}

ANSWER:
{answer}

Return ONLY this JSON shape with placeholders replaced: {{"faithfulness": <your score 1-5>, "justification": "<brief reason, no quotes>"}}
"""

_NC_PATTERN = re.compile(r'"?faithfulness"?\s*[:=]\s*"?([0-9](?:\.[0-9])?)"?', re.IGNORECASE)


def _fraction_to_score(fraction: float) -> float:
    if fraction >= 0.9:
        return 5.0
    if fraction >= 0.7:
        return 4.0
    if fraction >= 0.5:
        return 3.0
    if fraction >= 0.25:
        return 2.0
    return 1.0


def _extract_claims(answer: str, question: str) -> list[str] | None:
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    result = generate_json(prompt, temperature=LLM_TEMPERATURE_JUDGE)
    if isinstance(result, list):
        return [str(c) for c in result if str(c).strip()]
    # one retry with a raw call
    raw = generate(prompt, temperature=LLM_TEMPERATURE_JUDGE)
    from src.llm import _try_parse_json
    parsed = _try_parse_json(raw)
    if isinstance(parsed, list):
        return [str(c) for c in parsed if str(c).strip()]
    _log_failure(question, "claim_extraction", [raw])
    return None


def _verify_claims(claims: list[str], context_text: str, question: str) -> list[str] | None:
    import json as _json
    prompt = CLAIM_VERIFICATION_PROMPT.format(context=context_text, claims_json=_json.dumps(claims))
    result = generate_json(prompt, temperature=LLM_TEMPERATURE_JUDGE)
    if isinstance(result, list) and len(result) == len(claims):
        return [str(x).strip().upper() for x in result]
    raw = generate(prompt, temperature=LLM_TEMPERATURE_JUDGE)
    from src.llm import _try_parse_json
    parsed = _try_parse_json(raw)
    if isinstance(parsed, list) and len(parsed) == len(claims):
        return [str(x).strip().upper() for x in parsed]
    _log_failure(question, "claim_verification", [raw])
    return None


def _score_faithfulness_claims(question: str, answer: str, context_text: str) -> tuple[float | None, str]:
    claims = _extract_claims(answer, question)
    if claims is None:
        return None, "EVALUATOR_FAILURE: claim extraction not parseable"
    if len(claims) == 0:
        # Answer made no checkable factual claims (e.g. a refusal) -- vacuously
        # nothing to be unfaithful to. Relevance/context_precision still capture
        # whether refusing was the right call.
        return 5.0, "no factual claims to verify"

    statuses = _verify_claims(claims, context_text, question)
    if statuses is None:
        return None, "EVALUATOR_FAILURE: claim verification not parseable"

    found = sum(1 for s in statuses if s.startswith("FOUND"))
    fraction = found / len(statuses)
    score = _fraction_to_score(fraction)
    return score, f"{found}/{len(statuses)} claims grounded in context"


def _score_faithfulness_no_context(question: str, answer: str) -> tuple[float | None, str]:
    prompt = NO_CONTEXT_FAITHFULNESS_PROMPT.format(question=question, answer=answer)
    result = generate_json(prompt, temperature=LLM_TEMPERATURE_JUDGE)

    raw_attempts = []
    if not isinstance(result, dict) or not result or "faithfulness" not in result:
        for _ in range(2):
            raw = generate(prompt, temperature=LLM_TEMPERATURE_JUDGE)
            raw_attempts.append(raw)
            match = _NC_PATTERN.search(raw)
            if match:
                result = {"faithfulness": float(match.group(1))}
                break
        else:
            result = {}

    if not result or "faithfulness" not in result:
        _log_failure(question, "no_context_faithfulness", raw_attempts)
        return None, "EVALUATOR_FAILURE: no-context faithfulness not parseable"

    return float(result["faithfulness"]), result.get("justification", "judged against general knowledge")


# ---------------------------------------------------------------------------
# Public entry point -- same signature/return shape as before.
# ---------------------------------------------------------------------------

def judge(question: str, answer: str, context: dict) -> dict:
    context_text = _context_to_text(context)[:6000]
    is_empty_context = context_text.strip() == "(no context retrieved)"

    rc = _judge_relevance_context(question, context_text, answer)

    # faithfulness = GROUNDEDNESS: is the answer traceable to retrieved context?
    # faithfulness_world_knowledge = separate, only meaningful for no-retrieval
    # systems: is the answer factually correct by general knowledge?
    # These are reported as two distinct numbers (decided after validation showed
    # a no-retrieval system can never score well on groundedness by definition --
    # scoring it leniently against world knowledge instead was silently conflating
    # two different questions under one "faithfulness" number).
    faithfulness_world_knowledge = None
    if is_empty_context:
        faithfulness = 1.0  # nothing was retrieved, so nothing can be grounded
        faithfulness_world_knowledge, faith_note = _score_faithfulness_no_context(question, answer)
        faith_note = f"groundedness=1 (no context); world-knowledge check: {faith_note}"
    else:
        faithfulness, faith_note = _score_faithfulness_claims(question, answer, context_text)

    justification = rc.get("justification", "") or faith_note
    if rc.get("relevance") is not None and faith_note:
        justification = f"{faith_note}; {rc.get('justification', '')}".strip("; ")

    # context_precision=1 when context is empty is a deterministic fact (there is
    # nothing in an empty context that could be relevant), not something to ask the
    # model to judge. In practice it got this right on only 1/54 no_rag rows despite
    # the prompt saying so explicitly -- same failure mode as faithfulness: asking a
    # small local model to output a specific number it should just compute is
    # unreliable, so compute it in Python instead whenever the answer is already
    # known with certainty.
    context_precision = 1.0 if is_empty_context else rc.get("context_precision")

    return {
        "faithfulness": faithfulness,
        "faithfulness_world_knowledge": faithfulness_world_knowledge,
        "relevance": rc.get("relevance"),
        "context_precision": context_precision,
        "justification": justification,
    }
