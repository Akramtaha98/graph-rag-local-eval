"""Judge v2 (addresses reviewer Comments 1, 2, 6).
Changes relative to src/evaluate.py:
  1. NO 6,000-character cap: the judge sees the full chunks AND all graph triples (num_ctx is set explicitly).
  2. Claim verification is done ONE CLAIM PER CALL, so the verdict count can never differ from the claim count
     (the source of all 33 logged failures in v1).
  3. Answers with no extractable claims are NOT scored 5.0. They get faithfulness=None and status='no_claims'
     (use `is_refusal` to separate genuine refusals from extraction failures).
  4. Records the raw found fraction, the claims and the per-claim verdicts, for the claim audit.
  5. Relevance / context precision are kept holistic (as in v1) and must not be used for comparisons
     unless validated; they are returned unchanged for completeness.
Drop this file into graph_rag_project/src/ next to llm.py and put llm_v2.py next to it."""
import json
import re
from config import LLM_TEMPERATURE_JUDGE
from src.llm_v2 import generate_with_stats
from src.evaluate import _context_to_text, _fraction_to_score, RELEVANCE_CONTEXT_PROMPT  # reuse v1 pieces

CLAIM_EXTRACTION_PROMPT = """List each distinct factual claim made in the ANSWER below as a JSON array of
short strings, one per claim. A claim is any specific fact, number, name, date, or assertion. Do not include
hedging phrases, meta-commentary, or citations themselves as claims. If the answer makes no factual claims at all
(for example it only says information is unavailable), return an empty array [].

ANSWER:
{answer}

Return ONLY a JSON array of strings, nothing else."""

CLAIM_CHECK_PROMPT = """CONTEXT:
{context}

CLAIM: {claim}

Does CONTEXT itself state or directly support CLAIM? A claim being generally true does not count; only the CONTEXT counts.
Reply with exactly one word: FOUND or NOT_FOUND."""

REFUSAL_RE = re.compile(
    r"(does not|doesn't|do not|not) (contain|provide|mention|specify|include|state)|cannot (answer|be determined)|"
    r"no information|(insufficient|not enough) (information|context)", re.I)


def _parse_array(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(c).strip() for c in arr if str(c).strip()] if isinstance(arr, list) else None


def _verdict(raw):
    t = raw.strip().upper().replace(" ", "_")
    if t.startswith("NOT_FOUND"):
        return "NOT_FOUND"
    if t.startswith("FOUND"):
        return "FOUND"
    return None


def faithfulness_v2(question, answer, context_text):
    out = {"faithfulness": None, "found_fraction": None, "n_claims": 0, "claims": [], "verdicts": [],
           "status": "ok", "is_refusal": False, "max_prompt_tokens": 0}
    claims = None
    for _ in range(3):
        raw, ptok = generate_with_stats(CLAIM_EXTRACTION_PROMPT.format(answer=answer), temperature=LLM_TEMPERATURE_JUDGE)
        out["max_prompt_tokens"] = max(out["max_prompt_tokens"], ptok)
        claims = _parse_array(raw)
        if claims is not None:
            break
    if claims is None:
        out["status"] = "extraction_failure"
        return out
    if not claims:
        out["status"] = "no_claims"
        out["is_refusal"] = bool(REFUSAL_RE.search(answer))
        return out
    verdicts = []
    for c in claims:
        v = None
        for _ in range(3):
            raw, ptok = generate_with_stats(CLAIM_CHECK_PROMPT.format(context=context_text, claim=c),
                                            temperature=LLM_TEMPERATURE_JUDGE)
            out["max_prompt_tokens"] = max(out["max_prompt_tokens"], ptok)
            v = _verdict(raw)
            if v:
                break
        if v is None:
            out["status"] = "verification_failure"
            out["claims"], out["verdicts"] = claims, verdicts
            return out
        verdicts.append(v)
    frac = sum(v == "FOUND" for v in verdicts) / len(verdicts)
    out.update(found_fraction=frac, faithfulness=_fraction_to_score(frac), n_claims=len(claims),
               claims=claims, verdicts=verdicts)
    return out


def _relevance_v2(question, context_text, answer):
    """Same holistic prompt as v1 (unvalidated; descriptive use only), but run with the explicit context window."""
    prompt = RELEVANCE_CONTEXT_PROMPT.format(question=question, context=context_text, answer=answer)
    for _ in range(3):
        raw, _ = generate_with_stats(prompt, temperature=LLM_TEMPERATURE_JUDGE)
        r = re.search(r'"?relevance"?\s*[:=]\s*"?([1-5](?:\.[0-9])?)', raw, re.I)
        c = re.search(r'"?context_precision"?\s*[:=]\s*"?([1-5](?:\.[0-9])?)', raw, re.I)
        if r and c:
            return {"relevance": float(r.group(1)), "context_precision": float(c.group(1))}
    return {"relevance": None, "context_precision": None}


def judge_v2(question, answer, context):
    context_text = _context_to_text(context)            # no cap
    empty = context_text.strip() == "(no context retrieved)"
    rc = _relevance_v2(question, context_text, answer)
    if empty:
        return {"faithfulness": 1.0, "status": "no_context", "relevance": rc.get("relevance"),
                "context_precision": None, "found_fraction": None, "claims": [], "verdicts": [],
                "is_refusal": False, "max_prompt_tokens": 0, "context_chars": 0}
    f = faithfulness_v2(question, answer, context_text)
    f.update(relevance=rc.get("relevance"), context_precision=rc.get("context_precision"),
             context_chars=len(context_text))
    return f
