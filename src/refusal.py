"""Rule-based refusal classification (applied to the saved answers; does not change any score).
full    : the opening sentence says the context is insufficient and nothing after it answers the question
partial : the opening sentence says so, and the answer then offers an inference from the context
none    : otherwise"""
import re
_OPEN = re.compile(r"(does not|doesn't) contain (enough|sufficient)|not enough (information|context)|insufficient", re.I)
_PARTIAL = re.compile(r"however|infer|inference|but we can|based on the available", re.I)


def refusal_type(answer: str) -> str:
    a = answer.strip()
    first = re.split(r"(?<=[.!?])\s", a, 1)[0]
    if not _OPEN.search(first):
        return "none"
    return "partial" if _PARTIAL.search(a[len(first):]) else "full"
