"""Input and output checks around the LLM.

These are cheap, deterministic first-line defences. The main protection
against prompt injection is structural: the model only receives the farmer's
question plus retrieved passages from our own knowledge base, has no tools
and no access to user data, so a successful injection can at worst produce a
bad answer, never leak or change data.
"""
import re

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"(reveal|print|show|repeat) (your|the) (system )?(prompt|instructions)",
    r"you are now (?!a farmer)",
    r"\bjailbreak\b|\bDAN mode\b|developer mode",
    r"</?(system|source|context)>",
]
_injection = re.compile("|".join(INJECTION_PATTERNS), re.I)

PII_PATTERNS = {
    "aadhaar": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone": re.compile(r"(?<!\d)(\+91[- ]?)?[6-9]\d{9}(?!\d)"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}


class GuardrailViolation(ValueError):
    pass


def check_input(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
    if _injection.search(text):
        raise GuardrailViolation("Please ask a farming question in your own words.")
    return text


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Remove identifiers before text leaves our system for an LLM provider."""
    found = []
    for name, pat in PII_PATTERNS.items():
        if pat.search(text):
            found.append(name)
            text = pat.sub(f"[{name} removed]", text)
    return text, found


_PROMPT_BLOCK = re.compile(r"<(source|question)\b[^>]*>.*?</\1>", re.S | re.I)
_PROMPT_TAG = re.compile(r"</?(source|question)\b[^>]*>", re.I)
_ANSWER_LABEL = re.compile(r"^\s*(answer|उत्तर|जवाब)\s*[:：]\s*", re.I | re.M)

UNCLEAR_ANSWER = {
    "en": "Sorry, I couldn't put together a clear answer. Please ask your question again in a different way.",
    "hi": "माफ़ कीजिए, मैं साफ़ जवाब नहीं बना पाया। कृपया अपना सवाल दूसरे तरीके से फिर से पूछें।",
}


def strip_prompt_echo(answer: str, language: str = "en") -> str:
    """Small models sometimes copy the prompt (<source>/<question> blocks) into
    their reply. Keep only the model's own answer, or say it was unclear."""
    if not _PROMPT_TAG.search(answer):
        return _ANSWER_LABEL.sub("", answer, count=1)
    labels = list(_ANSWER_LABEL.finditer(answer))
    if labels:  # "...echoed prompt... Answer: <the real answer>"
        answer = answer[labels[-1].end():]
    answer = _PROMPT_TAG.sub("", _PROMPT_BLOCK.sub("", answer)).strip()
    return answer if len(answer) >= 20 else UNCLEAR_ANSWER.get(language, UNCLEAR_ANSWER["en"])


def check_output(answer: str, language: str = "en") -> str:
    answer = strip_prompt_echo(answer, language)
    answer, _ = redact_pii(answer)
    return answer.strip()[:4000]
