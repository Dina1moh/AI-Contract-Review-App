"""
Generation stage of the RAG pipeline: deterministic legal-information
extraction -> prompt (see prompt.py) -> local Qwen 2.5 1.5B via Ollama ->
one CUAD label.

classify_clause() is the orchestrator: it ties retrieval (retriever.py)
together with prompt-building (prompt.py) and generation to classify one
clause. Prompt wording/instructions live in prompt.py -- edit that file to
tune model behavior without touching this one.

Location: app/core/rag/generator.py
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable

import requests
from qdrant_client import QdrantClient

from ..config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_MAX_RETRY_ATTEMPTS,
    TOP_K,
)
from .retriever import retrieve_similar
from .prompt import build_prompt, load_label_definitions  # re-exported for callers

__all__ = [
    "extract_legal_information",
    "call_ollama",
    "parse_prediction",
    "classify_clause",
    "load_label_definitions",
]

ProgressCallback = Callable[[dict], None]


def _emit(callback: ProgressCallback | None, stage: str, message: str, **details) -> None:
    if callback:
        callback({"type": "progress", "stage": stage, "message": message, **details})


# --------------------------------------------------------------------------
# Deterministic legal-information extraction (regex-based)
# --------------------------------------------------------------------------
LEGAL_CONCEPT_PATTERNS: list[tuple[str, str]] = [
    ("change of control", r"\bchange of control\b"),
    ("governing law", r"\bgoverning law\b"),
    ("insurance", r"\binsur(?:e|ance|ed|ing)\b"),
    ("audit", r"\baudit(?:s|ed|ing| rights?)?\b"),
    ("license", r"\blicen[cs](?:e|es|ed|ing)?\b"),
    ("assignment", r"\bassign(?:ment|ed|ing)?\b"),
    ("transfer", r"\btransfer(?:red|ring|able)?\b"),
    ("termination", r"\bterminat(?:e|ed|ing|ion)\b"),
    ("renewal", r"\brenew(?:al|ed|ing)?\b"),
    ("notice", r"\bnotice\b"),
    ("liability", r"\bliabilit(?:y|ies)\b"),
    ("warranty", r"\bwarrant(?:y|ies)\b"),
    ("royalty", r"\broyalt(?:y|ies)\b"),
    ("revenue share", r"\brevenue(?:/profit)? sharing\b|\bprofit sharing\b"),
    ("price", r"\bprice(?:s|d|ing)?\b"),
    ("discount", r"\bdiscount(?:s|ed|ing)?\b"),
    ("non-compete", r"\bnon-compete\b"),
    ("non-solicit", r"\bnon-solicit\b"),
    ("exclusivity", r"\bexclusiv(?:e|ity)\b"),
    ("source code escrow", r"\bsource code escrow\b"),
    ("liquidated damages", r"\bliquidated damages\b"),
    ("minimum commitment", r"\bminimum commitment\b"),
    ("right of first refusal", r"\bright of first refusal\b"),
    ("right of first offer", r"\bright of first offer\b"),
    ("right of first negotiation", r"\bright of first negotiation\b"),
    ("first refusal", r"\bfirst refusal\b"),
    ("first offer", r"\bfirst offer\b"),
    ("first negotiation", r"\bfirst negotiation\b"),
    ("escrow", r"\bescrow\b"),
    ("perpetual", r"\bperpetual\b"),
    ("irrevocable", r"\birrevocable\b"),
    ("unlimited", r"\bunlimited\b"),
    ("uncapped", r"\buncapped\b"),
    ("volume", r"\bvolume\b"),
]

LEGAL_ACTION_PATTERNS: list[tuple[str, str]] = [
    ("grant", r"\bgrant(?:s|ed|ing)?\b"),
    ("license", r"\blicen[cs](?:e|es|ed|ing)?\b"),
    ("assign", r"\bassign(?:s|ed|ing|ment)?\b"),
    ("delegate", r"\bdelegat(?:e|es|ed|ing|ion)\b"),
    ("transfer", r"\btransfer(?:s|red|ring|able)?\b"),
    ("terminate", r"\bterminat(?:e|es|ed|ing|ion)?\b"),
    ("renew", r"\brenew(?:s|ed|ing|al)?\b"),
    ("notify", r"\bnotif(?:y|ies|ied|ication)\b"),
    ("pay", r"\bpay(?:s|ment|ing|able)?\b"),
    ("provide", r"\bprovid(?:e|es|ed|ing)\b"),
    ("audit", r"\baudit(?:s|ed|ing)?\b"),
    ("inspect", r"\binspect(?:s|ed|ing)?\b"),
    ("restrict", r"\brestrict(?:s|ed|ing|ion)?\b"),
    ("compete", r"\bcompet(?:e|es|ed|ing|ition)\b"),
    ("solicit", r"\bsolicit(?:s|ed|ing|ation)?\b"),
    ("purchase", r"\bpurchas(?:e|es|ed|ing|ed)\b"),
    ("sell", r"\bsell(?:s|ing|er|ers)?\b"),
    ("use", r"\buse(?:s|d|ing)?\b"),
    ("sublicense", r"\bsublicen[cs](?:e|es|ed|ing)?\b"),
    ("insure", r"\binsur(?:e|es|ed|ing|ance)\b"),
    ("disclose", r"\bdisclos(?:e|es|ed|ing|ure)\b"),
    ("hold harmless", r"\bhold harmless\b"),
]

RIGHT_PATTERNS: list[tuple[str, str]] = [
    ("right to", r"\bright to\b"),
    ("may", r"\bmay\b"),
    ("entitled to", r"\bentitled to\b"),
    ("authorized to", r"\bauthorized to\b"),
    ("permitted to", r"\bpermitted to\b"),
    ("shall have the right to", r"\bshall have the right to\b"),
]

OBLIGATION_PATTERNS: list[tuple[str, str]] = [
    ("shall", r"\bshall\b"),
    ("must", r"\bmust\b"),
    ("agree to", r"\bagree to\b"),
    ("agrees to", r"\bagrees to\b"),
    ("required to", r"\brequired to\b"),
    ("is required to", r"\bis required to\b"),
    ("undertakes to", r"\bundertakes to\b"),
]

RESTRICTION_PATTERNS: list[tuple[str, str]] = [
    ("shall not", r"\bshall not\b"),
    ("may not", r"\bmay not\b"),
    ("no ", r"\bno\b"),
    ("not", r"\bnot\b"),
    ("except", r"\bexcept\b"),
    ("only", r"\bonly\b"),
    ("solely", r"\bsolely\b"),
    ("without", r"\bwithout\b"),
    ("prohibited", r"\bprohibit(?:ed|s|ing)?\b"),
]

CONDITION_PATTERNS: list[tuple[str, str]] = [
    ("provided that", r"\bprovided that\b"),
    ("subject to", r"\bsubject to\b"),
    ("if", r"\bif\b"),
    ("unless", r"\bunless\b"),
    ("in the event", r"\bin the event\b"),
    ("upon", r"\bupon\b"),
    ("as long as", r"\bas long as\b"),
    ("only if", r"\bonly if\b"),
]

DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{1,2},\s+\d{4}"
    r"|"
    r"\d{1,2}\s+days?"
    r"|"
    r"\d{1,2}\s+months?"
    r"|"
    r"\d{1,2}\s+years?"
    r")\b",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?"
    r"|"
    r"\b(?:usd|eur|gbp)\s?\d[\d,]*(?:\.\d+)?\b"
    r"|"
    r"\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp)\b)",
    re.IGNORECASE,
)

PERCENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|percent|per cent)\b",
    re.IGNORECASE,
)

NUMERIC_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:days?|months?|years?|weeks?|hours?|days'|day|month|year|week|hour))?\b",
    re.IGNORECASE,
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        ordered.append(normalized)
    return ordered


def _collect_matches(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    matches: list[str] = []
    lowered = text.lower()
    for label, pattern in patterns:
        if re.search(pattern, lowered, re.IGNORECASE):
            matches.append(label)
    return _dedupe_preserve_order(matches)


def _collect_span_matches(text: str, pattern: re.Pattern[str]) -> list[str]:
    matches = [match.group(0).strip() for match in pattern.finditer(text)]
    return _dedupe_preserve_order(matches)


def _collect_legal_keywords(text: str) -> list[str]:
    vocabulary = [
        "termination", "terminate", "renewal", "renew", "notice",
        "written notice", "license", "licence", "assignment", "assign",
        "transfer", "liability", "warranty", "insurance", "audit",
        "exclusivity", "exclusive", "royalty", "revenue", "profit", "price",
        "discount", "non-compete", "non-solicit", "source code escrow",
        "liquidated damages", "minimum commitment", "first refusal",
        "first offer", "first negotiation", "governing law",
        "change of control", "perpetual", "irrevocable", "unlimited",
        "uncapped", "volume",
    ]
    lowered = text.lower()
    hits = [phrase for phrase in vocabulary if phrase in lowered]
    return _dedupe_preserve_order(hits)


def extract_legal_information(clause_text: str) -> dict[str, list[str]]:
    text = str(clause_text or "").strip()

    legal_concepts = _collect_matches(text, LEGAL_CONCEPT_PATTERNS)
    legal_actions = _collect_matches(text, LEGAL_ACTION_PATTERNS)
    rights = _collect_matches(text, RIGHT_PATTERNS)
    obligations = _collect_matches(text, OBLIGATION_PATTERNS)
    restrictions = _collect_matches(text, RESTRICTION_PATTERNS)
    conditions = _collect_matches(text, CONDITION_PATTERNS)
    dates = _collect_span_matches(text, DATE_PATTERN)
    money = _collect_span_matches(text, MONEY_PATTERN)
    percentages = _collect_span_matches(text, PERCENT_PATTERN)

    numeric_candidates = _collect_span_matches(text, NUMERIC_PATTERN)
    numeric_values = _dedupe_preserve_order(
        [value for value in numeric_candidates if value not in dates]
    )

    keywords = _dedupe_preserve_order(
        _collect_legal_keywords(text)
        + legal_concepts
        + legal_actions
        + rights
        + obligations
        + restrictions
        + conditions
    )

    return {
        "keywords": keywords,
        "legal_concepts": legal_concepts,
        "legal_actions": legal_actions,
        "rights": rights,
        "obligations": obligations,
        "restrictions": restrictions,
        "conditions": conditions,
        "dates": dates,
        "durations": _dedupe_preserve_order(
            [
                value
                for value in numeric_candidates
                if re.search(
                    r"\b(?:days?|months?|years?|weeks?|hours?)\b",
                    value,
                    re.IGNORECASE,
                )
            ]
        ),
        "monetary_values": money,
        "percentages": percentages,
        "numeric_values": numeric_values,
    }


# --------------------------------------------------------------------------
# Local Qwen via Ollama
# --------------------------------------------------------------------------
def call_ollama(prompt: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, OLLAMA_MAX_RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    # This machine needs explicit CPU + mmap settings to
                    # avoid GPU/CPU allocation failures.
                    "options": {
                        "temperature": 0,
                        "num_gpu": 0,
                        "use_mmap": True,
                        "num_ctx": 4096,
                    },
                },
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["response"])
        except Exception as exc:
            last_error = exc
            if attempt < OLLAMA_MAX_RETRY_ATTEMPTS:
                time.sleep(2 * attempt)

    raise RuntimeError(
        "Ollama call failed after "
        f"{OLLAMA_MAX_RETRY_ATTEMPTS} attempts. "
        f"Is `ollama serve` running and is '{OLLAMA_MODEL}' pulled?"
    ) from last_error


def parse_prediction(raw_response: str, labels: list[str]) -> str:
    try:
        parsed = json.loads(raw_response)
        candidate = str(parsed.get("clause_type", "")).strip()
        for label in labels:
            if candidate.lower() == label.lower():
                return label
    except (json.JSONDecodeError, AttributeError):
        pass

    match = re.search(r"\{.*?\}", raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            candidate = str(parsed.get("clause_type", "")).strip()
            for label in labels:
                if candidate.lower() == label.lower():
                    return label
        except (json.JSONDecodeError, AttributeError):
            pass

    found = [label for label in labels if label.lower() in raw_response.lower()]
    if found:
        return max(found, key=len)

    return "UNKNOWN"


# --------------------------------------------------------------------------
# Orchestrator: retrieval + feature extraction + prompt + generation
# --------------------------------------------------------------------------
def classify_clause(
    clause_text: str,
    query_vector: list[float],
    qdrant_client: QdrantClient,
    labels: list[str],
    label_definitions: dict[str, str],
    top_k: int = TOP_K,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    _emit(progress_callback, "search", "Searching similar CUAD examples in Qdrant")
    retrieved_examples = retrieve_similar(qdrant_client, query_vector, top_k=top_k)
    _emit(
        progress_callback,
        "search",
        f"Qdrant search complete ({len(retrieved_examples)} matches)",
        matches=len(retrieved_examples),
    )
    extracted_features = extract_legal_information(clause_text)
    prompt = build_prompt(
        clause_text=clause_text,
        examples=retrieved_examples,
        labels=labels,
        label_definitions=label_definitions,
        extracted_features=extracted_features,
    )
    _emit(progress_callback, "classification", "Running the local Ollama classifier")
    raw_response = call_ollama(prompt)
    predicted_label = parse_prediction(raw_response, labels)
    _emit(
        progress_callback,
        "classification",
        f"Classification complete: {predicted_label}",
        label=predicted_label,
    )

    return {
        "predicted_label": predicted_label,
        "raw_response": raw_response,
        "retrieved_examples": retrieved_examples,
        "retrieved_labels": [ex["clause_type"] for ex in retrieved_examples],
        "retrieved_scores": [round(float(ex["score"]), 4) for ex in retrieved_examples],
        "extracted_features": extracted_features,
        "prompt": prompt,
    }
