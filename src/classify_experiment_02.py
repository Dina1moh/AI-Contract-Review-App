"""
Experiment 2 CUAD clause classifier.

Pipeline:
    clause text
        -> Cohere query embedding
        -> Qdrant Top-K retrieval
        -> deterministic legal-information extraction
        -> CUAD label definitions
        -> retrieved TRAIN evidence
        -> enhanced prompt
        -> local Qwen 2.5 1.5B via Ollama
        -> one CUAD label
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cohere
import requests
from qdrant_client import QdrantClient, models

try:
    import config
except ModuleNotFoundError:  # pragma: no cover - fallback for package-style imports
    from src import config


DEFAULT_LABEL_DEFINITIONS: dict[str, str] = {
    "Affiliate License-Licensee": (
        "License rights are granted to the licensee's affiliates, "
        "subsidiaries, parent companies, or related entities."
    ),
    "Affiliate License-Licensor": (
        "License rights are granted to, reserved for, or shared with the "
        "licensor's affiliates, subsidiaries, parent companies, or related "
        "entities."
    ),
    "Anti-Assignment": (
        "A clause restricting assignment, delegation, transfer, or "
        "subcontracting of the agreement or its rights and duties."
    ),
    "Audit Rights": (
        "A clause giving one party the right to inspect, audit, or review "
        "books, records, systems, or compliance."
    ),
    "Cap On Liability": (
        "A clause setting an express maximum amount or formula that limits "
        "liability exposure."
    ),
    "Change Of Control": (
        "A clause triggered by a merger, acquisition, ownership change, or "
        "similar change in control."
    ),
    "Competitive Restriction Exception": (
        "A clause that carves out permitted competitive activity from an "
        "otherwise restrictive non-compete or exclusivity provision."
    ),
    "Covenant Not To Sue": (
        "A clause where a party promises not to sue, assert claims, or bring "
        "infringement actions."
    ),
    "Exclusivity": (
        "A clause making a relationship exclusive, sole, or restricted to "
        "one counterparty, channel, product, or territory."
    ),
    "Governing Law": (
        "A clause specifying the law, jurisdiction, or venue that governs "
        "interpretation or disputes."
    ),
    "Insurance": (
        "A clause requiring insurance coverage, policy types, limits, "
        "certificates, or proof of insurance."
    ),
    "Ip Ownership Assignment": (
        "A clause assigning or transferring intellectual property ownership "
        "or future IP rights to a party."
    ),
    "Irrevocable Or Perpetual License": (
        "A clause granting a license that is irrevocable, perpetual, "
        "permanent, or continuing indefinitely."
    ),
    "Joint Ip Ownership": (
        "A clause stating that intellectual property is jointly owned by two "
        "or more parties."
    ),
    "License Grant": (
        "A clause that grants a license, right, permission, or authorization "
        "to use specified IP, data, or materials."
    ),
    "Liquidated Damages": (
        "A clause setting a pre-agreed damages amount or formula for a "
        "specified breach or delay."
    ),
    "Minimum Commitment": (
        "A clause requiring a minimum purchase, spend, volume, output, or "
        "other contractual commitment."
    ),
    "Most Favored Nation": (
        "A clause requiring parity with the best price, terms, or treatment "
        "given to another counterparty."
    ),
    "No-Solicit Of Customers": (
        "A clause prohibiting solicitation of customers, clients, or "
        "accounts."
    ),
    "No-Solicit Of Employees": (
        "A clause prohibiting solicitation, hiring, or recruitment of "
        "employees, contractors, or personnel."
    ),
    "Non-Compete": (
        "A clause prohibiting competing business activity, products, "
        "services, or market participation."
    ),
    "Non-Transferable License": (
        "A clause stating that a license may not be assigned, transferred, or "
        "sublicensed."
    ),
    "Notice Period To Terminate Renewal": (
        "A clause requiring advance notice to stop automatic renewal or to "
        "terminate at the end of a renewal period."
    ),
    "Post-Termination Services": (
        "A clause requiring transition assistance, wind-down support, or "
        "continuing services after termination."
    ),
    "Price Restrictions": (
        "A clause limiting pricing, discounts, resale price, or price "
        "changes."
    ),
    "Renewal Term": (
        "A clause defining an automatic or optional renewal period or "
        "extension term."
    ),
    "Revenue/Profit Sharing": (
        "A clause allocating revenue, profit, royalties, commissions, or "
        "similar proceeds between parties."
    ),
    "Rofr/Rofo/Rofn": (
        "A clause giving a right of first refusal, right of first offer, or "
        "right of first negotiation."
    ),
    "Source Code Escrow": (
        "A clause requiring source code or related materials to be placed in "
        "escrow and released on defined triggers."
    ),
    "Termination For Convenience": (
        "A clause allowing a party to terminate without cause, often on "
        "notice and at its discretion."
    ),
    "Third Party Beneficiary": (
        "A clause giving enforceable rights to a non-party beneficiary."
    ),
    "Uncapped Liability": (
        "A clause stating that liability is uncapped, unlimited, or not "
        "subject to a monetary limit."
    ),
    "Unlimited/All-You-Can-Eat-License": (
        "A clause granting unlimited, broad, or uncapped usage rights without "
        "a usage cap or quantity limit."
    ),
    "Volume Restriction": (
        "A clause limiting quantity, volume, usage, production, or throughput."
    ),
    "Warranty Duration": (
        "A clause stating the duration or period of a warranty or warranty "
        "coverage."
    ),
}


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


def make_cohere_client() -> cohere.ClientV2:
    if not config.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is missing from .env")
    return cohere.ClientV2(api_key=config.COHERE_API_KEY)


def make_qdrant_client() -> QdrantClient:
    if config.QDRANT_URL:
        if config.QDRANT_API_KEY:
            return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)


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
        "termination",
        "terminate",
        "renewal",
        "renew",
        "notice",
        "written notice",
        "license",
        "licence",
        "assignment",
        "assign",
        "transfer",
        "liability",
        "warranty",
        "insurance",
        "audit",
        "exclusivity",
        "exclusive",
        "royalty",
        "revenue",
        "profit",
        "price",
        "discount",
        "non-compete",
        "non-solicit",
        "source code escrow",
        "liquidated damages",
        "minimum commitment",
        "first refusal",
        "first offer",
        "first negotiation",
        "governing law",
        "change of control",
        "perpetual",
        "irrevocable",
        "unlimited",
        "uncapped",
        "volume",
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


def embed_queries(
    cohere_client: cohere.ClientV2,
    texts: list[str],
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), config.COHERE_EMBED_BATCH_SIZE):
        batch = texts[start : start + config.COHERE_EMBED_BATCH_SIZE]
        response = None
        delay = 5

        for attempt in range(1, config.COHERE_MAX_RETRY_ATTEMPTS + 1):
            try:
                response = cohere_client.embed(
                    model=config.COHERE_MODEL,
                    input_type="search_query",
                    texts=batch,
                    embedding_types=["float"],
                )
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code != 429 or attempt == config.COHERE_MAX_RETRY_ATTEMPTS:
                    raise
                print(
                    f"Rate limited by Cohere; retrying query batch in {delay}s "
                    f"(attempt {attempt}/{config.COHERE_MAX_RETRY_ATTEMPTS})."
                )
                time.sleep(delay)
                delay *= 2

        if response is None:
            raise RuntimeError("Cohere did not return an embedding response.")

        all_embeddings.extend(response.embeddings.float)

    return all_embeddings


def retrieve_similar(
    qdrant_client: QdrantClient,
    query_vector: list[float],
    top_k: int = config.TOP_K,
) -> list[dict]:
    response = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="is_metadata",
                    match=models.MatchValue(value=True),
                )
            ]
        ),
    )
    hits = getattr(response, "points", response)
    return [
        {
            "score": hit.score,
            "clause_type": hit.payload["clause_type"],
            "clause_text": hit.payload["clause_text"],
        }
        for hit in hits
    ]


def load_label_definitions(labels: list[str]) -> dict[str, str]:
    path = config.LABEL_DEFINITIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, str] = {}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path} must contain a JSON object mapping labels to definitions."
            )
        existing = {str(k): str(v) for k, v in loaded.items()}

    missing = [label for label in labels if label not in existing]
    extra = [label for label in existing if label not in labels]

    if missing:
        for label in missing:
            definition = DEFAULT_LABEL_DEFINITIONS.get(label)
            if not definition:
                raise ValueError(
                    f"Missing definition for label '{label}' and no fallback was provided."
                )
            existing[label] = definition

        ordered = {label: existing[label] for label in labels}
        path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        existing = ordered

    if extra:
        raise ValueError(
            f"{path} contains unexpected labels not present in config.load_labels(): "
            + ", ".join(sorted(extra))
        )

    missing_after = [label for label in labels if label not in existing]
    if missing_after:
        raise ValueError(
            "Label-definition coverage is incomplete after repair: "
            + ", ".join(missing_after)
        )

    return {label: existing[label] for label in labels}


def _format_examples(examples: list[dict]) -> str:
    blocks: list[str] = []
    for index, example in enumerate(examples, start=1):
        score = example["score"]
        clause_type = example["clause_type"]
        clause_text = example["clause_text"]
        blocks.append(
            f"{index}. Label: {clause_type}\n"
            f"   Retrieval score: {score:.4f}\n"
            f"   Clause: \"{clause_text}\""
        )
    return "\n\n".join(blocks)


def _format_label_definitions(label_definitions: dict[str, str]) -> str:
    blocks: list[str] = []
    for label in label_definitions:
        blocks.append(f"- {label}: {label_definitions[label]}")
    return "\n".join(blocks)


def build_prompt(
    clause_text: str,
    examples: list[dict],
    labels: list[str],
    label_definitions: dict[str, str],
    extracted_features: dict[str, list[str]],
) -> str:
    label_list = "\n".join(f"- {label}" for label in labels)
    example_block = _format_examples(examples) if examples else "None"
    label_definition_block = _format_label_definitions(label_definitions)
    extracted_features_json = json.dumps(
        extracted_features,
        indent=2,
        ensure_ascii=False,
    )

    return f"""You are a legal contract clause classifier for CUAD.

Task:
Identify the MAIN LEGAL FUNCTION of the clause and return exactly one valid CUAD label.

Decision rules:
- Use the clause text, the extracted legal information, the label definitions, and the retrieved training examples together.
- Use retrieved examples as evidence, not as automatic answers.
- Distinguish rights, obligations, restrictions, permissions, conditions, limitations, termination mechanisms, and financial commitments.
- Prefer the primary contractual function when several concepts appear.
- Do not rely on keyword matching alone.
- Do not invent facts.
- Do not invent labels.
- Return exactly one label from the allowed list.
- Return ONLY a JSON object in this exact format:
  {{"clause_type": "<one allowed CUAD label>"}}

Original clause:
"{clause_text}"

Allowed CUAD labels:
{label_list}

CUAD label definitions:
{label_definition_block}

Retrieved Top-K TRAIN examples:
{example_block}

Extracted legal information:
{extracted_features_json}
"""


def call_ollama(prompt: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, config.OLLAMA_MAX_RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    # The local Ollama build on this machine needs explicit
                    # CPU + mmap settings to avoid GPU/CPU allocation
                    # failures while keeping the model and temperature fixed.
                    "options": {
                        "temperature": 0,
                        "num_gpu": 0,
                        "use_mmap": True,
                        "num_ctx": 2048,
                    },
                },
                timeout=config.OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload["response"])
        except Exception as exc:
            last_error = exc
            if attempt < config.OLLAMA_MAX_RETRY_ATTEMPTS:
                time.sleep(2 * attempt)

    raise RuntimeError(
        "Ollama call failed after "
        f"{config.OLLAMA_MAX_RETRY_ATTEMPTS} attempts. "
        f"Is `ollama serve` running and is '{config.OLLAMA_MODEL}' pulled?"
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


def classify_clause(
    clause_text: str,
    query_vector: list[float],
    qdrant_client: QdrantClient,
    labels: list[str],
    label_definitions: dict[str, str],
    top_k: int = config.TOP_K,
) -> dict:
    retrieved_examples = retrieve_similar(
        qdrant_client,
        query_vector,
        top_k=top_k,
    )
    extracted_features = extract_legal_information(clause_text)
    prompt = build_prompt(
        clause_text=clause_text,
        examples=retrieved_examples,
        labels=labels,
        label_definitions=label_definitions,
        extracted_features=extracted_features,
    )
    raw_response = call_ollama(prompt)
    predicted_label = parse_prediction(raw_response, labels)

    return {
        "predicted_label": predicted_label,
        "raw_response": raw_response,
        "retrieved_examples": retrieved_examples,
        "retrieved_labels": [ex["clause_type"] for ex in retrieved_examples],
        "retrieved_scores": [round(float(ex["score"]), 4) for ex in retrieved_examples],
        "extracted_features": extracted_features,
        "prompt": prompt,
    }


if __name__ == "__main__":
    import pandas as pd

    labels = config.load_labels()
    label_definitions = load_label_definitions(labels)

    test_df = pd.read_csv(config.TEST_DATA_PATH)
    if "is_metadata" not in test_df.columns:
        test_df["is_metadata"] = False
    test_df = test_df[~test_df["is_metadata"].astype(bool)].reset_index(drop=True)
    smoke_row = test_df.sample(n=min(200, len(test_df)), random_state=42).iloc[0]

    print(f"Loaded {len(labels)} labels.")
    print(f"Smoke-test clause: {smoke_row.document_id}")
    print(f"True label: {smoke_row.clause_type}")

    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()
    [query_vector] = embed_queries(cohere_client, [str(smoke_row.clause_text)])
    result = classify_clause(
        clause_text=str(smoke_row.clause_text),
        query_vector=query_vector,
        qdrant_client=qdrant_client,
        labels=labels,
        label_definitions=label_definitions,
        top_k=config.TOP_K,
    )

    if result["predicted_label"] not in labels:
        raise RuntimeError(
            f"Predicted label {result['predicted_label']!r} is not in config.load_labels()."
        )

    smoke_output = {
        "document_id": smoke_row.document_id,
        "true_label": smoke_row.clause_type,
        "retrieved_labels": result["retrieved_labels"],
        "retrieved_scores": result["retrieved_scores"],
        "extracted_features": result["extracted_features"],
        "raw_response": result["raw_response"],
        "parsed_clause_type": result["predicted_label"],
    }
    print(json.dumps(smoke_output, indent=2, ensure_ascii=False))
