"""
Classify a single contract clause into one of the CUAD clause-type labels
using retrieval-augmented few-shot prompting:

    clause text
        -> Cohere query embedding
        -> top-K similar TRAIN clauses from Qdrant (as few-shot examples)
        -> prompt built from those examples
        -> local Qwen 2.5 1.5B (via Ollama) predicts the label

This mirrors the architecture: RAG for grounding + prompt tuning instead of
fine-tuning, as agreed in the meeting.
"""

from __future__ import annotations

import json
import re
import time

import cohere
import requests
from qdrant_client import QdrantClient, models

import config


# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Embedding (query side -- asymmetric to how train clauses were embedded
# with input_type="search_document" in embed_train.py)
# --------------------------------------------------------------------------
def embed_queries(cohere_client: cohere.ClientV2, texts: list[str]) -> list[list[float]]:
    """Batch-embed query texts, matching embed_train.py's retry/backoff so a
    trial-tier rate limit doesn't kill a long evaluation run.
    """
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), config.COHERE_EMBED_BATCH_SIZE):
        batch = texts[start:start + config.COHERE_EMBED_BATCH_SIZE]

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

        all_embeddings.extend(response.embeddings.float)

    return all_embeddings


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve_similar(
    qdrant_client: QdrantClient,
    query_vector: list[float],
    top_k: int = config.TOP_K,
) -> list[dict]:
    """Top-K most similar TRAIN clauses, excluding metadata rows (they were
    excluded from the index already, but the filter is kept here in case
    the collection was ever rebuilt with INCLUDE_METADATA_CLAUSES=true).
    """
    response = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="is_metadata", match=models.MatchValue(value=True)
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


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def build_prompt(clause_text: str, examples: list[dict], labels: list[str]) -> str:
    label_list = "\n".join(f"- {label}" for label in labels)

    example_blocks = "\n\n".join(
        f'Example clause:\n"{ex["clause_text"]}"\nCorrect label: {ex["clause_type"]}'
        for ex in examples
    )

    return f"""You are a legal contract clause classifier.

Your task: read the clause below and assign it EXACTLY ONE label from this
fixed list of clause types. Do not invent a label that is not in the list.

Allowed labels:
{label_list}

Here are similar clauses from other contracts with their correct label, to
guide you:

{example_blocks}

Now classify this new clause:
"{clause_text}"

Respond with ONLY a JSON object in this exact format, nothing else:
{{"clause_type": "<one label from the allowed list above>"}}"""


# --------------------------------------------------------------------------
# Local Qwen via Ollama
# --------------------------------------------------------------------------
def call_ollama(prompt: str) -> str:
    last_error = None

    for attempt in range(1, config.OLLAMA_MAX_RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={
                    "model": config.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=config.OLLAMA_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as exc:
            last_error = exc
            if attempt < config.OLLAMA_MAX_RETRY_ATTEMPTS:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Ollama call failed after {config.OLLAMA_MAX_RETRY_ATTEMPTS} attempts. "
        f"Is `ollama serve` running and is '{config.OLLAMA_MODEL}' pulled? "
        f"(ollama run {config.OLLAMA_MODEL})"
    ) from last_error


# --------------------------------------------------------------------------
# Parse the model's answer back into one of the allowed labels
# --------------------------------------------------------------------------
def parse_prediction(raw_response: str, labels: list[str]) -> str:
    # 1) Try clean JSON parsing first.
    try:
        parsed = json.loads(raw_response)
        candidate = str(parsed.get("clause_type", "")).strip()
        for label in labels:
            if candidate.lower() == label.lower():
                return label
    except (json.JSONDecodeError, AttributeError):
        pass

    # 2) Fallback: the model sometimes wraps JSON in prose or markdown
    # fences. Pull out the first {...} block and retry.
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

    # 3) Last resort: does any allowed label literally appear in the raw
    # text? Prefer the longest match to avoid partial-name collisions
    # (e.g. "License Grant" vs "Non-Transferable License").
    found = [label for label in labels if label.lower() in raw_response.lower()]
    if found:
        return max(found, key=len)

    return "UNKNOWN"


# --------------------------------------------------------------------------
# Full single-clause pipeline
# --------------------------------------------------------------------------
def classify_clause(
    clause_text: str,
    query_vector: list[float],
    qdrant_client: QdrantClient,
    labels: list[str],
    top_k: int = config.TOP_K,
) -> dict:
    examples = retrieve_similar(qdrant_client, query_vector, top_k=top_k)
    prompt = build_prompt(clause_text, examples, labels)
    raw_response = call_ollama(prompt)
    predicted_label = parse_prediction(raw_response, labels)

    return {
        "predicted_label": predicted_label,
        "raw_response": raw_response,
        "retrieved_labels": [ex["clause_type"] for ex in examples],
        "retrieved_scores": [round(ex["score"], 4) for ex in examples],
    }


if __name__ == "__main__":
    # Quick manual smoke test on one hand-written clause.
    labels = config.load_labels()
    print(f"Loaded {len(labels)} labels.")

    sample_clause = (
        "Either party may terminate this Agreement for convenience upon "
        "sixty (60) days prior written notice to the other party."
    )

    cohere_client = make_cohere_client()
    qdrant_client = make_qdrant_client()

    [query_vector] = embed_queries(cohere_client, [sample_clause])
    result = classify_clause(sample_clause, query_vector, qdrant_client, labels)

    print(json.dumps(result, indent=2, ensure_ascii=False))
